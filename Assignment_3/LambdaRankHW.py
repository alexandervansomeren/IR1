__author__ = 'agrotov'

import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
from utils import normalized_discounted_cumulative_gain_at_k
import query

NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95


# TODO: Implement the lambda loss function
def lambda_loss(output, lambdas):
    return output * lambdas


class LambdaRankHW:
    NUM_INSTANCES = count()

    def __init__(self, algorithm, feature_count):
        self.algorithm = algorithm
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count, 1, BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)
        if algorithm == 'lambdarank':
            self.ndcg_at_max = normalized_discounted_cumulative_gain_at_k(optimal_ranking=[1] + [0] * 999, k=1000)
            self.previous_ndcg = 1

    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):
        try:
            now = time.time()
            for epoch in self.train(train_queries):
                if epoch['number'] % 10 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                        epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                    now = time.time()
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores

    def build_model(self, input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print "input_dim", input_dim, "output_dim", output_dim
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                         X_tensor_type=T.matrix,
                         batch_size=BATCH_SIZE,
                         learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch, deterministic=True, dtype="float32")

        # TODO: Change loss function
        # Point-wise loss function (squared error) - comment it out
        if self.algorithm == 'pointwise':
            loss_train = lasagne.objectives.squared_error(output, y_batch)
        # Pairwise loss function - comment it in
        elif self.algorithm == 'pairwise' or self.algorithm == 'lambdarank':
            loss_train = lambda_loss(output, y_batch)

        loss_train = loss_train.mean()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        # L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        # L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        # loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)

        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch], output_row_det,
        )

        # (2) Training function, updates the parameters, outpust loss
        train_func = theano.function(
            [X_batch, y_batch], loss_train,
            updates=updates,
            # givens={
            #     X_batch: dataset['X_train'][batch_slice],
            #     # y_batch: dataset['y_valid'][batch_slice],
            # },
        )

        print "finished create_iter_functions"
        return dict(
            train=train_func,
            out=score_func,
        )

    # TODO: Implement the aggregate (i.e. per document) lambda function
    def lambda_function(self, labels, scores):
        if 1 in labels:
            # assumes only one relevant document (homepage finding task)
            relevant_index = np.where(labels == 1)[0][0]
            relevant_score = scores[relevant_index]
            lambdas = 1. / (1 + np.exp(relevant_score - scores))
            lambdas[relevant_index] = 0
            lambdas[relevant_index] = np.sum(-1. * lambdas)
            if self.algorithm == 'lamdarank':
                ranking = labels[np.argsort(-scores, axis=0)]
                old_ndcg = self.ndcg_at_max.compute(ranking, k=1000)
                for doc_id in range(len(labels)):
                    if not doc_id == relevant_index:
                        swapped_ranking = [0]*1000
                        swapped_ranking[doc_id] = 1.0
                        new_ndcg = self.ndcg_at_max.compute(swapped_ranking, k=1000)
                        lambdas[doc_id] *= np.abs(old_ndcg - new_ndcg)
            return np.array(lambdas, dtype='float32')
        else:
            return np.zeros(len(labels), dtype='float32')

    def compute_lambdas_theano(self, query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)])
        # if self.algorithm == 'lamdarank':
        #     scores = self.score(query)
        #     relevance_labels = query.get_labels()
        #     relevance = relevance_labels[np.argsort(-scores, axis=0)]
        #     ndcg = self.ndcg_at_max.compute(relevance, 10, normalize=True)
        #     result *= np.abs(self.previous_ndcg - ndcg)
        #     self.previous_ndcg = ndcg
        return result

    def train_once(self, X_train, query, labels):

        resize_value = BATCH_SIZE
        if self.algorithm == 'pointwise':
            resize_value = min(resize_value, len(labels))

        X_train.resize((resize_value, self.feature_count), refcheck=False)

        if self.algorithm == 'pointwise':
            batch_train_loss = self.iter_funcs['train'](X_train, labels)

        else:
            lambdas = self.compute_lambdas_theano(query, labels)
            lambdas.resize((resize_value,))

            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)

        return batch_train_loss

    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()

        queries = train_queries.values()

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in xrange(len(queries)):
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()

                batch_train_loss = self.train_once(X_trains[random_index], queries[random_index], labels)
                batch_train_losses.append(batch_train_loss)

            avg_train_loss = np.mean(batch_train_losses)

            print(avg_train_loss)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }
