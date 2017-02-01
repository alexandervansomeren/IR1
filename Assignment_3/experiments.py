import argparse
import numpy as np
import query
import document
import LambdaRankHW
import pickle
import os
from utils import normalized_discounted_cumulative_gain_at_k
import pandas as pd

FLAGS = None


def main(algorithm, number_of_features=64, number_of_epochs=10):
    ndcg_at_10 = normalized_discounted_cumulative_gain_at_k(optimal_ranking=[1] + [0] * 10, k=10)

    ndcgs = []
    for fold in range(5):  # 5
        fold_dir = 'HP2003/Fold' + str(fold + 1)

        fname = algorithm + '_' \
                + str(number_of_epochs) + '_epochs_' \
                + str(number_of_features) + '_features_fold' + str(fold) + '.pickle'

        # Train
        if not os.path.isfile(fname):
            train_queries = query.load_queries(fold_dir + '/train.txt', number_of_features)

            model = LambdaRankHW.LambdaRankHW(algorithm, number_of_features)
            model.train_with_queries(train_queries, number_of_epochs)
            with open(fname, 'wb') as f:
                pickle.dump(model, f)
        else:
            with open(fname, 'rb') as f:
                model = pickle.load(f)

        # Validation
        validation_queries = query.load_queries(fold_dir + '/vali.txt', number_of_features)
        queries = validation_queries.values()
        for q in queries:
            scores = model.score(q)
            relevance_labels = q.get_labels()
            relevance = relevance_labels[np.argsort(-scores, axis=0)]
            if not 1 in relevance_labels:
                print("no relevant docs")
            ndcgs.append(ndcg_at_10.compute(relevance, 10, normalize=True))

            # Test
            # test_queries = query.load_queries(fold_dir + '/test.txt', number_of_features)
            # queries = test_queries.values()
            # for q in queries:
            #     scores = model.score(q)
            #     relevance_labels = q.get_labels()
            #     relevance = relevance_labels[np.argsort(-scores, axis=0)]
            #     if not 1 in relevance_labels:
            #         print("no relevant docs")
            #     ndcgs.append(ndcg_at_10.compute(relevance, 10, normalize=True))
        print("Average Fold")
        print(np.mean(np.array(ndcgs)))

    ndcg_at_10 = np.array(ndcgs)
    print '----------------'
    print 'Algorithm:       ' + algorithm
    print 'Epochs:          ' + str(number_of_epochs)
    print 'Average NDCG@10: ' + str(np.mean(ndcg_at_10))
    return np.mean(ndcg_at_10)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default='pointwise',
                        help='Learning to rank method [pointwise, pairwise, lambdarank].')

    FLAGS = parser.parse_args()

    algorithm = FLAGS.method

    result = main('lambdarank')
#      results_pointwise = {}
#      results_pairwise = {}
#      for n_epochs in range(0, 110, 10):
#          results_pointwise[n_epochs] = main('pointwise', number_of_epochs=n_epochs)
#          results_pairwise[n_epochs] = main('pairwise', number_of_epochs=n_epochs)
#      result_pointwise = pd.DataFrame.from_dict(results_pointwise, orient='index')
#      result_pointwise.columns = ['pointwise']
#      result_pairwise = pd.DataFrame.from_dict(results_pairwise, orient='index')
#      result_pairwise.columns = ['pairwise']
#      results = pd.concat([result_pairwise, result_pointwise], axis=1, join='inner')
#      results.to_pickle('epoch_validation_results.pickle')
