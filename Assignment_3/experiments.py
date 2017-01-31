import argparse
import numpy as np
import query
import document
import LambdaRankHW


FLAGS = None


def main():
    number_of_features = 64
    number_of_epochs = 10
    algorithm = FLAGS.method

    for fold in range(1):  # 5
        fold_dir = 'HP2003/Fold' + str(fold + 1)
        train_queries = query.load_queries(fold_dir + '/train.txt', number_of_features)

        model = LambdaRankHW.LambdaRankHW(algorithm, number_of_features)
        model.train_with_queries(train_queries, number_of_epochs)

    test_queries = query.load_queries(fold_dir + '/test.txt', number_of_features)
    queries = train_queries.values()
    for q in queries:
        scores = model.score(q)
        relevance_labels = q.get_labels()
        relevance = relevance_labels[np.argsort(-scores)]
        if not 1 in relevance_labels: 
            print("no relevant docs")

        

def discounted_cumulative_gain_at_k(ranking, labels, k):
    dcg = 0.0
    for rank, relevance in enumerate(ranking[0:k]):
        dcg += float(2 ** relevance - 1) / np.log2(rank + 2)
    return dcg




if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default='pairwise',
                        help='Learning to rank method [pointwise, pairwise, lambdarank].')

    FLAGS = parser.parse_args()

    main()

    # if FLAGS.method
