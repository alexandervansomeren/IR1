import argparse

import query
import document
import LambdaRankHW

FLAGS = None


def main():
    number_of_features = 64
    number_of_epochs = 1
    algorithm = FLAGS.method

    for fold in range(1):  # 5
        fold_dir = 'HP2003/Fold' + str(fold + 1)
        train_queries = query.load_queries(fold_dir + '/train.txt', number_of_features)

        model = LambdaRankHW.LambdaRankHW(algorithm, number_of_features)
        model.train_with_queries(train_queries, number_of_epochs)

        test_queries = query.load_queries(fold_dir + '/test.txt', number_of_features)








if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default='pairwise',
                        help='Learning to rank method [pointwise, pairwise, lambdarank].')

    FLAGS = parser.parse_args()

    main()

    # if FLAGS.method
