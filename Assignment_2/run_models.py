import os

import pyndri
import utils
import models
import cPickle as pickle


def main():
    # Get documents
    index = pyndri.Index('index/')

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    tf_filename = 'tf.pickle'

    # Construct term frequency and document frequency
    if os.path.isfile(tf_filename):
        with open(tf_filename, 'r') as f:
            tf = pickle.load(f)
    else:
        tf = models.construct_tf(topics, index)
        with open(tf_filename, "w") as f:
            pickle.dump(tf, f)

    # df = models.construct_df(tf)

    # Run models
    # tf_idf = models.tf_idf(tf, df)
    # bm25 = models.bm25(tf, df)


if __name__ == "__main__":
    main()
