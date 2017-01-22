import os

import pyndri
import utils
import models
import numpy as np


def main():
    # Get documents
    index = pyndri.Index('index/')

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    tf_filename = 'tf.npy'

    # Construct term frequency and document frequency
    if os.path.isfile(tf_filename):
        with open(tf_filename, 'r') as f:
            tf = np.load(f)
    else:
        tf = models.construct_tf(topics, index)
        with open(tf_filename, "w") as f:
            np.save(f, tf)

    df = tf.sum(axis=1)

    # Run models
    tf_idf = models.tf_idf(tf)
    # bm25 = models.bm25(tf, df, 1.2, 0.75)


if __name__ == "__main__":
    main()
