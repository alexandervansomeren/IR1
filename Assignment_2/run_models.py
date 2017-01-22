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
        with open(tf_filename, 'rb') as f:
            tf = np.load(f)
    else:
        tf = models.construct_tf(topics, index, max_query_terms=0, max_documents=0)
        with open(tf_filename, "wb") as f:
            np.save(f, tf)

    query_terms = models.collect_query_terms(topics, token2id)
    token2tf_index = {id2token[term_id]: index for index, term_id in enumerate(query_terms)}

    df = (tf > 0).sum(axis=1)

    # Run models
    tf_idf = models.tf_idf(tf, df)
    # bm25 = models.bm25(tf, df, 1.2, 0.75)


if __name__ == "__main__":
    main()
