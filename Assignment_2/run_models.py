import os
import pyndri
import utils
import models
import numpy as np
import json


def main():
    # Get documents
    index = pyndri.Index('index/')

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    tf_filename = 'tf.npy'
    term2index_filename = 'term2index.json'

    # Construct term frequency and document frequency
    if os.path.isfile(tf_filename) and os.path.isfile(term2index_filename):
        with open(tf_filename, 'rb') as f:
            tf = np.load(f)
        with open(term2index_filename, 'r') as f:
            term2index = json.load(f)
    else:
        tf, term2index = models.construct_tf(topics, index, max_query_terms=0, max_documents=500)
        with open(tf_filename, 'wb') as f:
            np.save(f, tf)
        with open(term2index_filename, 'w') as f:
            json.dump(term2index, f)

    df = (tf > 0).sum(axis=1)
    if 0 in df:
        df = float(df)
        df += 0.001        
    idf = np.log(float(tf.shape[1]) / df)

    # Run models
    tf_idf = models.tf_idf(tf, idf)
    # bm25 = models.bm25(tf, df, 1.2, 0.75)


if __name__ == "__main__":
    main()
