import os
import pyndri
import utils
import models
import numpy as np
import json


def main():

    # File names
    f_term_freq = 'tf.npy'
    f_term2index = 'term2index.json'
    f_tfidf = 'tfidf.npy'
    f_bm25 = 'bm25.npy'

    # Get documents
    index = pyndri.Index('index/')

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    # Construct term frequency matrix
    if os.path.isfile(f_term_freq) and os.path.isfile(f_term2index):
        with open(f_term_freq, 'rb') as f:
            tf = np.load(f)
        with open(f_term2index, 'r') as f:
            term2index = json.load(f)
    else:
        tf, term2index = models.construct_tf(topics, index, max_query_terms=0, max_documents=0)
        with open(f_term_freq, 'wb') as f:
            np.save(f, tf)
        with open(f_term2index, 'w') as f:
            json.dump(term2index, f)

    # Construct document frequency vector
    df = (tf > 0).sum(axis=1)
    if 0 in df:
        df += 1        
    idf = np.log(float(tf.shape[1]) / df)

    # Run models
    if not os.path.isfile(f_tfidf):
        tf_idf = models.tf_idf(tf, idf)
        with open('tfidf.npy', 'wb') as f:
            np.save(f, tf_idf)

    if not os.path.isfile(f_bm25):
        bm25 = models.bm25(tf, idf, 1.2, 0.75)
        with open('bm25.npy', 'wb') as f:
            np.save(f, bm25)

if __name__ == "__main__":
    main()
