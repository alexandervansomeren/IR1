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

    # Get models
    with open('tfidf.npy', 'rb') as f:
        tf_idf = np.load(f)
    with open('bm25.npy', 'rb') as f:
        bm25 = np.load(f)
    # Get dictionary mapping terms to model index    
    with open('term2index.json', 'r') as f:
        term2index = json.load(f)

    token2id, id2token, _ = index.get_dictionary()
    for _, query in topics.items():    
        query_indices = models.query2indices(query, term2index)
        tf_idf_score = models.tf_idf_score(tf_idf, query_indices)
        bm25_score = models.bm25_score(bm25, query_indices)

if __name__ == "__main__":
    main()
