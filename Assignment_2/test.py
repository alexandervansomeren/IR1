import models
import numpy as np
import json

def main():

    # Get documents
    index = pyndri.Index('index/')
    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    with open('tfidf.npy', 'rb') as f:
        tf_idf = np.load(f)
    with open('bm25.npy', 'rb') as f:
        bm25 = np.load(f)
    with open(term2index_filename, 'r') as f:
        term2index = json.load(f)

    for query_id, query in topics.items():    
        query_indices = models.query2indices(query, term2index)
        tf_idf_score = models.tf_idf_score(tf_idf, query_indices)
        bm25_score = models.bm25_score(bm25, query_indices)
        tf_idf_ranked = np.flip(np.argsort(tf_idf_score))
        bm25_ranked = np.flip(np.argsort(bm25_score))
        tf_idf_top3 = tf_idf_ranked[0:3]
        bm25_top3 = bm25_ranked[0:3]
        print(query)
        for i in range(3):
            print(index.document(i+1)
        
if __name__ == "__main__":
    main()
