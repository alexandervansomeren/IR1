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

    doc_names = []
    for i in range(index.document_base(), index.maximum_document()):
        doc_names.append(index.document(i)[0])

    tfidf_results = {}
    bm25_results = {}
    for query_id, query in topics.items():    
        query_indices = models.query2indices(query, term2index)
        tfidf_score = models.tf_idf_score(tf_idf, query_indices)
        bm25_score = models.bm25_score(bm25, query_indices)
        tfidf_results[query_id] = list(zip(tfidf_score, doc_names))
        bm25_results[query_id] = list(zip(bm25_score, doc_names))

    utils.write_run(model_name='tfidf', data=tfidf_results, 
                    out_f='ranking_tfidf.txt', max_objects_per_query=1000)
    utils.write_run(model_name='bm25', data=bm25_results,
                    out_f='ranking_bm25.txt', max_objects_per_query=1000)

# trec_eval -m all_trec -q ap_88_89/qrel_validation ranking_tfidf.txt | grep -E "^map\s"
# trec_eval -m all_trec -q ap_88_89/qrel_validation ranking_bm25.txt | grep -E "^map\s"
# trec_eval -m all_trec -q ap_88_89/qrel_test ranking_tfidf.txt | grep -E "^map\s"
# trec_eval -m all_trec -q ap_88_89/qrel_test ranking_bm25.txt | grep -E "^map\s"

if __name__ == "__main__":
    main()
