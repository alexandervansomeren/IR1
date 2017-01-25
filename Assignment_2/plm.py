import numpy as np
import models
import utils
import pyndri
from scipy.sparse import lil_matrix


def construct_positional_tf(topics, index, max_query_terms=0, max_documents=10000, max_doc_len=50):
    token2id, id2token, _ = index.get_dictionary()
    query_term_ids = models.collect_query_terms(topics, token2id)

    n_docs = index.document_count()
    # Take small sample for debugging purposes
    if max_query_terms > 0:
        query_term_ids = list(query_term_ids)[0:max_query_terms]
    if max_documents > 0:
        n_docs = max_documents

    # Count term frequencies per query term for each document
    p_tf = np.zeros([len(query_term_ids), max_doc_len, n_docs])
    for doc_id in range(n_docs):  # doc_id is shifted to the right in tf matrix (1 -> 0)
        for term_id, term in enumerate(query_term_ids):
            if term in index.document(doc_id + 1)[1]:
                print(doc_id + 1[1])
            break
            # p_tf[term_id, doc_id] = index.document(doc_id + 1)[1].count(term)
        break
    # Create dictionary to retrieve index of term in tf matrix
    term2index = {id2token[term_id]: index for index, term_id in enumerate(query_term_ids)}
    return p_tf, term2index


# Get documents
index = pyndri.Index('index/')

# Get queries
with open('./ap_88_89/topics_title', 'r') as f_topics:
    topics = utils.parse_topics(f_topics)
construct_positional_tf(topics, index, max_query_terms=0, max_documents=10000)
