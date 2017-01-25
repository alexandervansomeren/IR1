import numpy as np
import models
import utils
import pyndri
from scipy.sparse import lil_matrix


def plm(topics, index, max_query_terms=0, max_documents=0, max_doc_len=10):
    token2id, id2token, _ = index.get_dictionary()
    query_term_ids = models.collect_query_terms(topics, token2id)

    n_docs = index.document_count()

    # Take small sample for debugging purposes
    if max_query_terms > 0:
        query_term_ids = list(query_term_ids)[0:max_query_terms]
    if max_documents > 0:
        n_docs = max_documents

    k =

    for doc_id in range(n_docs):  # doc_id is shifted to the right in tf matrix (1 -> 0)
        doc = np.array(index.document(doc_id + 1)[1][0:max_doc_len])
        plm_matrix = np.zeros([len(query_term_ids), max_doc_len])

        # Set indices for query terms to 1 if they appear
        for term_id, term in enumerate(query_term_ids):
            if term in doc:
                indices = np.where(doc == term)
                for i in indices:
                    plm_matrix[term_id, i] = 1



    # Create dictionary to retrieve index of term in tf matrix
    term2index = {id2token[term_id]: index for index, term_id in enumerate(query_term_ids)}
    return p_tf, term2index


# Get documents
index = pyndri.Index('index/')

# Get queries
with open('./ap_88_89/topics_title', 'r') as f_topics:
    topics = utils.parse_topics(f_topics)
p_tf, term2index = construct_positional_tf(topics, index, max_query_terms=0)
# with open('p_tf.npy', 'wb') as f:
#     np.save(f, p_tf)
