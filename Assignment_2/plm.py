import numpy as np

import models
import utils
import pyndri
from scipy.sparse import lil_matrix


def plm(topics, index, max_query_terms=0, max_documents=0, max_doc_len=100):
    token2id, id2token, _ = index.get_dictionary()
    query_term_ids = models.collect_query_terms(topics, token2id)

    n_docs = index.document_count()

    # Take small sample for debugging purposes
    if max_query_terms > 0:
        query_term_ids = list(query_term_ids)[0:max_query_terms]
    if max_documents > 0:
        n_docs = max_documents

    k = gaussian_kernel(size=99, sigma=50)

    for doc_id in range(n_docs):  # doc_id is shifted to the right in tf matrix (1 -> 0)
        doc = np.array(index.document(doc_id + 1)[1][0:max_doc_len])
        plm_matrix = np.zeros([len(query_term_ids), max_doc_len])

        # Set indices for query terms to 1 if they appear
        for term_id, term in enumerate(query_term_ids):
            if term in doc:
                indices = np.where(doc == term)
                for i in indices:
                    plm_matrix[term_id, i] = 1

        # normalize
        print(np.apply_along_axis(convolve_or_skip, 1, plm_matrix, k))

        break


def convolve_or_skip(a, k):
    if np.count_nonzero(a) > 1:
        return np.convolve(a, k)


# Kernels
def gaussian_kernel(size=50, sigma=50):
    i = size / 2  # center
    k = np.zeros(size)
    for j in range(size):
        k[j] = np.exp((-(float(i) - j) ** 2) / (2 * float(sigma) ** 2))
    return k


def triangle_kernel(size=50, sigma=50):
    i = size / 2  # center
    k = np.zeros(size)
    for j in range(size):
        if (i - j) > sigma:
            k[j] = 1 - (np.abs(i - j) / sigma)
    return k


def cosine_hamming_kernel(size=50, sigma=50):
    i = size / 2  # center
    k = np.zeros(size)
    for j in range(size):
        if (i - j) > sigma:
            k[j] = 0.5 * (1 + np.cos(((np.pi * np.abs(i - j)) / sigma)))
    return k


def circle_kernel(size=50, sigma=50):
    i = size / 2  # center
    k = np.zeros(size)
    for j in range(size):
        if (i - j) > sigma:
            k[j] = np.sqrt(1 - (np.abs(i - j) / sigma) ** 2)
    return k


def passage_kernel(size=50, sigma=50):
    i = size / 2  # center
    k = np.zeros(size)
    for j in range(size):
        if (i - j) > sigma:
            k[j] = 1
    return k

# Get documents
index = pyndri.Index('index/')

# Get queries
with open('./ap_88_89/topics_title', 'r') as f_topics:
    topics = utils.parse_topics(f_topics)
plm(topics, index, max_query_terms=0)
# with open('p_tf.npy', 'wb') as f:
#     np.save(f, p_tf)
