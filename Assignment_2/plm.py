import json

import numpy as np

import models
import utils
import pyndri
import collections
from scipy.sparse import lil_matrix


def plm(index, word_prior, best_doc_indices, query_word_ids, k):
    scores_per_doc = {}
    for doc_id in best_doc_indices:  # doc_id is shifted to the right in tf matrix (1 -> 0)
        doc = np.array(index.document(doc_id + 1)[1])
        k = gaussian_kernel(size=len(doc), sigma=50)

        plm_matrix = np.zeros([len(query_word_ids), len(doc)])
        # Set indices for query terms to 1 if they appear
        for term_index, term_id in enumerate(query_word_ids):
            if term_id in doc:
                indices = np.where(doc == term_id)
                for i in indices:
                    plm_matrix[term_index, i] = 1

        # Apply kernel to get propagated count
        plm_matrix = np.apply_along_axis(convolve_or_skip, 1, plm_matrix, k)

        # Apply Dirichlet Prior Smoothing
        mu = 150
        plm_matrix = (plm_matrix + (mu * word_prior[:, np.newaxis])) / (plm_matrix.sum(axis=0) + mu)

        # Assume uniform P(w|Q), so scoring will be based the sum of matrix
        scores_per_index = plm_matrix.sum(axis=0)

        # Best position strategy
        score = scores_per_index.max()

        scores_per_doc[doc_id] = score
    # Sort documents by score
    ordered_scores_per_doc = sorted(scores_per_doc.items(), key=lambda x: -x[1])
    doc_ids = [d[0] for d in ordered_scores_per_doc]
    scores = [d[1] for d in ordered_scores_per_doc]
    return doc_ids, scores


# Apply convolution or skip if the row contains only zeros
def convolve_or_skip(a, k):
    if np.count_nonzero(a) > 1:
        return np.convolve(a, k, mode='same')
    else:
        return a


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

with open('tf.npy', 'rb') as f:
    tf = np.load(f)

f_tfidf = 'tfidf.npy'
with open(f_tfidf, 'rb') as f:
    tf_idf = np.load(f)

with open('term2index.json', 'r') as f:
    term2index = json.load(f)

word_prior = tf.sum(axis=1) / tf.sum()
token2id, id2token, _ = index.get_dictionary()

# Kernal
kernel_name = "gaussian"

if kernel_name == "gaussian":
    k = gaussian_kernel(size=99, sigma=50)
elif kernel_name == "triangle":
    k = triangle_kernel(size=99, sigma=50)
elif kernel_name == "cosine_hamming":
    k = cosine_hamming_kernel(size=99, sigma=50)
elif kernel_name == "circle":
    k = circle_kernel(size=99, sigma=50)

results = {}
for query_id, query in topics.items():
    # Get top 100 tf-idf
    query_indices = models.query2indices(query, term2index)
    tf_idf_score = models.tf_idf_score(tf_idf, query_indices)
    tf_idf_ranked_doc_indices = np.argsort(-tf_idf_score)
    best_1000_doc_indices = tf_idf_ranked_doc_indices[0:1000]
    if len(query_indices) != 0:
        print("ZERO LENGTH QUERY!")
        query_word_ids = models.query2word_ids(query, token2id)
        doc_ids, scores = plm(index, np.array([word_prior[i] for i in query_indices]), best_1000_doc_indices,
                              query_word_ids, k)
        doc_names = [index.document(doc_id + 1)[0] for doc_id in doc_ids]
        results[query_id] = list(zip(scores, doc_names))
model_name = 'plm' + kernel_name
utils.write_run(model_name=model_name, data=results,
                out_f='results/ranking_' + model_name + '.txt', max_objects_per_query=1000)
