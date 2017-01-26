import json
import os
import argparse
import pyndri
import gensim
import utils
import numpy as np
import lsm_models
import connector_classes
import models

FLAGS = None


def run_w2v(index, doc_names, topics, embedding_size, max_documents):
    print("Building / loading word2vec")
    wv2_model_filename = 'models/word2vec' + str(embedding_size) + '.model'
    if os.path.isfile(wv2_model_filename):
        w2v = lsm_models.Word2Vec(filename=wv2_model_filename,
                                  embedding_size=embedding_size,
                                  max_documents=max_documents)
    else:
        w2v = lsm_models.Word2Vec(embedding_size=embedding_size,
                                  max_documents=max_documents)
        w2v.train(index)

    print('Size Word2Vec model')
    print(len(w2v.model.wv.vocab))

    # Only for top 1000 tfidf!
    print("Building document representations")
    docs_representation_filename = 'tmp/doc2vecs' + str(embedding_size) + '.npy'
    if os.path.isfile(docs_representation_filename):
        with open(docs_representation_filename, 'rb') as f:
            docs_representation = np.load(f)
    else:
        docs_representation = w2v.docs2vec(index)
        with open(docs_representation_filename, 'wb') as f:
            np.save(f, docs_representation)

    f_tfidf = 'tfidf.npy'
    with open(f_tfidf, 'rb') as f:
        tf_idf = np.load(f)

    with open('term2index.json', 'r') as f:
        term2index = json.load(f)

    print("Scoring documents")
    w2v_results = {}
    for query_id, query in topics.items():
        # Get top 1000 tf-idf
        query_indices = models.query2indices(query, term2index)
        tf_idf_score = models.tf_idf_score(tf_idf, query_indices)
        tf_idf_ranked_doc_indices = np.argsort(-tf_idf_score)
        best_1000_doc_indices = tf_idf_ranked_doc_indices[0:1000]

        # Get query word2vec representation
        query_representation = w2v.query2vec(query)
        # Calculate the similarity with documents
        w2v_score = utils.cosine_similarity(query_representation, docs_representation[:, best_1000_doc_indices])
        w2v_results[query_id] = list(zip(w2v_score, doc_names))

        # print(query)
        # top_doc = index.document(np.argmax(w2v_score)+1)[1]
        # line = str(' ')
        # for word_id in top_doc:
        #    line = line + str(id2token.get(word_id,0)) + ' '
        # print(line)

    # Save results to file
    utils.write_run(model_name='w2v'+ str(embedding_size), data=w2v_results,
                    out_f='results/ranking_w2v' + str(embedding_size) + '.txt', max_objects_per_query=1000)


def run_lsi(index, doc_names, topics, num_topics):
    print("Building / loading LSI")
    dictionary = pyndri.extract_dictionary(index)
    corpus = connector_classes.IndriCorpus(index, dictionary)
    lsi_model_filename = 'models/lsi.model'
    if os.path.isfile(lsi_model_filename):
        lsi = lsm_models.LSI(filename=lsi_model_filename,
                             num_topics=num_topics)
    else:
        lsi = lsm_models.LSI(corpus=corpus,
                             num_topics=num_topics)
        lsi.save(lsi_model_filename)

    print("Scoring documents")
    lsi_results = {}
    # for query_id, query in topics.items():
    # Get query word2vec representation
    # query_representation = w2v.query2vec(query)
    # Calculate the similarity with documents
    # w2v_score = utils.cosine_similarity(query_representation, docs_representation)
    # w2v_results[query_id] = list(zip(w2v_score, doc_names))


    # Save results to file
    # utils.write_run(model_name='w2v', data=w2v_results,
    #                out_f='results/ranking_w2v.txt', max_objects_per_query=1000)


def run_lda(index, doc_names, topics, num_topics):
    print("Building / loading LDA")
    dictionary = pyndri.extract_dictionary(index)
    corpus = connector_classes.IndriCorpus(index, dictionary)
    lda_model_filename = 'models/lda.model'
    if os.path.isfile(lda_model_filename):
        lda = lsm_models.LDA(filename=lda_model_filename,
                             num_topics=num_topics)
    else:
        lda = lsm_models.LDA(corpus=corpus,
                             num_topics=num_topics)
        lda.save(ldi_model_filename)

    print("Scoring documents")
    lda_results = {}


def initialize_folders():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists('results'):
        os.makedirs('results')


def main():
    # Get documents
    index = pyndri.Index('index/')
    token2id, id2token, _ = index.get_dictionary()
    doc_names = utils.get_document_names(index)

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    # Create directories if they do not exist
    initialize_folders()

    # Run LSM for command line argument method
    # loop over 300
    if FLAGS.method == 'word2vec':
        for embedding_size in [50, 100, 150, 200]:
            run_w2v(index, doc_names, topics, embedding_size, index.document_count())
    elif FLAGS.method == 'lsi':
        run_lsi(index, doc_names, topics, 20)
    elif FLAGS.method == 'lda':
        run_lda(index, doc_names, topics, 20)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default='lsi',
                        help='Latent semanctic model [word2vec, lsi, lda, doc2vec].')

    FLAGS = parser.parse_args()

    main()
