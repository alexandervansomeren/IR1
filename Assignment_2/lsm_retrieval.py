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
    # Load model
    if os.path.isfile(wv2_model_filename):
        w2v = lsm_models.Word2Vec(filename=wv2_model_filename,
                                  embedding_size=embedding_size,
                                  max_documents=max_documents)
    # Train model
    else:
        w2v = lsm_models.Word2Vec(embedding_size=embedding_size,
                                  max_documents=max_documents)
        w2v.train(index)

    print("Building document representations")
    docs_representation_filename = 'tmp/doc2vecs' + str(embedding_size) + '.npy'
    if os.path.isfile(docs_representation_filename):
        with open(docs_representation_filename, 'rb') as f:
            docs_representation = np.load(f)
    else:
        docs_representation = w2v.docs2vec(index)
        with open(docs_representation_filename, 'wb') as f:
            np.save(f, docs_representation)


    print("Scoring documents")
    w2v_results = {}
    # Get top 1000 documents tf-idf ranking
    best_1000_doc_indices = utils.get_top_1000_tf_idf(topics)
    for query_id, query in topics.items():
        # Get query word2vec representation
        query_representation = w2v.query2vec(query)
        # Calculate the similarity with top 1000 document representations
        w2v_score = utils.cosine_similarity(query_representation, 
                                            docs_representation[:, best_1000_doc_indices])
        w2v_results[query_id] = list(zip(w2v_score, 
                                [doc_names[i] for i in best_1000_doc_indices]))

    # Save results to file
    utils.write_run(model_name='w2v'+ str(embedding_size), data=w2v_results,
                    out_f='results/ranking_w2v' + str(embedding_size) + '.txt', 
                    max_objects_per_query=1000)


def run_lsi(index, doc_names, topics, num_topics, max_documents):
    print("Building / loading LSI")
    dictionary = pyndri.extract_dictionary(index)
    corpus = connector_classes.IndriCorpus(index, dictionary, max_documents=max_documents)
    lsi_model_filename = 'models/lsi' + str(num_topics) + '.model'
    # Load model
    if os.path.isfile(lsi_model_filename):
        lsi = lsm_models.LSI(filename=lsi_model_filename,
                             num_topics=num_topics)
    # Train model
    else:
        lsi = lsm_models.LSI(corpus=corpus,
                             num_topics=num_topics)
        lsi.save(lsi_model_filename)

    print("Building document representations")
    docs_representation_filename = 'tmp/doc2projection' + str(num_topics) + '.npy'
    if os.path.isfile(docs_representation_filename):
        with open(docs_representation_filename, 'rb') as f:
            docs_representation = np.load(f)
    else:
        docs_representation = lsi.docs_projection(index)
        with open(docs_representation_filename, 'wb') as f:
            np.save(f, docs_representation)    

    print("Scoring documents")
    lsi_results = {}
    # Get top 1000 documents tf-idf ranking
    best_1000_doc_indices = utils.get_top_1000_tf_idf(topics)
    token2id,_,_ = index.get_dictionary()
    for query_id, query in topics.items():
        # Get projected representation for query
        query_word_ids = models.query2word_ids(query, token2id)
        query_projection = lsi.query_projection(query_word_ids)
        # Calculate the similarity with top 1000 document representations
        lsi_score = utils.cosine_similarity(query_projection,
                                            docs_representation[:, best_1000_doc_indices])
        lsi_results[query_id] = list(zip(lsi_score,
                                [doc_names[i] for i in best_1000_doc_indices]))

    # Save results to file
    utils.write_run(model_name='lsi', data=lsi_results,
                    out_f='results/ranking_lsi' + str(num_topics) + '.txt', 
                    max_objects_per_query=1000)


def run_lda(index, doc_names, topics, num_topics, max_documents):
    print("Building / loading LDA")
    dictionary = pyndri.extract_dictionary(index)
    corpus = connector_classes.IndriCorpus(index, dictionary, max_documents=max_documents)
    lda_model_filename = 'models/lda' + str(num_topics) + '.model'
    # Load model
    if os.path.isfile(lda_model_filename):
        lda = lsm_models.LDA(filename=lda_model_filename,
                             num_topics=num_topics)
    # Train model
    else:
        lda = lsm_models.LDA(corpus=corpus,
                             num_topics=num_topics)
        lda.save(lda_model_filename)

    print("Building document representations")
    docs_representation_filename = 'tmp/doc2topics' + str(num_topics) + '.npy'
    if os.path.isfile(docs_representation_filename):
        with open(docs_representation_filename, 'rb') as f:
            docs_representation = np.load(f)
    else:
        docs_representation = lda.docs2topic(index)
        with open(docs_representation_filename, 'wb') as f:
            np.save(f, docs_representation) 

    print("Scoring documents")
    lda_results = {}
    # Get top 1000 documents tf-idf ranking
    best_1000_doc_indices = utils.get_top_1000_tf_idf(topics)
    token2id,_,_ = index.get_dictionary()
    for query_id, query in topics.items():
        # Get topic distribution for query
        query_word_ids = models.query2word_ids(query, token2id)
        query_representation = lda.query2topic(query_word_ids)
        # Calculate the similarity with top 1000 document representations
        lda_score = utils.cosine_similarity(query_representation,
                                            docs_representation[:, best_1000_doc_indices])
        lda_results[query_id] = list(zip(lda_score,
                                [doc_names[i] for i in best_1000_doc_indices]))

    # Save results to file
    utils.write_run(model_name='lda', data=lda_results,
                    out_f='results/ranking_lda' + str(num_topics) + '.txt', 
                    max_objects_per_query=1000)


def run_d2v(index, doc_names, topics, size, max_documents):
    print("Building / loading Doc2Vec")
    #dictionary = pyndri.extract_dictionary(index)
    #corpus = connector_classes.IndriCorpus(index, dictionary, max_documents=max_documents)
    d2v_model_filename = 'models/doc2vec' + str(size) + '.model'
    # Load model
    if os.path.isfile(d2v_model_filename):
        d2v = lsm_models.Doc2Vec(filename=d2v_model_filename,
                             num_topics=num_topics)
    # Train model
    else:
        d2v = lsm_models.Doc2Vec(corpus=corpus,
                             num_topics=num_topics)
        d2v.save(d2v_model_filename)

    print("Building document representations")
    docs_representation_filename = 'tmp/doc2topics' + str(num_topics) + '.npy'
    if os.path.isfile(docs_representation_filename):
        with open(docs_representation_filename, 'rb') as f:
            docs_representation = np.load(f)
    else:
        docs_representation = d2v.docs2topic(index)
        with open(docs_representation_filename, 'wb') as f:
            np.save(f, docs_representation) 

    print("Scoring documents")
    d2v_results = {}
    # Get top 1000 documents tf-idf ranking
    best_1000_doc_indices = utils.get_top_1000_tf_idf(topics)
    token2id,_,_ = index.get_dictionary()
    for query_id, query in topics.items():
        # Get topic distribution for query
        query_word_ids = models.query2word_ids(query, token2id)
        query_representation = d2v.query2topic(query_word_ids)
        # Calculate the similarity with top 1000 document representations
        d2v_score = utils.cosine_similarity(query_representation,
                                            docs_representation[:, best_1000_doc_indices])
        d2v_results[query_id] = list(zip(d2v_score,
                                [doc_names[i] for i in best_1000_doc_indices]))

    # Save results to file
    utils.write_run(model_name='d2v', data=d2v_results,
                    out_f='results/ranking_d2v' + str(num_topics) + '.txt', 
                    max_objects_per_query=1000)


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
    if FLAGS.method == 'word2vec':
        for embedding_size in [50, 100, 150, 200]:
            run_w2v(index, doc_names, topics, embedding_size, index.document_count())
    elif FLAGS.method == 'lsi':
        for num_topics in [50, 100, 150, 200]:
            run_lsi(index, doc_names, topics, num_topics, index.document_count())
    elif FLAGS.method == 'lda':
        for num_topics in [50, 100, 150, 200]:
            run_lda(index, doc_names, topics, num_topics, index.document_count())
    elif FLAGS.method == 'doc2vec':
        for size in [50, 100, 150, 200]:
            run_d2v(index, doc_names, topics, size, index.document_count())

def run_all():
    # Get documents
    index = pyndri.Index('index/')
    token2id, id2token, _ = index.get_dictionary()
    doc_names = utils.get_document_names(index)

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    # Create directories if they do not exist
    initialize_folders()

    for embedding_size in [50, 100, 150, 200]:
        run_w2v(index, doc_names, topics, embedding_size, index.document_count())
    for num_topics in [50, 100, 150, 200]:
        run_lsi(index, doc_names, topics, num_topics, index.document_count())
    for num_topics in [50, 100, 150, 200]:
        run_lda(index, doc_names, topics, num_topics, index.document_count())


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default='word2vec',
                        help='Latent semanctic model [word2vec, lsi, lda, doc2vec].')

    FLAGS = parser.parse_args()

    #main()
    run_all()
