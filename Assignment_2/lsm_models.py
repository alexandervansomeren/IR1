import copy
import gensim
import logging
import pyndri
import sys
import connector_classes

def word2vec_model(index, embedding_size=300):
    logging.basicConfig(level=logging.INFO)
    logging.info('Initializing word2vec.')

    model = gensim.models.Word2Vec(
        size=embedding_size,  # Embedding size
        window=5,  # One-sided window size
        sg=True,  # Skip-gram. (False for CBOW)
        min_count=5,  # Minimum word frequency.
        sample=1e-3,  # Sub-sample threshold.
        hs=False,  # Hierarchical softmax.
        negative=10,  # Number of negative examples.
        iter=1,  # Number of iterations.
        workers=8,  # Number of workers.
    )

    logging.info('Loading vocabulary.')    

    dictionary = pyndri.extract_dictionary(index)
    sentences = connector_classes.IndriSentences(index, dictionary)

    logging.info('Constructing word2vec vocabulary.')

    # Build vocab.
    model.build_vocab(sentences, trim_rule=None)
    model.train(sentences)

    # If you're finished training a model (=no more updates, only querying)
    # model.init_sims(replace=True)

#     models = [word2vec_init]
# 
#     for epoch in range(1, 5 + 1):
#         logging.info('Epoch %d', epoch)
# 
#         model = copy.deepcopy(models[-1])
#         model.train(sentences)
# 
#         models.append(model)
# 
#     logging.info('Trained models: %s', models)

    return model


def lsi_model(index):
    logging.basicConfig(level=logging.INFO)
    logging.info('Initializing LSI.')

    model = gensim.models.LsiModel(num_topics=50) #200,  # Latent dimensions

    logging.info('Loading vocabulary.')    

    #index = pyndri.Index('index/')    
    dictionary = pyndri.extract_dictionary(index)

    ### TODO arguments
    corpus = connector_classes.IndriCorpus(index, dictionary)

    logging.info('Constructing word2vec vocabulary.')

    # Add documents
    model.add_documents(corpus)
    model.train(corpus)

    return model
