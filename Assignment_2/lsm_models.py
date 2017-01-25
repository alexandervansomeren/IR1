import copy
import gensim
import logging
import pyndri
import sys
import numpy as np
import connector_classes
from gensim.corpora.dictionary import Dictionary


class Word2Vec():

    def __init__(self, filename=None, embedding_size=300, max_documents=500):
        self.embedding_size=embedding_size
        self.max_documents=max_documents
        if filename == None:
            self.model = gensim.models.Word2Vec(
                size=self.embedding_size,  # Embedding size
                window=5,  # One-sided window size
                sg=True,  # Skip-gram. (False for CBOW)
                min_count=1,  # Minimum word frequency.
                sample=1e-3,  # Sub-sample threshold.
                hs=False,  # Hierarchical softmax.
                negative=10,  # Number of negative examples.
                iter=1,  # Number of iterations.
                workers=8)  # Number of workers.
        else:
            self.model = gensim.models.Word2Vec.load(filename)

    def train(self, index):  
        dictionary = pyndri.extract_dictionary(index)
        sentences = connector_classes.IndriSentences(index, dictionary, 
                                                     max_documents=self.max_documents)
        self.model.build_vocab(sentences, trim_rule=None)
        self.model.train(sentences)
        self.model.save('models/word2vec.model')
    
    def docs2vec(self, index):
        _, id2token, _ = index.get_dictionary()        
        docs_representation = np.zeros([self.embedding_size,self.max_documents])   
        for d in range(self.max_documents):#index.maximum_document()):
            doc = index.document(d+1)[1]
            doc_words = [id2token.get(word_id) for word_id in doc if word_id != 0]
            doc_words = [word for word in doc_words if index.term_count(word) >= 5]
            docvec = np.zeros([self.embedding_size, len(doc_words)])
            for i, word in enumerate(doc_words):
                docvec[:,i] = self.model[word]
            docs_representation[:,d] = np.mean(docvec, axis=1)
        return docs_representation

    # Build query representation as average of wordvectors
    def query2vec(self, query):
        query_tokens = query.lower().split(' ')
        query_tokens = [word for word in query_tokens if word in self.model.wv.vocab]
        wordvec = np.zeros([self.embedding_size, len(query_tokens)])
        for i, word in enumerate(query_tokens):
            wordvec[:,i] = self.model[word]
        return np.mean(wordvec, axis=1)


class LSI():

    def __init__(self, filename=None, num_topics=50):
        if filename == None:
            self.model = gensim.models.LsiModel(
                num_toptics=num_topics) # Latent dimensions
        else:
            self.model = gensim.models.LsiModel.load(filename)

    def train(self, index):
        dictionary = pyndri.extract_dictionary(index)
        # TODO make this work
        corpus = connector_classes.IndriCorpus(index, dictionary) 
        model.add_documents(corpus)
        model.save('models/lsi.model')
    

#class LDA():


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




