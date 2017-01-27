import copy
from collections import Counter

import gensim
import logging
import pyndri
import sys
import numpy as np
import connector_classes
from gensim.corpora.dictionary import Dictionary


class Word2Vec():
    def __init__(self, filename=None, embedding_size=300, max_documents=500):
        self.embedding_size = embedding_size
        self.max_documents = max_documents
        if filename == None:
            self.model = gensim.models.Word2Vec(
                size=self.embedding_size,  # Embedding size
                window=5,  # One-sided window size
                sg=True,  # Skip-gram. (False for CBOW)
                min_count=5,  # Minimum word frequency.
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
        self.model.save('models/word2vec' + str(self.embedding_size) + ' .model')

    def docs2vec(self, index):
        _, id2token, _ = index.get_dictionary()
        docs_representation = np.zeros([self.embedding_size, self.max_documents])
        for d in range(self.max_documents):  
            doc = index.document(d + 1)[1]
            doc_words = [id2token.get(word_id) for word_id in doc if word_id != 0]
            doc_words = [word for word in doc_words if index.term_count(word) >= 5]
            docvec = np.zeros([self.embedding_size, len(doc_words)])
            for i, word in enumerate(doc_words):
                docvec[:, i] = self.model[word]
            docs_representation[:, d] = np.mean(docvec, axis=1)
        return docs_representation

    # Build query representation as average of wordvectors
    def query2vec(self, query):
        query_tokens = query.lower().split(' ')
        query_tokens = [word for word in query_tokens if word in self.model.wv.vocab]
        wordvec = np.zeros([self.embedding_size, len(query_tokens)])
        for i, word in enumerate(query_tokens):
            wordvec[:, i] = self.model[word]
        return np.mean(wordvec, axis=1)


class LSI():
    def __init__(self, filename=None, corpus=None, num_topics=50, max_documents=500):
        self.num_topics = num_topics
        self.max_documents = max_documents
        if filename == None:
            self.model = gensim.models.LsiModel(
                corpus=corpus,
                num_topics=self.num_topics)  # Latent dimensions
        else:
            self.model = gensim.models.LsiModel.load(filename)

    def save(self, filename):
        self.model.save(filename)

    def docs_projection(self, index):
        docs_projection = np.zeros([self.num_topics, self.max_documents])
        for d in range(self.max_documents):
            doc = index.document(d + 1)[1]        
            bow = [(word_id,count) for word_id,count in 
                    dict(Counter(doc)).items() if word_id!= 0]
            #print(self.model[sorted(bow)])
            if len(bow) != 0:
                docs_projection[:,d] = np.array([p for _,p in self.model[sorted(bow)]])
            else:
                print("Document " + str(d))
                print(sorted(bow))
                print(self.model[sorted(bow)])
                print([p for _,p in self.model[sorted(bow)]])
                print(docs_projection.shape)
        return docs_projection

    def query_projection(self, query_word_ids):
        bow = [(word_id,count) for word_id,count in 
                dict(Counter(query_word_ids)).items() if word_id!= 0]
        if len(bow) != 0:
            query_projection = np.array([p for _,p in self.model[sorted(bow)]])
        else:
            print(query_word_ids)
            query_projection = np.zeros([self.num_topics])
        return query_projection                

class LDA():
    def __init__(self, filename=None, corpus=None, num_topics=50, max_documents=500):
        self.num_topics = num_topics
        self.max_documents = max_documents
        if filename == None:
            self.model = gensim.models.LdaModel(
                corpus=corpus,
                num_topics=num_topics)  # Latent dimensions
        else:
            self.model = gensim.models.LdaModel.load(filename)

    def save(self, filename):
        self.model.save(filename)

    def docs2topic(self, index):
        docs_topic_distribution = np.zeros([self.num_topics, self.max_documents])
        for d in range(self.max_documents):
            doc = index.document(d + 1)[1]        
            bow = [(word_id,count) for word_id,count in 
                    dict(Counter(doc)).items() if word_id!= 0]
            topic_distribution = self.model[sorted(bow)]
            for topic, prob in topic_distribution:
                docs_topic_distribution[topic,d] = prob
        return docs_topic_distribution

    def query2topic(self, query_word_ids):
        query_topic_distribution = np.zeros([self.num_topics])
        bow = [(word_id,count) for word_id,count in 
                dict(Counter(query_word_ids)).items() if word_id!= 0]
        topic_distribution = self.model[sorted(bow)]
        for topic, prob in topic_distribution:
            query_topic_distribution[topic] = prob
        return query_topic_distribution


class Doc2Vec():
    def __init__(self, filename=None, documents=None, size=50, max_documents=500):
        self.size = size
        self.max_documents = max_documents
        if filename == None:
            self.model = gensim.models.Doc2Vec(
                documents = documents, 
                size = self.size)
        else:
            self.model = gensim.models.Doc2Vec.load(filename)

    def save(self, filename):
        self.model.save(filename)

