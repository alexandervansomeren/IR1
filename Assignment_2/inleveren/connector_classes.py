import gensim
import pyndri
from collections import Counter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class IndriSentences(gensim.interfaces.CorpusABC):
    """Integrates an Index with Gensim's word2vec implementation."""

    def __init__(self, index, dictionary, max_documents=None):
        assert isinstance(index, pyndri.Index)

        self.index = index
        self.dictionary = dictionary
        self.max_documents = max_documents

    def _maximum_document(self):
        if self.max_documents is None:
            return self.index.maximum_document()
        else:
            return min(
                self.max_documents + self.index.document_base(),
                self.index.maximum_document())

    def __iter__(self):
        for int_doc_id in range(self.index.document_base(),
                                self._maximum_document()):
            tokens = self.index.document(int_doc_id)[1]

            yield tuple(
                self.dictionary[token_id]
                for token_id in tokens
                if token_id > 0 and token_id in self.dictionary)

    def __len__(self):
        return self._maximum_document() - self.index.document_base()

class IndriCorpus(gensim.interfaces.CorpusABC):
    """Integrates an Index with Gensim's LSI implementation."""
    
    def __init__(self, index, dictionary, max_documents=None):
        assert isinstance(index, pyndri.Index)

        self.index = index
        self.dictionary = dictionary
        self.max_documents = max_documents

    def _maximum_document(self):
        if self.max_documents is None:
            return self.index.maximum_document()
        else:
            return min(
                self.max_documents + self.index.document_base(),
                self.index.maximum_document())

    def __iter__(self):
        """
        The function that defines a corpus.
        Iterating over the corpus must yield sparse vectors, one for each document.
        """
        for int_doc_id in range(self.index.document_base(),
                                self._maximum_document()):
            doc = self.index.document(int_doc_id)[1]            

            bow = [(word_id,count) for word_id,count in dict(Counter(doc)).items() if word_id!= 0]
            yield sorted(bow)

    def __len__(self):     
        return self._maximum_document() - self.index.document_base()


class IndriDocs(gensim.interfaces.CorpusABC):
    """Integrates an Index with Gensim's Doc2Vec implementation."""

    def __init__(self, index, dictionary, max_documents=None):
        assert isinstance(index, pyndri.Index)

        self.index = index
        self.dictionary = dictionary
        self.max_documents = max_documents
        _,self.id2token,_ = index.get_dictionary()

    def _maximum_document(self):
        if self.max_documents is None:
            return self.index.maximum_document()
        else:
            return min(
                self.max_documents + self.index.document_base(),
                self.index.maximum_document())

    def __iter__(self):
        """
        The function that defines a corpus.
        Iterating over the corpus must yield sparse vectors, one for each document.
        """
        for int_doc_id in range(self.index.document_base(),
                                self._maximum_document()):
            doc = self.index.document(int_doc_id)[1]            
            tokens = [self.id2token[word_id] for word_id in doc if word_id != 0]
            yield TaggedDocument(tokens, [int_doc_id])

    def __len__(self):     
        return self._maximum_document() - self.index.document_base()

