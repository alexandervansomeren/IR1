import gensim
import pyndri

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
            ext_doc_id, tokens = self.index.document(int_doc_id)

            yield tuple(
                self.dictionary[token_id]
                for token_id in tokens
                if token_id > 0 and token_id in self.dictionary)

    def __len__(self):
        return self._maximum_document() - self.index.document_base()

class IndriCorpus(gensim.interfaces.CorpusABC):
    """Integrates an Index with Gensim's LSI implementation."""
    
    def __init__(self, input=None):
        super(TextCorpus, self).__init__()
        self.input = input
        self.dictionary = Dictionary()
        self.metadata = False
        if input is not None:
            self.dictionary.add_documents(self.get_texts())
        else:
            logger.warning("No input document stream provided; assuming "
                           "dictionary will be initialized some other way.")

    def __iter__(self):
        """
        The function that defines a corpus.
        Iterating over the corpus must yield sparse vectors, one for each document.
        """
        for text in self.get_texts():
            if self.metadata:
                yield self.dictionary.doc2bow(text[0], allow_update=False), text[1]
            else:
                yield self.dictionary.doc2bow(text, allow_update=False)

    def getstream(self):
        return utils.file_or_filename(self.input)

    def get_texts(self):
        """
        Iterate over the collection, yielding one document at a time. A document
        is a sequence of words (strings) that can be fed into `Dictionary.doc2bow`.
        Override this function to match your input (parse input files, do any
        text preprocessing, lowercasing, tokenizing etc.). There will be no further
        preprocessing of the words coming out of this function.
        """
        # Instead of raising NotImplementedError, let's provide a sample implementation:
        # assume documents are lines in a single file (one document per line).
        # Yield each document as a list of lowercase tokens, via `utils.tokenize`.
        with self.getstream() as lines:
            for lineno, line in enumerate(lines):
                if self.metadata:
                    yield utils.tokenize(line, lowercase=True), (lineno,)
                else:
                    yield utils.tokenize(line, lowercase=True)

    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            self.length = sum(1 for _ in self.get_texts())
        return self.length
