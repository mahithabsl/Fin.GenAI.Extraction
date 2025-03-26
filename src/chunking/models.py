class Chunk:
    def __init__(self, content, chunk_id, start_i, end_i, content_without_overlap):
        self.content = content
        self.chunk_id = chunk_id
        self.start_i = start_i
        self.end_i = end_i
        self.content_without_overlap = content_without_overlap

class Document:
    def __init__(self, content, title=""):
        self.content = content
        self.title = title
        self.chunks = []
        self._spacy_doc = None  # Will be set when processed

    @property
    def spacy_doc(self):
        return self._spacy_doc

    @spacy_doc.setter
    def spacy_doc(self, doc):
        self._spacy_doc = doc 