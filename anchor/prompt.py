import uuid
import numpy as np

class Prompt:
    def __init__(self, text, scores=[], id=None, embeddings=None):
        self.text = text
        self.id = id or uuid.uuid4()
        self.embeddings = embeddings
        self.scores = scores
    
    def _set_store(self, store):
        self.store = store
    
    def add_score(self, score):
        self.scores.append(score)
        self.store.update(self)
    
    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
        self.store.update(self)

    def __repr__(self):
        return f'Prompt(text={self.text}, id={self.id}, shape(embeddings)={np.shape(self.embeddings) if self.embeddings else None}, scores={self.scores})'
