import uuid

class Prompt:
    def __init__(self, text, scores=[], id=None, embeddings=None):
        self.text = text
        self.id = id or uuid.uuid4()
        self.embeddings = embeddings
        self.scores = scores
    
    def add_score(self, score):
        self.scores.append(score)
    
    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def __repr__(self):
        return f'Prompt(text={self.text}, id={self.id}, len(embeddings)={len(self.embeddings) if self.embeddings else None}, scores={self.scores})'
