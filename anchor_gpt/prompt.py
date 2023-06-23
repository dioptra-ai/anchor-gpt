import uuid
import numpy as np

class Prompt:
    def __init__(self, text, response, scores={}, id=None, embeddings=None):
        self.text = text
        self.response = response
        self.id = id or uuid.uuid4()
        self.embeddings = embeddings
        self.scores = scores

    def _set_store(self, store):
        self.store = store

    def add_score(self, score):
        """
        Add a score to the prompt.

        Parameters:
            score: any. Will be passed in the prompts to the PromptLogger retriever function to allow custom scoring.
        """
        self.scores.extent(score)
        self.store.update(self)

    def set_embeddings(self, embeddings):
        """
        Set the embeddings of the prompt.

        Parameters:
            embeddings: a list of floats.
        """
        self.embeddings = embeddings
        self.store.update(self)

    def __repr__(self):
        return f'Prompt(id={self.id}, text={self.text}, response={self.response}, shape(embeddings)={np.shape(self.embeddings) if self.embeddings else None}, scores={self.scores})'
