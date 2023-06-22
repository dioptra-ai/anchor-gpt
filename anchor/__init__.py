__version__ = '0.0.1'

import uuid

class GroundedPromptSet:
    def __init__(self, retriever, storage):
        self.retriever = retriever
        self.storage = storage
    
    def add(self, prompt):
        return self.storage.add(prompt)
    
    def get_by_id(self, id):
        return self.storage.get_by_id(id)

    def retrieve_n(self, n):
        return self.retriever(self.storage, n)

class GroundedPrompt:
    def __init__(self, prompt, id=None, prompt_embeddings=None):
        self.prompt = prompt
        self.id = id or uuid.uuid4()
        self.prompt_embeddings = prompt_embeddings
        self.scores = []
    
    def add_score(self, score):
        self.scores.append(score)
    
    def __repr__(self):
        return f'GroundedPrompt(prompt={self.prompt}, id={self.id}, prompt_embeddings={self.prompt_embeddings}, scores={self.scores})'

class BaseGroundedPromptStorage:
    def __init__(self):
        self.prompts = {}
    
    def add(self, prompt):
        self.prompts[prompt.id] = prompt
        return prompt
    
    def get_by_id(self, id):
        return self.prompts[id]
