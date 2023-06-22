__version__ = '0.0.1'

from .prompt_store import PromptStore
from .prompt import Prompt

class PromptLogger:
    def __init__(self, retriever, store=None):
        self.retriever = retriever
        self.store = store or PromptStore()
    
    def log(self, prompt):
        return self.store.add(prompt)
    
    def get_by_id(self, id):
        return self.store.get_by_id(id)

    def retrieve_n(self, n):
        return self.retriever(self.store, n)
