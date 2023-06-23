from .prompt_store import PromptStore
from .coreset import coreset

class PromptLogger:
    def __init__(self, retriever, store=None):
        self.retriever = retriever
        self.store = store or PromptStore()
    
    def log(self, prompt):
        prompt._set_store(self.store)

        return self.store.add(prompt)
    
    def get_by_ids(self, ids):
        return self.store.get_by_ids(ids)

    def retrieve_n(self, n):
        return self.retriever(self.store, n)

    def get_prompts_coreset(self, prompts, size, distance='euclidean'):
        prompt_ids = [p.id for p in prompts]
        selected_indexes = coreset(
            vector_ids=prompt_ids,
            already_selected=[],
            number_of_datapoints=size,
            transformer=lambda ids: [p.embeddings for p in self.store.get_by_ids(ids)],
            distance=distance
        )

        selected_ids = [prompt_ids[index] for index in selected_indexes]

        return self.store.get_by_ids(selected_ids)
