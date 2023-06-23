from .prompt_store import SQLitePromptStore
from .coreset import coreset

class PromptLogger:
    def __init__(self, retriever, store=None):
        '''
        retriever: a function that takes a PromptStore and a number and returns a list of prompts. See the example.py file for an example.
        store: a PromptStore object. If None, a SQLitePromptStore will be used.
        '''
        self.retriever = retriever
        self.store = store or SQLitePromptStore()
    
    def log(self, prompt):
        '''
        prompt: a Prompt object.
        '''
        prompt._set_store(self.store)

        return self.store.add(prompt)

    def retrieve_n(self, n):
        '''
        Retrieve the top n prompts according to the retriever function passed to the constructor.
        n: the number of prompts to retrieve.
        '''
        return self.retriever(self.store, n)

    def get_prompts_coreset(self, prompts, size, distance='euclidean'):
        '''
        Coreset is a technique that filters a dataset down to a subset while preserving diversity.
        prompts: a list of Prompt objects.
        size: the number of prompts to select.
        distance: the distance function to use for the coreset algorithm (default: euclidean).
        '''
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
