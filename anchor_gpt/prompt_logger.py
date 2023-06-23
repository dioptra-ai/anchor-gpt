import pandas as pd
from .prompt_store import prompt_to_db, db_to_prompt, SQLitePromptStore
from .coreset import coreset

class PromptLogger:
    def __init__(self, store=None):
        """
        A prompt logger that stores prompts in a PromptStore.

        Parameters:
            store: a PromptStore object. If None, a SQLitePromptStore will be used.
        """
        self.store = store or SQLitePromptStore()

    def log(self, prompt):
        """
        Log a prompt to the store.

        Parameters:
            prompt: a Prompt object.
        """
        prompt._set_store(self.store)

        return self.store.add(prompt)

    def retrieve(self, retriever, threshold):
        """
        Retrieve the top n prompts according to the retriever function.

        Parameters:
            retriever: a function that takes a PromptStore and threshold a number and returns a list of prompts.
            threshold: a theshold to be consumed by the retreiver.
        """
        return retriever(self.store, threshold)

    def deduplicate(self, prompts, size, distance='euclidean'):
        """
        Remove text duplicates and embeddings near duplicates using the coreset algorithm.
        More details on the coreset algorithm can be found here: https://arxiv.org/abs/1708.00489

        Parameters:
            prompts: a list of Prompt objects.
            size: the number of prompts to select.
            distance: the distance function to use for the coreset algorithm (default: euclidean).
        """

        data = [{'id': prompt.id, 'text': prompt.text} for prompt in prompts]

        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=['text'])

        prompt_ids = list(df['id'])
        selected_indexes = coreset(
            vector_ids=prompt_ids,
            already_selected=[],
            number_of_datapoints=size,
            transformer=lambda ids: [p.embeddings for p in self.store.get_by_ids(ids)],
            distance=distance
        )

        selected_ids = [prompt_ids[index] for index in selected_indexes]

        return self.store.get_by_ids(selected_ids)
