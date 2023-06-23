import random

from anchor import PromptLogger, Prompt

# Define a retriever function. The retriever function takes a store and a number n and returns
#  a list of prompts. In this case, we sort the prompts by their average score and return the top n.
def retriever(store, n):
    def prompt_average_score(prompt):
        return sum(prompt.scores) / len(prompt.scores)

    return sorted(store.select_prompts(), key=prompt_average_score, reverse=True)[:n]

prompt_logger = PromptLogger(retriever)

# Add grounded prompts with scores potentially coming from a KNN.
prompt1 = prompt_logger.log(Prompt('What is the meaning of life 1?', scores=[random.random(), random.random(), random.random()]))
prompt2 = prompt_logger.log(Prompt('What is the meaning of life 2?', scores=[random.random(), random.random(), random.random()]))
prompt3 = prompt_logger.log(Prompt('What is the meaning of life 3?', scores=[random.random(), random.random(), random.random()]))
prompt4 = prompt_logger.log(Prompt('What is the meaning of life 4?', scores=[random.random(), random.random(), random.random()]))

# Add more scores to the prompts, like user feedback or model predictions.
prompt1.add_score(random.random())
prompt1.add_score(random.random())
prompt1.add_score(random.random())
prompt1.set_embeddings([random.random() for _ in range(768)])

prompt2.add_score(random.random())
prompt2.add_score(random.random())
prompt2.add_score(random.random())
prompt2.set_embeddings([random.random() for _ in range(768)])

prompt3.add_score(random.random())
prompt3.add_score(random.random())
prompt3.add_score(random.random())
prompt3.set_embeddings([random.random() for _ in range(768)])

prompt4.add_score(random.random())
prompt4.add_score(random.random())
prompt4.add_score(random.random())
prompt4.add_score(random.random())
prompt4.set_embeddings([random.random() for _ in range(768)])

top_prompts = prompt_logger.retrieve_n(12)
prompts_coreset = prompt_logger.get_prompts_coreset(top_prompts, 2)

print(prompts_coreset)
