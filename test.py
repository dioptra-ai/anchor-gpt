import random

from anchor import PromptLogger, Prompt

# This is implemented by the user because it depends on the store's API and the arbitrary shape of the scores passed to prompt.add_score().
# In this example it's an in-memory store that only gives us a list of values, the scores are floats, and we choose to average them.
def retriever(store, n):
    def prompt_average_score(prompt):
        return sum(prompt.scores) / len(prompt.scores)

    return sorted(store.get_all_prompts(), key=prompt_average_score, reverse=True)[:n]

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

prompt2.add_score(random.random())
prompt2.add_score(random.random())
prompt2.add_score(random.random())

prompt3.add_score(random.random())
prompt3.add_score(random.random())
prompt3.add_score(random.random())

prompt4.add_score(random.random())
prompt4.add_score(random.random())
prompt4.add_score(random.random())
prompt4.add_score(random.random())

print(prompt_logger.retrieve_n(2))
