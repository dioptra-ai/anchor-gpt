import random

from anchor import GroundedPromptSet, GroundedPrompt, BaseGroundedPromptStorage

# This is implemented by the user because it depends on the storage's API and the arbitrary shape of the scores passed to prompt.add_score().
# In this example it's an in-memory storage that only gives us a list of values, the scores are floats, and we choose to average them.
def retriever(storage, n):
    def prompt_average_score(prompt):
        return sum(prompt.scores) / len(prompt.scores)
    
    return sorted(storage.prompts.values(), key=prompt_average_score, reverse=True)[:n]

prompt_set = GroundedPromptSet(retriever, BaseGroundedPromptStorage())


# Add grounded prompts with scores potentially coming from a KNN.
prompt1 = prompt_set.add(GroundedPrompt('What is the meaning of life 1?', scores=[random.random(), random.random(), random.random()]))
prompt2 = prompt_set.add(GroundedPrompt('What is the meaning of life 2?', scores=[random.random(), random.random(), random.random()]))
prompt3 = prompt_set.add(GroundedPrompt('What is the meaning of life 3?', scores=[random.random(), random.random(), random.random()]))
prompt4 = prompt_set.add(GroundedPrompt('What is the meaning of life 4?', scores=[random.random(), random.random(), random.random()]))

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

print(prompt_set.retrieve_n(2))
