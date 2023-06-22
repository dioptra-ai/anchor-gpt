from anchor import GroundedPromptSet, GroundedPrompt, BaseGroundedPromptStorage

def retriever(storage, n):
    def prompt_average_score(prompt):
        return sum(prompt.scores) / len(prompt.scores)
    
    return sorted(storage.prompts.values(), key=prompt_average_score, reverse=True)[:n]

storage = BaseGroundedPromptStorage()
prompt_set = GroundedPromptSet(retriever, storage)

prompt1 = prompt_set.add(GroundedPrompt('What is the meaning of life 1?'))
prompt2 = prompt_set.add(GroundedPrompt('What is the meaning of life 2?'))
prompt3 = prompt_set.add(GroundedPrompt('What is the meaning of life 3?'))
prompt4 = prompt_set.add(GroundedPrompt('What is the meaning of life 4?'))

import random
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
