import pytest
from anchor_gpt import PromptLogger, Prompt

def test_log():
    prompt_logger = PromptLogger()
    my_prompt = prompt_logger.log(Prompt(
        text='test prompt',
        response='test response',
        scores={'grounding_distances': [0.1, 0.2, 0.3]},
        embeddings=[0.1, 0.2, 0.3],
    ))

    retreived_prompt = prompt_logger.store.get_by_ids([my_prompt.id])[0]

    assert retreived_prompt.text == 'test prompt'
    assert retreived_prompt.response == 'test response'
    assert retreived_prompt.scores == {'grounding_distances': [0.1, 0.2, 0.3]}
    assert retreived_prompt.embeddings == [0.1, 0.2, 0.3]

    prompt_logger.store.purge()

def test_update():
    prompt_logger = PromptLogger()
    my_prompt = prompt_logger.log(Prompt(
        text='test prompt',
        response='test response',
        scores={'grounding_distances': [0.1, 0.2, 0.3]},
        embeddings=[0.1, 0.2, 0.3],
    ))

    my_prompt.update_scores({'user_feedback': 0.5})

    retreived_prompt = prompt_logger.store.get_by_ids([my_prompt.id])[0]
    assert retreived_prompt.scores == {'grounding_distances': [0.1, 0.2, 0.3], 'user_feedback': 0.5}

    my_prompt.update_scores({'user_feedback': 0.6})
    retreived_prompt = prompt_logger.store.get_by_ids([my_prompt.id])[0]
    assert retreived_prompt.scores == {'grounding_distances': [0.1, 0.2, 0.3], 'user_feedback': 0.6}

    prompt_logger.store.purge()

def test_retrieve():
    prompt_logger = PromptLogger()
    prompt_logger.log(Prompt(
        text='test prompt',
        response='test response',
        scores={'grounding_distances': 0.1},
        embeddings=[0.1, 0.2, 0.3],
    ))

    prompt_logger.log(Prompt(
        text='test prompt 2',
        response='test response',
        scores={'grounding_distances': 0.5},
        embeddings=[0.1, 0.2, 0.3],
    ))

    prompt_logger.log(Prompt(
        text='test prompt 3',
        response='test response',
        scores={'grounding_distances': 0.2},
        embeddings=[0.1, 0.2, 0.3],
    ))


    def retiever(store, threshold):
        def prompt_score(prompt):
            return prompt.scores['grounding_distances']
        return list(filter(lambda x: prompt_score(x) > threshold, store.select_prompts()))

    retreived_prompt = prompt_logger.retrieve(retiever, 0.3)[0]

    assert retreived_prompt.text == 'test prompt 2'

    prompt_logger.store.purge()


def test_dedup():
    prompt_logger = PromptLogger()
    prompt_logger.log(Prompt(
        text='test prompt',
        response='test response',
        scores={'grounding_distances': 0.1},
        embeddings=[0.1, 0.2, 0.3],
    ))

    prompt_logger.log(Prompt(
        text='test prompt 2',
        response='test response',
        scores={'grounding_distances': 0.5},
        embeddings=[-0.3, 0.5, 0.3],
    ))

    prompt_logger.log(Prompt(
        text='test prompt 3',
        response='test response',
        scores={'grounding_distances': 0.2},
        embeddings=[0.9, 0.2, 0.9],
    ))

    prompt_logger.log(Prompt(
        text='test prompt',
        response='test response',
        scores={'grounding_distances': 0.2},
        embeddings=[0.1, 0.2, 0.3],
    ))

    prompt_logger.log(Prompt(
        text='test prompt',
        response='test response',
        scores={'grounding_distances': 0.2},
        embeddings=[0.1, 0.2, 0.3],
    ))

    prompt_logger.log(Prompt(
        text='test prompt',
        response='test response',
        scores={'grounding_distances': 0.2},
        embeddings=[0.1, 0.2, 0.3],
    ))


    def retiever(store, threshold):
        def prompt_score(prompt):
            return prompt.scores['grounding_distances']
        return list(filter(lambda x: prompt_score(x) > threshold, store.select_prompts()))

    retreived_prompts = prompt_logger.retrieve(retiever, 0)

    dedup_prompt = prompt_logger.deduplicate(retreived_prompts, 3)

    dedup_text = [prompt.text for prompt in dedup_prompt]
    dedup_text.sort()
    assert dedup_text == ['test prompt', 'test prompt 2', 'test prompt 3']

    prompt_logger.store.purge()
