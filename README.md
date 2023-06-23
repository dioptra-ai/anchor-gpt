# anchor-gpt

Diagnose and find hallucinations in your grounded Large Language Model prompts with Anchor-GPT!

## Installation

```bash
pip install anchor-gpt
```

## Example Usage

```python

from anchor_gpt import PromptLogger, Prompt

# 1. Define a retriever function. The retriever function receives a PromptStore and a number and returns
#  a list of prompts. In this case, we sort the prompts by their average score and return the top n.
def retriever(store, n):
    def prompt_average_score(prompt):
        return sum(prompt.scores) / len(prompt.scores)

    return sorted(store.select_prompts(), key=prompt_average_score, reverse=True)[:n]

prompt_logger = PromptLogger(retriever)

# 2. Handle your prompts normally, and log them with their grounding scores.
@app.route("/chat", methods=["POST"])
def chat():
    # Do your grounding as normal:
    prompt_embeddings = model.encode(request.json["prompt"])
    vector_store_results = vector_store.query(prompt_embeddings, top_k=10)

    # Then log the prompt with the scores and embeddings. Prompts are stored locally in a SQLite database.
    prompt_logger.log(Prompt(
        text=request.json["prompt"], 
        scores=[r.distance for r in vector_store_results],
        embeddings=prompt_embeddings,
    ))

    # Do your chatbot prompt engineering stuff here, return the response.

# 3. Retrieve the worst grounded prompts, the best candidates to improve your grounding database.
@app.route("/hallucinations", methods=["GET"])
def hallucinations():
    # Retrieve a list of the top 100 most hallucinatory prompts.
    # This will call retriever defined above.
    top_prompts = prompt_logger.retrieve_n(100)

    # Optionally, calculate the coreset of the top 100 prompts.
    # This will return a list of the top 10 most hallucinatory prompts
    # with good diversity and coverage.
    coreset = prompt_logger.get_prompts_coreset(top_prompts, 10)

    return jsonify([p.text for p in coreset])
```