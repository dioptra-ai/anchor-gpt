import sqlite3
import uuid
import json

sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes_le)

from .prompt import Prompt

def prompt_to_db(prompt):
    return (prompt.id, prompt.text, str(prompt.scores) if prompt.scores else None, str(prompt.embeddings) if prompt.embeddings else None)

def db_to_prompt(db_prompt):
    return Prompt(
        id=uuid.UUID(bytes_le=db_prompt['id']), 
        text=db_prompt['text'], 
        scores=json.loads(db_prompt['scores']), 
        embeddings=json.loads(db_prompt['embeddings']) if db_prompt['embeddings'] else None
    )

class PromptStore:
    def __init__(self, name="anchor.db"):
        self.name = name
        self.con = sqlite3.connect(name)
        self.con.row_factory = sqlite3.Row
        self.con.execute("CREATE TABLE IF NOT EXISTS prompts (id UUID, text TEXT, scores JSON, embeddings JSON)")

    def add(self, prompt):
        self.con.execute("INSERT INTO prompts VALUES (?, ?, ?, ?)", prompt_to_db(prompt))
        self.con.commit()
        return prompt
    
    def get_by_id(self, id):
        result = self.con.execute("SELECT text, id, scores, embeddings FROM prompts WHERE id=?", (id,)).fetchone()

        return db_to_prompt(result) if result else None

    def get_all_prompts(self):

        return [db_to_prompt(row) for row in self.con.execute("SELECT text, id, scores, embeddings FROM prompts").fetchall()]
