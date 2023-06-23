import sqlite3
import uuid
import json

sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes_le)

from .prompt import Prompt

class PromptStore:
    def add(self, prompt):
        '''
        Add a prompt to the store.
        prompt: a Prompt object.
        '''
        raise NotImplementedError
    
    def update(self, prompt):
        '''
        Update a prompt in the store.
        prompt: a Prompt object.
        '''
        raise NotImplementedError
    
    def get_by_ids(self, ids):
        '''
        Get a list of prompts by their ids.
        ids: a list of UUIDs.
        '''
        raise NotImplementedError
    

def prompt_to_db(prompt):
    return (prompt.id, prompt.text, str(prompt.scores) if prompt.scores else None, str(prompt.embeddings) if prompt.embeddings else None)

def db_to_prompt(db_prompt):
    return Prompt(
        id=uuid.UUID(bytes_le=db_prompt['id']), 
        text=db_prompt['text'], 
        scores=json.loads(db_prompt['scores']), 
        embeddings=json.loads(db_prompt['embeddings']) if db_prompt['embeddings'] else None
    )

class SQLitePromptStore(PromptStore):
    '''
    A prompt store that uses SQLite as a backend.
    '''
    def __init__(self, name="anchor.db"):
        self.name = name
        self.connection = sqlite3.connect(name)
        # self.connection.set_trace_callback(print)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("CREATE TABLE IF NOT EXISTS prompts (id UUID, text TEXT, scores JSON, embeddings JSON)")

    def add(self, prompt):
        self.connection.execute("INSERT INTO prompts VALUES (?, ?, ?, ?)", prompt_to_db(prompt))
        self.connection.commit()
        return prompt

    def update(self, prompt):
        (id, text, scores, embeddings) = prompt_to_db(prompt)
        self.connection.execute("UPDATE prompts SET text=?, scores=?, embeddings=? WHERE id=?", (text, scores, embeddings, id))
        self.connection.commit()
        return prompt
    
    def get_by_ids(self, ids):
        result = self.connection.execute(f"SELECT * FROM prompts WHERE id IN ({','.join(['?'] * len(ids))})", ids).fetchall()

        return [db_to_prompt(row) for row in result]

    def select_prompts(self, fields='text, id, scores, embeddings', where='1=1', order_by='id', limit='-1'):

        return [db_to_prompt(row) for row in self.connection.execute(f"SELECT {fields} FROM prompts WHERE {where} ORDER BY {order_by} LIMIT {limit}")]
