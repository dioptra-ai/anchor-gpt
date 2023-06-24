import sqlite3
import uuid
import ast

sqlite3.register_adapter(uuid.UUID, lambda u: u.bytes_le)

from .prompt import Prompt

class PromptStore:
    """
    An extensible class for prompt stores.
    """
    def add(self, prompt):
        """
        Add a prompt to the store.

        Parameters:
            prompt: a Prompt object.
        """
        raise NotImplementedError

    def update(self, prompt):
        """
        Update a prompt in the store.

        Parameters:
            prompt: a Prompt object.
        """
        raise NotImplementedError

    def get_by_ids(self, ids):
        """
        Get a list of prompts by their ids.

        Parameters:
            ids: a list of UUIDs.
        """
        raise NotImplementedError

    def purge(self):
        """
        Purge the store of all prompts.

        """
        raise NotImplementedError


def prompt_to_db(prompt):
    """
    Convert a prompt to a tuple that can be inserted into the database.

    Parameters:
        prompt: a Prompt object.
    """
    return (prompt.id, prompt.text, prompt.response, str(prompt.scores) if prompt.scores else None, str(prompt.embeddings) if prompt.embeddings else None)

def db_to_prompt(db_prompt):
    """
    Convert a database row to a Prompt object.

    Parameters:
        db_prompt: a database row.
    """
    return Prompt(
        id=uuid.UUID(bytes_le=db_prompt['id']),
        text=db_prompt['text'],
        response=db_prompt['response'],
        scores=ast.literal_eval(db_prompt['scores']),
        embeddings=ast.literal_eval(db_prompt['embeddings']) if db_prompt['embeddings'] else None
    )

class SQLitePromptStore(PromptStore):
    """
    An impementation of a prompt store that uses a SQLite backend.
    """
    def __init__(self, name='anchor.db', debug=False):
        self.name = name
        self.connection = sqlite3.connect(name)
        if debug:
            self.connection.set_trace_callback(print)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('CREATE TABLE IF NOT EXISTS prompts (id UUID, text TEXT, response TEXT, scores JSON, embeddings JSON)')

    def add(self, prompt):
        self.connection.execute('INSERT INTO prompts VALUES (?, ?, ?, ?, ?)', prompt_to_db(prompt))
        self.connection.commit()
        return prompt

    def update(self, prompt):
        (id, text, response, scores, embeddings) = prompt_to_db(prompt)
        self.connection.execute('UPDATE prompts SET text=?, response=?, scores=?, embeddings=? WHERE id=?', (text, response, scores, embeddings, id))
        self.connection.commit()
        return prompt

    def purge(self):
        self.connection.execute('DELETE FROM prompts')
        self.connection.commit()

    def get_by_ids(self, ids):
        result = self.connection.execute(f'SELECT * FROM prompts WHERE id IN ({",".join(["?"] * len(ids))})', ids).fetchall()

        return [db_to_prompt(row) for row in result]

    def select_prompts(self, fields='text, response, id, scores, embeddings', where='1=1', order_by='id', limit='-1'):

        return [db_to_prompt(row) for row in self.connection.execute(f'SELECT {fields} FROM prompts WHERE {where} ORDER BY {order_by} LIMIT {limit}')]
