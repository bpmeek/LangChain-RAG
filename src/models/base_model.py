from .abstract_model import AbstractModel

import logging

log = logging.getLogger("sentence_transformers")
log.setLevel(logging.WARNING)


class BaseModel(AbstractModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

    def chat(self, text: str):
        self._chat_history.append(text)
        prompt = '\n'.join(self._chat_history)
        return self._get_response(prompt)

