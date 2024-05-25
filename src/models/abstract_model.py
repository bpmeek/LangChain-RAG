from abc import ABC
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AbstractModel(ABC):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = self._load_model()
        self._tokenizer = self._load_tokenizer()
        self._chat_history = []

    def _load_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self._model_name)

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self._model_name)

    def _get_response(self, prompt):
        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._tokenizer.decode(
            self._model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
            )[0],
            skip_special_tokens=True,
        )
        return outputs

    def chat(self, text: str):
        raise NotImplementedError
