import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from strenum import StrEnum


class Model(StrEnum):
    DISCORD = "Salesforce/discord_qg"


class QGPipeline:
    def __init__(self, model_id=Model.DISCORD):
        """
        :param model_id: name of huggingface transformers model
        """
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=self._torch_dtype
        )
        self.model.to(self._device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, documents: list[str], start_word="What") -> list[str]:
        """
        Generates one question per document (context)

        :param documents: contexts to generate questions for
        :param start_word: question start word
        :return: generated question
        """
        encoder_ids = self.tokenizer.batch_encode_plus(
            documents,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)
        
        decoder_input_ids = self.tokenizer.batch_encode_plus(
            [start_word] * len(documents), add_special_tokens=True, return_tensors="pt"
        ).to(self._device)["input_ids"][:, :-1]

        model_output = self.model.generate(
            **encoder_ids, decoder_input_ids=decoder_input_ids
        )
        generated_questions = self.tokenizer.batch_decode(
            model_output, skip_special_tokens=True
        )

        return generated_questions
