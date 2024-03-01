import warnings

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from alinet.qg import Model


class QGPipeline:
    def __init__(self, model_id=Model.BASELINE):
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

    def __call__(
        self,
        documents: list[str],
        start_word=None,
        max_tokens=32,
        num_beams=4,
    ) -> dict[int, str]:
        """
        Generates one question per document (context)

        :param documents: contexts to generate questions for
        :param start_word: question start word
        :param max_tokens: maximum length of the generation
        :param num_beams: number of beams for beam search
        :return: generated question
        """
        if len(documents) == 0:
            warnings.warn(
                "Empty list of documents passed to question generation model."
            )
            return []

        encoder_ids = self.tokenizer.batch_encode_plus(
            documents,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        decoder_input_ids = None
        if start_word:
            decoder_input_ids = self.tokenizer.batch_encode_plus(
                [start_word] * len(documents),
                add_special_tokens=True,
                return_tensors="pt",
            ).to(self._device)["input_ids"][:, :-1]

        model_output = self.model.generate(
            **encoder_ids,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_tokens,
            num_beams=num_beams,
        )
        generated_questions = self.tokenizer.batch_decode(
            model_output, skip_special_tokens=True
        )

        result_dict = {i: question for i, question in enumerate(generated_questions)}

        return result_dict