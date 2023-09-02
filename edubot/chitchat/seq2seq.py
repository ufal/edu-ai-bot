import random
from typing import Text, List
from abc import ABC

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from logzero import logger
from fuzzywuzzy import fuzz


class ChitchatHandler(ABC):
    def __init__(self):
        pass

    def get_reply(self, context: Text):
        raise NotImplementedError

    def _wrap_utterance(self, utterance: Text):
        return utterance

    def ask_chitchat(self, question: Text, conv_id: Text, context: List):
        context = "".join([self._wrap_utterance(u) for u in context])
        context += question
        reply = self.get_reply(context)
        return reply


class DummyChitchatHandler(ChitchatHandler):

    def get_reply(self, context: Text):
        return 'Toto bohužel nevím'


class Seq2SeqChitchatHandler(ChitchatHandler):

    def __init__(self, config, remote_service_handler, device):
        super().__init__()
        self.remote_service_handler = remote_service_handler
        # TODO might need to support different classes
        self.maxlen = config['MAXLEN']
        logger.info(f"Max-length for the chitchat: {self.maxlen}")
        model = AutoModelForSeq2SeqLM.from_pretrained(config['MODEL'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['MODEL'],
                                                       model_max_length=self.maxlen,
                                                       truncation_side='left')
        self.fallback_replies = config['FALLBACK_REPLIES']
        self.device = device
        self.model = model.to(self.device)
        self.decode_params = config['DECODE_PARAMS']
        self.inappropriate_kws = config['INAPPROPRIATE_KWS']

    def get_reply(self, context: Text):
        # translate into English
        en_transl = self.remote_service_handler.translate_cs2en(context)
        history = en_transl.strip().split("\n")
        question = history.pop()
        logger.debug(f"ENQ: {'|'.join(history)} --|-- {question}")

        # run chitchat model
        concat = self.tokenizer.eos_token.join(history + [question]) + self.tokenizer.eos_token
        question = self.tokenizer.encode(concat,
                                         max_length=self.maxlen,
                                         truncation=True,
                                         return_tensors='pt').to(self.device)
        # TODO: look at bad_word_ids
        # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.bad_words_ids(List[List[int]],
        reply_candidates = self.model.generate(question,
                                               pad_token_id=self.tokenizer.eos_token_id,
                                               num_return_sequences=5,
                                               **self.decode_params)
        for candidate in reply_candidates:
            candidate = self.tokenizer.decode(candidate, skip_special_tokens=True)
            if not self._is_inappropriate(candidate):
                reply = candidate
                break
            logger.info('Throwing out profane reply: %s', candidate)
        else:
            reply = random.choice(self.fallback_replies)

        logger.debug(f"ENA: {reply}")

        # translate back to CS
        reply = self.remote_service_handler.translate_en2cs(reply)
        reply = self.remote_service_handler.postprocess_gender(reply)
        return reply

    def _is_inappropriate(self, utterance: Text):
        return any((fuzz.partial_ratio(utterance.lower(), kw) > 75 for kw in self.inappropriate_kws))

    def _wrap_utterance(self, utterance: Text):
        return f"{utterance.strip()}\n"
