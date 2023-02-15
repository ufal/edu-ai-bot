import random
from collections import defaultdict
from typing import Text
from time import time as current_time_seconds
from dataclasses import dataclass, field
from abc import ABC

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from logzero import logger
from fuzzywuzzy import fuzz


class ContextStorage(ABC):
    def store_utterance(self, utterance: Text, conv_id: Text):
        raise NotImplementedError

    def retrieve_context(self, conv_id: Text):
        raise NotImplementedError


@dataclass
class InMemoryEntry:
    context: list = field(default_factory=lambda: [])
    time_created: float = field(default_factory=lambda: current_time_seconds())


class InMemoryContextStorage(ContextStorage):
    def __init__(self):
        self.context_cache = defaultdict(InMemoryEntry)

    def store_utterance(self, utterance: Text, conv_id: Text):
        self.context_cache[conv_id].context.append(utterance)

    def retrieve_context(self, conv_id: Text):
        return self.context_cache[conv_id].context

    def clear_cache(self):
        current_time = current_time_seconds()
        to_be_deleted = [conv_id for conv_id, entry in self.context_cache.items()
                     if current_time - entry.time_created > 7200]
        for id_to_delete in to_be_deleted:
            del self.context_cache[id_to_delete]


class ChitchatHandler(ABC):
    def __init__(self):
        self.context_storage = InMemoryContextStorage()

    def get_reply(self, context: Text):
        raise NotImplementedError

    def _wrap_utterance(self, utterance: Text):
        return utterance

    def ask_chitchat(self, question: Text, conv_id: Text):
        context = "".join(self.context_storage.retrieve_context(conv_id))
        context += question
        reply = self.get_reply(context)
        self.context_storage.store_utterance(self._wrap_utterance(question), conv_id)
        self.context_storage.store_utterance(self._wrap_utterance(reply), conv_id)
        self.context_storage.clear_cache()
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
