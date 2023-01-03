
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from logzero import logger


class DummyChitchatHandler:

    def ask_chitchat(self, query, context):
        return 'Toto bohužel nevím'


class ChitchatHandler:

    def __init__(self, config, remote_service_handler, device):
        self.remote_service_handler = remote_service_handler
        # TODO might need to support different classes
        self.tokenizer = AutoTokenizer.from_pretrained(config['MODEL'])
        model = AutoModelForSeq2SeqLM.from_pretrained(config['MODEL'])
        self.device = device
        self.model = model.to(self.device)
        self.decode_params = config['DECODE_PARAMS']

    def ask_chitchat(self, question: str, history: str):
        # translate into English
        en_transl = self.remote_service_handler.translate_cs2en(history + "\n" + question)
        history = en_transl.strip().split("\n")
        question = history.pop()
        logger.debug(f"ENQ: {'|'.join(history)} --|-- {question}")

        # run chitchat model
        concat = self.tokenizer.eos_token.join(history + [question]) + self.tokenizer.eos_token
        question = self.tokenizer.encode(concat, return_tensors='pt').to(self.device)
        reply = self.model.generate(question,
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    **self.decode_params)
        reply = self.tokenizer.decode(reply[0], skip_special_tokens=True)
        logger.debug(f"ENA: {reply}")

        # translate back to CS
        reply = self.remote_service_handler.translate_en2cs(reply)
        return reply
