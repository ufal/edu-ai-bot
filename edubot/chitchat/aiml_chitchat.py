import os
import aiml
import csv

from logzero import logger

class AIMLChitchat:
    def __init__(self, config, remote_service_handler):
        BRAIN_FILE=config["CHITCHAT"]["AIML_DATA_FILE"]
        STARTUP_FILE=config["CHITCHAT"]["AIML_STARTUP_FILE"]

        k = aiml.Kernel()

        logger.info("Parsing aiml files")
        k.bootstrap(learnFiles=STARTUP_FILE, commands="load aiml b")
        self.load_bot_properties(k, config["CHITCHAT"]["AIML_BOT_PROPERTIES_FILE"])
        logger.info("Saving brain file: " + BRAIN_FILE)
        k.saveBrain(BRAIN_FILE)

        self.kernel = k
        self.remote_service_handler = remote_service_handler

    def load_bot_properties(self, kernel, filename):
        with open(filename, 'rt') as fd:
            tsv_reader = csv.DictReader(fd, delimiter="\t")
            for row in tsv_reader:
                kernel.setBotPredicate(row['name'], row['value'])

    def ask_chitchat(self, input_text:str, conv_id):
        logger.debug(f"Question: {input_text}")
        en_transl = self.remote_service_handler.translate_cs2en(input_text)
        logger.debug(f"EN Question: {en_transl}")
        response = self.kernel.respond(en_transl)
        logger.debug(f"EN Answer: {response}")
        if not response:
            # trying without context
            response = self.kernel.respond("random pickup line")
            logger.debug(f"EN Backup Answer: {response}")
        if not response:
            response = random.choice(['Go on.', 'Oh, right.', 'I see.', 'I\'m not sure I understand you.'])
        cs_transl = self.remote_service_handler.translate_en2cs(response)
        logger.debug(f"CS Answer: {cs_transl}")
        return cs_transl
