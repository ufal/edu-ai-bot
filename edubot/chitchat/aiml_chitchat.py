import os
import aiml

from logzero import logger

class AIMLChitchat:
    def __init__(self, config, remote_service_handler):
        BRAIN_FILE=config["CHITCHAT"]["AIML_DATA_FILE"]
        STARTUP_FILE=config["CHITCHAT"]["AIML_STARTUP_FILE"]

        k = aiml.Kernel()

        # To increase the startup speed of the bot it is
        # possible to save the parsed aiml files as a
        # dump. This code checks if a dump exists and
        # otherwise loads the aiml from the xml files
        # and saves the brain dump.
        if os.path.exists(BRAIN_FILE):
            logger.info("Loading from brain file: " + BRAIN_FILE)
            k.loadBrain(BRAIN_FILE)
        else:
            logger.info("Parsing aiml files")
            k.bootstrap(learnFiles=STARTUP_FILE, commands="load aiml b")
            logger.info("Saving brain file: " + BRAIN_FILE)
            k.saveBrain(BRAIN_FILE)

        self.kernel = k
        self.remote_service_handler = remote_service_handler

    def ask_chitchat(self, input_text:str, conv_id):
        logger.debug(f"Question: {input_text}")
        en_transl = self.remote_service_handler.translate_cs2en(input_text)
        logger.debug(f"EN Question: {en_transl}")
        response = self.kernel.respond(en_transl)
        logger.debug(f"EN Answer: {response}")
        cs_transl = self.remote_service_handler.translate_en2cs(response)
        logger.debug(f"CS Answer: {cs_transl}")
        return cs_transl