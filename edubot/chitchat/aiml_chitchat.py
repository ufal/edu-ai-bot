import os
import aiml
import csv
import json
import random
import re

from logzero import logger

class AIMLChitchat:
    def __init__(self, config, remote_service_handler):
        BRAIN_FILE=config["CHITCHAT"]["AIML_DATA_FILE"]
        STARTUP_FILE=config["CHITCHAT"]["AIML_STARTUP_FILE"]

        k = aiml.Kernel()

        logger.info("Parsing aiml files")
        k.bootstrap(learnFiles=STARTUP_FILE, commands="load aiml b")
        self.load_bot_properties(k, config["CHITCHAT"]["AIML_BOT_PROPERTIES_FILE"], config["CHITCHAT"]["AIML_SUBSTITUTIONS_FILE"])
        logger.info("Saving brain file: " + BRAIN_FILE)
        k.saveBrain(BRAIN_FILE)

        self.kernel = k
        self.remote_service_handler = remote_service_handler

    def load_bot_properties(self, kernel, properties_filename, substitutions_filename):
        with open(properties_filename, 'rt') as fd:
            tsv_reader = csv.DictReader(fd, delimiter="\t")
            for row in tsv_reader:
                kernel.setBotPredicate(row['name'], row['value'])

        with open(substitutions_filename, 'rt', encoding='UTF-8') as fh:
            substs = json.load(fh)
            # we could use kernel.loadSubs from an .ini file, but that would overwrite the defaults, here we're just adding more
            for subst_k, subst_v in substs:
                kernel._subbers['normal'][subst_k.strip()] = subst_v.strip()

    def postprocess_response(self, resp):
        # basic cleanup of weird AIML stuff
        resp = re.sub(r'""\?', '', resp)
        resp = re.sub(r'[?!.]([" ])\?', r'\1?', resp)
        resp = re.sub(r'""', '"', resp)
        resp = re.sub(r'\bis  +is\b', 'is', resp, flags=re.I)
        resp = re.sub(r'"it"', 'it', resp, flags=re.I)
        resp = re.sub(r'^"(.*)"$', r'\1', resp)
        return resp

    def ask_chitchat(self, input_text:str, conv_id, context):
        logger.debug(f"Question: {input_text}")
        en_transl = self.remote_service_handler.translate_cs2en(input_text)
        logger.debug(f"EN Question: {en_transl}")
        response = self.kernel.respond(en_transl, sessionID=conv_id)
        response = self.postprocess_response(response)
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
