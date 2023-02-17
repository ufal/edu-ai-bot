#!/usr/bin/env python3

import os
import datetime
import json
import random
from argparse import ArgumentParser
from time import strftime
import re

import torch
import flask
import yaml
from yaml.loader import SafeLoader
from flask import request, jsonify
from logzero import logger

from edubot.remote_services import RemoteServiceHandler
from edubot.qa import QAHandler, OpenAIQA, OpenAIReformulate
from edubot.chitchat.seq2seq import Seq2SeqChitchatHandler, DummyChitchatHandler
from edubot.chitchat.aiml_chitchat import AIMLChitchat
from edubot.educlf.model import IntentClassifierModel
from langchain import OpenAI, PromptTemplate, LLMChain


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['POST'])
def ask():
    if not request.json or 'q' not in request.json:
        return "No query given.", 400
    logger.info(f"Query: {request.json['q']}")

    conv_id = str(request.json.get('conversation_id', 'default_id'))
    query = remote_service_handler.correct_diacritics(request.json['q'])
    logger.info(f"Korektor: {query}")

    exact = request.json.get('exact')
    force_wiki = request.json.get('w')
    context, title, url = None, None, None
    if force_wiki or query.startswith('/w'):
        query = re.sub(r'^/w\s+', '', query)
        intent, intent_conf = 'qawiki', 1.0
    elif query.startswith('/c'):
        query = re.sub(r'^/w\s+', '', query)
        intent, intent_conf = 'chch', 1.0
    else:
        intent, intent_conf = intent_clf_model.predict_example(query)[0] if intent_clf_model else (None, None)
    logger.info(f"Intent: {intent} ({intent_conf})")

    if intent in config['CHITCHAT_INTENTS']:
        response = chitchat_handler.ask_chitchat(query, conv_id)
    elif intent in handcrafted_responses:
        available_responses = handcrafted_responses[intent]
        response = random.choice(available_responses)
        response = response.replace('timenow()', strftime('%H:%M'))
        response = response.replace('datenow()', strftime('%d.%m.%Y'))
    else:
        context, retrieved_response, title, url = qa_handler.apply_qa(query, context=None, exact=exact)
        if not retrieved_response and not context:
            response = 'Promiňte, teď jsem nerozuměl.'
        else:
            if 'wikipedia' in url:
                url = 'https://cs.wikipedia.org/wiki/' + title.replace(' ', '_')
                if retrieved_response:
                    response = f'Myslím, že {retrieved_response} (Zdroj: {url})'
                else:
                    response = f'Tohle by vám mohlo pomoct: {context} (Zdroj: {url})'
            else:
                response = f'{context} (Zdroj: {url})'

    response_dict = {
        'a': response,
        'intent': intent
    }
    # file logging
    if ('LOGFILE_PATH' in config) and (config['LOGFILE_PATH'] is not None):
        logger.info(f"Logging into {config['LOGFILE_PATH']}")
        log_data = {'timestamp': str(datetime.datetime.now()),
                    'request': {'remote_addr': request.remote_addr,
                                'url': request.url,
                                'json': request.json},
                    'response': {'text': response_dict,
                                 'url': url,
                                 'context': context,
                                 'intent': intent,
                                 'korektor': query}}
        with open(config['LOGFILE_PATH'], 'a', encoding='UTF_8') as fh:
            fh.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            fh.flush()

    return jsonify(response_dict)


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('-p', '--port', type=int, default=8200, help="Port to listen on.")
    ap.add_argument('-ha', '--host-addr', type=str, default='0.0.0.0')
    ap.add_argument('-c', '--config', type=str, help='Path to yaml configuration file', default=os.path.join("configs", "default_config.yaml"))
    ap.add_argument('-l', '--logfile', type=str, help='Path to a file to log requests',)# default="DefaultLog.log")
    ap.add_argument('-d', '--debug', '--flask-debug', action='store_true', help='Show flask debug messages')
    ap.add_argument('--cuda', action='store_true', help='Use GPU (true by default)')
    ap.add_argument('--no-cuda', dest='cuda', action='store_false')
    ap.set_defaults(cuda=True)

    args = ap.parse_args()

    # get config
    logger.info(f"Loading config from: {args.config}")
    with open(args.config, 'rt') as fd:
        config = yaml.load(fd, Loader=SafeLoader)
    config['LOGFILE_PATH'] = args.logfile

    # set default device based on CUDA config
    device = torch.device('cuda') if (args.cuda and torch.cuda.is_available()) else torch.device('cpu:0')
    logger.info(f"Running on {device}")

    # load remote services handler
    remote_service_handler = RemoteServiceHandler(config)

    # load QA
    qa_handler = QAHandler(config, remote_service_handler, device)

    # load chitchat
    chitchat_model_name = config.get('CHITCHAT', {'MODEL': None})['MODEL']
    if chitchat_model_name == "AIML":
        chitchat_handler = AIMLChitchat(config, remote_service_handler)
    elif chitchat_model_name:
        chitchat_handler = Seq2SeqChitchatHandler(chitchat_model_name,
                                                  remote_service_handler,
                                                  device)
    else:
        logger.warning('No chitchat model defined, will run without it')
        chitchat_handler = DummyChitchatHandler()
    logger.info(f'Chitchat model: {chitchat_model_name} / {str(type(chitchat_handler))}')

    # load intent classifier
    intent_clf_model = IntentClassifierModel(None, device, None, None, config)
    intent_clf_model.load_from()

    # load handcrafted responses
    if not os.path.exists(config['HC_RESPONSES_PATH']):
        logger.warning('Could not find handcrafted responses, will run without them.')
        handcrafted_responses = dict()
    else:
        with open(config['HC_RESPONSES_PATH'], 'rt') as fd:
            handcrafted_responses = yaml.load(fd, Loader=SafeLoader)

    # run the stuff
    app.run(host=args.host_addr, port=args.port, debug=args.debug)
