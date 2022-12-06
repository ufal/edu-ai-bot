#!/usr/bin/env python3

import os
import datetime
import json
import random
from argparse import ArgumentParser

import torch
import flask
import yaml
from yaml.loader import SafeLoader
from flask import request, jsonify
from logzero import logger

from edubot.remote_services import RemoteServiceHandler
from edubot.utils import apply_qa


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['POST'])
def ask():
    print(handcrafted_responses)
    if not request.json or 'q' not in request.json:
        return "No query given.", 400
    logger.info(f"Query: {request.json['q']}")
    query = remote_service_handler.correct_diacritics(request.json['q'])
    logger.info(f"Korektor: {query}")
    exact = request.json.get('exact')
    context, title, url = None, None, None
    intent = intent_clf_model.predict_example(query)[0] if intent_clf_model else None
    logger.info(f"Intent: {intent}")

    if intent in custom_config['CHITCHAT_INTENTS']:
        response = remote_service_handler.ask_chitchat(query)
    elif intent in handcrafted_responses:
        available_responses = handcrafted_responses[intent]
        response = random.choice(available_responses)
    else:
        context, retrieved_response, title, url = apply_qa(remote_service_handler, qa_model, query, None, exact)
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
    if 'LOGFILE_PATH' in custom_config and custom_config['LOGFILE_PATH'] is not None:
        log_data = {'timestamp': str(datetime.datetime.now()),
                    'request': {'remote_addr': request.remote_addr,
                                'url': request.url,
                                'json': request.json},
                    'response': {'text': response_dict,
                                 'url': url,
                                 'context': context,
                                 'intent': intent,
                                 'korektor': query}}
        with open(custom_config['LOGFILE_PATH'], 'a', encoding='UTF_8') as fh:
            fh.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            fh.flush()

    return jsonify(response_dict)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-p', '--port', type=int, default=8200, help="Port to listen on.")
    ap.add_argument('-ha', '--host_addr', type=str, default='0.0.0.0')
    ap.add_argument('-c', '--config', type=str, help='Path to yaml configuration file')
    ap.add_argument('-l', '--logfile', type=str, help='Path to a file to log requests')
    ap.add_argument('-d', '--debug', '--flask-debug', action='store_true', help='Show flask debug messages')
    args = ap.parse_args()
    with open(args.config, 'rt') as fd:
        custom_config = yaml.load(fd, Loader=SafeLoader)
    if args.logfile:
        custom_config['LOGFILE_PATH'] = args.logfile

    stopwords = {'co', 'kdo', 'kdy', 'kde', 'jak', 'kolik', 'být', 'ten', '?', '.', ':', '!', 'znamenat',
                 'jaký', 'který', 'mít', 'proč'}
    remote_service_handler = RemoteServiceHandler(custom_config, stopwords)

    if os.path.isdir(custom_config['QA_MODEL_PATH']):
        from multilingual_qaqg.mlpipelines import pipeline

        qa_model = pipeline("multitask-qa-qg",
                            os.path.join(custom_config['QA_MODEL_PATH'], "checkpoint-185000"),
                            os.path.join(custom_config['QA_MODEL_PATH'], "mt5_qg_tokenizer"))
    else:
        logger.warn('Could not find QA directory, will run without it')
        qa_model = None

    if os.path.isdir(custom_config['INTENT_MODEL_PATH']):
        from edubot.educlf.model import IntentClassifierModel

        intent_clf_model = IntentClassifierModel(None, torch.device('cpu:0'), None, None)
        intent_clf_model.load_from(custom_config['INTENT_MODEL_PATH'])
    else:
        logger.warn('Could not find intent model directory, will run without intent model.')
        intent_clf_model = None

    if not os.path.exists(custom_config['HC_RESPONSES_PATH']):
        logger.warn('Could not find handcrafted responses, will run without them.')
        handcrafted_responses = dict()
    else:
        with open(custom_config['HC_RESPONSES_PATH'], 'rt') as fd:
            handcrafted_responses = yaml.load(fd, Loader=SafeLoader)

    app.run(host=args.host_addr, port=args.port, debug=args.debug)
