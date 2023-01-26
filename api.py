#!/usr/bin/env python3

import os
import datetime
import json
import random
from argparse import ArgumentParser
from time import strftime

import torch
import flask
import yaml
from yaml.loader import SafeLoader
from flask import request, jsonify
from logzero import logger

from edubot.remote_services import RemoteServiceHandler
from edubot.qa import QAHandler
from edubot.chitchat import Seq2SeqChitchatHandler, DummyChitchatHandler


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
    context, title, url = None, None, None
    if query.startswith('/w'):
        query = query[2:].strip()
        intent, intent_conf = 'qawiki', 1.0
    else:
        intent, intent_conf = intent_clf_model.predict_example(query)[0] if intent_clf_model else (None, None)
    logger.info(f"Intent: {intent} ({intent_conf})")

    if intent in custom_config['CHITCHAT_INTENTS']:
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
    ap.add_argument('-ha', '--host-addr', type=str, default='0.0.0.0')
    ap.add_argument('-c', '--config', type=str, help='Path to yaml configuration file', required=True)
    ap.add_argument('-l', '--logfile', type=str, help='Path to a file to log requests')
    ap.add_argument('-d', '--debug', '--flask-debug', action='store_true', help='Show flask debug messages')
    ap.add_argument('--cuda', action='store_true', help='Use GPU (true by default)')
    ap.add_argument('--no-cuda', dest='cuda', action='store_false')
    ap.set_defaults(cuda=True)

    args = ap.parse_args()

    # get config
    with open(args.config, 'rt') as fd:
        custom_config = yaml.load(fd, Loader=SafeLoader)
    if args.logfile:
        custom_config['LOGFILE_PATH'] = args.logfile

    # set default device based on CUDA config
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu:0')

    # load remote services handler
    with open(custom_config['STOPWORDS_PATH'], 'rt') as fd:
        stopwords = set((w.strip() for w in fd.readlines() if len(w.strip()) > 0))
    remote_service_handler = RemoteServiceHandler(custom_config, stopwords)

    # load QA
    if os.path.isdir(custom_config['QA_MODEL_PATH']):
        from multilingual_qaqg.mlpipelines import pipeline

        qa_model = pipeline("multitask-qa-qg",
                            os.path.join(custom_config['QA_MODEL_PATH'], "checkpoint-185000"),
                            os.path.join(custom_config['QA_MODEL_PATH'], "mt5_qg_tokenizer"),
                            use_cuda=args.cuda)
    else:
        logger.warn('Could not find QA directory, will run without it')
        qa_model = None

    if custom_config['SENTENCE_REPR_MODEL'].lower() in ['robeczech', 'eleczech']:
        from edubot.educlf.model import IntentClassifierModel
        sentence_repr_model = IntentClassifierModel(custom_config['SENTENCE_REPR_MODEL'],
                                                    device,
                                                    label_mapping=None,
                                                    out_dir=None)
    else:
        from sentence_transformers import SentenceTransformer
        sentence_repr_model = SentenceTransformer(custom_config['SENTENCE_REPR_MODEL'],
                                                  device=device)
    qa_handler = QAHandler(qa_model, sentence_repr_model, remote_service_handler)

    # load chitchat
    if custom_config.get('CHITCHAT', {'MODEL': None})['MODEL']:
        chitchat_handler = Seq2SeqChitchatHandler(custom_config['CHITCHAT'],
                                                  remote_service_handler,
                                                  device)
    else:
        logger.warn('No chitchat model defined, will run without it')
        chitchat_handler = DummyChitchatHandler()

    # load intent classifier
    if os.path.isdir(custom_config['INTENT_MODEL_PATH']):
        from edubot.educlf.model import IntentClassifierModel

        intent_clf_model = IntentClassifierModel(None, device, None, None)
        intent_clf_model.load_from(custom_config['INTENT_MODEL_PATH'])
    else:
        logger.warn('Could not find intent model directory, will run without intent model.')
        intent_clf_model = None

    # load handcrafted responses
    if not os.path.exists(custom_config['HC_RESPONSES_PATH']):
        logger.warn('Could not find handcrafted responses, will run without them.')
        handcrafted_responses = dict()
    else:
        with open(custom_config['HC_RESPONSES_PATH'], 'rt') as fd:
            handcrafted_responses = yaml.load(fd, Loader=SafeLoader)

    # run the stuff
    app.run(host=args.host_addr, port=args.port, debug=args.debug)
