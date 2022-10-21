#!/usr/bin/env python3

import os
import datetime
import json
from argparse import ArgumentParser

import torch
import flask
from flask import request, jsonify
from solr_query import ask_solr, filter_query, ask_chitchat, correct_diacritics
from logzero import logger

app = flask.Flask(__name__)
app.config["DEBUG"] = True

CHITCHAT_INTENTS = ['bye', 'dik', 'wellcome', 'chch']
INTENT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'educlf', 'trained_model')
QA_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'multilingual_qaqg')
LOGFILE_PATH = None  # can be set in parameters


if os.path.isdir(QA_MODEL_PATH):
    from multilingual_qaqg.mlpipelines import pipeline
    qa_model = pipeline("multitask-qa-qg",
                        os.path.join(QA_MODEL_PATH, "checkpoint-185000"),
                        os.path.join(QA_MODEL_PATH, "mt5_qg_tokenizer"))
else:
    logger.warn('Could not find QA directory, will run without it')
    qa_model = None

if os.path.isdir(INTENT_MODEL_PATH):
    from educlf.educlf.model import IntentClassifierModel
    intent_clf_model = IntentClassifierModel(None, torch.device('cpu:0'), None, None)
    intent_clf_model.load_from(INTENT_MODEL_PATH)
else:
    logger.warn('Could not find intent model directory, will run without intent model.')
    intent_clf_model = None


def apply_qa(query, context=None, exact=False):

    filtered_query_nac, filtered_query_nacv, query_type = filter_query(query)
    logger.info(f'Q: {query} | F: {filtered_query_nac} | {filtered_query_nacv}')
    if not context and filtered_query_nacv:
        for q, a, s in [(filtered_query_nacv, 'title_cz', 'logic'),
                        (filtered_query_nacv, 'first_paragraph_cz', 'logic'),
                        (f'"{filtered_query_nac}"', 'title_str', 'wiki'),
                        (f'"{filtered_query_nac}"', 'title_cz', 'wiki'),
                        (f'"{filtered_query_nac}"', 'first_paragraph_cz', 'wiki'),
                        (filtered_query_nac, 'title_cz', 'wiki'),
                        (filtered_query_nacv, 'first_paragraph_cz', 'wiki')]:
            if not q:  # skip if filtered_query_nac is empty
                continue
            db_result = ask_solr(query=q, attrib=a, source=s)
            if db_result.get('docs'):
                break

        if not db_result.get('docs'):
            logger.info(f'No result.')
            return None, None, None, None

        logger.debug("\n" + "\n".join([f'D: {doc["title"]}/{doc["score"]}' for doc in db_result['docs']]))

        answers = db_result['docs']
        title = answers[0]["title"]
    else:
        answers = [{'first_paragraph': None, 'url': None}]
        title = None

    if exact and query_type == 'default' and qa_model:
        # reranking by QA decoding score -- doesn't seem to work
        #resp_cands = []
        #for context in [a['first_paragraph'] for a in answers[:1]]:
            #resp_cands.append(qa_model({'question': query, 'context': context}))
        #logger.debug('RCs:\n' + "\n".join(['RC: %s | %f' % rc for rc in resp_cands]))
        #response, _ = max(resp_cands, key=lambda rc: rc[1])
        # feeding multiple contexts -- doesn't seem to work
        if not context:
            context = "\n".join([a['first_paragraph'] for a in answers[:1]])
        response, _ = qa_model({'question': query, 'context': context})
        return context, response, title, answers[0]["url"]
    return answers[0]["first_paragraph"], None, title, answers[0]["url"]


@app.route('/', methods=['POST'])
def ask():
    if not request.json or 'q' not in request.json:
        return "No query given.", 400
    logger.info(f"Query: {request.json['q']}")
    query = correct_diacritics(request.json['q'])
    logger.info(f"Korektor: {query}")
    exact = request.json.get('exact')
    context, response, title, url = None, None, None, None
    intent = intent_clf_model.predict_example(query)[0] if intent_clf_model else None
    logger.info(f"Intent: {intent}")
    if intent not in CHITCHAT_INTENTS:
        context, response, title, url = apply_qa(query, None, exact)
    if not response and not context:
        response = ask_chitchat(query)
        res = {'a': response}
    else:
        if 'wikipedia' in url:
            url = 'https://cs.wikipedia.org/wiki/' + title.replace(' ', '_')
            if response:
                res = {'a': f'Myslím, že {response} (Zdroj: {url})'}
            else:
                res = {'a': f'Tohle by vám mohlo pomoct: {context} (Zdroj: {url})'}
        else:
            res = {'a': f'{context} (Zdroj: {url})'}
    res['intent'] = intent

    # file logging
    if LOGFILE_PATH:
        log_data = {'timestamp': str(datetime.datetime.now()),
                    'request': {'remote_addr': request.remote_addr,
                                'url': request.url,
                                'json': request.json},
                    'response': {'text': res,
                                 'url': url,
                                 'context': context,
                                 'intent': intent,
                                 'korektor': query}}
        with open(LOGFILE_PATH, 'a', encoding='UTF_8') as fh:
            fh.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            fh.flush()

    return jsonify(res)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-p', '--port', type=int, default=8200, help="Port to listen on.")
    ap.add_argument('-l', '--logfile', type=str, help='Path to a file to log requests')
    ap.add_argument('-d', '--debug', '--flask-debug', action='store_true', help='Show flask debug messages')
    args = ap.parse_args()
    if args.logfile:
        LOGFILE_PATH = args.logfile

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
