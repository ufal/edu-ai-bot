#!/usr/bin/env python3

import os
import datetime
import json
from argparse import ArgumentParser

import flask
from flask import request, jsonify
from solr_query import ask_solr, filter_query
from logzero import logger

app = flask.Flask(__name__)
app.config["DEBUG"] = True

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


def apply_qa(query, context=None, exact=False):

    filtered_query, query_type = filter_query(query)
    logger.info(f'Q: {query} | F: {filtered_query}')
    if not context and filtered_query:
        for q, a in [(f'"{filtered_query}"', 'title_str'),
                     (f'"{filtered_query}"', 'title_cz'),
                     (f'"{filtered_query}"', 'first_paragraph_cz'),
                     (filtered_query, 'title_cz'),
                     (filtered_query, 'first_paragraph_cz')]:
            db_result = ask_solr(query=q, attrib=a)
            if db_result.get('docs'):
                break

        if not db_result.get('docs'):
            logger.info(f'No result.')
            return None, None, None

        logger.debug("\n" + "\n".join([f'D: {doc["title"]}/{doc["score"]}' for doc in db_result['docs']]))

        answers = db_result['docs']
        title = answers[0]["title"]
    else:
        answers = [{'first_paragraph': None}]
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
        return context, response, title
    return answers[0]["first_paragraph"], None, title


@app.route('/', methods=['POST'])
def ask():
    query = request.json['q']
    exact = request.json.get('exact')
    context, response, title = apply_qa(query, None, exact)
    if not response and not context:
        res = {'a': 'Toto bohužel nevím.'}
    else:
        url = 'https://cs.wikipedia.org/wiki/' + title.replace(' ', '_')
        if response:
            res = {'a': f'Myslím, že {response} (Zdroj: {url})'}
        else:
            res = {'a': f'Tohle by vám mohlo pomoct: {context} (Zdroj: {url})'}

    # file logging
    if LOGFILE_PATH:
        log_data = {'timestamp': str(datetime.datetime.now()),
                    'request': {'remote_addr': request.remote_addr,
                                'url': request.url,
                                'json': request.json},
                    'response': {'text': res,
                                 'url': url,
                                 'context': context}}
        with open(LOGFILE_PATH, 'a', encoding='UTF_8') as fh:
            fh.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            fh.flush()

    return jsonify(res)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-p', '--port', type=int, default=8200, help="Port to listen on.")
    ap.add_argument('-l', '--logfile', type=str, help='Path to a file to log requests')
    args = ap.parse_args()
    if args.logfile:
        LOGFILE_PATH = args.logfile

    app.run(host='0.0.0.0', port=args.port)
