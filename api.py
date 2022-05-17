#!/usr/bin/env python3

import os
import flask
from argparse import ArgumentParser
from flask import request, jsonify
from solr_query import ask_solr
from logzero import logger

app = flask.Flask(__name__)
app.config["DEBUG"] = True

QA_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'multilingual_qaqg')

if os.path.isdir(QA_MODEL_PATH):
    from multilingual_qaqg.mlpipelines import pipeline
    qa_model = pipeline("multitask-qa-qg",
                        os.path.join(QA_MODEL_PATH, "checkpoint-185000"),
                        os.path.join(QA_MODEL_PATH, "mt5_qg_tokenizer"))
else:
    logger.warn('Could not find QA directory, will run without it')
    qa_model = None


@app.route('/', methods=['POST'])
def ask():
    query = request.json['q']
    logger.info(f'Q: {query}')
    db_result = ask_solr(query=query)

    if not db_result.get('docs'):
        logger.info(f'No result.')
        return jsonify({'a': 'Toto bohužel nevím.'})

    for doc in db_result['docs']:
        logger.info(f'D: {doc["title"]}')
    answer = db_result['docs'][0]
    url = 'https://cs.wikipedia.org/wiki/' + answer["title"].replace(' ', '_')
    if request.json.get('exact') and qa_model:
        response = qa_model({'question': query, 'context': answer['first_paragraph']})
        return jsonify({'a': f'Myslím, že {response} (Zdroj: {url})'})
    return jsonify({'a': f'Tohle by vám mohlo pomoct: {answer["first_paragraph"]} (Zdroj: {url})'})


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-p', '--port', type=int, default=8200, help="Port to listen on.")
    args = ap.parse_args()

    app.run(host='0.0.0.0', port=args.port)
