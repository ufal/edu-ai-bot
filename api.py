#!/usr/bin/env python3

import os
import flask
from argparse import ArgumentParser
from flask import request, jsonify
from solr_query import ask_solr, filter_query
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
    filtered_query, query_type = filter_query(query)
    logger.info(f'Q: {query} | F: {filtered_query}')
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
        return jsonify({'a': 'Toto bohužel nevím.'})

    logger.debug("\n" + "\n".join([f'D: {doc["title"]}/{doc["score"]}' for doc in db_result['docs']]))

    answers = db_result['docs']
    url = 'https://cs.wikipedia.org/wiki/' + answers[0]["title"].replace(' ', '_')
    if request.json.get('exact') and query_type == 'default' and qa_model:
        # reranking by QA decoding score -- doesn't seem to work
        #resp_cands = []
        #for context in [a['first_paragraph'] for a in answers[:1]]:
            #resp_cands.append(qa_model({'question': query, 'context': context}))
        #logger.debug('RCs:\n' + "\n".join(['RC: %s | %f' % rc for rc in resp_cands]))
        #response, _ = max(resp_cands, key=lambda rc: rc[1])
        # feeding multiple contexts -- doesn't seem to work
        context = "\n".join([a['first_paragraph'] for a in answers[:1]])
        response, _ = qa_model({'question': query, 'context': context})
        return jsonify({'a': f'Myslím, že {response} (Zdroj: {url})'})
    return jsonify({'a': f'Tohle by vám mohlo pomoct: {answers[0]["first_paragraph"]} (Zdroj: {url})'})


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-p', '--port', type=int, default=8200, help="Port to listen on.")
    args = ap.parse_args()

    app.run(host='0.0.0.0', port=args.port)
