#!/usr/bin/env python3

import flask
from argparse import ArgumentParser
from flask import request, jsonify
from solr_query import ask_solr
from logzero import logger


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['POST'])
def ask():
    query = request.json['q']
    logger.info(f'Q: {query}')
    answer = ask_solr(query=query)

    for doc in answer['docs']:
        logger.info(f'D: {doc["title"]}')
    return jsonify({'a': answer})


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-p', '--port', type=int, default=8200, help="Port to listen on.")
    args = ap.parse_args()

    app.run(host='0.0.0.0', port=args.port)
