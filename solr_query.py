import json
import requests

from logzero import logger
from werkzeug.urls import url_fix

URL_TO_SOLR = "http://quest.ms.mff.cuni.cz/namuddis/qasolr/wiki_test/query?q={query}&wt=json"

def ask_solr(*, query):
    response = requests.get(
        url_fix(URL_TO_SOLR.format(query=query))
    )
    j = json.loads(response.content.decode('utf8'))['response']
    logger.info(j)

    return j

    # response = json.load(connection.response)
    # logger.info(response)
    # return response
