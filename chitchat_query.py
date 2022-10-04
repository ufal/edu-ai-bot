
import requests
from logzero import logger


CHITCHAT_URL = 'http://localhost:8200'


def ask_chitchat(query):
    logger.info('Request "%s" at %s' % (query, CHITCHAT_URL))
    resp = requests.post(CHITCHAT_URL, json={'q': query})
    try:
        reply = resp.json()['a'].strip()
        logger.info('Reply: "%s"' % reply)
        return reply
    except Exception as e:
        logger.error(str(e))
        return 'Toto bohužel nevím.'
