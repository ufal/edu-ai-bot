
import json
import requests

from logzero import logger
from werkzeug.urls import url_fix
from edubot.utils import dotdict


class RemoteServiceHandler:

    def __init__(self, config, stopwords):
        self.urls = config['URLs']
        self.stopwords = stopwords

    def ask_solr(self, query, attrib=None, source='wiki'):
        url = self.urls['WIKI']
        if source == 'logic':
            url = self.urls['LOGIC']
        if attrib is not None:
            if isinstance(attrib, list):
                query = ' OR '.join([f'{a}:{query}' for a in attrib])
            else:
                query = f'{attrib}:{query}'
        query = f'({query}) AND url:{url}'
        response = requests.get(
            url_fix(self.urls['URL_SOLR'].format(query=query))
        )
        j = json.loads(response.content.decode('utf8'))['response']
        logger.debug(query + "\n" + str(j))

        return j

    def filter_query(self, query):
        try:
            udpipe = requests.get(url_fix(self.urls['URL_UDPIPE'].format(query=query)))
            tagged = [line.split("\t") for line in udpipe.json()['result'].split("\n")
                      if "\t" in line and not line.startswith('#')]
            tagged = [dotdict({'form': w[1], 'lemma': w[2], 'tag': w[4]}) for w in tagged]
        except Exception as e:
            logger.warn('UDpipe problem:' + str(e))
            return query
        logger.debug("\n" + "\n".join(["\t".join([w['form'], w['lemma'], w['tag']]) for w in tagged]))
        filtered_nac = " ".join([w.form for w in tagged
                                 if w.tag[0] in set(['N', 'A', 'C']) and w.lemma not in self.stopwords])
        filtered_nacv = " ".join([w.form for w in tagged
                                  if w.tag[0] in set(['N', 'A', 'C', 'V']) and w.lemma not in self.stopwords])
        qtype = 'default'
        if not filtered_nacv:
            qtype = 'empty'
        elif tagged[0].tag[0] == 'V':
            qtype = 'Y/N'
        elif tagged[0].lemma == 'proč':
            qtype = 'why'
        return filtered_nac, filtered_nacv, qtype

    def correct_diacritics(self, text: str):
        r = requests.post(self.urls['URL_KOREKTOR'], {'data': text, 'model': 'czech-diacritics_generator'})
        return r.json()['result']

    def ask_chitchat(self, query):
        logger.info('Request "%s" at %s' % (query, self.urls['URL_CHITCHAT']))
        resp = requests.post(self.urls['URL_CHITCHAT'], json={'q': query})
        try:
            reply = resp.json()['a'].strip()
            logger.info('Reply: "%s"' % reply)
            return reply
        except Exception as e:
            logger.error(str(e))
            return 'Toto bohužel nevím.'