
import json
import requests

from logzero import logger
from werkzeug.urls import url_fix
from utils import dotdict

URL_TO_SOLR = "http://quest.ms.mff.cuni.cz/namuddis/qasolr/wiki_test/query?q={query}&wt=json&fl=title,first_paragraph,url,score"
URL_TO_UDPIPE = "http://lindat.mff.cuni.cz/services/udpipe/api/process?tokenizer&tagger&data={query}"


def ask_solr(*, query, attrib=None, source='wiki'):
    url = 'https?//cs.wikipedia.org*'
    if source == 'logic':
        url = 'https?//www.logickaolympiada.cz*'
    if attrib is not None:
        if isinstance(attrib, list):
            query = ' OR '.join([f'{a}:{query}' for a in attrib])
        else:
            query = f'{attrib}:{query}'
    query = f'({query}) AND url:{url}'
    response = requests.get(
        url_fix(URL_TO_SOLR.format(query=query))
    )
    j = json.loads(response.content.decode('utf8'))['response']
    logger.debug(query + "\n" + str(j))

    return j


STOP_WORDS = set([
    'co', 'kdo', 'kdy', 'kde', 'jak', 'kolik', 'být', 'ten', '?', '.', ':', '!',
    'znamenat', 'jaký', 'který', 'mít', 'proč',
])


def filter_query(query):
    try:
        udpipe = requests.get(url_fix(URL_TO_UDPIPE.format(query=query)))
        tagged = [line.split("\t") for line in udpipe.json()['result'].split("\n")
                  if "\t" in line and not line.startswith('#')]
        tagged = [dotdict({'form': w[1], 'lemma': w[2], 'tag': w[4]}) for w in tagged]
    except Exception as e:
        logger.warn('UDpipe problem:' + str(e))
        return query
    logger.debug("\n" + "\n".join(["\t".join([w['form'], w['lemma'], w['tag']]) for w in tagged]))
    filtered_nac = " ".join([w.form for w in tagged
                             if w.tag[0] in set(['N', 'A', 'C']) and w.lemma not in STOP_WORDS])
    filtered_nacv = " ".join([w.form for w in tagged
                              if w.tag[0] in set(['N', 'A', 'C', 'V']) and w.lemma not in STOP_WORDS])
    qtype = 'default'
    if not filtered_nacv:
        qtype = 'empty'
    elif tagged[0].tag[0] == 'V':
        qtype = 'Y/N'
    elif tagged[0].lemma == 'proč':
        qtype = 'why'
    return filtered_nac, filtered_nacv, qtype
