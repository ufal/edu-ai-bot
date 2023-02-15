
import json
import requests
import conllu
import csv

from logzero import logger
from werkzeug.urls import url_fix
from edubot.utils import dotdict
from edubot.cs_morpho import Generator


class RemoteServiceHandler:

    def __init__(self, config, stopwords):
        self.urls = config['URLs']
        with open(config['STOPWORDS_PATH'], 'rt') as fd:
            self.stopwords = set((w.strip() for w in fd.readlines() if len(w.strip()) > 0))
        self.morpho = Generator()
        self.female_to_male = {}
        with open(config['GENDERED_WORDS_PATH'], 'rt') as fd:
            tsv_reader = csv.DictReader(fd, delimiter="\t")
            for row in tsv_reader:
                self.female_to_male[row['female']] = row['male']

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
            url_fix(self.urls['SOLR'].format(query=query))
        )
        j = json.loads(response.content.decode('utf8'))['response']
        logger.debug(query + "\n" + str(j))

        return j

    def filter_query(self, query):
        try:
            udpipe = requests.get(url_fix(self.urls['UDPIPE_TAG'].format(query=query)))
            tagged = conllu.parse(udpipe.json()['result'])
            tagged = [dotdict({'form': w['form'], 'lemma': w['lemma'], 'tag': w['xpos']}) for s in tagged for w in s]
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
        r = requests.post(self.urls['KOREKTOR'], {'data': text, 'model': 'czech-diacritics_generator'})
        return r.json()['result']

    def postprocess_gender(self, text: str):
        """Postprocess 1st person gender coming from MT (change feminine to masculine)."""

        def regen_as_masc(tok):
            """Regenerate a given form as masculine."""
            logger.debug(f'Regen as masc: ' + tok['lemma'])
            new_lemma = self.female_to_male.get(tok['lemma'], tok['lemma'])
            number = 'S' if tok['xpos'][3] in 'SW' else 'P'
            gender = '[MY]'
            new_tag = tok['xpos'][:2] + gender + number + tok['xpos'][4:7] + '?' + tok['xpos'][8:12] + '?' + tok['xpos'][13:]
            f = self.morpho.generate(new_lemma, new_tag)
            if f:
                tok['form'] = f[0]
            return

        def fix_person(tree):
            """Recursively search for 1st person adjectives/verb participles and fix them."""
            if tree.token['feats'] and tree.token['feats'].get('Gender') in ['Fem', 'Fem,Neut']:
                if any([c.token['feats'] and c.token['feats'].get('Person') == '1' for c in tree.children]):
                    regen_as_masc(tree.token)
                    for c in tree.children:
                        if c.token['deprel'] == 'conj' and c.token['feats'] and c.token['feats'].get('Gender') in ['Fem', 'Fem,Neut']:
                            regen_as_masc(c.token)
            for child in tree.children:  # DFS
                fix_person(child)

        try:
            udpipe = requests.get(url_fix(self.urls['UDPIPE_PARSE'].format(query=text)))
            sents = conllu.parse(udpipe.json()['result'])
            sent_text = ''
            for sent in sents:
                tree = sent.to_tree()
                fix_person(tree)
                sent_text += ''.join([w['form'] + ('' if w['misc'] and w['misc'].get('SpaceAfter') == 'No' else ' ')
                                      for w in sent])
            return sent_text.strip()
        except Exception as e:
            logger.warn(e)
            return text

    def translate(self, text: str, service: str):
        r = requests.post(service, data={'input_text': text})
        return r.content.decode('utf8')

    def translate_en2cs(self, text: str):
        return self.translate(text, self.urls['LINDAT_EN2CS'])

    def translate_cs2en(self, text: str):
        return self.translate(text, self.urls['LINDAT_CS2EN'])
