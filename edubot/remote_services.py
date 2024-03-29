
import json
import requests
import conllu
import csv

from logzero import logger
from werkzeug.urls import url_fix
from edubot.utils import dotdict
from edubot.cs_morpho import Generator, Analyzer


class RemoteServiceHandler:

    def __init__(self, config):
        self.urls = config['URLs']
        with open(config['STOPWORDS_PATH'], 'rt') as fd:
            self.stopwords = set((w.strip() for w in fd.readlines() if len(w.strip()) > 0))
        self.morpho = Generator()
        self.tagger = Analyzer()
        self.female_to_male = {}
        with open(config['GENDERED_WORDS_PATH'], 'rt') as fd:
            tsv_reader = csv.DictReader(fd, delimiter="\t")
            for row in tsv_reader:
                self.female_to_male[row['female']] = row['male']
        with open(config['IDENTITY_VERBS_PATH'], 'rt') as fd:
            self.identity_verbs = set((w.strip() for w in fd.readlines() if len(w.strip()) > 0))
        # abbrev replacements: prepare all cases
        self.abbrev_replace = config['ABBREV_REPLACE']
        for abbr, expand in list(self.abbrev_replace.items()):
            tagged = self.tagger.analyze(expand)
            self.abbrev_replace[abbr] = {'lemma': ' '.join([w.lemma for w in tagged])}
            for target_case in ['1', '2', '3', '4', '6', '7']:
                self.abbrev_replace[abbr][target_case] = ' '.join(self.morpho.inflect_phrase(tagged, target_case))

    def ask_solr(self, query, attrib=None, source='wiki'):
        if source == 'wiki':
            url = self.urls['WIKI']
        else:
            url = source + '?http*'
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
        # logger.debug(query + "\n" + str(j))

        return j

    def filter_query(self, query):
        tagged = self.tagger.analyze(query)
        logger.debug("\n" + "\n".join(["\t".join([w.form, w.lemma, w.tag]) for w in tagged]))
        allowed_tags = set(['N', 'B', 'A', 'C', 'F'])  # https://ufal.mff.cuni.cz/techrep/tr64.pdf

        def is_allowed(w):
            return w.tag[0] in allowed_tags and w.lemma not in self.stopwords

        words_nac = self.replace_abbrevs([w for w in tagged if is_allowed(w)], spaced=True)
        lemmas_nac = self.replace_abbrevs([w for w in tagged if is_allowed(w)], use_lemmas=True, spaced=True)
        allowed_tags.add('V')
        words_nacv = self.replace_abbrevs([w for w in tagged if is_allowed(w)], spaced=True)
        lemmas_nacv = self.replace_abbrevs([w for w in tagged if is_allowed(w)], use_lemmas=True, spaced=True)
        qtype = 'default'
        if not words_nacv:
            qtype = 'empty'
        elif tagged[0].tag[0] == 'V':
            qtype = 'Y/N'
        elif tagged[0].lemma == 'proč':
            qtype = 'why'
        return dotdict({'words_nac': words_nac, 'lemmas_nac': lemmas_nac,
                        'words_nacv': words_nacv, 'lemmas_nacv': lemmas_nacv}), qtype

    def replace_abbrevs(self, tagged: list, use_lemmas=False, spaced=False):
        out = ''
        for prev_word, word in zip([None] + tagged, tagged):
            out += ' ' if (spaced or word.space_before) else ''
            if word.lemma.upper() in self.abbrev_replace:
                if use_lemmas:
                    out += self.abbrev_replace[word.lemma.upper()]['lemma']
                else:
                    # determine the abbrev case
                    target_case = '1'
                    if prev_word:
                        if prev_word.tag[0] == 'R':
                            target_case = prev_word.tag[4]
                        if prev_word.tag[0] == 'N':
                            target_case = '2'
                    out += self.abbrev_replace[word.lemma.upper()][target_case]
            else:
                out += word.lemma if use_lemmas else word.form
        return out.strip()

    def correct_diacritics(self, text: str):
        r = requests.post(self.urls['KOREKTOR'], {'data': text, 'model': 'czech-diacritics_generator'})
        return r.json()['result']

    def postprocess_gender(self, text: str):
        """Postprocess 1st person gender coming from MT (change feminine to masculine)."""

        def regen_as_masc(tok):
            """Regenerate a given form as masculine."""
            # logger.debug(f'Regen as masc: ' + tok['lemma'])
            new_lemma = self.female_to_male.get(tok['lemma'], tok['lemma'])  # fem-masc lemma changes
            if tok['xpos'][:2] == 'Vs':  # morphodita x udpipe discrepancy in lemmas, get the morphodita lemma
                new_lemma = self.tagger.analyze(tok['form'])[0][1]
            number = 'S' if tok['xpos'][3] in 'SW' else 'P'
            gender = '[MY]'
            new_tag = tok['xpos'][:2] + gender + number + tok['xpos'][4:7] + '?' + tok['xpos'][8:12] + '?' + tok['xpos'][13:]
            # retrieve the new form & set it inside the token
            f = self.morpho.generate(new_lemma, new_tag)
            if f:
                tok['form'] = (f[0][0].upper() if tok['form'][0].isupper() else f[0][0]) + f[0][1:]  # preserve capitalization
            return

        def fix_person(tree, is_1st_ps=False):
            """Recursively search for 1st person adjectives/verb participles and fix them."""
            # found 1st ps close by
            if is_1st_ps or tree.token['feats'].get('Person') == '1' or any([c.token['feats'].get('Person') == '1' for c in tree.children]):
                # logger.debug('Regen: ' + str(tree.token))
                # it's gendered -- change gender of this one
                if tree.token['feats'].get('Gender') in ['Fem', 'Fem,Neut']:
                    # logger.debug('Hit: ' + str(tree.token))
                    regen_as_masc(tree.token)
                # recurse into children
                for c in tree.children:
                    if (c.token['feats'].get('Gender') in ['Fem', 'Fem,Neut']
                        # coordination, adjectives, complements, copulas
                        and ((c.token['deprel'] in ['conj', 'amod', 'xcomp', 'cop', 'aux:pass'])
                             # oblique argument (instrumental) for verbs of appointment/identity
                             or (tree.token['lemma'] in self.identity_verbs and c.token['deprel'] in ['obl:arg', 'obl']))):
                            # logger.debug(f'Regen {c.token["deprel"]}: ' + str(c.token))
                            fix_person(c, is_1st_ps=True)

            # XXX could skip the children already fixed above, but maybe too much bother?
            for child in tree.children:  # DFS
                fix_person(child)

        try:
            udpipe = requests.get(url_fix(self.urls['UDPIPE_PARSE'].format(query=text)))
            sents = conllu.parse(udpipe.json()['result'])
            sent_text = ''
            for sent in sents:
                for token in sent:
                    token['feats'] = {} if not token['feats'] else token['feats']
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
