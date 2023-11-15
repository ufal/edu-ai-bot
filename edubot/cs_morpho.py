#!/usr/bin/env python3


"""Simple wrappers for Morphodita Czech morphology analyzer and generator."""

from ufal.morphodita import Tagger, Forms, TaggedLemmas, TokenRanges, Morpho, TaggedLemmasForms, TagsetConverter
import os
import requests
from logzero import logger
import zipfile
from io import BytesIO
from dataclasses import dataclass
import re

DEFAULT_MODEL_URL = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4794/czech-morfflex2.0-pdtc1.0-220710.zip'
DEFAULT_BASE = os.path.dirname(__file__)
DEFAULT_PATH = os.path.join(DEFAULT_BASE, 'czech-morfflex2.0-pdtc1.0-220710')
DEFAULT_TAGGER = os.path.join(DEFAULT_PATH, 'czech-morfflex2.0-pdtc1.0-220710.tagger')
DEFAULT_MORPHO = os.path.join(DEFAULT_PATH, 'czech-morfflex2.0-220710.dict')


def download_default_models():
    logger.warn(f'Downloading morphodita models from {DEFAULT_MODEL_URL}...')
    r = requests.get(DEFAULT_MODEL_URL)
    with zipfile.ZipFile(BytesIO(r.content), 'r') as zh:
        zh.extractall(DEFAULT_BASE)


@dataclass
class TaggedWord:
    form: str
    lemma: str
    tag: str
    space_before: bool


class Analyzer:
    """Morphodita analyzer/tagger wrapper."""

    def __init__(self, tagger_model=DEFAULT_TAGGER, morpho_model=DEFAULT_MORPHO, strip_lemmas=True):
        if tagger_model == DEFAULT_TAGGER and not os.path.isfile(DEFAULT_TAGGER):
            download_default_models()
        self._tagger = Tagger.load(tagger_model)
        self._morpho = Morpho.load(morpho_model)
        if strip_lemmas:
            self._format_lemmas = TagsetConverter.newStripLemmaIdConverter(self._morpho)
        else:
            self._format_lemmas = TagsetConverter.newIdentityConverter()
        self._tokenizer = self._tagger.newTokenizer()
        self._forms_buf = Forms()
        self._tokens_buf = TokenRanges()
        self._lemmas_buf = TaggedLemmas()

    def analyze(self, text: str):
        self._tokenizer.setText(text)
        out = []
        while self._tokenizer.nextSentence(self._forms_buf, self._tokens_buf):
            self._tagger.tag(self._forms_buf, self._lemmas_buf)
            prev_tok_end = 0
            for form, lemma, tok_range in zip(self._forms_buf, self._lemmas_buf, self._tokens_buf):
                self._format_lemmas.convert(lemma)
                out.append(TaggedWord(form=form, lemma=lemma.lemma, tag=lemma.tag, space_before = prev_tok_end < tok_range.start))
                prev_tok_end = tok_range.start + tok_range.length
        return out


class Generator:
    """Morphodita generator wrapper, with support for inflecting
    noun phrases (stop/city names, personal names)."""

    def __init__(self, morpho_model=DEFAULT_MORPHO):
        if morpho_model == DEFAULT_MORPHO and not os.path.isfile(DEFAULT_MORPHO):
            download_default_models()
        self._morpho = Morpho.load(morpho_model)
        self._out_buf = TaggedLemmasForms()

    def generate(self, lemma, tag_wildcard, capitalized=None):
        """Get variants for one word from the Morphodita generator. Returns
        empty list if nothing found in the dictionary."""
        # run the generation for this word
        self._morpho.generate(lemma, tag_wildcard, self._morpho.GUESSER, self._out_buf)
        # see if we found any forms, return empty if not
        if not self._out_buf:
            return []

        # prepare capitalization
        def process_capitalization(word):
            if capitalized is True:
                return word[0].upper() + word[1:]
            elif capitalized is False:
                return word[0].lower() + word[1:]
            return word
        # process the results (sort so that standard variants, with nothing at the end of the tag, are preferred)
        return [process_capitalization(form)
                for form, _ in sorted([(ft.form, ft.tag)
                                       for ft in self._out_buf[0].forms], key=lambda ft: ft[1])]

    def inflect_phrase(self, phrase: list[TaggedWord], target_case: str, personal_names: bool = False):
        """Inflect a stop/city/personal name in the given case (return
        lists of inflection variants for all words).

        @param phrase: list of TaggedWord objects with form/lemma/tag from the analyzer to be inflected
        @param case: the target case (Czech, 1-7)
        @param personal_names: should be False for stops/cities, True for personal names
        """
        forms = []
        prev_tag = ''
        for word in phrase:
            form_list = self._inflect_word(word, target_case, prev_tag, personal_names)
            if not form_list:
                form_list = [word.form]
            forms.append(form_list[0])
            prev_tag = word.tag
        return forms

    def _inflect_word(self, word, target_case, prev_tag, personal_names=False):
        """Inflect one word in the given case (return a list of variants,
        None if the generator fails)."""
        # inflect each word in nominative not following a noun in nominative
        # (if current case is not nominative), avoid numbers
        if (re.match(r'^[^C]...1', word.tag)
                and (not re.match(r'^NN..1', prev_tag) or personal_names)
                and word.form not in ['římská']
                and target_case != '1'):
            # change the case in the tag, allow all variants
            new_tag = re.sub(r'^(....)1(.*)(.)$', r'\g<1>' + target_case + r'\g<2>?', word.tag)
            # -ice: test both sg. and pl. versions
            if (word.form.endswith('ice') and word.form[0] == word.form[0].upper()
                    and not re.match(r'(nemocnice|ulice|vrátnice)', word.form, re.IGNORECASE)):
                new_tag = re.sub(r'^(...)S', r'\1[SP]', new_tag)
            # try inflecting, return empty list if not found in the dictionary
            capitalized = word.form[0] == word.form[0].upper()
            new_forms = self.generate(word.lemma, new_tag, capitalized)
            return new_forms
        else:
            return [word.form]
