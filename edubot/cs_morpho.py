#!/usr/bin/env python3


"""Simple wrappers for Morphodita Czech morphology analyzer and generator."""

from ufal.morphodita import Tagger, Forms, TaggedLemmas, TokenRanges, Morpho, TaggedLemmasForms
import os
import requests
from logzero import logger
import zipfile
from io import BytesIO

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


class Analyzer:
    """Morphodita analyzer/tagger wrapper."""

    def __init__(self, tagger_model=DEFAULT_TAGGER):
        if tagger_model == DEFAULT_TAGGER and not os.path.isfile(DEFAULT_TAGGER):
            download_default_models()
        self._tagger = Tagger.load(tagger_model)
        self._tokenizer = self._tagger.newTokenizer()
        self._forms_buf = Forms()
        self._tokens_buf = TokenRanges()
        self._lemmas_buf = TaggedLemmas()

    def analyze(self, stop_text):
        self._tokenizer.setText(stop_text)
        while self._tokenizer.nextSentence(self._forms_buf, self._tokens_buf):
            self._tagger.tag(self._forms_buf, self._lemmas_buf)
            return [(form, lemma.lemma, lemma.tag)
                    for (form, lemma) in zip(self._forms_buf, self._lemmas_buf)]


class Generator:
    """Morphodita generator wrapper, with support for inflecting
    noun phrases (stop/city names, personal names)."""

    def __init__(self, morpho_model=DEFAULT_MORPHO):
        if morpho_model == DEFAULT_MORPHO and not os.path.isfile(DEFAULT_MORPHO):
            download_default_models()
        self._morpho = Morpho.load(morpho_model)
        self._out_buf = TaggedLemmasForms()

    def generate(self, lemma, tag_wildcard):
        """Get variants for one word from the Morphodita generator. Returns
        empty list if nothing found in the dictionary."""
        # run the generation for this word
        self._morpho.generate(lemma, tag_wildcard, self._morpho.GUESSER, self._out_buf)
        # see if we found any forms, return empty if not
        if not self._out_buf:
            return []
        return [form_tag.form for form_tag in self._out_buf[0].forms]
