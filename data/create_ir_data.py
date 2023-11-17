#!/usr/bin/env python

import pandas as pd
from edubot.cs_morpho import Analyzer
from edubot.remote_services import RemoteServiceHandler
from argparse import ArgumentParser
import yaml
import csv


def process_and_append_data(df, out_fd, expand_f, url_key, start=0):
    data = df.to_dict('records')
    data = data[1:]
    data = [{'url': f'{url_key} ' + (i[2] if isinstance(i[2], str) else '-'),
             'title': expand_f(i[0], lemmatize=False),
             'title_cz': expand_f(i[0], lemmatize=True),
             'first_paragraph': expand_f(i[1].replace('\n', ' '), lemmatize=False),
             'first_paragraph_cz': expand_f(i[1].replace('\n', ' '), lemmatize=True), }
            for i in data if isinstance(i[1], str)]
    df2 = pd.DataFrame.from_dict(data)
    df2 = df2.set_index(pd.RangeIndex(start=start, stop=start + len(data)))
    df2.index.name = 'id'
    df2.to_csv(out_fd, sep='\t', mode='a', header=False)


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Configuration file path')
    ap.add_argument('--custom', type=str, required=True, help='Custom wiki file path (xlsx)')
    ap.add_argument('--wiki', type=str, required=True, help='Wikipedia dump file path (tsv)')
    ap.add_argument('target_file', type=str, help='Target file path')
    args = ap.parse_args()

    with open(args.config, 'rt') as fd:
        config = yaml.load(fd, Loader=yaml.loader.SafeLoader)
    services = RemoteServiceHandler(config)
    tagger = Analyzer()

    def lemmatize(text):
        analyzed_text = tagger.analyze(text)
        return ' '.join((i.lemma for i in analyzed_text))

    def expand(text, lemmatize):
        analyzed_text = tagger.analyze(text)
        # preserve spacing for word forms, but insert spaces everywhere for lemmas
        return services.replace_abbrevs(analyzed_text, use_lemmas=lemmatize, spaced=lemmatize)

    with open(args.target_file, 'w', encoding='UTF-8') as out_fd:
        # WIKI data
        wiki_df = pd.read_csv(args.wiki, sep='\t', quoting=csv.QUOTE_NONE, header=None, names=['id', 'url', 'title', 'first_paragraph'])
        wiki_df = wiki_df[wiki_df['title'].notnull()]
        wiki_df = wiki_df[wiki_df['title'] != '¶']
        wiki_df['title_cz'] = wiki_df['title'].apply(lemmatize)
        wiki_df['first_paragraph_cz'] = wiki_df['first_paragraph'].apply(lemmatize)
        wiki_df = wiki_df.set_index('id')
        wiki_df.to_csv(out_fd, sep='\t', mode='a', header=True)

        # add LO data
        lo_df = pd.read_excel(args.custom, sheet_name='QA LO', header=None)
        process_and_append_data(lo_df, out_fd, expand, url_key="LOGIC", start=9000001)
        # add NPI data
        npi_df = pd.read_excel(args.custom, sheet_name='QA NPI', header=None)
        process_and_append_data(npi_df, out_fd, expand, url_key="NPI", start=9100001)
