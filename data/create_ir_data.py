#!/usr/bin/env python

import pandas as pd
from edubot.cs_morpho import Analyzer

INPUT_FILE = 'Dotazy Wiki chatbot.xlsx'
WIKI_DATA = 'cs_wiki_paragraphs_redirects_resolved.tsv'
TARGET_FILE = 'joined_ir_data.tsv'
HEADER = 'id\turl\ttitle\tfirst_paragraph\n'


def process_and_append_data(df, out_fd, lemmatize_f, url_key, start=0):
    data = df.to_dict('records')
    data = data[1:]
    data = [{'url': f'{url_key} ' + (i[2] if isinstance(i[2], str) else '-'),
             'title': i[0],
             'first_paragraph': i[1].replace('\n', ' '),
             'first_paragraph_cz': lemmatize_f(i[1].replace('\n', ' ')), }
            for i in data if isinstance(i[1], str)]
    df2 = pd.DataFrame.from_dict(data)
    df2 = df2.set_index(pd.RangeIndex(start=start, stop=start+len(data)))
    df2.index.name = 'id'
    df2.to_csv(out_fd, sep='\t', mode='a', header=False)


def get_lemmatizer():
    analyzer = Analyzer()

    def lemmatize(text):
        analyzed_text = analyzer.analyze(text)
        return ' '.join((i[1] for i in analyzed_text))
    return lemmatize


if __name__ == '__main__':
    with open(TARGET_FILE, 'w', encoding='UTF-8') as out_fd:
        out_fd.write(HEADER)
        lemmatize = get_lemmatizer()

        # WIKI data
        wiki_df = pd.read_csv(WIKI_DATA, sep='\t', header=None)
        wiki_df = wiki_df[wiki_df['first_paragraph'].str.contains('\t¶\t') == False]
        wiki_df['first_paragraph_cz'] = wiki_df['first_paragraph'].apply(lemmatize)
        wiki_df.to_csv(out_fd, sep='\t', mode='a', header=True)

        # add LO data
        lo_df = pd.read_excel(INPUT_FILE, sheet_name='QA LO', header=None)
        process_and_append_data(lo_df, out_fd, lemmatize, url_key="LOGIC", start=9000001)
        # add NPI data
        npi_df = pd.read_excel(INPUT_FILE, sheet_name='QA NPI', header=None)
        process_and_append_data(npi_df, out_fd, lemmatize, url_key="NPI", start=9100001)
