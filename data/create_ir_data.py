#!/usr/bin/env python

import pandas as pd
import math
import io


INPUT_FILE = 'Dotazy Wiki chatbot.xlsx'
WIKI_DATA = 'cs_wiki_paragraphs_redirects_resolved.tsv'
TARGET_FILE = 'joined_ir_data.tsv'

HEADER = 'id\turl\ttitle\tfirst_paragraph\n'


# open the output buffer
out = open(TARGET_FILE, 'w', encoding='UTF-8')
out.write(HEADER)

# feed all wiki data into the output buffer (line-by-line, low-level)
with open(WIKI_DATA, 'r', encoding='UTF-8') as fh:
    for line in fh:
        if '\t¶\t' in line:  # skip weird article name that won't work with Solr
            continue
        # TODO incorporate the lemmatization
        # inst_id, inst_url, inst_title, inst_firstpar = line.rstrip().split('\t')
        out.write(line)

# add LO data
lo_data = pd.read_excel(INPUT_FILE, sheet_name='QA LO', header=None)
lo_data = lo_data.to_dict('records')
lo_data = lo_data[1:]
lo_data = [{'url': 'LOGIC ' + i[2], 'title': i[0], 'first_paragraph': i[1].replace('\n', ' ')} for i in lo_data]
df = pd.DataFrame.from_dict(lo_data)
df = df.set_index(pd.RangeIndex(start=9000001, stop=9000001+len(lo_data)))
df.index.name = 'id'
df.to_csv(out, sep='\t', mode='a', header=False)

# add NPI data
npi_data = pd.read_excel(INPUT_FILE, sheet_name='QA NPI', header=None)
npi_data = npi_data.to_dict('records')
npi_data = npi_data[1:]
npi_data = [{'url': 'NPI ' + (i[2] if isinstance(i[2], str) else '-'), 'title': i[0], 'first_paragraph': i[1].replace('\n', ' ')}
            for i in npi_data if isinstance(i[1], str)]
df2 = pd.DataFrame.from_dict(npi_data)
df2 = df2.set_index(pd.RangeIndex(start=9100001, stop=9100001+len(npi_data)))
df2.index.name = 'id'
df2.to_csv(out, sep='\t', mode='a', header=False)

# close output file
out.close()

