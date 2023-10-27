#!/usr/bin/env python

import pandas as pd

# Will get all data from the intents spreadsheet + our additional file
# + will add more on date/time
# + output everything as merged_intents.tsv

INPUT_FILE = 'Dotazy Wiki chatbot.xlsx'
ADDED_FILE = 'intents_to_add.tsv'
TARGET_FILE = 'merged_intents.tsv'

data = pd.read_excel('Dotazy Wiki chatbot.xlsx', sheet_name='chitchat', header=None)
data = list(data[0])
data = [{'input': i, 'intent': 'chch'} for i in data]
out_data = data

data = pd.read_excel(INPUT_FILE, sheet_name='ovládání', header=None)
data = data.to_dict('records')
data = data[3:]
data = [{'input': i[0], 'intent': i[1]} for i in data]
out_data.extend(data)

data = pd.read_excel(INPUT_FILE, sheet_name='QA ema', header=None)
data = list(data[0])
data = [{'input': i, 'intent': 'qa_ema'} for i in data]
out_data.extend(data)

data = pd.read_excel(INPUT_FILE, sheet_name='QA LO', header=None)
data = data.to_dict('records')
data = data[1:]
data = [{'input': i[0], 'intent': 'qa_lo'} for i in data]
out_data.extend(data)


data = pd.read_excel(INPUT_FILE, sheet_name='QA NPI', header=None)
data = data.to_dict('records')
data = data[1:]
data = [{'input': i[0], 'intent': 'qa_npi'} for i in data]
out_data.extend(data)

data = pd.read_excel(INPUT_FILE, sheet_name='QA wiki', header=None)
data = data.to_dict('records')
data = data[1:]
data = [{'input': i[0], 'intent': 'qawiki'} for i in data]
out_data.extend(data)

data = pd.read_csv(ADDED_FILE, sep='\t', header=None)
data = data.to_dict('records')
qawiki = [{'input': i[1], 'intent': i[0]} for i in data if i[0] == 'qawiki']
out_data.extend(qawiki)
chch = [{'input': i[1], 'intent': i[0]} for i in data if i[0] == 'chch']
out_data.extend(chch)
souhlas = [{'input': i[1], 'intent': 'chch'} for i in data if i[0] == 'souhlas']
out_data.extend(souhlas)

Ps = [{'input': i[1], 'intent': i[0]} for i in data if i[0].startswith('P_')]
Ps.extend([{'input': i, 'intent': 'P_kolik_hodin'} for i in ["kolik máme hodin", "kolik máš hodin", "víš kolik je hodin", "řekneš mi kolik je hodin", "řekni mi kolik je hodin", "jaký je čas"]])
Ps.extend([{'input': i, 'intent': 'P_kolikateho'} for i in ["kolikátého je dneska", "kolikátého je", "dej mi dnešní datum", "dnešní datum", "co je dnes za den", "jaký je dnes den", "Co máme za den?", "Co máme dnes za den?"]])
Ps = Ps * 5
out_data.extend(Ps)

df = pd.DataFrame.from_dict(out_data)
df.to_csv(TARGET_FILE, sep='\t', index=None, columns=['intent', 'input'])
