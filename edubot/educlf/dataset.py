import random
from csv import reader as csv_reader
from datasets import Dataset

def load_data(fn):
    data = {'labels': [], 'utterance': []}
    train_size = 0.8
    raw_data = []
    with open(fn, 'rt', newline='', encoding='utf-8') as csvfd:
        datareader = csv_reader(csvfd, delimiter='\t')
        for row in datareader:
            if len(row) != 2:
                continue
            raw_data.append((row[0], row[1].strip('"').strip().lower()))
    random.shuffle(raw_data)
    data['labels'], data['utterance'] = zip(*raw_data)
    train_data = {'labels': data['labels'][:int(len(data['labels']) * train_size)],
                  'utterance': data['utterance'][:int(len(data['labels']) * train_size)]}
    eval_data = {'labels': data['labels'][int(len(data['labels']) * train_size):],
                 'utterance': data['utterance'][int(len(data['labels']) * train_size):]}
    return Dataset.from_dict(train_data), Dataset.from_dict(eval_data)
