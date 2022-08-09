#!/usr/bin/env python

import re
import os
import json


def read_vert(fname):
    text = ' '.join([l.rstrip().split('\t')[0] for l in open(fname, 'r', encoding='UTF-8').readlines()])
    text = re.sub(r' <g/> ', '', text)
    text = re.sub(r'</?s> ', '', text)
    text = re.sub(r' </s>$', '', text)
    return text


DATA_SIZES = {'train': 8061, 'dev': 1403, 'test': 4101}
count = 1
for portion in ['train', 'dev', 'test']:

    data = []

    while len(data) < DATA_SIZES[portion]:
        subdir = '%06d' % count
        question = read_vert(os.path.join(subdir, '01question.vert'))
        answer = read_vert(os.path.join(subdir, '02answer.vert'))
        context = read_vert(os.path.join(subdir, '03text.vert'))
        url = open(os.path.join(subdir, '04url.txt'), 'r', encoding='UTF-8').readlines()[0].strip()

        data.append({'question': question,
                     'answer': answer,
                     'context': context,
                     'url': url})
        count += 1

    json.dump(data, open(portion + '.json', 'w', encoding='UTF-8'), ensure_ascii=False, indent=2)
