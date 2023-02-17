import json
import re
import os
from abc import ABC
from csv import reader as csv_reader

class DataParser(ABC):

    def __init__(self, data_root, first_paragraph_only=False):
        self.data_root = data_root
        self.first_paragraph_only = first_paragraph_only
        self.data = []

    def load_from_file(self, data_file):
        raise NotImplementedError()

    @staticmethod
    def parser_factory(data_type, data_root, first_paragraph_only=False):
        if data_type.lower() == 'cs-squad':
            return CsSquadParser(data_root, first_paragraph_only)
        elif data_type == 'sqad':
            return SqadParser(data_root, first_paragraph_only)
        elif data_type == 'edubot':
            return EdubotParser(data_root, first_paragraph_only)
        else:
            raise NotImplementedError(f'Unknown data type: {data_type}')


class CsSquadParser(DataParser):

    def load_from_file(self, data_file):
        raw_data = json.load(open(data_file, 'r', encoding='UTF-8'))
        for article in raw_data['data']:
            art_title = article['title']
            if self.first_paragraph_only:
                article['paragraphs'] = article['paragraphs'][:1]
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    answers = set([a['text_translated'] for a in qa['answers']] + [a['text'] for a in qa['answers']])
                    self.data.append({'id': qa['id'],
                                      'question': qa['question'],
                                       'answers': list(answers),
                                       'is_impossible': qa.get('is_impossible', False),
                                       'context': context,
                                       'title': art_title})


class SqadParser(DataParser):
    DATA_SIZES = {'train': 8061, 'dev': 1403, 'test': 4101}

    @staticmethod
    def read_vert(fname):
        text = ' '.join([l.rstrip().split('\t')[0] for l in open(fname, 'r', encoding='UTF-8').readlines()])
        text = re.sub(r' <g/> ', '', text)
        text = re.sub(r'</?s> ', '', text)
        text = re.sub(r' </s>$', '', text)
        return text

    def load_from_file(self, data_subset):
        if data_subset == 'train':
            offset = 0
        elif data_subset == 'dev':
            offset = self.DATA_SIZES['train']
        else:
            offset = self.DATA_SIZES['train'] + self.DATA_SIZES['dev']
        length = self.DATA_SIZES[data_subset]

        for i in range(offset, offset + length):
            subdir = '%06d' % i
            question = self.read_vert(os.path.join(self.data_root, subdir, '01question.vert'))
            answer = self.read_vert(os.path.join(self.data_root, subdir, '02answer.vert'))
            context = self.read_vert(os.path.join(self.data_root, subdir, '03text.vert'))
            url = open(os.path.join(self.data_root, subdir, '04url.txt'), 'r', encoding='UTF-8').readlines()[0].strip()

            self.data.append({'question': question,
                              'answers': [answer],
                              'context': context,
                              'url': url,
                              'id': str(i),
                              'title': ''
                              })

class EdubotParser(DataParser):

    def load_from_file(self, data_file):
        if data_file.endswith('.tsv'):
            self.load_from_tsv_file(data_file)
        elif data_file.endswith('.json'):
            self.load_from_json(data_file)
        else:
            raise NotImplementedError(f'Unknown file type: {data_file}')

    def load_from_tsv_file(self, data_file):
        with open(data_file, 'rt', newline='') as infd:
            reader = csv_reader(infd, delimiter='\t')
            try:
                data_columns = next(reader)
            except StopIteration:
                print('The provided file doesn\'t contain a header!')
                return
            # row: 0:question, 1:link, 2:type, 3:answer, 4:long_answer, 5:context, 6:title
            self.data = [{'question': row[0],
                          'answers': [row[3]],
                          'context': row[5],
                          'id': str(n),
                          'title': row[6]} for n, row in enumerate(reader)]

    def load_from_json(self, data_file):
        with open(data_file, 'rt') as fd:
            for line in fd:
                entry = json.loads(line)
                self.data.append(entry)
