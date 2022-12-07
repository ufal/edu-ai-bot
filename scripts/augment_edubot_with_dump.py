import glob
import json
import argparse
from urllib.parse import unquote
from csv import reader as csv_reader


def load_edubot_from_file(data_file):
    def extract_title(field):
        title = field.split('/')[-1]
        title = ' '.join(unquote(title).split('_'))
        return title

    with open(data_file, 'rt', newline='') as infd:
        reader = csv_reader(infd, delimiter='\t')
        try:
            data_columns = next(reader)
        except StopIteration:
            print('The provided file doesn\'t contain a header!')
            return
        # row: 0:question, 1:link, 2:type, 3:answer, 4:long_answer, 5:context, 6:title
        data = [{'question': row[0],
                 'link': row[1],
                 'answers': [row[3]],
                 'long_answer': row[4],
                 'title': extract_title(row[1]),
                 'id': str(n)} for n, row in enumerate(reader)]
        return data


def main(args):
    dump = dict()
    edu_data = load_edubot_from_file(args.in_file)
    for f in glob.glob(f'{args.dump_dir}/*'):
        print(f'processing {f}')
        with open(f, 'rt') as fd:
            for line in fd:
                entry = json.loads(line)
                dump[entry['title']] = entry

    with open(args.output, 'wt', encoding='utf-8') as ofd:
        for data_entry in edu_data:
            title = data_entry['title']
            if title not in dump:
                continue
            data_entry['context'] = dump[title]['text'].split('\n')[0]
            line = json.dumps(data_entry)
            print(line, file=ofd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir', type=str, help='Path to a directory containing Wiki dumps.')
    parser.add_argument('--in_file', type=str, help='Input .tsv file with Edubot data.')
    parser.add_argument('--output', type=str, help='Output file to write the augmented data.')
    args = parser.parse_args()
    main(args)