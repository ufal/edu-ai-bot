
import json
from argparse import ArgumentParser
from logzero import logger
import logzero

from api import apply_qa


def run_on_data(data_file, first_par_only=True):

    data = []
    if data_file.endswith('.json'):
        raw_data = json.load(open(data_file, 'r', encoding='UTF-8'))

        for article in raw_data['data']:
            art_title = article['title']
            if first_par_only:
                article['paragraphs'] = article['paragraphs'][:1]
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    answers = set([a['text_translated'] for a in qa['answers']] + [a['text'] for a in qa['answers']])
                    data.append({'id': qa['id'],
                                 'question': qa['question'],
                                 'answers': answers,
                                 'is_impossible': qa['is_impossible'],
                                 'context': context,
                                 'title': art_title})

    title_hits = 0
    answer_hits_pred_context = 0
    answer_hits_gold_context = 0
    tested = 0

    logzero.loglevel(logzero.WARN)
    for qa in data:
        if qa['is_impossible']:  # skip non-answerable for now
            continue
        _, answer_gold_context, _ = apply_qa(qa['question'], qa['context'], exact=True)
        pred_context, answer_pred_context, pred_title = apply_qa(qa['question'], None, exact=True)

        tested += 1
        title_hits += int(pred_title == qa['title'])
        answer_hits_pred_context += int(answer_pred_context in qa['answers'])
        answer_hits_gold_context += int(answer_gold_context in qa['answers'])

        logger.warning('\n\tQ: %s | G: %s / P: %s \n\tG: %s\n\tP: %s\n\tp: %s'
                       % (qa['question'], qa['title'], pred_title, str(qa['answers']), answer_gold_context, answer_pred_context))

    logger.warning(f'Total: {tested}\n\tCorrect titles: {title_hits}'
                   + f'\n\tCorrect answers w. gold context: {answer_hits_gold_context}'
                   + f'\n\tCorrect answers w. pred. context: {answer_hits_pred_context}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('eval_file', type=str, help='Evaluation data')

    args = ap.parse_args()
    run_on_data(args.eval_file)
