from argparse import ArgumentParser
from logzero import logger
import logzero
from eval import DataParser

from api import apply_qa


def run_on_data(data):
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
    ap.add_argument('eval_file', type=str, help='Evaluation data. Should be split name for sqad, filename for others.')
    ap.add_argument('data_root', type=str, help='Evaluation data root directory')
    ap.add_argument('data_type',type=str, help='The format of evaluation data. Can be [cs-SQuAD|SQAD|edubot]')

    args = ap.parse_args()
    parser = DataParser.parser_factory(args.data_type, args.data_root, first_paragraph_only=True)
    parser.load_edubot_from_file(args.eval_file)
    run_on_data(parser.data)
