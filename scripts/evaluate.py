import re

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logzero
from edubot.eval import DataParser
import os
from argparse import ArgumentParser
import torch
import time
import requests
import yaml
from yaml.loader import SafeLoader
from logzero import logger
import fuzzywuzzy.fuzz
from nltk.metrics.scores import f_measure
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from edubot.remote_services import RemoteServiceHandler
from edubot.qa import QAHandler, OpenAIQA

def get_title_by_url(url):
    resp = requests.get(url)
    title = re.findall(r'<title>(.*?)</title>', resp.text)
    return title[0] if title else None


def evaluate_answer(answer, context, gt_answer, gt_context=None):
    answer_tk = word_tokenize(answer)
    gt_answer_tk = word_tokenize(gt_answer)
    context_tk = word_tokenize(context)
    gt_context_tk = word_tokenize(gt_context) if gt_context else None

    answer_f1 = f_measure(set(answer_tk), set(gt_answer_tk))
    answer_context_f1 = f_measure(set(answer_tk), set(context_tk))
    answer_bleu = sentence_bleu([gt_answer_tk], answer_tk, smoothing_function=SmoothingFunction().method1)
    answer_context_bleu = sentence_bleu([context_tk], answer_tk, smoothing_function=SmoothingFunction().method1)

    answer_token_ratio = fuzzywuzzy.fuzz.partial_token_sort_ratio(answer, gt_answer)
    answer_context_token_ratio = fuzzywuzzy.fuzz.partial_token_sort_ratio(answer, context)

    context_bleu = sentence_bleu([gt_context_tk], context_tk, smoothing_function=SmoothingFunction().method1)\
        if gt_context else 0
    context_token_ratio = fuzzywuzzy.fuzz.partial_token_sort_ratio(context, gt_context)\
        if gt_context else 0

    return {'answer_f1': answer_f1,
            'answer_context_f1': answer_context_f1,
            'answer_bleu': answer_bleu,
            'answer_context_bleu': answer_context_bleu,
            'answer_token_ratio': answer_token_ratio,
            'answer_context_token_ratio': answer_context_token_ratio,
            'context_bleu': context_bleu,
            'context_token_ratio': context_token_ratio}


def average_eval(total_eval_results):
    total_eval_results_avg = {}
    for key in total_eval_results:
        total_eval_results_avg[key] = sum(total_eval_results[key]) / len(total_eval_results[key])
    return total_eval_results_avg


def run_on_data(data, qa_handler):
    tested = 0
    exec_times = []
    total_eval_results_predicted = {'answer_f1': [],
                                    'answer_context_f1': [],
                                    'answer_bleu': [],
                                    'answer_context_bleu': [],
                                    'answer_token_ratio': [],
                                    'answer_context_token_ratio': [],
                                    'context_bleu': [],
                                    'context_token_ratio': [],
                                    'url_accuracy': [],
                                    'context_use_ratio': []}
    total_eval_results_gold = {'answer_f1': [],
                               'answer_context_f1': [],
                               'answer_bleu': [],
                               'answer_context_bleu': [],
                               'answer_token_ratio': [],
                               'answer_context_token_ratio': [],
                               'context_use_ratio': []}
    logzero.loglevel(logzero.INFO)
    for qa in data:
        tested += 1
        if tested % 10 == 0:
            print('=========== RETRIEVED CONTEXTS  ===========')
            averaged_results = average_eval(total_eval_results_predicted)
            for key in averaged_results:
                print(f'{key:30} {averaged_results[key]:.4f}')

            print('============ GOLD CONTEXTS  ===========')
            averaged_results = average_eval(total_eval_results_gold)
            for key in averaged_results:
                print(f'{key:30} {averaged_results[key]:.4f}')
            print(f'Average execution time: {sum(exec_times) / len(exec_times):.4f}')
        if tested == 100:
            break

        logger.warning("Question " + str(tested))
        if 'is_impossible' in qa and qa['is_impossible']:  # skip non-answerable for now
            logger.warning("Non-answerable " + str(qa))
            continue

        st = time.time()
        qares_gold = qa_handler.apply_qa(qa['question'], qa['context'], exact=True)
        exec_times.append(time.time() - st)

        st = time.time()
        qares_ir = qa_handler.apply_qa(qa['question'], None, exact=True)
        exec_times.append(time.time() - st)

        answer_gold_context = qares_gold.reply
        pred_context, answer_pred_context = qares_ir.retrieved, qares_ir.reply
        pred_link = get_title_by_url(qares_ir.all_results[0]['url'])
        gold_link = get_title_by_url(qa['link'])
        if answer_pred_context is None or answer_gold_context is None:
            logger.warning("None answer " + str(qa))
            continue
        logger.warning(f'Average execution time: {sum(exec_times) / len(exec_times):.4f}')
        answer_pred_context = answer_pred_context.strip()
        answer_gold_context = answer_gold_context.strip()
        ir_ctx_used = int(qares_ir.source != 'model')
        gold_ctx_used = int(qares_gold.source != 'model')

        logger.info(f"Question:\t\t\t\t {qa['question']}")
        logger.info(f"Retrieved context (used:{ir_ctx_used}):\t\t {answer_pred_context}")
        logger.info(f"Gold context (used:{gold_ctx_used}):\t\t\t {answer_gold_context}")
        logger.info(f"Ground truth answer:\t {qa['answers'][0]}")
        logger.info("-" * 100)

        print(f"Question:\t\t\t {qa['question']}")
        print(f"Retrieved context (used:{ir_ctx_used}):\t\t {answer_pred_context}")
        print(f"Gold context (used:{gold_ctx_used}):\t\t\t {answer_gold_context}")
        print(f"Ground truth answer:\t\t {qa['answers'][0]}")
        print("-" * 100)
        eval_results_predicted = evaluate_answer(answer_pred_context, pred_context, qa['answers'][0], qa['context'])
        total_eval_results_predicted['answer_f1'].append(eval_results_predicted['answer_f1'] if eval_results_predicted['answer_f1'] else 0)
        total_eval_results_predicted['answer_context_f1'].append(eval_results_predicted['answer_context_f1'] if eval_results_predicted['answer_context_f1'] else 0)
        total_eval_results_predicted['answer_bleu'].append(eval_results_predicted['answer_bleu'] if eval_results_predicted['answer_bleu'] else 0)
        total_eval_results_predicted['answer_context_bleu'].append(eval_results_predicted['answer_context_bleu'] if eval_results_predicted['answer_context_bleu'] else 0)
        total_eval_results_predicted['answer_token_ratio'].append(eval_results_predicted['answer_token_ratio'] if eval_results_predicted['answer_token_ratio'] else 0)
        total_eval_results_predicted['answer_context_token_ratio'].append(
            eval_results_predicted['answer_context_token_ratio'] if eval_results_predicted['answer_context_token_ratio'] else 0)
        total_eval_results_predicted['context_bleu'].append(eval_results_predicted['context_bleu'] if eval_results_predicted['context_bleu'] else 0)
        total_eval_results_predicted['context_token_ratio'].append(eval_results_predicted['context_token_ratio'] if eval_results_predicted['context_token_ratio'] else 0)
        total_eval_results_predicted['url_accuracy'].append(1 if pred_link == gold_link else 0)
        total_eval_results_predicted['context_use_ratio'].append(ir_ctx_used)

        eval_results_gold = evaluate_answer(answer_gold_context, qa['context'], qa['answers'][0])
        total_eval_results_gold['answer_f1'].append(eval_results_gold['answer_f1'] if eval_results_gold['answer_f1'] else 0)
        total_eval_results_gold['answer_context_f1'].append(eval_results_gold['answer_context_f1'] if eval_results_gold['answer_context_f1'] else 0)
        total_eval_results_gold['answer_bleu'].append(eval_results_gold['answer_bleu'] if eval_results_gold['answer_bleu'] else 0)
        total_eval_results_gold['answer_context_bleu'].append(eval_results_gold['answer_context_bleu'] if eval_results_gold['answer_context_bleu'] else 0)
        total_eval_results_gold['answer_token_ratio'].append(eval_results_gold['answer_token_ratio'] if eval_results_gold['answer_token_ratio'] else 0)
        total_eval_results_gold['answer_context_token_ratio'].append(
            eval_results_gold['answer_context_token_ratio'] if eval_results_gold['answer_context_token_ratio'] else 0)
        total_eval_results_gold['context_use_ratio'].append(gold_ctx_used)

    # we don't need to return anything, results are printed on the console at the start of the loop
    return


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--eval_file', type=str, help='Evaluation data. Should be split name for sqad, filename for others.')
    ap.add_argument('--data_root', type=str, help='Evaluation data root directory')
    ap.add_argument('--data_type',type=str, help='The format of evaluation data. Can be [cs-SQuAD|SQAD|edubot]')
    ap.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')
    ap.add_argument('--cuda', action='store_true', help='Use GPU (true by default)')
    ap.add_argument('--no-cuda', dest='cuda', action='store_false')
    ap.set_defaults(cuda=True)

    args = ap.parse_args()
    parser = DataParser.parser_factory(args.data_type, args.data_root, first_paragraph_only=True)
    parser.load_from_file(args.eval_file)

    logger.info(f"Loading config from: {args.config}")
    with open(args.config, 'rt') as fd:
        config = yaml.load(fd, Loader=SafeLoader)

    remote_service_handler = RemoteServiceHandler(config)

    device = torch.device('cuda') if (args.cuda and torch.cuda.is_available()) else torch.device('cpu:0')
    qa_handler = QAHandler(config, remote_service_handler, device)

    run_on_data(parser.data, qa_handler)
