import sys
sys.path.append('../')
import logzero
from edubot.eval import DataParser
import os
from argparse import ArgumentParser
import torch
import time
import yaml
from yaml.loader import SafeLoader
from logzero import logger
import fuzzywuzzy.fuzz
from nltk.metrics.scores import f_measure
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from edubot.remote_services import RemoteServiceHandler
from edubot.qa import QAHandler, OpenAIQA
from langchain import OpenAI, PromptTemplate, LLMChain


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
                                    'context_token_ratio': []}
    total_eval_results_gold = {'answer_f1': [],
                               'answer_context_f1': [],
                               'answer_bleu': [],
                               'answer_context_bleu': [],
                               'answer_token_ratio': [],
                               'answer_context_token_ratio': []}
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
        _, answer_gold_context, _, _ = qa_handler.apply_qa(qa['question'], qa['context'], exact=True)
        exec_times.append(time.time() - st)
        st = time.time()
        pred_context, answer_pred_context, pred_title, url = qa_handler.apply_qa(qa['question'], None, exact=True)
        exec_times.append(time.time() - st)
        if answer_pred_context is None or answer_gold_context is None:
            logger.warning("None answer " + str(qa))
            continue
        logger.warning(f'Average execution time: {sum(exec_times) / len(exec_times):.4f}')
        answer_pred_context = answer_pred_context.strip()
        answer_gold_context = answer_gold_context.strip()


        logger.info(f"Question:\t\t\t\t {qa['question']}")
        logger.info(f"Retrieved context:\t\t {answer_pred_context}")
        logger.info(f"Gold context:\t\t\t {answer_gold_context}")
        logger.info(f"Ground truth answer:\t {qa['answers'][0]}")
        logger.info("-" * 100)

        print(f"Question:\t\t\t\t {qa['question']}")
        print(f"Retrieved context:\t\t {answer_pred_context}")
        print(f"Gold context:\t\t\t {answer_gold_context}")
        print(f"Ground truth answer:\t {qa['answers'][0]}")
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

        eval_results_gold = evaluate_answer(answer_gold_context, qa['context'], qa['answers'][0])
        total_eval_results_gold['answer_f1'].append(eval_results_gold['answer_f1'] if eval_results_gold['answer_f1'] else 0)
        total_eval_results_gold['answer_context_f1'].append(eval_results_gold['answer_context_f1'] if eval_results_gold['answer_context_f1'] else 0)
        total_eval_results_gold['answer_bleu'].append(eval_results_gold['answer_bleu'] if eval_results_gold['answer_bleu'] else 0)
        total_eval_results_gold['answer_context_bleu'].append(eval_results_gold['answer_context_bleu'] if eval_results_gold['answer_context_bleu'] else 0)
        total_eval_results_gold['answer_token_ratio'].append(eval_results_gold['answer_token_ratio'] if eval_results_gold['answer_token_ratio'] else 0)
        total_eval_results_gold['answer_context_token_ratio'].append(
            eval_results_gold['answer_context_token_ratio'] if eval_results_gold['answer_context_token_ratio'] else 0)

    return total_eval_results_predicted, total_eval_results_gold


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--eval_file', type=str, help='Evaluation data. Should be split name for sqad, filename for others.')
    ap.add_argument('--data_root', type=str, help='Evaluation data root directory')
    ap.add_argument('--data_type',type=str, help='The format of evaluation data. Can be [cs-SQuAD|SQAD|edubot]')
    ap.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')

    args = ap.parse_args()
    parser = DataParser.parser_factory(args.data_type, args.data_root, first_paragraph_only=True)
    parser.load_from_file(args.eval_file)

    logger.info(f"Loading config from: {args.config}")
    with open(args.config, 'rt') as fd:
        custom_config = yaml.load(fd, Loader=SafeLoader)
    with open(custom_config['STOPWORDS_PATH'], 'rt') as fd:
        stopwords = set((w.strip() for w in fd.readlines() if len(w.strip()) > 0))
    remote_service_handler = RemoteServiceHandler(custom_config, stopwords)
    if 'openai/' in custom_config['QA_MODEL_PATH']:
        qa_model = OpenAIQA(custom_config['QA_MODEL_PATH'].split('/')[-1])
    elif os.path.isdir(custom_config['QA_MODEL_PATH']):
        from multilingual_qaqg.mlpipelines import pipeline

        qa_model = pipeline("multitask-qa-qg",
                            os.path.join(custom_config['QA_MODEL_PATH'], "checkpoint-185000"),
                            os.path.join(custom_config['QA_MODEL_PATH'], "mt5_qg_tokenizer"),
                            use_cuda=args.cuda)
    else:
        logger.warning('Could not find QA directory, will run without it')
        qa_model = None

    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda') if cuda_available else torch.device('cpu:0')
    if custom_config['SENTENCE_REPR_MODEL'].lower() in ['robeczech', 'eleczech']:
        from edubot.educlf.model import IntentClassifierModel

        sentence_repr_model = IntentClassifierModel(custom_config['SENTENCE_REPR_MODEL'],
                                                    device,
                                                    label_mapping=None,
                                                    out_dir=None)
    else:
        from sentence_transformers import SentenceTransformer

        sentence_repr_model = SentenceTransformer(custom_config['SENTENCE_REPR_MODEL'],
                                                  device=device)
    reformulate_model_path = custom_config.get('REFORMULATE_MODEL_PATH', None)
    if reformulate_model_path is not None and 'openai/' in reformulate_model_path:
        reformulate_model_name = reformulate_model_path.split('/')[-1]
        reformulate_llm = OpenAI(model_name=reformulate_model_name,
                                 temperature=0,
                                 top_p=0.8,
                                 openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
        reformulate_prompt = PromptTemplate(input_variables=['question'],
                                            template="""Začni odpověď na otázku.
    Otázka:
    {question}
    Odpověď:""")
        reformulate_model = LLMChain(llm=reformulate_llm, prompt=reformulate_prompt)
    else:
        reformulate_model = None

    qa_handler = QAHandler(qa_model,
                           sentence_repr_model,
                           remote_service_handler,
                           reformulate_model)
    run_on_data(parser.data, qa_handler)
