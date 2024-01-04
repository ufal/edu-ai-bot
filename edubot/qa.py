from typing import Text, Iterable, Tuple, Dict, Any, List
from numbers import Number
from logzero import logger
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from edubot.educlf.model import IntentClassifierModel


class OpenAIQA:

    CONTEXT_IRRELEVANT_PHRASES = [
        '(?:není|nejsou) součástí poskytnutého kontextu',
        'kontext ne(?:poskytuje|obsahuje|uvádí|ní relevantní)',
        'v(?: poskytnutém)? kontextu není',
        'kontextu nelze určit',
        r'není(?: \w+){0,5} (?:ke?|v)(?: poskytnutému?)? kontextu',
        r'kontext (?:se týká|poskytuje|obsahuje)(?: "?\w+"?){0,10}(?:, (?:ne|nikoliv?)| a neobsahuje| a ne)',
        '(?:není možné|nelze) (?:odpovědět|určit) (?:na základě|s využitím)(?: poskytnutého)? kontextu',
    ]

    def __init__(self, model_name):
        # support both legacy & new "chat" models
        openai_class = OpenAI
        if "gpt-4" in model_name or "gpt-3.5-turbo-1106" in model_name:
            openai_class = ChatOpenAI

        answer_llm = openai_class(model_name=model_name,
                                  temperature=0,
                                  top_p=0.8,
                                  openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
        answer_prompt = PromptTemplate(input_variables=['context', 'question'],
                                       template="""Odpověz na otázku s využitím kontextu.
        Využij pouze informace z kontextu, kopíruj text co nejvíc je to možné.
        Pokud kontext neobsahuje potřebné informace, odpověz jen „Kontext není relevantní“.
        Buď stručný a odpověz maximálně jednou větou. Nepoužívej více vět.
        Kontext:
        {context}
        Otázka:
        {question}
        Odpověď:""")
        self.llm_answer_chain = LLMChain(llm=answer_llm, prompt=answer_prompt)

        answer_no_context = PromptTemplate(input_variables=['question'],
                                       template="""Odpověz na otázku.
        Buď stručný a odpověz maximálně jednou větou. Nepoužívej více vět.
        Otázka:
        {question}
        Odpověď:""")
        self.llm_backup_chain = LLMChain(llm=answer_llm, prompt=answer_no_context)
        self.no_context_pattern = re.compile(r'\b(?:' + '|'.join(self.CONTEXT_IRRELEVANT_PHRASES) + r')\b', re.I)

    def __call__(self, kwarg_dict: Dict[Text, Any]) -> Tuple[Text, Text]:
        assert 'context' in kwarg_dict and 'question' in kwarg_dict
        logger.debug(f'OpenAI query {str(kwarg_dict)}')
        score = 1.0
        try:
            if kwarg_dict['context']:
                response = self.llm_answer_chain.run(**kwarg_dict)
                if self.no_context_pattern.search(response):
                    logger.debug(f'Re-runing w/o context (response: {response.strip()})')
                    response = self.llm_backup_chain.run(**kwarg_dict)
                    score = 0.0
            else:
                response = self.llm_backup_chain.run(**kwarg_dict)
                score = 0.0
        except Exception as e:
            logger.error('Exception in OpenAI/Langchain')
            logger.exception(e)
            response = 'Toto by možná mohlo pomoct: ' + kwarg_dict['context']
        return response.strip(), score


class OpenAIReformulate:

    def __init__(self, model_name):
        reformulate_llm = OpenAI(model_name,
                                 temperature=0,
                                 top_p=0.8,
                                 openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
        reformulate_prompt = PromptTemplate(input_variables=['question'],
                                            template="""Začni odpověď na otázku.
Otázka:
{question}
Odpověď:""")
        self.llm_chain = LLMChain(llm=reformulate_llm, prompt=reformulate_prompt)

    def run(self, question):
        return self.llm_chain.run(question=question)


class HuggingFaceQA:

    def __init__(self, model_name, device):
        self.device = device
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, kwarg_dict):
        question, context = kwarg_dict['question'], kwarg_dict['context']
        inputs = self.tokenizer(question, context, return_tensors="pt")
        outputs = self.model(**inputs.to(self.device))
        start_position = outputs.start_logits[0].argmax()
        end_position = outputs.end_logits[0].argmax()
        answer_ids = inputs["input_ids"][0][start_position:end_position]
        response = self.tokenizer.decode(answer_ids)
        return response, 1.0  # TODO fix confidence scores somehow


@dataclass
class QAResults:
    retrieved: str = None
    reply: str = None
    url: str = None
    source: str = None
    all_results: list = None


class QAHandler:

    def __init__(self, config, remote_service_handler, device):

        self.remote_service_handler = remote_service_handler
        config = config['QA']

        # load QA model
        if config['MODEL_TYPE'] == 'huggingface':
            self.qa_model = HuggingFaceQA(config['HUGGINGFACE_MODEL'], device)
        elif config['MODEL_TYPE'] == 'openai':
            self.qa_model = OpenAIQA(config['OPENAI_MODEL'])
        elif config['MODEL_TYPE'] == 'local' and os.path.isdir(config['LOCAL_MODEL']):
            from multilingual_qaqg.mlpipelines import pipeline

            self.qa_model = pipeline("multitask-qa-qg",
                                     os.path.join(config['LOCAL_MODEL'], "checkpoint-185000"),
                                     os.path.join(config['LOCAL_MODEL'], "mt5_qg_tokenizer"),
                                     use_cuda=(device.type == 'cuda'))
        else:
            logger.warning('Could not find valid QA model setting, will run without it')
            self.qa_model = None
        logger.info(f'QA model: {config["MODEL_TYPE"]} / {config[config["MODEL_TYPE"].upper() + "_" + "MODEL"]} / {str(type(self.qa_model))}')

        # init reranking models
        self.repr_model = self.init_repr_model(config.get('SENTENCE_REPR_MODEL'), device)
        logger.info(f'Sentence repr model: {config.get("SENTENCE_REPR_MODEL")} / {str(type(self.repr_model))}')

        self.repr_model_indomain = self.init_repr_model(config.get('SENTENCE_REPR_MODEL_INDOMAIN'), device)
        logger.info(f'Sentence repr model in-domain: {config.get("SENTENCE_REPR_MODEL_INDOMAIN")} / {str(type(self.repr_model_indomain))}')

        # init reformulation model (currently not used)
        reformulate_model_path = config.get('REFORMULATE_MODEL_PATH', None)
        if reformulate_model_path is not None and 'openai/' in reformulate_model_path:
            self.reformulate_model = OpenAIReformulate(reformulate_model_path.split('/')[-1])
        else:
            self.reformulate_model = None
        logger.info(f'Reformulate model: {reformulate_model_path} / {str(type(self.reformulate_model))}')

        # init additional settings
        self.site_to_pref = config.get('SITE_TO_PREF', {'default': ['WIKI']})
        self.inappropriate_regex = re.compile(r'\b(' + '|'.join(config.get('INAPPROPRIATE_KWS', ['$^'])) + r')\b', flags=re.I)
        self.distance_threshold = config.get('DISTANCE_THRESHOLD', 2.0)
        self.distance_threshold_indomain = config.get('DISTANCE_THRESHOLD_INDOMAIN', 2.0)

    def init_repr_model(self, model_name, device):
        """Simple initializer for representation models (supporting IntentClassifierModel (legacy) as well as SentenceTransformer)"""

        if model_name and (model_name.lower() in ['robeczech', 'eleczech']):
            return IntentClassifierModel(model_name, device, label_mapping=None, out_dir=None)
        elif model_name:
            return SentenceTransformer(model_name, device=device)
        else:
            return None

    def get_solr_configs(self, site, intent, filtered_query):
        """Get SOLR search configs -- what query to build, which attributes to search, what sources to filter."""
        cfgs = []
        sources = self.site_to_pref.get(site, self.site_to_pref['default'])
        if site == 'force':
            sources = [re.sub(r'qa_?', '', intent).upper()]  # take source from intent forcefully
        for src in sources:
            if src == 'WIKI':  # wiki is a bit more detailed
                cfgs.extend([(f'"{filtered_query.words_nac}"', 'title_str', 'wiki'),
                            (f'"{filtered_query.lemmas_nac}"', 'title_cz', 'wiki'),
                            (f'"{filtered_query.lemmas_nac}"', 'first_paragraph_cz', 'wiki'),
                            (filtered_query.lemmas_nac, 'title_cz', 'wiki'),
                            (filtered_query.lemmas_nacv, 'first_paragraph_cz', 'wiki')])
            else:  # other sources are stricter
                cfgs.extend([(filtered_query.lemmas_nacv, 'title_cz', src),
                             (filtered_query.lemmas_nacv, 'first_paragraph_cz', src),])
        return cfgs

    def apply_qa(self, query, context=None, intent='qawiki', exact=False, site='default'):
        """Main QA entry point, running the query & models."""
        if self.reformulate_model is not None:
            reformulated_query = self.reformulate_model.run(question=query) or query
        else:
            reformulated_query = query
        filtered, query_type = self.remote_service_handler.filter_query(reformulated_query)
        if isinstance(self.qa_model, OpenAIQA):
            query_type = 'default'
        logger.info(f'Q: {query} | F: {filtered.words_nac} | {filtered.words_nacv} | {filtered.lemmas_nacv}')

        solr_configs = self.get_solr_configs(site, intent, filtered)

        src = None  # source where we found something
        answers = None
        if not context and filtered.words_nacv:
            for q, a, src in solr_configs:
                if not q:  # skip if filtered query is empty
                    continue
                db_result = self.remote_service_handler.ask_solr(query=q, attrib=a, source=src)

                if db_result.get('docs'):
                    logger.debug(f"{src} \n" + "\n".join([f'D: {doc["title"]}/{doc["score"]}' for doc in db_result['docs']]))
                    answers = self.filter_inappropriate(db_result['docs'])
                    if not answers:
                        logger.info('No result left after filtering.')
                        continue

                    if self.repr_model:
                        if src == 'wiki':
                            # rank wiki articles by content (=answer) as title isn't very indicative
                            answers = self.rank_utterances(self.repr_model, query, [(a['first_paragraph'], a) for a in answers])
                        else:
                            rank_model = self.repr_model_indomain or self.repr_model
                            threshold = self.distance_threshold
                            if intent == 'qa_' + src.lower():  # source matching intent -- higher threshold
                                threshold = self.distance_threshold_indomain
                            # rank indomain articles by title (=question) as this is closer to the query
                            answers = self.rank_utterances(rank_model, query, [(a['title'], a) for a in answers], threshold)
                            if not answers:
                                logger.info('No result left after thresholding.')
                                continue
                    # stop searching if we found something (we didn't hit "continue")
                    break

        if answers is None:
            if not context and not (intent == 'qawiki' and exact and query_type == 'default' and self.qa_model):  # nothing found
                return QAResults()
            else:  # either context is provided or we try our luck with the QA model hallucinating w/o context
                answers = [{"first_paragraph": "", "url": "", "score": 0.0, "title": ""}]
                src = 'context' if context else 'model'

        for answer in answers:
            answer["content"] = answer["first_paragraph"]
            del answer["first_paragraph"]
            answer["score"] = float(answer["score"])
            if src == 'wiki':
                # replace ID with title for wiki URLs
                answer["url"] = 'https://cs.wikipedia.org/wiki/' + answer['title'].replace(' ', '_')
            else:
                # remove source prefix from URL for others than wiki
                answer["url"] = re.sub(r'^[A-Z]+ http', 'http', answer["url"])

        # run the QA model for wiki (or for testing)
        if (src in ['wiki', 'model'] or context) and exact and query_type == 'default' and self.qa_model:
            context = context or answers[0]['content']
            response, score = self.qa_model({'question': query, 'context': context})
            return QAResults(retrieved=context, reply=response, url=(answers[0]["url"] if score else ''), source=(src if score else 'model'), all_results=answers)
        # return 1st result as reply
        return QAResults(retrieved=answers[0]['content'], reply=None, url=answers[0]["url"], source=src, all_results=answers)

    def rank_utterances(self, rank_model, reference: Text, candidates: Iterable[Tuple[Text, Any]], threshold: Number = 2.0)\
            -> List[Dict[Text, Text]]:
        """
        :param rank_model: The model to use for the ranking
        :param reference: The reference text
        :param candidates: Iterable of tuples (keytext, Any), keytext is used for sorting
        :param threshold: Max. threshold for cosine distance -- anything over that is filtered out
        :return: candidates sorted by distances, with the "score" value replaced by distance. The keys used for representation aren't returned.
        """
        reference_repr = rank_model.encode(reference)
        candidates_repr = (rank_model.encode(c[0]) for c in candidates)
        distances = sorted([(cosine(reference_repr, cr), c[1]) for cr, c in zip(candidates_repr, candidates)], key=lambda c: c[0])
        logger.debug("\n" + "\n".join([(f'D: {doc["title"]}/{sim:.4f}' if sim < threshold else f'D: {doc["title"]}/{sim:.4f} !!') for sim, doc in distances]))
        distances = [c for c in distances if c[0] < threshold]
        for d, c in distances:
            c["score"] = f'{d:.6f}'
        return [c for _, c in distances]

    def filter_inappropriate(self, docs):
        res = []
        for d in docs:
            if self.inappropriate_regex.search(d['title']):
                logger.debug(f'Filtered out: {d["title"]}')
            else:
                res.append(d)
        return res
