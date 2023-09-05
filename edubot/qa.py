from typing import Text, Iterable, Tuple, Dict, Any, List
from logzero import logger
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
from langchain import OpenAI, PromptTemplate, LLMChain


class OpenAIQA:
    def __init__(self, model_name):
        answer_llm = OpenAI(model_name=model_name,
                            temperature=0,
                            top_p=0.8,
                            openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
        answer_prompt = PromptTemplate(input_variables=['context', 'question'],
                                       template="""Odpověz na otázku s využitím kontextu.
        Využij pouze informace z kontextu, kopíruj text co nejvíc je to možné.
        Buď stručný a odpověz maximálně jednou větou. Nepoužívej více vět.
        Kontext:
        {context}
        Otázka:
        {question}
        Odpověď:""")
        self.llm_answer_chain = LLMChain(llm=answer_llm, prompt=answer_prompt)

    def __call__(self, kwarg_dict: Dict[Text, Any]) -> Tuple[Text, Text]:
        assert 'context' in kwarg_dict and 'question' in kwarg_dict
        logger.debug(f'OpenAI query {str(kwarg_dict)}')
        try:
            response = self.llm_answer_chain.run(**kwarg_dict)
        except Exception as e:
            logger.error('Exception in OpenAI/Langchain')
            logger.exception(e)
            response = 'Toto by možná mohlo pomoct: ' + kwarg_dict['context']
        return response.strip(), 1.0  # scores are not implemented in LangChain yet (https://github.com/hwchase17/langchain/issues/1063)


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


class QAHandler:

    def __init__(self, config, remote_service_handler, device):

        self.remote_service_handler = remote_service_handler

        # load QA model
        if config['QA']['MODEL_TYPE'] == 'huggingface':
            self.qa_model = HuggingFaceQA(config['QA']['HUGGINGFACE_MODEL'], device)
        elif config['QA']['MODEL_TYPE'] == 'openai':
            self.qa_model = OpenAIQA(config['QA']['OPENAI_MODEL'])
        elif config['QA']['MODEL_TYPE'] == 'local' and os.path.isdir(config['QA']['LOCAL_MODEL']):
            from multilingual_qaqg.mlpipelines import pipeline

            self.qa_model = pipeline("multitask-qa-qg",
                                     os.path.join(config['QA']['LOCAL_MODEL'], "checkpoint-185000"),
                                     os.path.join(config['QA']['LOCAL_MODEL'], "mt5_qg_tokenizer"),
                                     use_cuda=(device.type == 'cuda'))
        else:
            logger.warning('Could not find valid QA model setting, will run without it')
            self.qa_model = None
        logger.info(f'QA model: {config["QA"]["MODEL_TYPE"]} / {config["QA"][config["QA"]["MODEL_TYPE"].upper() + "_" + "MODEL"]} / {str(type(self.qa_model))}')

        if config.get('SENTENCE_REPR_MODEL', '').lower() in ['robeczech', 'eleczech']:
            from edubot.educlf.model import IntentClassifierModel
            self.repr_model = IntentClassifierModel(config['SENTENCE_REPR_MODEL'],
                                                    device,
                                                    label_mapping=None,
                                                    out_dir=None)
        elif 'SENTENCE_REPR_MODEL' in config:
            from sentence_transformers import SentenceTransformer
            self.repr_model = SentenceTransformer(config['SENTENCE_REPR_MODEL'],
                                                  device=device)
        else:
            self.repr_model = None
        logger.info(f'Sentence repr model: {config.get("SENTENCE_REPR_MODEL")} / {str(type(self.repr_model))}')

        reformulate_model_path = config.get('REFORMULATE_MODEL_PATH', None)
        if reformulate_model_path is not None and 'openai/' in reformulate_model_path:
            self.reformulate_model = OpenAIReformulate(reformulate_model_path.split('/')[-1])
        else:
            self.reformulate_model = None
        logger.info(f'Reformulate model: {reformulate_model_path} / {str(type(self.reformulate_model))}')

        self.site_to_pref = config['QA'].get('SITE_TO_PREF', {'default': ['WIKI']})

    def get_solr_configs(self, site, filtered_query_nac, filtered_query_nacv):
        """Get SOLR search configs -- what query to build, which attributes to search, what sources to filter."""
        cfgs = []
        for src in self.site_to_pref.get('site', self.site_to_pref['default']):
            if src == 'WIKI':  # wiki is a bit more detailed
                cfgs.extend([(f'"{filtered_query_nac}"', 'title_str', 'wiki'),
                            (f'"{filtered_query_nac}"', 'title_cz', 'wiki'),
                            (f'"{filtered_query_nac}"', 'first_paragraph_cz', 'wiki'),
                            (filtered_query_nac, 'title_cz', 'wiki'),
                            (filtered_query_nacv, 'first_paragraph_cz', 'wiki')])
            else:  # other sources are stricter
                cfgs.extend([(filtered_query_nacv, 'title_cz', src),
                             (filtered_query_nacv, 'first_paragraph_cz', src),])
        return cfgs

    def apply_qa(self, query, context=None, exact=False, site='default'):
        """Main QA entry point, running the query & models."""
        if self.reformulate_model is not None:
            reformulated_query = self.reformulate_model.run(question=query) or query
        else:
            reformulated_query = query
        filtered_query_nac, filtered_query_nacv, query_type =\
            self.remote_service_handler.filter_query(reformulated_query)
        if isinstance(self.qa_model, OpenAIQA):
            query_type = 'default'
        logger.info(f'Q: {query} | F: {filtered_query_nac} | {filtered_query_nacv}')

        solr_configs = self.get_solr_configs(site, filtered_query_nac, filtered_query_nacv)

        src = None  # source
        if not context and filtered_query_nacv:
            for q, a, src in solr_configs:
                if not q:  # skip if filtered query is empty
                    continue
                db_result = self.remote_service_handler.ask_solr(query=q, attrib=a, source=src)
                if db_result.get('docs'):
                    break

            if not db_result.get('docs'):
                logger.info('No result.')
                return None, None, None, None

            logger.debug("\n" + "\n".join([f'D: {doc["title"]}/{doc["score"]}' for doc in db_result['docs']]))

            answers = db_result['docs']

            if self.repr_model:
                if src == 'wiki':
                    # rank wiki articles by content (=answer) as title isn't very indicative
                    ranked_answers = self.rank_utterances(query, [(a['first_paragraph'], a) for a in answers])
                else:
                    # but rank other articles by title (=question) as this is closer to the query
                    ranked_answers = self.rank_utterances(query, [(a['title'], a) for a in answers])
                _, chosen_answer = ranked_answers[0]
            else:
                chosen_answer = answers[0]

            title = chosen_answer["title"]
        else:
            chosen_answer = {'first_paragraph': None, 'url': None}
            title = None

        if src == 'wiki' and exact and query_type == 'default' and self.qa_model:
            if not context:
                context = chosen_answer['first_paragraph']
            response, _ = self.qa_model({'question': query, 'context': context})
            return context, response, title, chosen_answer["url"]
        return chosen_answer['first_paragraph'], None, title, chosen_answer["url"]

    def rank_utterances(self, reference: Text, candidates: Iterable[Tuple[Text, Any]])\
            -> List[Tuple[float, Dict[Text, Text]]]:
        """
        :param reference: The reference text
        :param candidates: Iterable of tuples (keytext, Any), keytext is used for sorting
        :return: sorted candidates list along with zipped distances. The keys used for representation aren't returned.
        """
        reference_repr = self.repr_model.encode(reference)
        candidates_repr = (self.repr_model.encode(c[0]) for c in candidates)
        distances = [(cosine(reference_repr, cr), c[1]) for cr, c in zip(candidates_repr, candidates)]
        return sorted(distances, key=lambda c: c[0])
