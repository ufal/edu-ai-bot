from typing import Text, Iterable, Tuple, Dict, Any, List
from logzero import logger
from scipy.spatial.distance import cosine
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
        response = self.llm_answer_chain.run(**kwarg_dict)
        return response, kwarg_dict['context']


class QAHandler:

    def __init__(self, qa_model, repr_model, remote_service_handler, reformulate_model=None):
        self.qa_model = qa_model
        self.repr_model = repr_model
        self.remote_service_handler = remote_service_handler
        self.reformulate_model = reformulate_model


    def apply_qa(self, query, context=None, exact=False):
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

        if not context and filtered_query_nacv:
            for q, a, s in [#(filtered_query_nacv, 'title_cz', 'logic'),   # XXX logic olympiad not used at the moment, TODO make this configurable
                            #(filtered_query_nacv, 'first_paragraph_cz', 'logic'),
                            (f'"{filtered_query_nac}"', 'title_str', 'wiki'),
                            (f'"{filtered_query_nac}"', 'title_cz', 'wiki'),
                            (f'"{filtered_query_nac}"', 'first_paragraph_cz', 'wiki'),
                            (filtered_query_nac, 'title_cz', 'wiki'),
                            (filtered_query_nacv, 'first_paragraph_cz', 'wiki')]:
                if not q:  # skip if filtered_query_nac is empty
                    continue
                db_result = self.remote_service_handler.ask_solr(query=q, attrib=a, source=s)
                if db_result.get('docs'):
                    break

            if not db_result.get('docs'):
                logger.info(f'No result.')
                return None, None, None, None

            logger.debug("\n" + "\n".join([f'D: {doc["title"]}/{doc["score"]}' for doc in db_result['docs']]))

            answers = db_result['docs']

            ranked_answers = self.rank_utterance_list_by_similarity(query,
                                                                    [(a['first_paragraph'], a) for a in answers])
            distance, chosen_answer = ranked_answers[0]
            title = chosen_answer["title"]
        else:
            chosen_answer = {'first_paragraph': None, 'url': None}
            title = None

        if exact and query_type == 'default' and self.qa_model:
            if not context:
                context = chosen_answer['first_paragraph']
            response, _ = self.qa_model({'question': query, 'context': context})
            return context, response, title, chosen_answer["url"]
        return chosen_answer['first_paragraph'], None, title, chosen_answer["url"]

    def rank_utterance_list_by_similarity(self, reference: Text, candidates: Iterable[Tuple[Text, Any]])\
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
