from typing import Text, List, Tuple, Dict
from logzero import logger
from scipy.spatial.distance import cosine


def rank_utterance_list_by_similarity(model, reference: Text, candidates: List[Tuple[Text, Dict[Text, Text]]])\
        -> List[Tuple[float,  Dict[Text, Text]]]:
    """
    :param model: Model used for sentence representations
    :param reference: The reference text
    :param candidates: List of tuples (keytext, Any), keytext is used for sorting
    :return: sorted candidates list along with zipped distances. The keys used for representation aren't returned.
    """
    reference_repr = model.encode(reference)
    candidates_repr = [model.encode(c[0]) for c in candidates]
    distances = [(cosine(reference_repr, cr), c[1]) for cr, c in zip(candidates_repr, candidates)]
    return sorted(distances, key=lambda c: c[0] )


def apply_qa(remote_service_handler, qa_model, repr_model, query, context=None, exact=False):
    filtered_query_nac, filtered_query_nacv, query_type = remote_service_handler.filter_query(query)
    logger.info(f'Q: {query} | F: {filtered_query_nac} | {filtered_query_nacv}')
    if not context and filtered_query_nacv:
        for q, a, s in [(filtered_query_nacv, 'title_cz', 'logic'),
                        (filtered_query_nacv, 'first_paragraph_cz', 'logic'),
                        (f'"{filtered_query_nac}"', 'title_str', 'wiki'),
                        (f'"{filtered_query_nac}"', 'title_cz', 'wiki'),
                        (f'"{filtered_query_nac}"', 'first_paragraph_cz', 'wiki'),
                        (filtered_query_nac, 'title_cz', 'wiki'),
                        (filtered_query_nacv, 'first_paragraph_cz', 'wiki')]:
            if not q:  # skip if filtered_query_nac is empty
                continue
            db_result = remote_service_handler.ask_solr(query=q, attrib=a, source=s)
            if db_result.get('docs'):
                break

        if not db_result.get('docs'):
            logger.info(f'No result.')
            return None, None, None, None

        logger.debug("\n" + "\n".join([f'D: {doc["title"]}/{doc["score"]}' for doc in db_result['docs']]))

        answers = db_result['docs']

        ranked_answers = rank_utterance_list_by_similarity(repr_model,
                                                           query,
                                                           [(a['first_paragraph'], a) for a in answers])
        print('RANKED')
        for a in ranked_answers:
            print(a)
        distance, chosen_answer = ranked_answers[0]
        title = chosen_answer["title"]
    else:
        chosen_answer = {'first_paragraph': None, 'url': None}
        title = None

    if exact and query_type == 'default' and qa_model:
        if not context:
            context = chosen_answer['first_paragraph']
        response, _ = qa_model({'question': query, 'context': context})
        return context, response, title, chosen_answer["url"]
    return chosen_answer["first_paragraph"], None, title, chosen_answer["url"]


class dotdict(dict):

    def __init__(self, dct=None):
        if dct is not None:
            dct = dotdict.transform(dct)
        else:
            dct = {}
        super(dotdict, self).__init__(dct)

    @staticmethod
    def transform(dct):
        new_dct = {}
        for k, v in dct.items():
            if isinstance(v, dict):
                new_dct[k] = dotdict(v)
            else:
                new_dct[k] = v
        return new_dct

    __getattr__ = dict.__getitem__

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            super(dotdict, self).__setitem__(key, dotdict(value))
        else:
            super(dotdict, self).__setitem__(key, value)

    def __setattr__(self, key, value):
        self[key] = value

    __delattr__ = dict.__delitem__

    def __getstate__(self):
        result = self.__dict__.copy()
        return result

    def __setstate__(self, dict):
        self.__dict__ = dict
