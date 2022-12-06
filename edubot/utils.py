from logzero import logger

def apply_qa(remote_service_handler, qa_model, query, context=None, exact=False):
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
        title = answers[0]["title"]
    else:
        answers = [{'first_paragraph': None, 'url': None}]
        title = None

    if exact and query_type == 'default' and qa_model:
        # reranking by QA decoding score -- doesn't seem to work
        #resp_cands = []
        #for context in [a['first_paragraph'] for a in answers[:1]]:
            #resp_cands.append(qa_model({'question': query, 'context': context}))
        #logger.debug('RCs:\n' + "\n".join(['RC: %s | %f' % rc for rc in resp_cands]))
        #response, _ = max(resp_cands, key=lambda rc: rc[1])
        # feeding multiple contexts -- doesn't seem to work
        if not context:
            context = "\n".join([a['first_paragraph'] for a in answers[:1]])
        response, _ = qa_model({'question': query, 'context': context})
        return context, response, title, answers[0]["url"]
    return answers[0]["first_paragraph"], None, title, answers[0]["url"]


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
