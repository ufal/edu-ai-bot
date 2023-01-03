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
