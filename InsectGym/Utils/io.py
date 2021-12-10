def sterilize(obj):
    """Make an object more ameniable to dumping as json
    """
    if type(obj) in (str, float, int, bool, type(None)):
        return obj
    elif isinstance(obj, dict):
        return {k: sterilize(v) for k, v in obj.items()}
    list_ret = []
    dict_ret = {}
    for a in dir(obj):
        if a == '__iter__' and callable(obj.__iter__):
            list_ret.extend([sterilize(v) for v in obj])
        elif a == '__dict__':
            dict_ret.update({k: sterilize(v) for k, v in obj.__dict__.items() if k not in ['__module__', '__dict__', '__weakref__', '__doc__']})
        elif a not in ['__doc__', '__module__']:
            aval = getattr(obj, a)
            if type(aval) in (str, float, int, bool, type(None)):
                dict_ret[a] = aval
            elif a != '__class__' and a != '__objclass__' and isinstance(aval, type):
                dict_ret[a] = sterilize(aval)
    if len(list_ret) == 0:
        if len(dict_ret) == 0:
            return repr(obj)
        return dict_ret
    else:
        if len(dict_ret) == 0:
            return list_ret
    return (list_ret, dict_ret)