# def tuple_values_to_string(t,sep:str):
#     s =''
#     for i in range(len(t)-1):
#         s += str(t[i])
#         s += sep
#     s += str(t[-1])
#     return s

def SqlString(str_):
    return '\"{}\"'.format(str_)
