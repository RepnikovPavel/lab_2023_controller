def WinString(str_:str):
    return '\"{}\"'.format(str_)
def WinPath(str_:str):
    return str_.replace('\\','/')