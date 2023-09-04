import os

def delete_all_data_from_directory(directory: str):
    if not os.path.exists(directory):
        return
    if not os.path.isdir(directory):
        return
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))

def remove_dir(directory: str):
    os.remove(directory)

def mkdir_if_not_exists(dir: str):

    if not os.path.exists(dir):
        os.makedirs(dir)

def delete_file_if_exists(filename:str):
    if os.path.exists(filename):
        os.remove(filename)
