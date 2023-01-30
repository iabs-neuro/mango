import os


def folder_ensure(fpath):
    os.makedirs(fpath, exist_ok=True)


def sort_vector(a, asc=True):
    return sorted(zip(range(len(a)), a), key=lambda item: item[1], reverse=asc)
