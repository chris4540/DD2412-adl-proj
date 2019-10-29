import os


def mkdir(dirname):
    folder = os.path.join(os.getcwd(), dirname)
    os.makedirs(folder, exist_ok=True)
