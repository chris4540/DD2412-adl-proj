import os


def mkdir(dirname):
    save_dir = os.path.join(os.getcwd(), dirname)
    os.makedirs(save_dir, exist_ok=True)

