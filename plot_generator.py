"""
Sample the image from the generator

Please download the folder from the project storage folder with this command
$ gsutil -m cp -r  gs://dd2412-proj-exp-data/re-exp/plot_gen/zeroshot_cifar10_T-40-2_S-16-1_seed_45/ .
"""
import os
import tensorflow as tf
tf.enable_v2_behavior() # Must run this in order to have similar behaviour as TF2.0
import matplotlib
matplotlib.use('Agg')   # use Agg backend for no display environemet
from matplotlib import pyplot as plt
import numpy as np
from utils.seed import set_seed
from utils.preprocess import get_cifar10_mean_std
from net.generator import NavieGenerator

# ----------
# Config
# ----------
class Config:
    z_dim = 100
    seed = 45
    n_pics = 5

def get_z_val():
    ret = tf.random.normal([Config.n_pics, Config.z_dim])
    return ret

def generate_pics_from_weight_file(weight_file, z_val):
    # load generator from weight file
    generator = NavieGenerator(input_dim=Config.z_dim)
    generator.load_weights(weight_file)

    pseudo_imgs = generator(z_val, training=False)

    mean, std = get_cifar10_mean_std()
    # put back mean and std
    ret = pseudo_imgs * std + mean
    # pseudo_imgs
    return ret
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    set_seed(Config.seed)
    files = [
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i100.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i200.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i300.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i400.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i500.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i700.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i1000.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i1500.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i2000.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i2500.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i10000.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i15000.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i20000.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i50000.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i70000.h5",
        "zeroshot_cifar10_T-40-2_S-16-1_seed_45/generator_i79999.h5",
    ]

    ncols = len(files)
    fig, axs = plt.subplots(Config.n_pics, ncols, figsize=(ncols, Config.n_pics))
    plt.setp(axs, xticks=[], yticks=[])
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    z_val = get_z_val()

    for j, file in enumerate(files):
        imgs = generate_pics_from_weight_file(file, z_val)
        for i in range(Config.n_pics):
            ax = axs[i, j]
            ax.axis('off')
            ax.imshow(imgs[i, :], interpolation='nearest')

    plt.savefig("cifar10-generator-samples.pdf", bbox_inches='tight')
