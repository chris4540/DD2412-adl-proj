import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('exp_results/cmp_methods.csv')
    df = df.set_index('m')
    styles = {
        'No teacher': "r.-",
        "KD+AT": "g.-",
        "KD+AT full data": "m-",
        'Zeroshot(m=0)': "b-",
        'Zeroshot(m=100)': "c*-",
    }
    ax = df.plot(kind='line', style=styles)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Number of Images per class (M)")
    ax.set_title("CIFAR-10")
    plt.savefig("method_cmp.pdf", bbox_inches='tight')
