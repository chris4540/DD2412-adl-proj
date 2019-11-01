import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_mean_trans_err(df):
    ret = np.abs(df['teacher'] - df['student']).mean()
    return ret

if __name__ == '__main__':
    fewshot_kdat_df = pd.read_csv('fewshot.csv', index_col=0)
    zero_df = pd.read_csv('zeroshot.csv', index_col=0)
    normalkd_df = pd.read_csv('normal_kd.csv', index_col=0)

    print('Mean trans err of fewshot: ', calculate_mean_trans_err(fewshot_kdat_df))
    print('Mean trans err of zeroshot: ', calculate_mean_trans_err(zero_df))
    print('Mean trans err of normal-kd: ', calculate_mean_trans_err(normalkd_df))
    # ================================
    fewshot_kdat_df.rename({'student': 'fewshot'})
    ax = fewshot_kdat_df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Adversarial steps")
    ax.set_ylabel("Probability to match other classes")
    plt.savefig('fewshot.png', bbox_inches='tight')

    zero_df.rename({'student': 'zeroshot'})
    ax = zero_df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_ylabel("Probability to match other classes")
    ax.set_xlabel("Adversarial steps")
    plt.savefig('zeroshot.png',bbox_inches='tight')

    normalkd_df.rename({'student': 'normal KD'})
    ax = normalkd_df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_ylabel("Probability to match other classes")
    ax.set_xlabel("Adversarial steps")
    plt.savefig('normalkd.png', bbox_inches='tight')
    # ================================
