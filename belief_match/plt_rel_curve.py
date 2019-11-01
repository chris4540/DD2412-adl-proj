import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cal_mean_trans_err(df):
    ret = np.abs(df['teacher'] - df['student']).mean()
    return ret

def cal_rel_mean_trans_err(df):
    max_element = df.max().max()
    ret = cal_mean_trans_err(df) / max_element
    return ret


if __name__ == '__main__':
    fewshot_kdat_df = pd.read_csv('fewshot.csv', index_col=0)
    zero_df = pd.read_csv('zeroshot.csv', index_col=0)
    normalkd_df = pd.read_csv('normal_kd.csv', index_col=0)

    print('Mean trans err of fewshot: ', cal_mean_trans_err(fewshot_kdat_df))
    print('Rel. Mean trans err of fewshot: ', cal_rel_mean_trans_err(fewshot_kdat_df))
    #
    print('Mean trans err of zeroshot: ', cal_mean_trans_err(zero_df))
    print('Rel. Mean trans err of zeroshot: ', cal_rel_mean_trans_err(zero_df))
    #
    print('Mean trans err of normal-kd: ', cal_mean_trans_err(normalkd_df))
    print('Rel. Mean trans err of normal-kd: ', cal_rel_mean_trans_err(normalkd_df))
    # ==========================================================
    # Few shot
    fewshot_kdat_df.rename({'student': 'fewshot'})
    ax = fewshot_kdat_df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Adversarial steps")
    ax.set_ylabel("Probability to match other classes")
    plt.savefig('fewshot.png', bbox_inches='tight')
    # normalized
    df = fewshot_kdat_df / fewshot_kdat_df.max().max()
    ax = df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Adversarial steps")
    ax.set_ylabel("Probability to match other classes")
    plt.savefig('renormal_fewshot.png', bbox_inches='tight')
    # =============================================================
    # Zero shot
    zero_df.rename({'student': 'zeroshot'})
    ax = zero_df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_ylabel("Probability to match other classes")
    ax.set_xlabel("Adversarial steps")
    plt.savefig('zeroshot.png',bbox_inches='tight')
    # normalized
    df = zero_df / zero_df.max().max()
    ax = df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Adversarial steps")
    ax.set_ylabel("Probability to match other classes")
    plt.savefig('renormal_zeroshot.png', bbox_inches='tight')
    # =============================================================
    # Normal KD
    normalkd_df.rename({'student': 'Normal KD'})
    ax = normalkd_df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_ylabel("Probability to match other classes")
    ax.set_xlabel("Adversarial steps")
    plt.savefig('normalkd.png', bbox_inches='tight')
    # normalized
    df = normalkd_df / normalkd_df.max().max()
    ax = df.plot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Adversarial steps")
    ax.set_ylabel("Probability to match other classes")
    plt.savefig('renormal_normalkd.png', bbox_inches='tight')
    # =============================================================
