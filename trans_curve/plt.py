import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fewshot_kdat_df = pd.read_csv('fewshot_kdat.csv', index_col=0)
    zero_df = pd.read_csv('zero.csv', index_col=0)

    # map column name

    # ================================
    ax = fewshot_kdat_df.plot()
    ax.set_ylim(0, 0.15)

    plt.savefig('fewshot_kdat.png')

    ax = zero_df.plot()
    ax.set_ylim(0, 0.15)
    plt.savefig('zero.png')
