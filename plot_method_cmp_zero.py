"""
Plot zeroshot with m=100 vs without using any image
"""
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df_with_real = pd.read_csv("./exp_results/cifar10_T-40-2_S-16-1_seed_23-m100_training_log.csv")
    df_wo_real = pd.read_csv("./exp_results/cifar10_T-40-2_S-16-1_seed_45_training_log.csv")

    df_with_real = df_with_real.set_index("epoch")
    df_wo_real = df_wo_real.set_index("epoch")

    test_acc_with_real = df_with_real['test_acc']
    test_acc_wo_real = df_wo_real['test_acc']

    # re-sample
    sample_epochs = [f for f in range(1000, 80000, 1000)]
    sample_epochs.append(79999)
    test_acc_with_real = test_acc_with_real.loc[sample_epochs]
    test_acc_wo_real = test_acc_wo_real.loc[sample_epochs]

    df = pd.DataFrame({
        "With real data": test_acc_with_real,
        "Without real data": test_acc_wo_real,
    })

    ax = df.plot(kind="line")
    plt.show()
