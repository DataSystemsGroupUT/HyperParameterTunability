from scipy.stats import wilcoxon
import math
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


# ### Using Scipy Library to calculate Wilcoxon


if __name__ == "__main__":

    df = pd.read_csv('../output_csv/cls_statistic.csv')
    model_list = ["AB_std", "ET_std", "RF_std", "GB_std", "DT_std", "SVM_std"]
    plt.figure(figsize=(15, 10))
    sns.violinplot(data=df[model_list], orient="v")
    plt.ylabel("Standard Deviations")
    plt.xlabel("Classifiers")
    plt.show()
