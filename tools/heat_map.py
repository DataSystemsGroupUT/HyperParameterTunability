import scipy.stats as ss
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


# 1. read data
df = pd.read_csv('../output_csv/cls_statistic.csv')
model_list = ["AB_max", "ET_max", "RF_max", "GB_max", "DT_max", "SVM_max"]
# plt.figure(figsize=(15,10))
# sns.violinplot(data=df[model_list],orient="v")
# plt.ylabel("Standard Deviations")
# plt.xlabel("Classifiers");
df_max = df[model_list]


# 2. calculate rank data

df_copy = df_max.copy()
for index, row in df_max.iterrows():
    df_copy.loc[index] = len(row) - (ss.rankdata(row)).round()+1


# 3. Create new dataframe for output
df_rank = pd.DataFrame(columns=["AB_max", "ET_max", "RF_max",
                       "GB_max", "DT_max", "SVM_max"], index=[1, 2, 3, 4, 5, 6])
for index, row in df_rank.iterrows():
    for column in df_rank.columns:
        out = df_copy[df_copy[column] == index].count()
        df_rank.loc[index, column] = out[0]


# 4. show the plot
plt.figure(figsize=(15, 10))
ax = sns.heatmap(df_rank, annot=True, fmt="d")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
