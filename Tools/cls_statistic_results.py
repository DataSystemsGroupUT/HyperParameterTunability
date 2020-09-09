import pandas as pd
import pickle
import scipy.stats
import seaborn as sns
import math
import warnings
from Tools.helper import determine_relevant
from Tools.helper import marginal_plots
from Tools.helper import obtain_marginal_contributions
from Tools.helper import cls_kde_plot


# 1.create a csv file for our analysis
# dataset_names=list()
# with  open("dataset_names.txt")as f:
#     for line in f:
#         dataset_names.append(line)

# df=pd.DataFrame(dataset_names,columns=["datasets_name"])
# df.to_csv("data_analysis.csv")


# 2. read CSV file for 200 datasets
df_data_analysis=pd.DataFrame()
dict_cls=dict()


for cls in ["RF","AB","SVM","DT","GB","ET"]:
    
    file_path="../performance_data/"+cls+"_results_total.csv"
    cls_results_total=pd.read_csv(file_path)
    
    dict_mean=dict()
    dict_max=dict()
    dict_min=dict()
    dict_std=dict()
    
    for dataset in cls_results_total["dataset"].unique():
        current_dataset_results=cls_results_total[cls_results_total["dataset"]==dataset]
        
        dict_mean[dataset]=current_dataset_results["CV_auc"].mean()
        dict_max[dataset]=current_dataset_results["CV_auc"].max()
        dict_min[dataset]=current_dataset_results["CV_auc"].min()
        dict_std[dataset]=current_dataset_results["CV_auc"].std()
        
        
    df1=pd.DataFrame(dict_mean.items(),columns=["datasets",cls+"_mean"])
    df2=pd.DataFrame(dict_max.items(),columns=["datasets",cls+"_max"])
    df3=pd.DataFrame(dict_min.items(),columns=["datasets",cls+"_min"])
    df4=pd.DataFrame(dict_std.items(),columns=["datasets",cls+"_std"])
    dfs=pd.merge(pd.merge(pd.merge(df1,df2,on='datasets'),df3,on='datasets'),df4, on="datasets")
    dict_cls[cls]=dfs
    
    
dfs=pd.merge(pd.merge(pd.merge(
    pd.merge(
        pd.merge(
            dict_cls["RF"],dict_cls["GB"],on='datasets')
        ,dict_cls["DT"],on='datasets')
    ,dict_cls["SVM"], on="datasets"),dict_cls["ET"], on="datasets"),dict_cls["AB"], on="datasets")
dfs.to_csv("cls_statistic.csv")