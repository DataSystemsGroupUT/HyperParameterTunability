import collections
import copy
import json
import matplotlib.pyplot as plt
from statistics import median
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats
import seaborn as sns
import warnings

"""
Credit to https://github.com/janvanrijn/openml-pimp
"""



def rank_dict(dictionary, reverse=False):
    '''
    Get a dictionary and return a rank dictionary
    for example dic={'a':10,'b':2,'c':6}
    will return dic={'a':1.0,'b':3.0,'c':2.0}
    
    '''
    dictionary = copy.copy(dictionary)
    
    if reverse:
        
        for key in dictionary.keys():
            dictionary[key] = 1 - dictionary[key]
                    
    sortdict = collections.OrderedDict(sorted(dictionary.items()))
    ranks = scipy.stats.rankdata(list(sortdict.values()))
    result = {}
    
    for idx, (key, value) in enumerate(sortdict.items()):
        result[key] = ranks[idx]
        
    return result


def sum_dict_values(a, b, allow_subsets=False):
    '''
    Get two dictionary sum them together!
    '''
    result = {}
    a_total = sum(a.values())
    b_total = sum(b.values())
    a_min_b = set(a.keys()) - set(b.keys())
    b_min_a = set(b.keys()) - set(a.keys())
    
    #     if len(b_min_a) > 0:
    #         raise ValueError('dict b got illegal keys: %s' %str(b_min_a))
            
    #     if not allow_subsets and len(a_min_b):
    #         raise ValueError('keys not the same')
        
    for idx in a.keys():
        if idx in b:
            result[idx] = a[idx] + b[idx]
        else:
            result[idx] = a[idx]
                
    #     if sum(result.values()) != a_total + b_total:
    #         raise ValueError()
        
    return result

def obtain_marginal_contributions(df):

    '''
    This is the main function that calls Top functions
    '''
    
    all_ranks = dict()
    all_tasks = list()
    total_ranks = None
    num_tasks = 0
    marginal_contribution = collections.defaultdict(list)

    lst_datasets=list(df.dataset.unique())

    for dataset in lst_datasets:


        a=df[df.dataset==dataset]
        a=a.drop("dataset",axis=1)
        param=dict()


        for index, row in a.iterrows():
            marginal_contribution[row["param"]].append(row["importance"])
            param.update( {row["param"] : row["importance"]} )

        ranks = rank_dict(param, reverse=True)
        if total_ranks is None:
            total_ranks = ranks
        else:
            total_ranks = sum_dict_values( ranks,total_ranks, allow_subsets=False)
            num_tasks += 1
    total_ranks = divide_dict_values(total_ranks, num_tasks)
    return total_ranks, marginal_contribution, lst_datasets

def marginal_plots(sorted_values, keys, fig_title):

    sorted_values=sorted_values[0:8]
    keys=keys[0:8]
    plt.figure(figsize=(12,10))
    plt.violinplot(list(sorted_values), list(range(len(sorted_values))), showmeans=True )
    plt.plot([-0.5, len(sorted_values) - 0.5], [0, 0], 'k-', linestyle='--', lw=1)
    keys = [format_name(key) for key in keys]
    plt.xticks(list(range(len(sorted_values))), list(keys), rotation=45, ha='right')
    plt.ylabel('marginal contribution')
    plt.title(fig_title)
    plt.show()
    plt.savefig("output_plots/"+fig_title+".jpg" ,bbox_inches = 'tight',pad_inches = 0)
    plt.close()

def format_name(name):
    '''
    Format hyperparameter names!
    '''
    mapping_plain = {
        'strategy': 'imputation',
        'max_features': 'max. features',
        'min_samples_leaf': 'min. samples leaf',
        'min_samples_split': 'min. samples split',
        'criterion': 'split criterion',
        'learning_rate': 'learning rate',
        'max_depth': 'max. depth',
        'n_estimators': 'iterations',
        'algorithm': 'algorithm',
    }
    
    mapping_short = {
        'strategy': 'imputation',
        'max_features': 'max. feat.',
        'min_samples_leaf': 'samples leaf',
        'min_samples_split': 'samples split',
        'criterion': 'split criterion',
        'learning_rate': 'learning r.',
        'max_depth': 'max. depth',
        'n_estimators': 'iterations',
        'algorithm': 'algo.',
    }

    parts = name.split('__')
    
    for idx, part in enumerate(parts):
        if part in mapping_plain:
            if len(parts) < 3:
                parts[idx] = mapping_plain[part]
            else:
                parts[idx] = mapping_short[part]

                
    return ' / '.join(parts)


def divide_dict_values( d, denominator):
    ''' 
    divide d/demoniator
    '''
    result = {}
    
    for idx in d.keys():
        result[idx] = d[idx] / denominator
        
    return result

def determine_relevant( data, max_items=None, max_interactions=None):




    sorted_values = []
    keys=[]
    interactions_seen = 0


    for key in sorted(data, key=lambda k: median(data[k]), reverse=True):
        if '__' in key:
            interactions_seen += 1
            if interactions_seen > max_interactions:
                continue

        sorted_values.append(data[key])
        keys.append(key)


    if max_items is not None:
        sorted_values = sorted_values[:max_items]
        keys = keys[:max_items]

    return sorted_values, keys


def cls_kde_plot(file_path,cls,important_hyperparameter,x1,x2,y1,y2,b=0,kernel=None,scale=None):
    
    
    #file_path="../PerformanceData/total/AB_results_total.csv"
    df= pd.read_csv(file_path)
    df_total=pd.DataFrame()


    for item in df.dataset.unique():


        df_dataset=df.loc[df['dataset'] == item ]
        #max_auc=max(df_dataset["CV_auc"])
        df_row=df_dataset.loc[df_dataset['CV_auc'] == max(df_dataset["CV_auc"]) ]
        df_total = df_total.append(df_row)
    
    if kernel!= None:
        df_total=df_total[df_total[important_hyperparameter]==kernel]
        important_hyperparameter="gamma"
        
    plt.figure(figsize=(7,9))
    
    #set bandwidth for kde
    if b!= 0:
        sns.kdeplot(df_total[important_hyperparameter],bw=b)
    else:
        sns.kdeplot(df_total[important_hyperparameter])
        

    if kernel != None:
        plt_title=cls+"-"+kernel+":"+important_hyperparameter    
        plt.title(plt_title )
    else: 
        plt_title=cls+":"+important_hyperparameter
        plt.title(plt_title )


    plt.xlim(x1,x2)
    plt.ylim(y1,y2)
    if scale!=None:
        plt.xscale(scale)
    plt.savefig("../output_plots/"+plt_title+".jpg" ,bbox_inches = 'tight',pad_inches = 0)
    plt.close()