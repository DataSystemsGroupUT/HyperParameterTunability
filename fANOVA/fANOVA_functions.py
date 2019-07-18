import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import fanova
import fanova.visualizer as viz

from sklearn.preprocessing import LabelEncoder

def do_fANOVA(csv_name, algorithm, st = 0, end = 200):
    '''
    Derive importance of hyperparameter combinations
    on the performance data for the given algorithm
    
    Input:
           csv_name - (DataFrame) contains the performance data                       
           algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
           st - (int) starts from the specified number of dataset
           end - (int) ends at the specified number of dataset
    Output:
           writes the results on a csv file
    
    '''
    
    data = pd.read_csv(csv_name)
    
    cs1, cs2 = config_space[algorithm]
        
    cols = col_names[algorithm]

    data = data.iloc[:,cols]

    data.imputation.fillna('none',inplace=True)
    
    data = label_encoding(data, algorithm)
    
    datasets = data.dataset.unique()[st:end]

    results = pd.DataFrame()
    
    
    for indx, d_name in enumerate(datasets):
        print('Dataset {}({})/{}'.format(indx + 1, d_name, len(datasets)))
        selected = data.dataset == d_name
        data_filter = data.loc[selected,:]
        
        missing_values = sum(data_filter.imputation == 3) == 0
        
        try:
            df, time_taken = fanova_to_df(data_filter, algorithm, 
                                          missing_values, cs1, cs2)
            
            df['dataset'] = d_name
            df['imputation'] = missing_values
            df['time_taken'] = time_taken

            results = pd.concat([results, df],axis=0)
            results.to_csv('{}_fANOVA_results.csv'.format(algorithm),
                           header=True,
                           index=False)
        except Exception as e:
            print('************************************ \n \
            The following error occured for {} dataset:\n {} \n \
            *************************************'.format(d_name, e))


def fanova_to_df(data, algorithm, missing_values, cs1, cs2):
    '''
    Derive importance of hyperparameter combinations
    for the given algorithm
    
    Input:
           data - (DataFrame) contains the performance data 
                  for a dataset
           algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
           missing_values - (boolean) whether imputation has 
                            been done on the dataset
           cs1, cs2 - configuration space objects
    Output:
           df - (DataFrame) contains the variance contributions
                per hyperparameter combination
           time_taken - performance time in sec
    
    '''
    
    
    if missing_values:
        X = data.loc[:,sorted(data.columns[1:-1])].values
        y = data.iloc[:, -1].values
        cs = cs1
    else:
        X = data.loc[:,sorted(data.columns[1:-2])].values
        y = data.iloc[:, -1].values
        cs = cs2      
          
        
    f = fanova.fANOVA(X, y,
                      n_trees=32,
                      bootstrapping=True,
                      config_space = cs)
    
    
    start = time.perf_counter()
    print('Singles')
    imp1 = get_single_importance(f)
    print('Pairs')
    imp2 = f.get_most_important_pairwise_marginals()
    print('Triples')
    if missing_values:
        imp3_1 = get_triple_importance(f, algorithm)
        imp3_2 = get_triple_impute(f, algorithm)
        imp3 = dict_merge(imp3_1, imp3_2)
    else:
        imp3 = get_triple_importance(f, algorithm)

    imp = dict_merge(imp1, imp2, imp3)
    end = time.perf_counter()

    time_taken = end - start
    print('time taken is {} min'.format(time_taken / 60))
    
    df = pd.DataFrame({'param':list(imp.keys()),
                       'importance': list(imp.values())},
                      index=None)
    return df, time_taken   


def cs_RF():
    '''
    Defining the configuration space in case of 
    Random Forest and Extra Trees Classifiers
    
    '''
    cs1 = CS.ConfigurationSpace()
    cs2 = CS.ConfigurationSpace()

    hp1 = CSH.CategoricalHyperparameter('bootstrap',
                                      choices=['0', '1'])
    hp2 = CSH.CategoricalHyperparameter('criterion',
                                      choices=['0', '1'])
    hp3 = CSH.CategoricalHyperparameter('imputation',
                                      choices=['0', '1', '2'])


    hp4 = CSH.UniformFloatHyperparameter('max_features', lower=0.1,
                                       upper=0.9, log=False)
    hp5 = CSH.UniformIntegerHyperparameter('min_samples_leaf', lower=1,
                                         upper=20, log=False)
    hp6 = CSH.UniformIntegerHyperparameter('min_samples_split', lower=2,
                                         upper=20, log=False)


    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, hp5, hp6])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp2, hp4, hp5, hp6])
    
    return cs1, cs2

def cs_AB():
    '''
    Defining the configuration space in case of 
    AdaBoost Classifier
    
    '''
    cs1 = CS.ConfigurationSpace()
    cs2 = CS.ConfigurationSpace()

    hp1 = CSH.CategoricalHyperparameter('algorithm', choices=['0', '1'])
    hp2 = CSH.CategoricalHyperparameter('imputation', choices=['0', '1', '2'])


    hp3 = CSH.UniformIntegerHyperparameter('max_depth', lower=1, upper=10, log=False)
    hp4 = CSH.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=2, log=True)
    hp5 = CSH.UniformIntegerHyperparameter('n_estimators', lower=50, upper=500, log=False)


    # imputation case
    cs1.add_hyperparameters([hp1, hp2, hp3, hp4, hp5])

    # no imputation case
    cs2.add_hyperparameters([hp1, hp3, hp4, hp5])
    
    return cs1, cs2

config_space = {'RandomForest': cs_RF(),
               'AdaBoost': cs_AB(),
               'ExtraTrees': cs_RF()}

RF_cols = ["dataset", "bootstrap", "criterion",         
          "max_features", "min_samples_leaf", "min_samples_split", 
          "n_estimators", "imputation", 'CV_accuracy']

AB_cols = ['dataset', 'algorithm', 'max_depth',
           'learning_rate', 'n_estimators', 'imputation',
           'CV_accuracy']

col_names = {'RandomForest': RF_cols,
               'AdaBoost': AB_cols,
               'ExtraTrees': RF_cols}
    
def label_encoding(data, algorithm):
    '''
    Performing label encoding for the categorical hyperparameters
    of the given algorithm
    
    Input:
           data - (DataFrame) contains the performance data
           algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
    Output:
           data - (DataFrame) contains only numerical features
    
    '''
    le = LabelEncoder()
    
    if algorithm == 'RandomForest' or algorithm == 'ExtraTrees': 
        imputation = sorted(data.imputation.unique())
        bootstrap = sorted(data.bootstrap.unique())
        criterion = sorted(data.criterion.unique())
        print('{} - {}'.format(imputation, np.arange(len(imputation))))
        print('{} - {}'.format(bootstrap, np.arange(len(bootstrap))))
        print('{} - {} \n'.format(criterion, np.arange(len(criterion))))

        data.imputation = le.fit_transform(data.imputation)
        data.bootstrap = le.fit_transform(data.bootstrap)
        data.criterion = le.fit_transform(data.criterion)
    elif algorithm == 'AdaBoost':
        imputation = sorted(data.imputation.unique())
        ab_algorithm = sorted(data.algorithm.unique())
        print('{} - {}'.format(imputation, np.arange(len(imputation))))
        print('{} - {}'.format(ab_algorithm, np.arange(len(ab_algorithm))))

        data.imputation = le.fit_transform(data.imputation)
        data.algorithm = le.fit_transform(data.algorithm)
        
    
    return data


def get_single_importance(f):
    '''
    Derive importance of each hyperparameter
    
    Input:
           f - (fANOVA) object
    Output:
           imp1 - (dict) key: hyperparameter name
                         value: variance contribution
    
    '''
    names = f.cs.get_hyperparameter_names()
    
    imp1 = {}
    for name in names:
        imp1_ind = f.quantify_importance([name])
        value = imp1_ind[(name,)]['individual importance']
        imp = {name:value}
        imp1.update(imp)
        
    return imp1

def get_importance(f, *params):
    '''
    Derive importance of the specified
    combination of hyperparameters
    
    Input:
           f - (fANOVA) object
           *params - (str) names of hyperparameters
    Output:
           imp - (dict) key: hyperparameter combination
                         value: variance contribution
    
    '''
    imp = f.quantify_importance(list(params))
    value = imp[params]['individual importance']
    imp = {params:value}
    return imp

def dict_merge(*args):
    '''
    Merges several python dictionaries
    
    Input:
           *args - (dict) python dictionaries
    Output:
           imp - (dict) merged dictionary
    
    '''
    imp = {}
    for dictt in args:
        imp.update(dictt)
    return imp

def get_triple_importance(f, algorithm):
    '''
    Derive importance of specified triple combinations
    of hyperparameters per algorithm
    
    Input:
           f - (fANOVA) object
           algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
    Output:
           imp - (dict) key: hyperparameter name
                        value: variance contribution
    
    '''
    if algorithm == 'RandomForest' or algorithm == 'ExtraTrees': 
        imp1 = get_importance(f, 'bootstrap', 'max_features', 'min_samples_leaf')
        imp2 = get_importance(f, 'bootstrap', 'max_features', 'min_samples_split')
        imp = dict_merge(imp1, imp2)
    elif algorithm == 'AdaBoost':
        imp = get_importance(f, 'algorithm', 'max_depth', 'learning_rate')
            
    return imp

def get_triple_impute(f, algorithm):
    '''
    Derive importance of specified triple combinations
    of hyperparameters per algorithm in case of 
    data imputation
    
    Input:
           f - (fANOVA) object
           algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
    Output:
           imp - (dict) key: hyperparameter name
                        value: variance contribution
    
    '''
    if algorithm == 'RandomForest' or algorithm == 'ExtraTrees': 
        imp = get_importance(f, 'imputation', 'max_features', 'min_samples_leaf')
    elif algorithm == 'AdaBoost':
        imp = get_importance(f, 'imputation', 'max_depth', 'learning_rate')
    
    return imp



    

            
