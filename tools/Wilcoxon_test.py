from scipy.stats import wilcoxon
import math
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ### Using Scipy Library to calculate Wilcoxon
def willcoxon_for_two_model(data1, data2):
    '''
    this function return willcoxon test for two models.
    '''

    stat, p = wilcoxon(data1, data2)
    #print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha or math.isnan(p):
        #print('Same distribution (fail to reject H0)==>>0')
        return 0
    else:
        #print('Different distribution (reject H0)==>> 1')
        return 1


def willcoxon_all(df):
    '''
    this Fuction compute a 6*6 matrix with Wilcoxon signed-rank test for our models.

    '''

    df_out = pd.DataFrame(index=model_list, columns=model_list)
    df_out = df_out.fillna(0)

    for model1 in model_list:
        for model2 in model_list:
            if model1 == model2:
                df_out[model1][model2] = 0
            else:
                df_out[model1][model2] = willcoxon_for_two_model(
                    df[model1], df[model2])

    return df_out


if __name__ == "__main__":

    df = pd.read_csv('cls_statistic.csv')
    model_list = ["AB_max", "ET_max", "RF_max", "GB_max", "DT_max", "SVM_max"]

    print('There is no significant difference (fail to reject H0)  ==>> 0')
    print('There is  significant difference (reject H0)     ==>> 1')

    print(willcoxon_all(df[model_list]))
