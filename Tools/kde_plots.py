# imports
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
from tools.helper import determine_relevant
from tools.helper import marginal_plots
from tools.helper import obtain_marginal_contributions
from tools.helper import cls_kde_plot


if __name__ == "__main__":
    # Create KDE Plots

    #Adaboost
    cls_kde_plot("performance_data/AB_results_total.csv","adaboost","max_depth",1,10,0.07,0.11)

    #Decision Tree
    cls_kde_plot("performance_data/DT_results_total.csv","Decision Tree","max_features",0,1,0,3.5)

    #Random Forest
    cls_kde_plot("performance_data/RF_results_total.csv","Random Forest","min_samples_leaf",0,20,0.03,0.08,b=0.65)


    #Extra Tree
    cls_kde_plot("performance_data/ET_results_total.csv","Extra Tree","min_samples_leaf",1,20,0.01,0.08)

    # Gradient Boosting
    cls_kde_plot("performance_data/GB_results_total.csv","Gradient Boosting","learning_rate",0,1,0.3,1.3)

    # SVM_ sigmoid
    cls_kde_plot("performance_data/SVM_results_total.csv","SVM","kernel",0.014,10,0,1.22,kernel="sigmoid",scale="log")


    # SVM_RBF
    cls_kde_plot("performance_data/SVM_results_total.csv","SVM","kernel",0.01,10,0,0.14,kernel="rbf",scale="log")