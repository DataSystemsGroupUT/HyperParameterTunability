"""Run whole results here!"""

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
    # you can see in this loop how we recall functions over 200 datasets for 6 classifiers.
    for cls in {"RF", "SVM", "ET", "DT", "AB", "GB"}:
        df = pd.read_csv("../PerformanceData/"+cls+"_fANOVA_results.csv")
        total_ranks, marginal_contribution, _ = obtain_marginal_contributions(
            df)
        sorted_values, keys = determine_relevant(
            marginal_contribution, max_interactions=3)
        marginal_plots(sorted_values, keys, cls)
