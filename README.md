{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# HyperParameterTunability"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following repository contains \n",
    "* the scripts for collecting performance data of 6 \n",
    "machine learning algorithms on 200 classification tasks from OpenML environment\n",
    "* the collected performance data of SVM, Decision Tree, Random Forest, AdaBoost, Gradient Boosting and Extra Trees Classifiers.\n",
    "* tools for \n",
    "    - importing and modifying the collected data\n",
    "    - searching correlation between the dataset metafeatures and classifier performances\n",
    "    - conducting statistical tests to compare performance of the classifiers over the tasks \n",
    "* script for extracting metafeatures of the datasets\n",
    "* script for performing fANOVA on the performance data\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### To start collecting data for a given classifier over all datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from DataCollection.functions import *\n",
    "\n",
    "path_to_datasets = 'Datasets/'\n",
    "classification_per_algorithm(path=path_to_datasets, algorithm='DecisionTree')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Conduct fANOVA on the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from fANOVA.fanova_functions import *\n",
    "do_fanova(dataset_name='PerformanceData/AB_results_total.csv', algorithm='AdaBoost')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Extract Metafeatures"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from MetafeatureExtraction.metafeatures import *\n",
    "extract_for_all(path_to_datasets)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Create the Database object to import the collected data in desired formats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from Tools.database import Database\n",
    "db = Database()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>dataset</th>\n",
       "      <th>AB</th>\n",
       "      <th>ET</th>\n",
       "      <th>RF</th>\n",
       "      <th>DT</th>\n",
       "      <th>GB</th>\n",
       "      <th>SVM</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AP_Breast_Omentum.csv</td>\n",
       "      <td>0.981060</td>\n",
       "      <td>0.976235</td>\n",
       "      <td>0.976462</td>\n",
       "      <td>0.973912</td>\n",
       "      <td>0.983555</td>\n",
       "      <td>0.914538</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>AP_Breast_Prostate.csv</td>\n",
       "      <td>0.995238</td>\n",
       "      <td>0.995238</td>\n",
       "      <td>0.995238</td>\n",
       "      <td>0.995238</td>\n",
       "      <td>0.995238</td>\n",
       "      <td>0.961498</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>AP_Endometrium_Lung.csv</td>\n",
       "      <td>0.968363</td>\n",
       "      <td>0.958392</td>\n",
       "      <td>0.957018</td>\n",
       "      <td>0.929240</td>\n",
       "      <td>0.968363</td>\n",
       "      <td>0.894591</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>AP_Endometrium_Prostate.csv</td>\n",
       "      <td>0.992857</td>\n",
       "      <td>0.992857</td>\n",
       "      <td>0.992857</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.984615</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>AP_Endometrium_Uterus.csv</td>\n",
       "      <td>0.854854</td>\n",
       "      <td>0.837953</td>\n",
       "      <td>0.859561</td>\n",
       "      <td>0.827924</td>\n",
       "      <td>0.860409</td>\n",
       "      <td>0.758801</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       dataset        AB        ET        RF        DT  \\\n",
       "0        AP_Breast_Omentum.csv  0.981060  0.976235  0.976462  0.973912   \n",
       "1       AP_Breast_Prostate.csv  0.995238  0.995238  0.995238  0.995238   \n",
       "2      AP_Endometrium_Lung.csv  0.968363  0.958392  0.957018  0.929240   \n",
       "3  AP_Endometrium_Prostate.csv  0.992857  0.992857  0.992857  1.000000   \n",
       "4    AP_Endometrium_Uterus.csv  0.854854  0.837953  0.859561  0.827924   \n",
       "\n",
       "         GB       SVM  \n",
       "0  0.983555  0.914538  \n",
       "1  0.995238  0.961498  \n",
       "2  0.968363  0.894591  \n",
       "3  1.000000  0.984615  \n",
       "4  0.860409  0.758801  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "per_dataset_acc = db.get_per_dataset_accuracies()\n",
    "per_dataset_acc.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "metafeatures = db.get_metafeatures()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Conduct statistical tests (Friedman and Nemenyi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FriedmanchisquareResult(statistic=335.3428571428567, pvalue=2.5012413336096264e-70)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from Tools.stat_tests import *\n",
    "ranked_datasets = db.get_ranked_datasets()\n",
    "do_friedman_test(ranked_datasets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x140.4 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AB</th>\n",
       "      <th>DT</th>\n",
       "      <th>ET</th>\n",
       "      <th>GB</th>\n",
       "      <th>RF</th>\n",
       "      <th>SVM</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>AB</th>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.001</td>\n",
       "      <td>0.173720</td>\n",
       "      <td>0.900000</td>\n",
       "      <td>0.011845</td>\n",
       "      <td>0.001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>DT</th>\n",
       "      <td>0.001000</td>\n",
       "      <td>-1.000</td>\n",
       "      <td>0.001000</td>\n",
       "      <td>0.001000</td>\n",
       "      <td>0.001000</td>\n",
       "      <td>0.900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ET</th>\n",
       "      <td>0.173720</td>\n",
       "      <td>0.001</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.105836</td>\n",
       "      <td>0.900000</td>\n",
       "      <td>0.001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>GB</th>\n",
       "      <td>0.900000</td>\n",
       "      <td>0.001</td>\n",
       "      <td>0.105836</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.005603</td>\n",
       "      <td>0.001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RF</th>\n",
       "      <td>0.011845</td>\n",
       "      <td>0.001</td>\n",
       "      <td>0.900000</td>\n",
       "      <td>0.005603</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SVM</th>\n",
       "      <td>0.001000</td>\n",
       "      <td>0.900</td>\n",
       "      <td>0.001000</td>\n",
       "      <td>0.001000</td>\n",
       "      <td>0.001000</td>\n",
       "      <td>-1.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           AB     DT        ET        GB        RF    SVM\n",
       "AB  -1.000000  0.001  0.173720  0.900000  0.011845  0.001\n",
       "DT   0.001000 -1.000  0.001000  0.001000  0.001000  0.900\n",
       "ET   0.173720  0.001 -1.000000  0.105836  0.900000  0.001\n",
       "GB   0.900000  0.001  0.105836 -1.000000  0.005603  0.001\n",
       "RF   0.011845  0.001  0.900000  0.005603 -1.000000  0.001\n",
       "SVM  0.001000  0.900  0.001000  0.001000  0.001000 -1.000"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "do_nemenyi_test(ranked_datasets, plot=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
