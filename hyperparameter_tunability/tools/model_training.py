import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_rand
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score,\
    recall_score, precision_score, roc_auc_score, mean_squared_error, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class Model(object):
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.label_encoder = params['label_encoder']
        self.X, self.y = self.preprocessing()
        self.cv_splits = params['cv_splits']
        self.nr_iterations = params['nr_iterations']
        self.classifier = params['classifier']
        self.score = params['evaluation_score']
        self.best_model = None
        self.best_model_with_pca = None
        self.best_score = None
        self.verbose = params['verbose']

        if params['do_pca']:
            self.X_pca = self.do_pca(self.X)
        else:
            self.X_pca = None

    def preprocessing(self):
        dataset = self.dataset
        X = pd.get_dummies(dataset.iloc[:, :-1])
        y = dataset.iloc[:, -1]
        if self.label_encoder:
            le = LabelEncoder()
            y = le.fit_transform(y.copy())
        return (X, y)

    def do_random_search(self, selected_cols=None, return_score=False):
        classifier_name = self.classifier
        model = self.get_model(classifier_name)
        model_parameters = self.get_params(classifier_name)

        if selected_cols is not None:
            dataset = self.dataset
            X = pd.get_dummies(dataset)
            self.X = X.loc[:, selected_cols]
        else:
            X = self.X

        cv = StratifiedKFold(n_splits=self.cv_splits)
        random_search = RandomizedSearchCV(estimator=model,
                                           param_distributions=model_parameters,
                                           n_iter=self.nr_iterations,
                                           cv=cv, iid=True,
                                           scoring=self.score)
        start = time()
        random_search.fit(self.X, self.y)
        end = time()
        time_taken_without_pca = end - start
        best_idx = np.flatnonzero(random_search.cv_results_[
                                  'rank_test_score'] == 1)[0]
        self.best_score = random_search.cv_results_[
            'mean_test_score'][best_idx]
        self.best_model = random_search.best_estimator_

        if self.verbose:
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % (time_taken_without_pca, self.nr_iterations))
            self.report(random_search.cv_results_, 1)

        if self.X_pca is not None:
            start = time()
            random_search.fit(self.X_pca, self.y)
            end = time()
            time_taken_with_pca = end - start
            self.best_model_with_pca = random_search.best_estimator_

            if self.verbose:
                print("RandomizedSearchCV took %.2f seconds for %d candidates"
                      " parameter settings." % (time_taken_with_pca, self.nr_iterations))
                self.report(random_search.cv_results_, 1)

        if return_score:
            return self.best_score

    def plot_tree(self, save_plot=False):
        # for Decision Tree only
        model = self.best_model
        plot_tree(model, feature_names=self.X.columns)
        if save_plot:
            plt.savefig('tree.png')
        plt.show()

    @staticmethod
    def feature_importance(features, importances, pca=False, threshold=0, top=None):
        if top is None:
            top = len(features)

        if pca:
            features = ['PC{}'.format(i+1) for i in range(len(importances))]
            out_df = pd.DataFrame({'features': features,
                                   'importance': importances})
        else:
            out_df = pd.DataFrame({'features': features,
                                   'importance': importances})
        out_df = out_df.loc[out_df.importance > threshold, :]
        return out_df.sort_values('importance',
                                  ascending=False).iloc[:top, :]

    @staticmethod
    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    @staticmethod
    def get_model(classifier_name):
        model_space = {'DT': DecisionTreeClassifier(),
                       'RF': RandomForestClassifier(),
                       'AB': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                       'GB': GradientBoostingClassifier(),
                       'LR': LogisticRegression()}
        return model_space[classifier_name]

    @staticmethod
    def get_params(classifier_name):
        params_dt = {'max_features': sp_rand(0.1, 0.9),
                     'min_samples_leaf': sp_randint(1, 21),
                     'min_samples_split': sp_randint(2, 21),
                     'criterion': ['entropy', 'gini']}

        params_random_forest = {'n_estimators': sp_randint(1, 100),
                                'bootstrap': [True, False],
                                'max_features': sp_rand(0.1, 0.9),
                                'min_samples_leaf': sp_randint(1, 21),
                                'min_samples_split': sp_randint(2, 21),
                                'criterion': ['entropy', 'gini']}

        params_adaboost = {'base_estimator__max_depth': sp_randint(1, 11),
                           'algorithm': ['SAMME', 'SAMME.R'],
                           'n_estimators': sp_randint(50, 501),
                           'learning_rate': sp_rand(0.01, 2)}

        params_gboosting = {'learning_rate': sp_rand(0.01, 1),
                            'criterion': ['friedman_mse', 'mse'],
                            'n_estimators': sp_randint(50, 501),
                            'max_depth': sp_randint(1, 11),
                            'min_samples_split': sp_randint(2, 21),
                            'min_samples_leaf': sp_randint(1, 21),
                            'max_features': sp_rand(0.1, 0.9)}

        params_log_regression = {'penalty': ['l2', 'none'],
                                 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                                 'C': sp_rand(2 ** (-5), 2 ** 15)}

        parameter_space = {'DT': params_dt, 'RF': params_random_forest,
                           'AB': params_adaboost, 'GB': params_gboosting,
                           'LR': params_log_regression}
        return parameter_space[classifier_name]

    @staticmethod
    def do_pca(data, var_threshold=0.8):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        nr_components = 1
        pca = PCA(n_components=nr_components)
        data_pca = pca.fit_transform(data)
        less_var = sum(pca.explained_variance_ratio_) < var_threshold

        while less_var:
            nr_components += 1
            pca = PCA(n_components=nr_components)
            data_pca = pca.fit_transform(data)
            less_var = sum(pca.explained_variance_ratio_) < var_threshold
        return data_pca
