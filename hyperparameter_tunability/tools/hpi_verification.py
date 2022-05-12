"""
    Hyperparameter importance verifcation with hyperopt.
    In this file, user can find all the materails,
    in order to verfiy the result of fANOVA.
    by fixing each hyperparameter once to
    see what would be the rank of each one.
"""
from hyperopt import hp, tpe, fmin, Trials, rand
from sklearn.model_selection import train_test_split
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler


class HPIVerification:
    """Base class for hyperparameter importance verfication for clustering."""
    def __init__(self, n_iter_opt, X, y, algorithm):
        self.n_iter_opt = n_iter_opt
        self.X = X
        self.y = y
        self.algorithm = algorithm
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

    N = 1
    # define the ML models
    models = {
                # 'AdaBoost': AdaBoostClassifier(base_estimator =
                # DecisionTreeClassifier()),
                'random_forest': RandomForestClassifier(),
                # 'SVM': SVC(),
                # 'ExtraTrees': ExtraTreesClassifier(),
                # 'GradientBoosting': GradientBoostingClassifier(),
                # 'DecisionTree': DecisionTreeClassifier()
                }

    # create hyperparameter search space for each hyperparameter.
    params_random_forest = {
                            'bootstrap': np.random.choice([True, False], N),
                            'max_features': np.random.uniform(0.1, 0.9, N),
                            'min_samples_leaf': np.random.choice(np.arange(1, 21, 1), N),
                            'min_samples_split': np.random.choice(np.arange(2, 21, 1), N),
                            'criterion': np.random.choice(['entropy', 'gini'], N)}

    parameters = {
                    # 'AdaBoost': params_adaboost,
                    'random_forest': params_random_forest,
                    # 'SVM': params_svm,
                    # 'ExtraTrees': params_random_forest,
                    # 'GradientBoosting': params_gboosting,
                    # 'DecisionTree': params_decision_tree
                    }

    hyperopt_params_random_forest = {
                    "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 21, 1)),
                    "bootstrap": hp.choice("bootstrap", [True, False]),
                    "min_samples_split": hp.choice("min_samples_split", np.arange(2, 21, 1)),
                    "criterion": hp.choice("criterion", ['entropy', 'gini']),
                    'max_features': hp.uniform("max_features", 10 ** (-1), 9*(10 ** (-1))),
                }

    hyperopt_parameters = {
                    'random_forest': hyperopt_params_random_forest,
                }

    def x_preprocessing(self,):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        return self.X_train, self.X_test

    def train_evaluate(self, params):
        model = self.models[self.algorithm]
        model.set_params(**params)
        # apply preprocessing on top of dataset.
        self.X_train, self.X_test = self.x_preprocessing()
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        score = roc_auc_score(self.y_test, predictions)
        return score

    # as most of the optimization frameworks, try to minimize the objective function, we multiple to -1.
    def objective(self, params):
        return -1.0 * self.train_evaluate(params)

    def hp_search(self, hpi):
        """Filter the search space based on hpi.

        Args:
            hpi (str): the hyperparmeter which we want to fix it during optimization.

        Returns:
            k_run(ndarray): The generated random samples.

        """
        k_run = self.parameters[self.algorithm][hpi]
        # if hpi in "init":
        #     k_run = np.random.choice(['k-means++', 'random'], self.N)
        #     k_run_str = [str(n) for n in k_run]
        #     return k_run_str

        return k_run

    def hyper_opt_run(self):
        """Returns the resluts of applying hyperopt by fixing
        at the time one hyperparameter and tuning across other hyperparameters.

        Returns:
            hp_trials_dict(dict): A dictionary which
            keeps all the result for K diffrent configuration
            for fixed hyperparamter and n different itereation
            for optimization.
        """
        hp_trials_dict = {}

        hpi_dict = self.parameters[self.algorithm]
        for hpi in hpi_dict.keys():
            SEARCH_PARAMS = self.hyperopt_parameters[self.algorithm]
            SEARCH_PARAMS_dict = SEARCH_PARAMS.copy()
            SEARCH_PARAMS_dict.pop(hpi)
            k_run = self.hp_search(hpi)

            trials_lst = []
            best_lst = []

            for i in range(0, self.N):

                FIXED_PARAMS = {
                    hpi: k_run[i],
                }
                # **FIXED_PARAMS
                params = {**SEARCH_PARAMS_dict, **FIXED_PARAMS}
                trials = Trials()
                best = fmin(fn=self.objective,
                            space=params,
                            algo=rand.suggest,
                            max_evals=self.n_iter_opt,
                            trials=trials)

                trials_lst.append(trials.results)
                best_lst.append(best)

            hp_trials_dict[hpi] = trials_lst

        return hp_trials_dict

    def hpi_avg_lossess_dataset(self, hp_trials_lst):
        """
        calculates average lossess based on K run and returns ranked hpi.

        Args:
            hp_trials_lst (dict): A dictionary which
            keeps all the result for K diffrent configuration for fixed hyperparamter and
            n different itereation for optimization.

        Returns:
            avg_losses(dict): return average lossess based on fixed each hyperparameter once per time.
        """
        losses = {}
        avg_losses = {}
        lst_hpi_losses = {}

        hpi_dict = self.parameters[self.algorithm]

        for hpi in hpi_dict.keys():
            lst_hpi_losses = hp_trials_lst[hpi]
            losses[hpi] = [[lst_hpi_losses[y][x]["loss"] for x in range(self.n_iter_opt)] for y in range(self.N)]
            avg_losses[hpi] = [sum(x)/len(losses[hpi]) for x in zip(*losses[hpi])]

        return avg_losses

    def hpi_avg_lossess(self, dicts, db_name):
        len_db = len(db_name)
        merged = dicts[db_name[0]]
        dicts.pop(db_name[0])
        db_name.pop(0)
        for d in db_name:
            for hpi in dicts[d]:
                merged[hpi] = [sum(k) for k in zip(dicts[d][hpi], merged[hpi])]

        for hpi in merged:
            merged[hpi] = [item / len_db for item in merged[hpi]]

        return merged

    def rank_hpi(self, avg_losses):
        """_summary_
        Args:
            avg_losses (dict): _description_

        Returns:
            ranked_hpi: _description_
        """
        ranked_hpis = []
        ranked_hpis = rankdata([avg_losses[key] for key in avg_losses.keys()], method="ordinal", axis=0)
        hpi_names = list(avg_losses.keys())
        with open('ranked_hpis.txt', 'w') as f:
            for item in ranked_hpis:
                f.write("%s\n" % item)

        return ranked_hpis, hpi_names

    def plot(self, ranked_hpis, hpi_names):
        """draw hyperparameters based on thier rank.

        Args:
            ranked_hpis (list): It contains rank of each hyperparameters.
            hpi_names (list): name of hyperparameters.
        Returns:
            show the plot
        """
        for ranked_hpi, hpi_name in zip(ranked_hpis, hpi_names):
            plt.plot(range(0, self.n_iter_opt), ranked_hpi, label=str(hpi_name))

        plt.legend()
        plt.show()
        plt.savefig('plotttttttt.png')
        return print("plot was drawn! ")
