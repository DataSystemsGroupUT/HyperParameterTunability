from hyperopt import hp, tpe, fmin, Trials, rand
from sklearn.model_selection import train_test_split
import numpy as np
import json
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

class HPIVerification_Randomized:
    """Base class for hyperparameter importance verfication for clustering."""
    def __init__(self, n_iter_opt, X, y, algorithm):
        self.n_iter_opt = n_iter_opt
        self.X = X
        self.y = y
        self.algorithm = algorithm
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

    N = 1


    # create hyperparameter search space for each hyperparameter.
    params_random_forest = {
                        'min_samples_leaf': np.random.choice(np.arange(1, 2, 1), N),
                        'bootstrap': np.random.choice([ False], N),
                        'min_samples_split': np.random.choice(np.arange(8, 9, 1), N),
                        'criterion': np.random.choice(['entropy', ], N),
                        'max_features': np.random.uniform(0.8, 0.9, N),
                        }

    parameters = {
                    # 'AdaBoost': params_adaboost,
                    'random_forest': params_random_forest,
                    # 'SVM': params_svm,
                    # 'ExtraTrees': params_random_forest,
                    # 'GradientBoosting': params_gboosting,
                    # 'DecisionTree': params_decision_tree
                    }

    randomized_params_random_forest = {
                    "min_samples_leaf": np.arange(1, 21, 1),
                    "bootstrap":  [True, False],
                    "min_samples_split":  np.arange(2, 21, 1),
                    "criterion":  ['entropy', 'gini'],
                    'max_features': stats.uniform(0.1, 0.9),
                }

    randomized_parameters = {
                    'random_forest': randomized_params_random_forest,
                }

    def y_preprocessing(self,):
        le = LabelEncoder()
        self.y_train = le.fit_transform(self.y_train)
        num_labels = len(np.unique(self.y_train))
        y_sparse = None

        if num_labels > 2:
            multilabel = True
            lb = LabelBinarizer()
            lb.fit(self.y_train)
            y_sparse = lb.transform(self.y_train)
        else:
            multilabel = False

        return multilabel, y_sparse, self.y_train

    def x_preprocessing(self,):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        
        return self.X_train, self.X_test

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
        if hpi in {"criterion", "bootstrap"}:
            k_run = self.randomized_params_random_forest[hpi]
        return k_run

    def fixed_hpi_estimator(self, fixed_param,):
        # define the ML models
        models = {
                # 'AdaBoost': AdaBoostClassifier(base_estimator =
                # DecisionTreeClassifier()),
                'random_forest': RandomForestClassifier(**fixed_param),
                # 'SVM': SVC(),
                # 'ExtraTrees': ExtraTreesClassifier(),
                # 'GradientBoosting': GradientBoostingClassifier(),
                # 'DecisionTree': DecisionTreeClassifier()
                }
        return models[self.algorithm]

    def randomized_opt_run(self):
        """Returns the resluts of applying randomized by fixing
        at the time one hyperparameter and tuning across other hyperparameters.

        Returns:
            hp_trials_dict(dict): A dictionary which
            keeps all the result for K diffrent configuration
            for fixed hyperparamter and n different itereation
            for optimization.
        """
        hp_trials_dict = {}

        hpi_dict = self.parameters[self.algorithm]
        multilabel, y_sparse, self.y_train = self.y_preprocessing()
        for hpi in hpi_dict.keys():
            SEARCH_PARAMS = self.randomized_parameters[self.algorithm]
            SEARCH_PARAMS_dict = SEARCH_PARAMS.copy()
            SEARCH_PARAMS_dict.pop(hpi)
            k_run = self.hp_search(hpi)

            trials_lst = []
            
            for i in range(0, len(k_run)):

                FIXED_PARAMS = {
                    hpi: k_run[i],
                }
                # **FIXED_PARAMS
                params = {**SEARCH_PARAMS_dict}
                est = self.fixed_hpi_estimator(FIXED_PARAMS)
                clf = RandomizedSearchCV(est, param_distributions=params, scoring="roc_auc", n_jobs=-1, cv=5, n_iter=self.n_iter_opt, error_score="raise")

                self.X_train, self.X_test = self.x_preprocessing()
                search = clf.fit(self.X_train, self.y_train)
                trials_lst.append(search.cv_results_['mean_test_score'])


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
            if hpi in {"criterion", "bootstrap"}:
                _N = len(self.randomized_params_random_forest[hpi])
                lst_hpi_losses = hp_trials_lst[hpi]
                losses[hpi] = [[lst_hpi_losses[y][x] for x in range(self.n_iter_opt)] for y in range(_N)]
                avg_losses[hpi] = [sum(x)/len(losses[hpi]) for x in zip(*losses[hpi])]
            else:
                lst_hpi_losses = hp_trials_lst[hpi]
                losses[hpi] = [[lst_hpi_losses[y][x] for x in range(self.n_iter_opt)] for y in range(self.N)]
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
