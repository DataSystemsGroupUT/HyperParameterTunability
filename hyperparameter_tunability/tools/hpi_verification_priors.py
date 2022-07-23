"""
    Important Hyperparameter priors verifcation with randomsearch.
    In this file, user can find all the materails,
    in order to verfiy the result of best priors.
    by creating two search space on is uniformly distrbutaed and the another one is based on KDE.
"""
from hyperopt import hp, tpe, fmin, Trials, rand
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder


class HPIVerification_Priors:
    """Base class for best priors for the important hyperparameter. """
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
                'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                'random_forest': RandomForestClassifier(),
                'SVM': SVC(),
                'ExtraTrees': ExtraTreesClassifier(),
                'GradientBoosting': GradientBoostingClassifier(),
                'DecisionTree': DecisionTreeClassifier()
                }

    # create hyperparameter search space for each hyperparameter.
    modified_params_random_forest = {
                    "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 2, 1)),
                    "bootstrap": hp.choice("bootstrap", [True, False]),
                    "min_samples_split": hp.choice("min_samples_split", np.arange(2, 21, 1)),
                    "criterion": hp.choice("criterion", ['entropy', 'gini']),
                    'max_features': hp.uniform("max_features", 10 ** (-1), 9*(10 ** (-1))),
                    }
    uniform_params_random_forest = {
                    "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 21, 1)),
                    "bootstrap": hp.choice("bootstrap", [True, False]),
                    "min_samples_split": hp.choice("min_samples_split", np.arange(2, 21, 1)),
                    "criterion": hp.choice("criterion", ['entropy', 'gini']),
                    'max_features': hp.uniform("max_features", 10 ** (-1), 9*(10 ** (-1))),
                }
    modified_params_decision_tree = {'max_features': hp.uniform("max_features", 6.5*(10 ** (-1)), 8.1*(10 ** (-1))),
                            'min_samples_leaf': hp.choice("min_samples_leaf", np.arange(1, 21, 1)),
                            'min_samples_split': hp.choice("min_samples_split", np.arange(2, 21, 1)),
                            'criterion': hp.choice("criterion", ['entropy', 'gini'])
                            }
    uniform_params_decision_tree = {'max_features': hp.uniform("max_features", 10 ** (-1), 9*(10 ** (-1))),
                        'min_samples_leaf': hp.choice("min_samples_leaf", np.arange(1, 21, 1)),
                        'min_samples_split': hp.choice("min_samples_split", np.arange(2, 21, 1)),
                        'criterion': hp.choice("criterion", ['entropy', 'gini'])
                        }

    modified_params_adaboost = {'base_estimator__max_depth': hp.choice("base_estimator__max_depth", np.arange(6, 11, 1)),
                    'algorithm':  hp.choice("algorithm", ['SAMME', 'SAMME.R']),
                    # iterations in the paper
                    'n_estimators': hp.choice("n_estimators", np.arange(50, 100, 1)),
                    'learning_rate':  hp.uniform("learning_rate", 0.01, 2.0),
                    }
    
    uniform_params_adaboost = {'base_estimator__max_depth': hp.choice("base_estimator__max_depth", np.arange(1, 11, 1)),
                    'algorithm':  hp.choice("algorithm", ['SAMME', 'SAMME.R']),
                    # iterations in the paper
                    'n_estimators': hp.choice("n_estimators", np.arange(50, 100, 1)),
                    'learning_rate':  hp.uniform("learning_rate", 0.01, 2.0),
                    }

    modified_params_svm = {'kernel': hp.choice("kernel", ['rbf', 'sigmoid']),
                'C': hp.uniform("C", 2 ** (-5), 2 ** 15),
                'coef0': hp.uniform("coef0", -1, 1),
                'gamma': hp.uniform("gamma", 10 ** (-4), 10 ** (-1)), 
                'shrinking':  hp.choice("shrinking", [True, False]), 
                'tol': hp.uniform("tol",10 ** (-5), 10 ** (-1)),  
                }
    
    uniform_params_svm = {'kernel': hp.choice("kernel", ['rbf', 'sigmoid']),
                'C': hp.uniform("C", 2 ** (-5), 2 ** 15),
                'coef0': hp.uniform("coef0", -1, 1),
                'gamma': hp.uniform("gamma", 2 ** (-15), 2 ** 3), 
                'shrinking':  hp.choice("shrinking", [True, False]), 
                'tol': hp.uniform("tol",10 ** (-5), 10 ** (-1)),  
                }

    modified_params_gboosting = {'learning_rate':  hp.uniform("learning_rate", 0.01, 0.4),
                        'criterion': hp.choice("criterion", ['friedman_mse', 'mse']),  
                        'n_estimators': hp.choice("n_estimators", np.arange(50, 100, 1)),
                        'max_depth': hp.choice("max_depth", np.arange(1, 10, 1)),
                        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 10, 1)),
                        "min_samples_split": hp.choice("min_samples_split", np.arange(2, 10, 1)),
                        'max_features': hp.uniform("max_features", 10 ** (-1), 9*(10 ** (-1))),
                        
                        }
    
    uniform_params_gboosting = {'learning_rate':  hp.uniform("learning_rate", 0.01, 1.0),
                        'criterion': hp.choice("criterion", ['friedman_mse', 'mse']),  
                        'n_estimators': hp.choice("n_estimators", np.arange(50, 100, 1)),
                        'max_depth': hp.choice("max_depth", np.arange(1, 10, 1)),
                        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 10, 1)),
                        "min_samples_split": hp.choice("min_samples_split", np.arange(2, 10, 1)),
                        'max_features': hp.uniform("max_features", 10 ** (-1), 9*(10 ** (-1))),
                        
                        }

 
    modified_parameters = {
                    'AdaBoost': modified_params_adaboost,
                    'random_forest': modified_params_random_forest,
                    'SVM': modified_params_svm,
                    'ExtraTrees': modified_params_random_forest,
                    'GradientBoosting': modified_params_gboosting,
                    'DecisionTree': modified_params_decision_tree
                    }

    uniform_parameters = {
                    'random_forest': uniform_params_random_forest,
                    'AdaBoost': uniform_params_adaboost,
                    'SVM': uniform_params_svm,
                    'ExtraTrees': uniform_params_random_forest,
                    'GradientBoosting': uniform_params_gboosting,
                    'DecisionTree': uniform_params_decision_tree
                }

    def x_preprocessing(self,):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        return self.X_train, self.X_test

    def y_preprocessing(self,):
        le = LabelEncoder()
        self.y_train = le.fit_transform(self.y_train)
        self.y_test = le.transform(self.y_test)
        
        return self.y_train, self.y_test
  
    def train_evaluate(self, params):
        model = self.models[self.algorithm]
        model.set_params(**params)
        # apply preprocessing on top of dataset.
        self.X_train, self.X_test = self.x_preprocessing()
        self.y_train, self.y_test = self.y_preprocessing()
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        score = roc_auc_score(self.y_test, predictions)
        return score

    # as most of the optimization frameworks, try to minimize the objective function, we multiple to -1.
    def objective(self, params):
        return -1.0 * self.train_evaluate(params)

    def opt_run(self):
        """Returns the resluts of applying random search for
          two different seach space(uniform & fixed)

        Returns:
            uniform_best_loss(dict): A dictionary which
            keeps all the result for uniform configuration.
            
            modified_best_loss(dict):A dictionary which
            keeps all the result for modified configuration. 
        """

        uniform_search_param = self.uniform_parameters[self.algorithm]
        modified_search_param = self.modified_parameters[self.algorithm]

        uniform_trials = Trials()
        fmin(   fn=self.objective,
                space=uniform_search_param,
                algo=rand.suggest,
                max_evals=self.n_iter_opt,
                trials=uniform_trials)
        uniform_best_loss = uniform_trials.best_trial['result']['loss']
        
        modified_trials = Trials()
        fmin(   fn=self.objective,
                space=modified_search_param,
                algo=rand.suggest,
                max_evals=self.n_iter_opt,
                trials=modified_trials)
        modified_best_loss = modified_trials.best_trial['result']['loss']

        return uniform_best_loss, modified_best_loss

