import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from Functions import ClassificationPerDataset, ClassificationPerAlgorithm


N = 500

params_random_forest = {'n_estimators': np.repeat(100, N),
                        'bootstrap': np.random.choice([True, False], N),
                        'max_features': np.random.uniform(0.1, 0.9, N),
                        'min_samples_leaf': np.random.choice(np.arange(1, 21, 1), N),
                        'min_samples_split': np.random.choice(np.arange(2, 21, 1), N),
                        'criterion': np.random.choice(['entropy', 'gini'], N)}


params_adaboost = {'base_estimator__max_depth': np.random.choice(np.arange(1, 11, 1), N),
                   'algorithm': np.random.choice(['SAMME', 'SAMME.R'], N),
                   'n_estimators': np.random.choice(np.arange(50, 501, 1), N),  # iterations in the paper
                   'learning_rate': np.random.uniform(0.01, 2, N)}


params_svm = {'kernel': np.random.choice(['rbf', 'sigmoid'], N),
              'C': np.random.uniform(np.exp(2 ** (-5)), np.exp(2 ** (-15)), N),
              'coef0': np.random.uniform(-1, 1, N),
              'gamma': np.random.uniform(np.exp(2 ** (-15)), np.exp(2 ** (3)), N),
              'shrinking': np.random.choice([True, False], N),
              'tol': np.random.uniform(np.exp(10 ** (-5)), np.exp(10 ** (-1)), N)}

params_gboosting = {'learning_rate': np.random.uniform(0.01, 1, N),
                    'criterion': np.random.choice(['friedman_mse', 'mae', 'mse'], N),
                   'n_estimators': np.random.choice(np.arange(50, 501, 1), N),
                   'max_depth': np.random.choice(np.arange(1, 11, 1), N),
                   'min_samples_split': np.random.choice(np.arange(2, 21, 1), N),
                   'min_samples_leaf': np.random.choice(np.arange(1, 21, 1), N),
                   'max_features': np.random.uniform(0.1, 0.9, N)}
                   
                   


parameters = {'AdaBoost': params_adaboost,
              'RandomForest': params_random_forest,
              'SVM': params_svm,
              'ExtraTrees': params_random_forest,
              'GradientBoosting': params_gboosting}

models = {'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
          'RandomForest': RandomForestClassifier(),
          'SVM': SVC(),
          'ExtraTrees': ExtraTreesClassifier(),
          'GradientBoosting': GradientBoostingClassifier()}



ClassificationPerAlgorithm('datasets/', 'ExtraTrees', models, parameters)

