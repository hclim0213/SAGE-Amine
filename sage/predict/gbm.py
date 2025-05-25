"""
Copyright (c) 2024 Hocheol Lim.
"""
import numpy as np
from .utils import calculate_metrics, get_scoring
# scaler_x = MinMaxScaler()
# X_train_scaled = scaler_x.fit_transform(X_train)
# X_test_scaled = scaler_x.transform(X_test)

# ps = PredefinedSplit(test_fold=data_train['split'])
# grid_search = get_grid_search(keyword='regression', model=temp_model, ps=ps)

# print(temp_x, '\t', temp_model,'\t', 'best', '\t', grid_search.best_params_)
# print(grid_search.cv_results_)
  
# with open(temp_x+'_'+temp_model+'.pkl', 'wb') as f:
#     pickle.dump(grid_search, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_param_grid(keyword: str):
    from sklearn.gaussian_process.kernels import RBF
    
    if keyword == 'LR':
        param_grid = {'fit_intercept': [True]}
    
    if keyword == 'Lasso':
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    if keyword == 'Ridge':
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    if keyword == 'ElasNet':
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    if keyword == 'SVM':
        param_grid = {
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'C': [0.01, 0.1, 1, 10, 100],
        }
    
    if keyword == 'GB':
        param_grid = {
        'kernel': [None],
        'alpha': [1e-10, 1e-6, 1e-2],
        'n_restarts_optimizer': [0, 5, 10, 20, 25],
        }
    
    if keyword == 'MLP':
        param_grid = {
        'hidden_layer_sizes': [(64,),(128,),(64,64,),(128,128,)],
        'alpha': [0.0001, 0.05],
        'solver': ['lbfgs', 'adam'],
        }

    if keyword == 'RF':
        param_grid = {
        'n_estimators':[100, 500, 1000, 2000, 3000],
        'max_depth':[10,20,30],
        }

    if keyword == 'XGB':
        param_grid = {
        'booster': ['gbtree', 'dart'],
        'n_estimators':[100, 500, 1000, 2000, 3000],
        'max_depth':[10,20,30],
        'learning_rate':[0.01,0.05,0.1],
        }

    if keyword == 'LGBM':
        param_grid = {
        'boosting_type': ['gbdt', 'dart'],
        'n_estimators':[100, 500, 1000, 2000, 3000],
        'learning_rate':[0.01,0.05,0.1],
        }
    
    if keyword == 'CB':
        param_grid = {
        'n_estimators':[100, 500, 1000, 2000, 3000],
        'depth':[10,20,30],
        'learning_rate':[0.01,0.05,0.1],
        }
    
    return param_grid

def get_model_regression(keyword: str):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    
    if keyword == 'LR':
        model = LinearRegression()

    if keyword == 'Lasso':
        model = Lasso()

    if keyword == 'Ridge':
        model = Ridge()

    if keyword == 'ElasNet':
        model = ElasticNet(random_state=42)

    if keyword == 'SVM':
        model = SVR(verbose=False, max_iter=-1, random_state=42)
    
    if keyword == 'GB':
        model = GaussianProcessRegressor(random_state=42)
    
    if keyword == 'MLP':
        model = MLPRegressor(random_state=42, max_iter=5000, learning_rate='adaptive', verbose=False)

    if keyword == 'RF':
        model = RandomForestRegressor(random_state=42, n_jobs=1)

    if keyword == 'XGB':
        model = XGBRegressor(random_state=42, n_jobs=1)

    if keyword == 'LGBM':
        model = LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1)
    
    if keyword == 'CB':
        model = CatBoostRegressor(random_state=42, verbose=False)
    
    return model

def get_model_classification(keyword: str):
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    
    if keyword == 'LR':
        model = LogisticRegression()

    if keyword == 'Lasso':
        model = SGDClassifier(random_state=42, penalty='l1')

    if keyword == 'Ridge':
        #model = RidgeClassifier()
        model = SGDClassifier(random_state=42, penalty='l2')
    
    if keyword == 'ElasNet':
        model = SGDClassifier(random_state=42, penalty='elasticnet')

    if keyword == 'SVM':
        model = SVC(class_weight='balanced', verbose=False, max_iter=-1, random_state=42)

    if keyword == 'MLP':
        model = MLPClassifier(random_state=42, max_iter= 5000, learning_rate='adaptive', verbose=False)

    if keyword == 'RF':
        model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)

    if keyword == 'XGB':
        # from sklearn.utils.class_weight impot compute_class_weight
        # classes = np.unique(y_train)
        # weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        # model = XGBClassifier(sample_weight=weights, random_state=42, n_jobs=1, use_label_encoder=False)
        
        model = XGBClassifier(random_state=42, n_jobs=1, use_label_encoder=False)

    if keyword == 'LGBM':
        model = LGBMClassifier(class_weight='balanced', random_state=42, verbosity=-1, n_jobs=1)
    
    if keyword == 'CB':
        model = CatBoostClassifier(auto_class_weights='balanced', random_state=42, verbose=False)
    
    return model

def get_grid_search(keyword: str, model: str, cv=5, ps=None, n_jobs=-1):
    from sklearn.model_selection import GridSearchCV
    
    if keyword == 'classfication' and ps == None:
        clf = GridSearchCV(estimator=get_model_classification(model), param_grid=get_param_grid(model), cv=cv, scoring=get_scoring(keyword), refit='f1', n_jobs=n_jobs, return_train_score=True)
    elif keyword == 'classfication' and keyword == 'CB':
        clf = GridSearchCV(estimator=get_model_classification(model), param_grid=get_param_grid(model), cv=ps, scoring=get_scoring(keyword), refit='f1', n_jobs=n_jobs, return_train_score=True)
    elif keyword == 'classfication':
        clf = GridSearchCV(estimator=get_model_classification(model), param_grid=get_param_grid(model), cv=ps, scoring=get_scoring(keyword), refit='f1', n_jobs=1, return_train_score=True)
    
    if keyword == 'regression' and ps == None:
        clf = GridSearchCV(estimator=get_model_regression(model), param_grid=get_param_grid(model), cv=cv, scoring=get_scoring(keyword), refit='r2', n_jobs=n_jobs, return_train_score=True)
    elif keyword == 'regression' and keyword == 'CB':
        clf = GridSearchCV(estimator=get_model_regression(model), param_grid=get_param_grid(model), cv=ps, scoring=get_scoring(keyword), refit='r2', n_jobs=1, return_train_score=True)
    elif keyword == 'regression':
        clf = GridSearchCV(estimator=get_model_regression(model), param_grid=get_param_grid(model), cv=ps, scoring=get_scoring(keyword), refit='r2', n_jobs=n_jobs, return_train_score=True)
    
    return clf