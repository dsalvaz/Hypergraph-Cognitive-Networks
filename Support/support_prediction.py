import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict

################ CANDIDATE REGRESSORS ###################
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import NuSVR

##################### EVALUATION ########################
#from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
#from sklearn.inspection import permutation_importance

def format_dataframes (nodes, attr, graph, louv, eva, lemon, hyper):
    """

    :param attr:
    :param graph:
    :param louv:
    :param eva:
    :param lemon:
    :param hyper:
    :return:
    """

    df_attr = pd.DataFrame(attr)
    df_graph = pd.DataFrame(graph)
    #df_louv = pd.DataFrame(louv)
    #df_eva = pd.DataFrame(eva)
    df_lemon = pd.DataFrame(lemon)
    df_hyper = pd.DataFrame(hyper)

    df_attr['word'] = nodes
    df_graph['word'] = nodes
    #df_louv['word'] = nodes
    #df_eva['word'] = nodes
    df_lemon['word'] = nodes
    df_hyper['word'] = nodes

    nan_words = ['aloft', 'carol', 'drunk', 'filling'] ## MISSED PREPROCESSING
    df_attr = df_attr[df_attr['word'].isin(nan_words) == False]
    df_graph = df_graph[df_graph['word'].isin(nan_words) == False]
    #df_louv = df_louv[df_louv['word'].isin(nan_words) == False]
    #df_eva = df_eva[df_eva['word'].isin(nan_words) == False]
    df_lemon = df_lemon[df_lemon['word'].isin(nan_words) == False]
    df_hyper = df_hyper[df_hyper['word'].isin(nan_words) == False]

    dfs = [df_attr,
          df_graph,
          #df_louv,
          #df_eva,
          df_lemon,
          df_hyper
          ]

    dfs_names = ['attr',
                'egonet',
                #'louv',
                #'eva',
                'lemon',
                'hego'
                ]

    return dfs, dfs_names


def ml_cv_pipeline(dfs, dfs_names, to_pred, which='rf', n_cv=5):
    """

    :param dfs:
    :param dfs_names:
    :param to_pred:
    :param which:
    :return:
    """

    res = defaultdict(lambda: defaultdict(dict))

    for i, d in enumerate(dfs):

        var_to_pred = dfs[0][to_pred]  # dfs[0] is "df_attr"
        vars_to_del = ['word', to_pred]
        attributes = [col for col in d.columns if col not in vars_to_del]
        X = d[attributes].values
        y = var_to_pred

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if which == 'rf':
            reg = RandomForestRegressor(criterion='mse', n_estimators=300, max_features=0.5, max_depth=None)
        elif which == 'linear':
            reg = LinearRegression()
        elif which == 'ada':
            reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=5), n_estimators=100,
                                    learning_rate=0.5)
        elif which == 'svr':
            reg = NuSVR(C=1.0, kernel='rbf', nu=0.6)

        scores_rmse = cross_validate(reg, X, y, cv=n_cv, scoring='neg_root_mean_squared_error',
                                          return_estimator=True)
        scores_r2 = cross_validate(reg, X, y, cv=n_cv, scoring='r2', return_estimator=True)

        res[dfs_names[i]]['RMSE']['M'] = np.mean(scores_rmse['test_score'])
        res[dfs_names[i]]['RMSE']['STD'] = np.std(scores_rmse['test_score'])

        res[dfs_names[i]]['R2']['M'] = np.mean(scores_r2['test_score'])
        res[dfs_names[i]]['R2']['STD'] = np.std(scores_r2['test_score'])

    return res

def shap_pipeline(dfs, dfs_names, to_pred, which='rf', savefig=False):
    """

    :param dfs:
    :param dfs_names:
    :param to_pred:
    :param which:
    :return:
    """

    for i, d in enumerate(dfs):

        var_to_pred = dfs[0][to_pred]  # dfs[0] is "df_attr"
        vars_to_del = ['word', to_pred]
        attributes = [col for col in d.columns if col not in vars_to_del]
        X = d[attributes].values
        y = var_to_pred

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if which == 'rf':
            reg = RandomForestRegressor(criterion='mse', n_estimators=300, max_features=0.5, max_depth=None)
        elif which == 'linear':
            reg = LinearRegression()
        elif which == 'ada':
            reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=5), n_estimators=100,
                                    learning_rate=0.5)
        elif which == 'svr':
            reg = NuSVR(C=1.0, kernel='rbf', nu=0.6)

        reg.fit(X_train, y_train)
        explainer = shap.TreeExplainer(reg)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=attributes, show=False, )

        plt.gcf().axes[-1].set_aspect('auto')
        plt.tight_layout()
        plt.gcf().axes[-1].set_box_aspect(50)

        if savefig==True:
            plt.savefig('Figures\shap_rf_' + str(dfs_names[i]) + '.png', bbox_inches='tight')
            plt.show()
        else:
            plt.show()


"""
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

########### RANDOM FOREST PARAMETERS ###########
param_list = {
    'n_estimators': [300],
    'max_depth': [None, 4, 8],
    'max_features': ['auto', 0.5, 'log2'],
    # 'min_samples_split': [2, 5, 10, 20],
    # 'min_samples_leaf': [1, 5, 10, 20]
}
###############################################
"""
