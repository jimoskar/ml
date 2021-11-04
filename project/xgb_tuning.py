import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import lightgbm as lgb
import sklearn.model_selection as model_selection
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy import stats
import xgboost as xgb
import optuna 
from sklearn.model_selection import KFold

def rmlse(y_true, y_pred):
    # Alternatively: sklearn.metrics.mean_squared_log_error(y_true, y_pred) ** 0.5
    assert (y_true >= 0).all() 
    assert (y_pred >= 0).all()
    log_error = np.log1p(y_pred) - np.log1p(y_true)  # Note: log1p(x) = log(1 + x)
    return np.mean(log_error ** 2) ** 0.5

def objective_xgb(trial, X, y):

    param = {   'booster': 'gbtree',
                'max_depth':trial.suggest_int('max_depth', 1, 11),
                'reg_alpha':trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                'reg_lambda':trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                'min_child_weight':trial.suggest_int('min_child_weight', 0, 5),
                'gamma':trial.suggest_int('gamma', 0, 5),
                'learning_rate':trial.suggest_loguniform('learning_rate',0.001,0.5),
                'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
                'nthread' : -1
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = np.empty(5)

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        model = xgb.train(param, dtrain, num_boost_round=1200, evals=[(dtest, "validation")], 
                    early_stopping_rounds = 50, verbose_eval=False) #callbacks=[pruning_callback])
    
        preds = np.exp(model.predict(dtest)) - 1
        y_true = np.exp(y_test) - 1
        cv_scores[idx] = rmlse(y_true, preds)

    return np.mean(cv_scores)


all_data = pd.read_csv('resources/data_position_poi.csv')

NUMERIC_FEATURES = ["latitude", "longitude", "constructed", "area_total",
            "rooms", "balconies", "loggias", "metro_distance", "park_distance",
            "square_distance", "stories","floor", "ceiling", "bathrooms_shared", "bathrooms_private", "phones"]
CATEGORICAL_FEATURES = ["seller", "district", "material", "condition", "heating", "new", 
                "layout", "windows_court", "windows_street", "parking", "garbage_chute", "elevator_passenger", "elevator_without", "elevator_service"]

all_data[CATEGORICAL_FEATURES] = all_data[CATEGORICAL_FEATURES].astype('category')

#log transform the target:
all_data['price'] = np.log1p(all_data['price'])

#log transform skewed numeric features:

skewed_feats = all_data[NUMERIC_FEATURES].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Encoding categorical data
all_data = pd.get_dummies(all_data, columns  = CATEGORICAL_FEATURES)

all_data = all_data.fillna(all_data.mean())

data = all_data.loc[all_data['split'] == 'train', :]
data = data.drop(columns=['split'])

data_test = all_data.loc[all_data['split'] == 'test', :]
data_test = data_test.drop(columns=['split', 'price'])


X_train = data.drop(columns=['price', 'address', 'street'])
y = data.price

xgb_study = optuna.create_study(direction="minimize", study_name="XGB Regressor")
func = lambda trial: objective_xgb(trial, X_train, y)
xgb_study.optimize(func, n_trials=20)
