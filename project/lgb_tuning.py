import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import lightgbm as lgb
from scipy.stats import skew
from scipy import stats
import optuna 
from sklearn.model_selection import KFold
from optuna.integration import LightGBMPruningCallback
import warnings
warnings.filterwarnings("ignore")

def rmlse(y_true, y_pred):
    # Alternatively: sklearn.metrics.mean_squared_log_error(y_true, y_pred) ** 0.5
    assert (y_true >= 0).all() 
    assert (y_pred >= 0).all()
    log_error = np.log1p(y_pred) - np.log1p(y_true)  # Note: log1p(x) = log(1 + x)
    return np.mean(log_error ** 2) ** 0.5


def objective_lgb(trial, X, y, area, categorical):
    params = {
        'verbose': -1,
        'metric': 'rmse', 
        'random_state': 42,
        'n_estimators': 10000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02, 0.03]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('cat_smooth', 3, 60)
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pruning_callback = LightGBMPruningCallback(trial, "rmse")

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_test = np.log(y[test_idx]/area[test_idx])
        y_train = np.log(y[train_idx]/area[train_idx])

        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test,free_raw_data=False)
        model = lgb.train(params, 
                        lgb_train, valid_sets=lgb_eval, 
                        verbose_eval=False, callbacks = [pruning_callback], early_stopping_rounds=100,
                        categorical_feature=categorical)
    
        preds = np.exp(model.predict(X_test)) * area[test_idx]
        y_true = np.exp(y_test) * area[test_idx]
        cv_scores[idx] = rmlse(y_test, preds)
    
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


data = all_data.loc[all_data['split'] == 'train', :]
data = data.drop(columns=['split'])

data_test = all_data.loc[all_data['split'] == 'test', :]
data_test = data_test.drop(columns=['split', 'price'])


X = data.drop(columns=['price', 'address', 'street'])
y = data.price
area = data.area_total

study = optuna.create_study(direction="minimize", study_name="LGB Regressor")
func = lambda trial: objective_lgb(trial, X, y, area, CATEGORICAL_FEATURES)
study.optimize(func, n_trials=100)

'''
params = study.best_params
params['random_state'] = 42
params['metric'] = 'rmse'
params['n_estimators'] = 10000
params['verbose'] =  -1

lgb_train = lgb.Dataset(X, y, free_raw_data=False)

lgb_mod = lgb_mod = lgb.train(
    params,
    lgb_train,
    categorical_feature= CATEGORICAL_FEATURES
)
lgb.plot_importance(lgb_mod)
plt.show()

X_test = data_test.drop(columns=['address', 'street'])
preds_test = np.exp(lgb_mod.predict(X_test)) * data_test['area_total']
submission = pd.DataFrame()
submission['id'] = data_test.index
submission['price_prediction'] = preds_test
submission.to_csv('submissions/lgbm_submission.csv', index=False)
'''