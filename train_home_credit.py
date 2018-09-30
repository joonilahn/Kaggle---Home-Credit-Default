# Many thanks to Georgios Sarantitis (https://www.kaggle.com/georsara1/lightgbm-all-tables-included-0-778)

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from time import gmtime, strftime
import gc, json, argparse

# Default parameters chosen by using Bayesian Obtimization
DEFAULT_PARAMS = {'boosting_type': 'gbdt',
              'n_jobs' : 8,
              'n_estimators' : 10000,
              'max_depth' : 8,
              'objective': 'binary',
              'num_leaves': 34,
              'learning_rate': 0.02,
              'max_bin': 63,
              'subsample': 0.8715623,
              'subsample_freq': 1,
              'colsample_bytree': 0.9497036,
              'reg_alpha': 0.041545473,
              'reg_lambda': 0.0735294,
              'min_split_gain': 0.0222415,
              'min_child_weight': 60,
              'num_class' : 1,
              'silent' : -1,
              'verbose' : -1,
              # 'device': 'gpu',
              'metric' : 'auc'
              }

def obj_to_cat(df):
    '''
    Convert object dtypes to categorical dtypes
    '''
    cat_features = [f_ for f_ in df.columns if df[f_].dtype == 'object']
    for f_ in cat_features:
        df[f_] = df[f_].astype('category')
    df[cat_features] = df[cat_features].apply(lambda x: x.cat.codes)
    return df

def preprocess_dfs(df, dfname):
    '''
    Calculate mean, max, min values for all numerical features
    and count the number of data for each SK_ID_CURR
    '''
    # new_df1
    # Get categorical features
    cat_features = [f_ for f_ in df.columns if df[f_].dtype == 'object']
    
    if len(cat_features) > 0:
        # Onehot encoding
        df_dummies = pd.get_dummies(df[cat_features])
        df_concat = pd.concat([df['SK_ID_CURR'], df_dummies], axis=1)
        new_df1 = df_concat.groupby('SK_ID_CURR').sum().reset_index()

    # new_df2
    new_df2 = obj_to_cat(df)    
    
    # For bureau dataframe, drop 'SK_ID_BUREAU', else drop 'SK_ID_PREV'
    if dfname == 'bureau':
        drop_column = 'SK_ID_BUREAU'
    else:
        drop_column = 'SK_ID_PREV'

    new_df2 = new_df2.drop(drop_column, axis=1)
    cols = [s + '_' + l for s in new_df2.columns.tolist()
               if s!='SK_ID_CURR'
               for l in ['mean','max','min']]
    new_df2 = new_df2.groupby('SK_ID_CURR').agg(['mean','max','min']).reset_index()
    new_df2.columns=['SK_ID_CURR'] + cols

    if len(cat_features) > 0:
        new_df2 = new_df2.drop([s + '_' + l for s in cat_features for l in ['mean','max','min']], axis=1)
        
    new_df2[dfname + '_cnt'] = df[[drop_column, 'SK_ID_CURR']].groupby('SK_ID_CURR').count()[drop_column]
    
    if len(cat_features) > 0:
        new_df2 = new_df2.merge(right=new_df1, how='left', on='SK_ID_CURR')
    
    return new_df2

def merge_dfs(leftdf, rightdfs):
    for rightdf in rightdfs:
        leftdf = leftdf.merge(right=rightdf.reset_index(), how='left', on='SK_ID_CURR')
    return leftdf

def create_feats(df):
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    # Create new features
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    return df

def train_model(X_train, y_train, X_test, params, print_every=100):
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X_train.shape[0])
    sub_preds = np.zeros(X_test.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in X_train.columns if f not in ['SK_ID_CURR']]
    
    # set parameters
    n_jobs = params['n_jobs']
    n_estimators = params['n_estimators']
    learning_rate = params['learning_rate']
    num_leaves = params['num_leaves']
    colsample_bytree = params['colsample_bytree']
    subsample = params['subsample']
    max_depth = params['max_depth']
    reg_alpha = params['reg_alpha']
    reg_lambda = params['reg_lambda']
    min_split_gain = params['min_split_gain']
    min_child_weight = params['min_child_weight']
    silent = params['silent']
    verbose = params['verbose']
    metric = params['metric']
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
        trn_x, trn_y = X_train[feats].iloc[trn_idx], y_train.iloc[trn_idx]
        val_x, val_y = X_train[feats].iloc[val_idx], y_train.iloc[val_idx]

        clf = LGBMClassifier(
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            silent=silent,
            metric=metric,
            verbose=verbose)

        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=print_every, early_stopping_rounds=100
               )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(X_test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        # fold_importance_df = pd.DataFrame()
        # fold_importance_df["feature"] = feats
        # fold_importance_df["importance"] = clf.feature_importances_
        # fold_importance_df["fold"] = n_fold + 1
        # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    cv_score = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % cv_score) 

    # Plot feature importances
#     cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
#                     by="importance", ascending=False)[:50].index

#     best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
#     plt.figure(figsize=(8,10))
#     sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
#     plt.title('LightGBM Features (avg over folds)')
#     plt.tight_layout()

    X_test['TARGET'] = sub_preds
    
    return test, cv_score

def save_params(params):
    with open('best_param.json', 'w') as fp:
        json.dump(params, fp)

def main(params):
    # Read datasets
    print('Reading datasets')
    datapath = 'input/'
    descriptions = pd.read_csv(datapath + 'HomeCredit_columns_description.csv', encoding='iso-8859-1')
    traindf = pd.read_csv(datapath + 'application_train.csv') 
    testdf = pd.read_csv(datapath + 'application_test.csv')
    bureau = pd.read_csv(datapath + 'bureau.csv')
    pos = pd.read_csv(datapath + 'POS_CASH_balance.csv')
    balance = pd.read_csv(datapath + 'credit_card_balance.csv')
    prev_application = pd.read_csv(datapath + 'previous_application.csv')
    instpayments = pd.read_csv(datapath + 'installments_payments.csv')

    # Get categorical features
    cat_features = [f_ for f_ in testdf.columns if testdf[f_].dtype == 'object']

    # Convert object dtypes to categorical dtypes
    print('Preprocessing datasets')
    traindf = obj_to_cat(traindf)
    testdf = obj_to_cat(testdf)

    # Preprocess the other tables
    bureau = preprocess_dfs(bureau, 'bureau')
    pos = preprocess_dfs(pos, 'pos')
    balance = preprocess_dfs(balance, 'balance')
    prev_application = preprocess_dfs(prev_application, 'prev_applications')
    instpayments = preprocess_dfs(instpayments, 'instpayments')

    # Merge all tables
    print('Merging datasets')
    traindf = merge_dfs(traindf, [bureau, pos, balance, prev_application, instpayments])
    testdf = merge_dfs(testdf, [bureau, pos, balance, prev_application, instpayments])

    # Create new features 
    traindf = create_feats(traindf)
    testdf = create_feats(testdf)

    # Create train data and labels
    X = traindf.drop('TARGET', axis=1)
    y = traindf.TARGET

    del traindf, bureau, pos, balance, prev_application, instpayments
    gc.collect()

    # Train
    print('Started Training')
    test, score = train_model(X, y, testdf, params)

    # Create the submission data
    t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    test[['SK_ID_CURR', 'TARGET']].to_csv('submission' + t + '.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=float, dest='params',
                        default=DEFAULT_PARAMS,
                        help='Set parameters from json file')
    args = parser.parse_args()
    main(args.params)