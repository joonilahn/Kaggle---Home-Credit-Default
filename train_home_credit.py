# Many thanks to Georgios Sarantitis (https://www.kaggle.com/georsara1/lightgbm-all-tables-included-0-778)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from time import gmtime, strftime
import gc

DEFAULT_PARAMS = {'boosting_type': 'gbdt',
              'max_depth' : 7,
              'objective': 'binary',
              'num_leaves': 64,
              'learning_rate': 0.05,
              'max_bin': 63,
              'subsample_for_bin': 200,
              'subsample': 1,
              'subsample_freq': 1,
              'colsample_bytree': 0.7,
              'reg_alpha': 5,
              'reg_lambda': 3,
              'min_split_gain': 0.5,
              'min_child_weight': 1,
              'min_child_samples': 5,
              'scale_pos_weight': 1,
              'num_class' : 1,
              'device': 'gpu',
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

def preprocess_bureau(df, dfname):
    '''
    Calculate mean, max, min values for all numerical features
    and count the number of data for each SK_ID_CURR
    '''
    new_bureau = obj_to_cat(df)
    new_bureau = new_bureau.drop('SK_ID_BUREAU', axis=1)
    cols = [s + '_' + l for s in new_bureau.columns.tolist()
               if s!='SK_ID_CURR' for l in ['mean','max','min']]
    new_bureau = new_bureau.groupby('SK_ID_CURR').agg(['mean','max','min']).reset_index()
    new_bureau.columns=['SK_ID_CURR'] + cols
    new_bureau[dfname + '_cnt'] = df[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
    return new_bureau

def preprocess_others(df, dfname):
    '''
    Calculate mean, max, min values for all numerical features
    and count the number of data for each SK_ID_CURR
    '''
    new_df = obj_to_cat(df)
    new_df = new_df.drop('SK_ID_PREV', axis=1)
    cols = [s + '_' + l for s in new_df.columns.tolist()
               if s!='SK_ID_CURR' for l in ['mean','max','min']]
    new_df = new_df.groupby('SK_ID_CURR').agg(['mean','max','min']).reset_index()
    new_df.columns=['SK_ID_CURR'] + cols
    new_df[dfname + '_cnt'] = df[['SK_ID_PREV', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_PREV']
    return new_df

def merge_dfs(leftdf, rightdfs):
    for rightdf in rightdfs:
        leftdf = leftdf.merge(right=rightdf.reset_index(), how='left', on='SK_ID_CURR')
    return leftdf

def main(params)
    # Read datasets
    print('Reading datasets')
    datapath = 'dataset/'
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
    bureau = preprocess_bureau(bureau, 'bureau')
    pos = preprocess_others(pos, 'pos')
    balance = preprocess_others(balance, 'balance')
    prev_application = preprocess_others(prev_application, 'prev_applications')
    instpayments = preprocess_others(instpayments, 'instpayments')

    # Merge all tables
    print('Merging datasets')
    traindf = merge_dfs(traindf, [bureau, pos, balance, prev_application, instpayments])
    testdf = merge_dfs(testdf, [bureau, pos, balance, prev_application, instpayments])

    # Create train data and labels
    X = traindf.drop('TARGET', axis=1)
    y = traindf.TARGET

    del traindf, bureau, pos, balance, prev_application, instpayments
    gc.collect()

    # Split the data into train data and validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build Dataset for lightgbm
    train_data = lgb.Dataset(X_train, label=y_train, 
                             categorical_feature=cat_features)
    valid_data = lgb.Dataset(X_val, label=y_val, 
                             categorical_feature=cat_features)

    # # Set parameters
    # params = {'boosting_type': 'gbdt',
    #           'max_depth' : 7,
    #           'objective': 'binary',
    #           'nthread': 5,
    #           'num_leaves': 64,
    #           'learning_rate': 0.05,
    #           'max_bin': 512,
    #           'subsample_for_bin': 200,
    #           'subsample': 1,
    #           'subsample_freq': 1,
    #           'colsample_bytree': 0.7,
    #           'reg_alpha': 5,
    #           'reg_lambda': 3,
    #           'min_split_gain': 0.5,
    #           'min_child_weight': 1,
    #           'min_child_samples': 5,
    #           'scale_pos_weight': 1,
    #           'num_class' : 1,
    #           'metric' : 'auc'
    #           }

    # Train
    print('Started Training')
    lgbm = lgb.train(params,
                     train_data,
                     2500,
                     valid_sets=valid_data,
                     early_stopping_rounds= 40,
                     verbose_eval= 20
                     )

    # Create the submission data
    y_pred = lgbm.predict(testdf)
    submissiondf = pd.DataFrame({'SK_ID_CURR': testdf['SK_ID_CURR'], 'TARGET': y_pred})
    t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    submissiondf.to_csv('submission' + t + '.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=float, dest='params',
                        default=DEFAULT_PARAMS,
                        help='Set parameters from json file')
    args = parser.parse_args()
    main(args.params)