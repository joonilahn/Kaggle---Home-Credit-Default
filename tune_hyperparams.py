# Many thanks to Georgios Sarantitis (https://www.kaggle.com/georsara1/lightgbm-all-tables-included-0-778)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from time import gmtime, strftime
import gc, json

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

def main():
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
    bureau = preprocess_dfs(bureau, 'bureau')
    pos = preprocess_dfs(pos, 'pos')
    balance = preprocess_dfs(balance, 'balance')
    prev_application = preprocess_dfs(prev_application, 'prev_applications')
    instpayments = preprocess_dfs(instpayments, 'instpayments')

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

    # Grid search
    best_auc = 0.0
    print('Start searching the optimal parameters')
    for depth in [-1, 3, 5]:
      for nleaves in [20, 40, 60]:
          for colsample in [0.7, 0.75, 0.8]:
              for min_split in [0.5, 0.7]:
                  print('max_depth: {0:}, num_leaves: {1:}, col_sample: {2:}, min_split: {3:}'\
                        .format(depth, nleaves, colsample, min_split))
                  params = {'boosting_type': 'gbdt',
                            'max_depth' : depth,
                            'objective': 'binary',
                            'num_leaves': nleaves,
                            'learning_rate': 0.05,
                            'max_bin': 63,
                            'subsample': 1,
                            'subsample_freq': 1,
                            'colsample_bytree': colsample,
                            'reg_alpha': 5,
                            'reg_lambda': 3,
                            'min_split_gain': min_split,
                            'min_child_weight': 1,
                            'min_child_samples': 5,
                            'scale_pos_weight': 1,
                            'num_class' : 1,
                            'device': 'gpu',
                            'metric' : 'auc'
                            }

                  train_data = lgb.Dataset(X_train, label=y_train, 
                                   categorical_feature=cat_features)
                  valid_data = lgb.Dataset(X_val, label=y_val, 
                                   categorical_feature=cat_features)

                  lgbm = lgb.train(params,
                                   train_data,
                                   2500,
                                   valid_sets=valid_data,
                                   early_stopping_rounds= 40,
                                   verbose_eval= False
                                   )

                  if lgbm.best_score['valid_0']['auc'] > best_auc:
                      best_auc = lgbm.best_score['valid_0']['auc']
                      best_lgb = lgbm
                      best_params = params

    # best_params = params
    print('Tuning regularization parameters')
    for r_alpha in [1, 3, 5]:
        for r_lambda in [1, 2, 3]:
            print('reg_alpha: {0:}, reg_lambda: {1:}'.format(r_alpha, r_lambda))
            params = best_params
            params['reg_alpha'] = r_alpha
            params['reg_lambda'] = r_lambda
            
            train_data = lgb.Dataset(X_train, label=y_train, 
                             categorical_feature=cat_features)
            valid_data = lgb.Dataset(X_val, label=y_val, 
                             categorical_feature=cat_features)

            lgbm = lgb.train(params,
                             train_data,
                             2500,
                             valid_sets=valid_data,
                             early_stopping_rounds= 40,
                             verbose_eval= False
                             )

            if lgbm.best_score['valid_0']['auc'] > best_auc:
                best_auc = lgbm.best_score['valid_0']['auc']
                best_lgb = lgbm
                best_params = params

    # Save the parameters in 'params.json'
    with open('params.json', 'w') as fp:
        json.dump(best_params, fp)

    # Create the submission data
    print('Creating a submission file')
    y_pred = best_lgb.predict(testdf)
    submissiondf = pd.DataFrame({'SK_ID_CURR': testdf['SK_ID_CURR'], 'TARGET': y_pred})
    t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    submissiondf.to_csv('submission' + t + '.csv', index=False)

if __name__ == '__main__':
    main()