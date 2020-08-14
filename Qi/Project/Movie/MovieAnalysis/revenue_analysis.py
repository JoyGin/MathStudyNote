import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import ast
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

def prepare(df):
    global json_cols
    global train_dict
    
    df['rating'] = df['rating'].fillna(1.5)
    df['totalVotes'] = df['totalVotes'].fillna(6)
    df['weightedRating'] = ( df['rating']*df['totalVotes'] + 6.367 * 1000 ) / ( df['totalVotes'] + 1000 )

    df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[ (df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[ (df['release_year'] > 18)  & (df['release_year'] < 100), "release_year"] += 1900
    
    releaseDate = pd.to_datetime(df['release_date']) 
    df['release_dayofweek'] = releaseDate.dt.dayofweek 
    df['release_quarter'] = releaseDate.dt.quarter     

    df['_budget_runtime_ratio'] = df['budget']/df['runtime'] 
    df['_budget_popularity_ratio'] = df['budget']/df['popularity']
    df['_budget_year_ratio'] = df['budget']/(df['release_year']*df['release_year'])
    df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']

    df['_popularity_totalVotes_ratio'] = df['totalVotes']/df['popularity']
    df['_rating_popularity_ratio'] = df['rating']/df['popularity']
    df['_rating_totalVotes_ratio'] = df['totalVotes']/df['rating']
    df['_totalVotes_releaseYear_ratio'] = df['totalVotes']/df['release_year']
    df['_budget_rating_ratio'] = df['budget']/df['rating']
    df['_runtime_rating_ratio'] = df['runtime']/df['rating']
    df['_budget_totalVotes_ratio'] = df['budget']/df['totalVotes']
    
    df['has_homepage'] = 0
    df.loc[pd.isnull(df['homepage']) ,"has_homepage"] = 1
    
    df['isbelongs_to_collectionNA'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']) ,"isbelongs_to_collectionNA"] = 1
    
    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0 ,"isTaglineNA"] = 1 

    df['isOriginalLanguageEng'] = 0 
    df.loc[ df['original_language'] == "en" ,"isOriginalLanguageEng"] = 1
    
    df['isTitleDifferent'] = 1
    df.loc[ df['original_title'] == df['title'] ,"isTitleDifferent"] = 0 

    df['isMovieReleased'] = 1
    df.loc[ df['status'] != "Released" ,"isMovieReleased"] = 0 

    # get collection id
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x : np.nan if len(x)==0 else x[0]['id'])
    
    df['original_title_letter_count'] = df['original_title'].str.len() 
    df['original_title_word_count'] = df['original_title'].str.split().str.len() 


    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()
    
    df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))
    df['cast_count'] = df['cast'].apply(lambda x : len(x))
    df['crew_count'] = df['cast'].apply(lambda x : len(x))

    df['meanruntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')
    df['meanPopularityByYear'] = df.groupby("release_year")["popularity"].aggregate('mean')
    df['meanBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('mean')
    df['meantotalVotesByYear'] = df.groupby("release_year")["totalVotes"].aggregate('mean')
    df['meanTotalVotesByRating'] = df.groupby("rating")["totalVotes"].aggregate('mean')

    for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies'] :
        df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col+'_etc' for n in [d['name'] for d in x]])))).map(lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)
    df.drop(['genres_etc'], axis = 1, inplace = True)
    
    df = df.drop(['id', 'revenue','belongs_to_collection','genres','homepage','imdb_id','overview','runtime'
    ,'poster_path','production_companies','production_countries','release_date','spoken_languages'
    ,'status','title','Keywords','cast','crew','original_language','original_title','tagline', 'collection_id'
    ],axis=1)
    
    df.fillna(value=0.0, inplace = True) 

    return df

train = pd.read_csv('data/prediction/train.csv')
test = pd.read_csv('data/prediction/test.csv')


train.loc[train['id'] == 16, 'revenue'] = 192864
train.loc[train['id'] == 90, 'budget'] = 30000000
train.loc[train['id'] == 118, 'budget'] = 60000000
train.loc[train['id'] == 149, 'budget'] = 18000000
train.loc[train['id'] == 313, 'revenue'] = 12000000
train.loc[train['id'] == 451, 'revenue'] = 12000000
train.loc[train['id'] == 464, 'budget'] = 20000000
train.loc[train['id'] == 470, 'budget'] = 13000000
train.loc[train['id'] == 513, 'budget'] = 930000
train.loc[train['id'] == 797, 'budget'] = 8000000
train.loc[train['id'] == 819, 'budget'] = 90000000
train.loc[train['id'] == 850, 'budget'] = 90000000
train.loc[train['id'] == 1007, 'budget'] = 2
train.loc[train['id'] == 1112, 'budget'] = 7500000
train.loc[train['id'] == 1131, 'budget'] = 4300000
train.loc[train['id'] == 1359, 'budget'] = 10000000
train.loc[train['id'] == 1542, 'budget'] = 1
train.loc[train['id'] == 1570, 'budget'] = 15800000
train.loc[train['id'] == 1571, 'budget'] = 4000000
train.loc[train['id'] == 1714, 'budget'] = 46000000
train.loc[train['id'] == 1721, 'budget'] = 17500000
train.loc[train['id'] == 1865, 'revenue'] = 25000000
train.loc[train['id'] == 1885, 'budget'] = 12
train.loc[train['id'] == 2091, 'budget'] = 10
train.loc[train['id'] == 2268, 'budget'] = 17500000
train.loc[train['id'] == 2491, 'budget'] = 6
train.loc[train['id'] == 2602, 'budget'] = 31000000
train.loc[train['id'] == 2612, 'budget'] = 15000000
train.loc[train['id'] == 2696, 'budget'] = 10000000
train.loc[train['id'] == 2801, 'budget'] = 10000000
train.loc[train['id'] == 335, 'budget'] = 2
train.loc[train['id'] == 348, 'budget'] = 12
train.loc[train['id'] == 470, 'budget'] = 13000000
train.loc[train['id'] == 513, 'budget'] = 1100000
train.loc[train['id'] == 640, 'budget'] = 6
train.loc[train['id'] == 696, 'budget'] = 1
train.loc[train['id'] == 797, 'budget'] = 8000000
train.loc[train['id'] == 850, 'budget'] = 1500000
train.loc[train['id'] == 1199, 'budget'] = 5
train.loc[train['id'] == 1282, 'budget'] = 9
train.loc[train['id'] == 1347, 'budget'] = 1
train.loc[train['id'] == 1755, 'budget'] = 2
train.loc[train['id'] == 1801, 'budget'] = 5
train.loc[train['id'] == 1918, 'budget'] = 592
train.loc[train['id'] == 2033, 'budget'] = 4
train.loc[train['id'] == 2118, 'budget'] = 344
train.loc[train['id'] == 2252, 'budget'] = 130
train.loc[train['id'] == 2256, 'budget'] = 1
train.loc[train['id'] == 2696, 'budget'] = 10000000

test.loc[test['id'] == 3033, 'budget'] = 250
test.loc[test['id'] == 3051, 'budget'] = 50
test.loc[test['id'] == 3084, 'budget'] = 337
test.loc[test['id'] == 3224, 'budget'] = 4
test.loc[test['id'] == 3594, 'budget'] = 25
test.loc[test['id'] == 3619, 'budget'] = 500
test.loc[test['id'] == 3831, 'budget'] = 3
test.loc[test['id'] == 3935, 'budget'] = 500
test.loc[test['id'] == 4049, 'budget'] = 995946
test.loc[test['id'] == 4424, 'budget'] = 3
test.loc[test['id'] == 4460, 'budget'] = 8
test.loc[test['id'] == 4555, 'budget'] = 1200000
test.loc[test['id'] == 4624, 'budget'] = 30
test.loc[test['id'] == 4645, 'budget'] = 500
test.loc[test['id'] == 4709, 'budget'] = 450
test.loc[test['id'] == 4839, 'budget'] = 7
test.loc[test['id'] == 3125, 'budget'] = 25
test.loc[test['id'] == 3142, 'budget'] = 1
test.loc[test['id'] == 3201, 'budget'] = 450
test.loc[test['id'] == 3222, 'budget'] = 6
test.loc[test['id'] == 3545, 'budget'] = 38
test.loc[test['id'] == 3670, 'budget'] = 18
test.loc[test['id'] == 3792, 'budget'] = 19
test.loc[test['id'] == 3881, 'budget'] = 7
test.loc[test['id'] == 3969, 'budget'] = 400
test.loc[test['id'] == 4196, 'budget'] = 6
test.loc[test['id'] == 4221, 'budget'] = 11
test.loc[test['id'] == 4222, 'budget'] = 500
test.loc[test['id'] == 4285, 'budget'] = 11
test.loc[test['id'] == 4319, 'budget'] = 1
test.loc[test['id'] == 4639, 'budget'] = 10
test.loc[test['id'] == 4719, 'budget'] = 45
test.loc[test['id'] == 4822, 'budget'] = 22
test.loc[test['id'] == 4829, 'budget'] = 20
test.loc[test['id'] == 4969, 'budget'] = 20
test.loc[test['id'] == 5021, 'budget'] = 40
test.loc[test['id'] == 5035, 'budget'] = 1
test.loc[test['id'] == 5063, 'budget'] = 14
test.loc[test['id'] == 5119, 'budget'] = 2
test.loc[test['id'] == 5214, 'budget'] = 30
test.loc[test['id'] == 5221, 'budget'] = 50
test.loc[test['id'] == 4903, 'budget'] = 15
test.loc[test['id'] == 4983, 'budget'] = 3
test.loc[test['id'] == 5102, 'budget'] = 28
test.loc[test['id'] == 5217, 'budget'] = 75
test.loc[test['id'] == 5224, 'budget'] = 3
test.loc[test['id'] == 5469, 'budget'] = 20
test.loc[test['id'] == 5840, 'budget'] = 1
test.loc[test['id'] == 5960, 'budget'] = 30
test.loc[test['id'] == 6506, 'budget'] = 11
test.loc[test['id'] == 6553, 'budget'] = 280
test.loc[test['id'] == 6561, 'budget'] = 7
test.loc[test['id'] == 6582, 'budget'] = 218
test.loc[test['id'] == 6638, 'budget'] = 5
test.loc[test['id'] == 6749, 'budget'] = 8
test.loc[test['id'] == 6759, 'budget'] = 50
test.loc[test['id'] == 6856, 'budget'] = 10
test.loc[test['id'] == 6858, 'budget'] = 100
test.loc[test['id'] == 6876, 'budget'] = 250
test.loc[test['id'] == 6972, 'budget'] = 1
test.loc[test['id'] == 7079, 'budget'] = 8000000
test.loc[test['id'] == 7150, 'budget'] = 118
test.loc[test['id'] == 6506, 'budget'] = 118
test.loc[test['id'] == 7225, 'budget'] = 6
test.loc[test['id'] == 7231, 'budget'] = 85
test.loc[test['id'] == 5222, 'budget'] = 5
test.loc[test['id'] == 5322, 'budget'] = 90
test.loc[test['id'] == 5350, 'budget'] = 70
test.loc[test['id'] == 5378, 'budget'] = 10
test.loc[test['id'] == 5545, 'budget'] = 80
test.loc[test['id'] == 5810, 'budget'] = 8
test.loc[test['id'] == 5926, 'budget'] = 300
test.loc[test['id'] == 5927, 'budget'] = 4
test.loc[test['id'] == 5986, 'budget'] = 1
test.loc[test['id'] == 6053, 'budget'] = 20
test.loc[test['id'] == 6104, 'budget'] = 1
test.loc[test['id'] == 6130, 'budget'] = 30
test.loc[test['id'] == 6301, 'budget'] = 150
test.loc[test['id'] == 6276, 'budget'] = 100
test.loc[test['id'] == 6473, 'budget'] = 100
test.loc[test['id'] == 6842, 'budget'] = 30

test['revenue'] = np.nan

# features from https://www.kaggle.com/kamalchhirang/eda-simple-feature-engineering-external-data
train = pd.merge(train, pd.read_csv('data/prediction/TrainAdditionalFeatures.csv'), how='left', on=['imdb_id'])
test = pd.merge(test, pd.read_csv('data/prediction/TestAdditionalFeatures.csv'), how='left', on=['imdb_id'])

additionalTrainData = pd.read_csv('data/prediction//additionalTrainData.csv')
additionalTrainData['release_date'] = additionalTrainData['release_date'].astype('str')
additionalTrainData['release_date'] = additionalTrainData['release_date'].str.replace('-', '/')
train = pd.concat([train, additionalTrainData])

#train = pd.merge(train, additionalTrainData, how='left', on=['imdb_id'],axis=1)
print(train.columns)
print(train.shape)
train['revenue'] = np.log1p(train['revenue'])
y = train['revenue'].values

json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

for col in tqdm(json_cols + ['belongs_to_collection']) :
    train[col] = train[col].apply(lambda x : get_dictionary(x))
    test[col] = test[col].apply(lambda x : get_dictionary(x))
    
def get_json_dict(df) :
    global json_cols
    result = dict()
    for e_col in json_cols :
        d = dict()
        rows = df[e_col].values
        for row in rows :
            if row is None : continue
            for i in row :
                if i['name'] not in d :
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result

train_dict = get_json_dict(train)
test_dict = get_json_dict(test)

# remove cateogry with bias and low frequency
for col in json_cols :
    
    remove = []
    train_id = set(list(train_dict[col].keys()))
    test_id = set(list(test_dict[col].keys()))   
    
    remove += list(train_id - test_id) + list(test_id - train_id)
    for i in train_id.union(test_id) - set(remove) :
        if train_dict[col][i] < 10 or i == '' :
            remove += [i]
            
    for i in remove :
        if i in train_dict[col] :
            del train_dict[col][i]
        if i in test_dict[col] :
            del test_dict[col][i]
            
all_data = prepare(pd.concat([train, test]).reset_index(drop = True))
train = all_data.loc[:train.shape[0] - 1,:]
test = all_data.loc[train.shape[0]:,:] 


random_seed = 2019
k = 10
fold = list(KFold(k, shuffle = True, random_state = random_seed).split(train))
np.random.seed(random_seed)


def xgb_model(trn_x, trn_y, val_x, val_y, test, verbose) :
    
    params = {'objective': 'reg:linear', 
              'eta': 0.01, 
              'max_depth': 6, 
              'subsample': 0.6, 
              'colsample_bytree': 0.7,  
              'eval_metric': 'rmse', 
              'seed': random_seed, 
              'silent': True,
    }
    
    record = dict()
    model = xgb.train(params
                      , xgb.DMatrix(trn_x, trn_y)
                      , 100000
                      , [(xgb.DMatrix(trn_x, trn_y), 'train'), (xgb.DMatrix(val_x, val_y), 'valid')]
                      , verbose_eval=verbose
                      , early_stopping_rounds=500
                      , callbacks = [xgb.callback.record_evaluation(record)])
    best_idx = np.argmin(np.array(record['valid']['rmse']))

    val_pred = model.predict(xgb.DMatrix(val_x), ntree_limit=model.best_ntree_limit)
    test_pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)

    return {'val':val_pred, 'test':test_pred, 'error':record['valid']['rmse'][best_idx], 'importance':[i for k, i in model.get_score().items()]}

def lgb_model(trn_x, trn_y, val_x, val_y, test, verbose) :

    params = {'objective':'regression',
         'num_leaves' : 30,
         'min_data_in_leaf' : 20,
         'max_depth' : 9,
         'learning_rate': 0.004,
         #'min_child_samples':100,
         'feature_fraction':0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         'lambda_l1': 0.2,
         "bagging_seed": random_seed,
         "metric": 'rmse',
         #'subsample':.8, 
          #'colsample_bytree':.9,
         "random_state" : random_seed,
         "verbosity": -1}

    record = dict()
    model = lgb.train(params
                      , lgb.Dataset(trn_x, trn_y)
                      , num_boost_round = 100000
                      , valid_sets = [lgb.Dataset(val_x, val_y)]
                      , verbose_eval = verbose
                      , early_stopping_rounds = 500
                      , callbacks = [lgb.record_evaluation(record)]
                     )
    best_idx = np.argmin(np.array(record['valid_0']['rmse']))

    val_pred = model.predict(val_x, num_iteration = model.best_iteration)
    test_pred = model.predict(test, num_iteration = model.best_iteration)
    
    return {'val':val_pred, 'test':test_pred, 'error':record['valid_0']['rmse'][best_idx], 'importance':model.feature_importance('gain')}

def cat_model(trn_x, trn_y, val_x, val_y, test, verbose) :
    
    model = CatBoostRegressor(iterations=100000,
                                 learning_rate=0.004,
                                 depth=5,
                                 eval_metric='RMSE',
                                 colsample_bylevel=0.8,
                                 random_seed = random_seed,
                                 bagging_temperature = 0.2,
                                 metric_period = None,
                                 early_stopping_rounds=200
                                )
    model.fit(trn_x, trn_y,
                 eval_set=(val_x, val_y),
                 use_best_model=True,
                 verbose=False)
    
    val_pred = model.predict(val_x)
    test_pred = model.predict(test)
   
    print(model.get_scale_and_bias())
    #print(model)
    return {'val':val_pred, 
            'test':test_pred, 
            'error':model.get_best_score()['validation']['RMSE']}


result_dict = dict()
val_pred = np.zeros(train.shape[0])
test_pred = np.zeros(test.shape[0])
final_err = 0
verbose = False

for i, (trn, val) in enumerate(fold) :
    print(i+1, "fold.    RMSE")
    
    trn_x = train.loc[trn, :]
    trn_y = y[trn]
    val_x = train.loc[val, :]
    val_y = y[val]
    
    fold_val_pred = []
    fold_test_pred = []
    fold_err = []
    
    #""" xgboost
    '''
    start = datetime.now()
    result = xgb_model(trn_x, trn_y, val_x, val_y, test, verbose)
    fold_val_pred.append(result['val']*0.2)
    fold_test_pred.append(result['test']*0.2)
    fold_err.append(result['error'])
    print("xgb model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')
    #"""
    
    #""" lightgbm
    start = datetime.now()
    result = lgb_model(trn_x, trn_y, val_x, val_y, test, verbose)
    fold_val_pred.append(result['val']*0.4)
    fold_test_pred.append(result['test']*0.4)
    fold_err.append(result['error'])
    print("lgb model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')
    #"""
    '''
    #""" catboost model
    start = datetime.now()
    result = cat_model(trn_x, trn_y, val_x, val_y, test, verbose)
    fold_val_pred.append(result['val']*1.0)
    fold_test_pred.append(result['test']*1.0)
    fold_err.append(result['error'])
    print("cat model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')
    #"""
    
    # mix result of multiple models
    val_pred[val] += np.mean(np.array(fold_val_pred), axis = 0)
    #print(fold_test_pred)
    #print(fold_test_pred.shape)
    #print(fold_test_pred.columns)
    test_pred += np.mean(np.array(fold_test_pred), axis = 0) / k
    final_err += (sum(fold_err) / len(fold_err)) / k
    
    print("---------------------------")
    print("avg   err.", "{0:.5f}".format(sum(fold_err) / len(fold_err)))
    print("blend err.", "{0:.5f}".format(np.sqrt(np.mean((np.mean(np.array(fold_val_pred), axis = 0) - val_y)**2))))
    
    print('')
    
print("fianl avg   err.", final_err)
print("fianl blend err.", np.sqrt(np.mean((val_pred - y)**2)))


sub = pd.read_csv('data/prediction/sample_submission.csv')
df_sub = pd.DataFrame()
df_sub['id'] = sub['id']
df_sub['revenue'] = np.expm1(test_pred)
#df_sub['revenue'] = test_pred
#print(df_sub['revenue'])
df_sub.to_csv("submission.csv", index=False)