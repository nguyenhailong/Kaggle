import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
#from nltk.stem.porter import *
#stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer =  WordNetLemmatizer()

import re
#import enchant
import random
random.seed(2401)
import xgboost as xgb
runfile('spell_check_dict.py')

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
df_attr = pd.read_csv('../input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
num_train = df_train.shape[0]

stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':0}

def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        if s in spell_check_dict:
            s = spell_check_dict[s]
        
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) ##'desgruda' palavras que estão juntas
        
        s = s.replace("  "," ")
        s = re.sub(r"([0-9]),([0-9])", r"\1\2", s)
        s = s.replace(","," ")
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    
        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")
        
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air condition")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        #s = (" ").join([stemmer.stem(z) for z in s.lower().split(" ")])
        #s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s = (" ").join([wordnet_lemmatizer.lemmatize(z) for z in s.split(" ")])
        return s
    else:
        return "null"

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


#comment out the lines below use df_all.csv for further grid search testing
#if adding features consider any drops on the 'cust_regression_vals' class

#df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
#df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
#
#df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
#df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
#df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
#df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
#
#df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
#df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
#df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
#df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
#
#df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
#df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
#df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
#df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
#df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
#
#df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
#df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
#
#df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
#df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
#df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
#df_brand = pd.unique(df_all.brand.ravel())
#d={}
#i = 1
#for s in df_brand:
#    d[s]=i
#    i+=1
#df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
#df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))
#df_all.to_csv('../input/df_all_spell_check.csv')


df_all = pd.read_csv('../input/df_all_spell_check.csv', encoding="ISO-8859-1", index_col=0)

# Extra feature engineering
df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))

df_all['query_1st_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[1]))
df_all['query_1st_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[2]))

puid_count = np.asmatrix(np.unique(df_all['product_uid'],return_counts=True))
puid_count = np.transpose(puid_count)
puid_count = pd.DataFrame(puid_count,columns=['product_uid','count'])
df_all = pd.merge(df_all, puid_count, how='left', on='product_uid')

df_all['jc_ratio_title'] = df_all['word_in_title']/(df_all['len_of_query'] + df_all['len_of_title']-df_all['word_in_title'])
df_all['jc_ratio_description'] = df_all['word_in_description']/(df_all['len_of_query']+ df_all['len_of_description'] - df_all['word_in_description'])

#df_all['brand_null'] = df_all['brand']=='null'
#df_all['brand_NA'] = (df_all['brand']==' na') | (df_all['brand']=='na')
#df_all['unbranded'] = df_all['brand']=='unbranded'

#attr_product_uid = np.unique(df_attr['product_uid'])
#df_all['product_has_attribute'] = [ uid in attr_product_uid for uid in df_all['product_uid']]

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]

#Clean memory
del df_all,df_attr,df_brand,df_pro_desc, puid_count
import gc
gc.collect()

id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train[:]
X_test = df_test[:]
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))

start_time = time.time()
#rfr = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2401, verbose = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
countVect = CountVectorizer()
tsvd = TruncatedSVD(n_components=100, random_state = 2016)


def gen_txt_features(X_train):
    cst = cust_regression_vals().fit_transform(X_train)
    ##
    s1 = cust_txt_col(key='search_term').fit_transform(X_train)
    tfidf1 = tfidf.fit_transform(s1)
    txt1 = tsvd.fit_transform(tfidf1)
    
    s2 = cust_txt_col(key='product_title').fit_transform(X_train)
    tfidf2 = tfidf.fit_transform(s2)
    txt2 = tsvd.fit_transform(tfidf2)
    
    s3 = cust_txt_col(key='product_description').fit_transform(X_train)
    tfidf3 = tfidf.fit_transform(s3)
    txt3 = tsvd.fit_transform(tfidf3)
    
    s4 = cust_txt_col(key='brand').fit_transform(X_train)
    tfidf4 = tfidf.fit_transform(s4)
    txt4 = tsvd.fit_transform(tfidf4)
    ##  
    s1 = cust_txt_col(key='search_term').fit_transform(X_train)
    countVect1 = countVect.fit_transform(s1)
    cnt1 = tsvd.fit_transform(countVect1)
    
    s2 = cust_txt_col(key='product_title').fit_transform(X_train)
    countVect2 = countVect.fit_transform(s2)
    cnt2 = tsvd.fit_transform(countVect2)
    
    s3 = cust_txt_col(key='product_description').fit_transform(X_train)
    countVect3 = countVect.fit_transform(s3)
    cnt3 = tsvd.fit_transform(countVect3)
    
    s4 = cust_txt_col(key='brand').fit_transform(X_train)
    countVect4 = countVect.fit_transform(s4)
    cnt4 = tsvd.fit_transform(countVect4)
    
    feature_data = np.concatenate((cst,txt1*0.5,txt2*0.25,txt3*0.5,txt4*0.5,cnt1*0.5,cnt2*0.25,cnt3*0.5,cnt4*0.5),axis=1)
    return (feature_data)
#param_grid = {'rfr__max_features': [15], 'rfr__max_depth': [20]}

train_fea = gen_txt_features(X_train)
test_fea = gen_txt_features(X_test)

np.savetxt("../input/train_fea.csv", train_fea, delimiter=",")
np.savetxt("../input/test_fea.csv", test_fea, delimiter=",")
np.savetxt("../input/y_train.csv",y_train)


clf = xgb.XGBRegressor(silent = False)#max_depth=6, subsample=0.8, n_estimators= 1000, learning_rate= 0.03, silent = False)

param_grid = {#'model__min_child_weight': [10],
                  'subsample': [0.8],
                  'max_depth': [5,6],
                  'learning_rate': [0.03,0.02,0.01],
                  'n_estimators': [1000,2000]
                  }
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
#model.fit(X_train, y_train)
model.fit(train_fea, y_train)


print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(test_fea)
y_pred[y_pred > 3] = 3
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../submissions/test_'+ str(-model.best_score_) + '_xgb.csv',index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))

#------ Draft & Testing
#xgb.plot_importance(model)
#
#vectorizer = TfidfVectorizer(  ngram_range=(1, 1), stop_words='english', encoding="ISO-8859-1")
#countVect = CountVectorizer()
#x = countVect.fit_transform(df_all['search_term'])
#
#tsvd = TruncatedSVD(n_components=50, random_state = 2016)
#temp = vectorizer.fit_transform(df_all['search_term'])
#a1 =tsvd.fit_transform(temp)
#temp2 = vectorizer.fit_transform(df_all['product_title'])
#temp3= vectorizer.fit_transform(df_all['product_description'])
#temp4 = vectorizer.fit_transform(df_all['brand'])