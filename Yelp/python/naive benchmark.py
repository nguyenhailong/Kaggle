#%matplotlib inline

# Sample script naive benchmark that yields 0.609 public LB score WITHOUT any image information

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in data, files are assumed to be in the "../input/" directory.
train = pd.read_csv('../input/train.csv')
train_biz_id = pd.read_csv('../input/train_photo_to_biz_ids.csv')
test_biz_id = pd.read_csv('../input/test_photo_to_biz.csv')
submit = pd.read_csv('../submissions/sample_submission.csv')

biz_id_train = np.unique(train_biz_id['business_id'])
biz_id_test = np.unique(test_biz_id['business_id'])
# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1 if str(i) in str(s).split(' ') else 0 for i in range(9)]))
Y = train['labels'].apply(to_bool)

# get means proportion of each class
py = Y.mean()
plt.bar(Y.columns,py,color='steelblue',edgecolor='white')

# predict classes that are > 0.5, 2,3,5,6,8
# try using only six labels instead of five
submit['labels'] = '1 2 3 5 6 8'
for i in range(len(submit)):
    submit.at[i,'labels'] = submit.at[i,'labels'] + ' '+ str(np.random.choice([0,4,7],1,replace=False)).strip('[]')

submit.to_csv('../submissions/naive.csv',index=False)
