# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 23:31:14 2016

@author: long
"""

import numpy as np
import pandas as pd

sub_list = [#'../submissions/sub0.471688809587rfr.csv', 
            '../submissions/test_0.462414522323_xgb.csv',
            '../submissions/test_0.46340574982_xgb.csv', 
            '../submissions/test_0.462827443648_xgb.csv',
            #'../submissions/test_0.478819452685_ExTree.csv'
            '../submissions/test_0.453887_Long_xgb_features822_depth8_eta0.02_round602_sub0.83_col0.77.csv',
            '../submissions/test_0.454297_Long_xgb_features822_depth6_eta0.02_round1180_sub0.83_col0.77.csv',
            '../submissions/test_0.464334465068_xgb_norvig_correction.csv',
            '../submissions/test_0.462414522323_xgb.csv',
            '../submissions/test_0.462012709009_xgb.csv'
            ]
sub1 = pd.read_csv(sub_list[0])
for i in range(1,len(sub_list)):
    sub2 = pd.read_csv(sub_list[i])
    sub1['relevance'] = sub1['relevance'] + sub2['relevance']

sub1['relevance'] = sub1['relevance']/len(sub_list)
sub1.to_csv('../submissions/ensemble_7.csv',index=False)
