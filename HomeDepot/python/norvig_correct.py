# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 08:22:19 2016

@author: long
"""


#text_file = open("../input/big_prod_desc.txt", "w")
#for desc in df_pro_desc['product_description']:
#    text_file.write(desc + '\n')
#text_file.close()

import re, collections

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

big_text = ''.join(df_all['product_description'])
#NWORDS = train(words(open('../input/big_prod_desc.txt').read()))
NWORDS = train(words(big_text))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def norvig_correct(word):
    if len(word)>2:
        candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
        return max(candidates, key=NWORDS.get)
    else:
        return word