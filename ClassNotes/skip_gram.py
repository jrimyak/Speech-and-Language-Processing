'''
Predict the context or the neighbors of a word

he context of a word can be representedd through a set of skip-gram pairs of (target_word, context_word) where contextent word appears in the neighboring contect of target_word


Example 

The wide road shimmered in the hot sun
8 words - length 
for each of the context words are defined by the window size (usually 2 or 3)
the window size determines the span of words on either side
WINDOW SIZE         Text                                SKip-grams                                  
2           The wide road shimmered in the hot sun.     (wide, the), (wide, road), (wide, shimmered)
2           The wide road shimmered in the hot sun.     (shimmered, wide), (shimmered, road), (shimmered, in), (shimmered, the)
2           The wide road shimmered in the hot sun.     (sun, the), (sun, hot)

Window size of 3 is the same idea... (window size shouldd be fixed)

objecct fucntion of the skip gram model is to maximize the probability of predicting context wordds given the target wordd. for a sequence of words w1, w2, ... wT, the objecctive can be written as the average log probability 
v and v' are the target andd context vector repsresntation of wrods and W is the vocabulary size
1/T sum {t=1, T} sum {-c < j < c; j!=0} log(w{t+j} | w{t})
where c is the size of the training contextt. the basic skip gram formulation defines this probabiblity using the softmax function 
observing wI in the context of wO
p(wO, wI) = exp(v'_wO^T v_wI) / sum { w=1, w=W} exp(v'_W^T v_wI) 
Problem W can be really large which makes this computationally hard

the noise contrastive estimation loss funcction is an efficent approximation for a full softmax. the NCE loss and be called negative sampling 

what is a negative context word for wide? "in the hot sun" doesn't appear in window size hence its negative context words
ex:
when window size = 2
(hot, shimmered)
(wide, hot)
(wide, sun)

'''

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

