#importing necessary modules
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from tensorflow.keras.layers import Dropout, Embedding
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from tensorflow.keras.models import Model, load_model
import tensorflow

import pandas as pd
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from numpy import array
from numpy import asarray
from numpy import zeros
import pickle


def process_text(input_string):
    '''
    returns output list where 1-5 corresponds to [tops, trousers, outerwear, dresses, skirts]

    sample input: process_text(input_string = "The benign king gazes beneath the whore's blouse")
    sample output:[1]
    '''
    #take a look at the data
    df = pd.read_csv('utils/NLP_files/TIL_NLP_train_dataset.csv')
    X = list(df["word_representation"])


    mod_X = []
    for sen in X:
        tmp = sen.split(" ")
        ans = [x for x in tmp]
        mod_sen = (" ").join(ans)
        mod_X.append(mod_sen)

    df_val = pd.read_csv('utils/NLP_files/TIL_NLP_test_dataset.csv')
    df_val.head(10)
    X_val=df_val['word_representation'].values

    #tokenizing sequence
    tokenizer = Tokenizer(num_words=4250)
    tokenizer.fit_on_texts(mod_X)
    
    mod_X = tokenizer.texts_to_sequences(mod_X)
    X_val = tokenizer.texts_to_sequences(X_val)
    vocab_size = len(tokenizer.word_index) + 1
    pre_pad_X = mod_X
    pre_pad_X_val = X_val
    mod_X = pad_sequences(mod_X, padding='post', maxlen=24)
    X_val = pad_sequences(X_val, padding='post', maxlen=24)

    all_lens = []

    for i in range(len(pre_pad_X)):
        all_lens.append(len(pre_pad_X[i]))

    encoded_words = pd.read_pickle("utils/NLP_files/encoded_words.pkl")

    #encode words from the dictionary
    tmp = input_string.split(" ")
    ans = [encoded_words[x] for x in tmp if x in encoded_words]
    res = (" ").join(ans)

    #tokenize and pad
    token_string = tokenizer.texts_to_sequences([res])
    padded_token_str = pad_sequences(token_string, padding='post', maxlen=24)
    # print(padded_token_str)

    #load trained model
    #assign predicted values, rounding up and down respectively
    model = load_model('utils/NLP_files/my_model.h5')
    preds_val = model.predict(padded_token_str)
    preds_val[preds_val>=0.5] = 1
    preds_val[preds_val<0.5] = 0


    # print(preds_val)
    res = [i + 1 for i in range(len(preds_val[0])) if preds_val[0][i] == 1]
    # print(res)
    # desired order for CV is [tops, trousers, outerwear, dresses, skirts]
    res2 = []
    for i in res:
        if i == 1:
            res2.append(3)
        elif i == 2:
            res2.append(1)
        elif i == 3:
            res2.append(2)
        else:
            res2.append(i)
    # print(res2)
    # preds_val_df=pd.DataFrame(data=preds_val, columns = ["outwear","top","trousers","women dresses","women skirts"])

    return res2
    
print(process_text(input_string = "The benign king gazes beneath the whore's blouse"))