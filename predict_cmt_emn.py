import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.models import load_model
model=load_model(r'text_base.keras')

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


while True:
    comment=[input('enter a comment')]
    comment_seqence=tokenizer.texts_to_sequences(comment)
    paded_comment=pad_sequences(comment_seqence,maxlen=300,padding='post')

    res=model.predict(paded_comment)
    res=np.argmax(res)
    print(res)

    maps={ 0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear',5:'surprise'}
    print(f'detected emotion is {maps[res]}')

