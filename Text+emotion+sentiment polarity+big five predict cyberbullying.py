# -*- coding: utf-8 -*-
"""
Created on Sun May  2 05:50:29 2021

@author: raybeam
"""
from numpy import array
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers import Concatenate
from tensorflow.keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


data_sets = pd.read_csv("C:/Users/raybeam/Desktop/wiki_data_new_senti.csv")
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
text_remove = "@\S+|https?:\S+|http?:\S|[^A-Za-z]+"
def preprocess(text, lemmatize=True):
    # Remove link,user and special characters
    text = re.sub(text_remove, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if lemmatize:
                tokens.append(lemmatizer.lemmatize(token))
            else:
                tokens.append(token)
    return " ".join(tokens)
data_sets.comment_text = data_sets.comment_text.apply(lambda x: preprocess(x))
y = data_sets['target']
X = data_sets.drop(['target'],axis=1)
data_sets.comment_text
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X1_train=X_train["comment_text"]
X1_train.shape
X1_test=X_test["comment_text"]
X1_test.shape
X1_train=X1_train.astype("str")
X1_test=X1_test.astype("str")
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
title_tokeniser = Tokenizer(num_words=35276)
title_tokeniser.fit_on_texts(X1_train)
X1_train = tokenizer.texts_to_sequences(X1_train)
maxlen = 2500
X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
from tensorflow.keras.preprocessing.text import Tokenizer
title_tokeniser = Tokenizer(num_words=35276)
title_tokeniser.fit_on_texts(X1_test)
X1_test = tokenizer.texts_to_sequences(X1_test)
maxlen = 2500
X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)
word_dict_insult = {}
for sentence_insult in data_sets.comment_text:
    for word in sentence_insult.split(" "):
        if word not in word_dict_insult:
            word_dict_insult[word] = True
max_sentence_length_insult = 0
for sentence_insult in data_sets.comment_text:
    sentence_insult = sentence_insult.split(" ")
    sentence_len_insult = len(sentence_insult)
    if sentence_len_insult > max_sentence_length_insult:
        max_sentence_length_insult = sentence_len_insult

print("len of total features: {}".format(len(word_dict_insult)))
print("max sentence len: {}".format(max_sentence_length_insult))
from keras.preprocessing.text import text_to_word_sequence
tokenized_data_insult = []  
for sentence_insult in data_sets.comment_text:
    tokens_insult = text_to_word_sequence(sentence_insult)
    tokenized_data_insult.append(tokens_insult)    
vocab_size_insult = len(word_dict_insult)
max_text_length_insult = max_sentence_length_insult
x_tokenizer_insult = text.Tokenizer(num_words=35276)
x_tokenizer_insult.fit_on_texts(tokenized_data_insult)
x_tokenized_insult = x_tokenizer_insult.texts_to_sequences(tokenized_data_insult)
x_train_data_insult = sequence.pad_sequences(x_tokenized_insult, maxlen=max_text_length_insult)    
print(max_text_length_insult)
#get word index vocab
word_index = x_tokenizer_insult.word_index
#Building word2vec model
from gensim.models import Word2Vec
embed_dim=250
model_insult=Word2Vec(tokenized_data_insult,min_count=3,
                 window=5,
                 size=embed_dim, 
                 sg=1,
                 workers=4)
words=list(model_insult.wv.vocab)
print("Total vocabulary size:",len(words))

#Pushing the vector values of each word into a file for creating an embedding matrix below.
filename="word2vec_embeddings.txt"
model_insult.wv.save_word2vec_format(filename, binary=False)
#Building word2vec model
from gensim.models import Word2Vec
embed_dim=250
vocab_size = 35276
tokenizer = Tokenizer()
embeddings_index={}
f=open(os.path.join('word2vec_embeddings.txt'),encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
embedding_matrix = np.zeros((vocab_size,embed_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(vocab_size)


X1_train.shape
X2_train = X_train[['O', 'C', 'E','A','N','Anger','Disgust','Fear','Joy','Sadness','senti_score']].values
X2_test = X_test[['O', 'C', 'E','A','N','Anger','Disgust','Fear','Joy','Sadness','senti_score']].values

from tensorflow.keras import layers
### Models
### Models
# Specify wide model
############

input_1 = layers.Input(shape=(maxlen,))
input_2 = layers.Input(shape=(11)) #3個column, useful funny cool
#Create the first submodel that accepts data from first input layer
embedding_layer = layers.Embedding(input_dim=vocab_size,output_dim=embed_dim,
                            weights=[embedding_matrix], trainable=False,
                            mask_zero=True)(input_1)
BiLSTM_Layer_1 = layers.Bidirectional(layers.LSTM(128))(embedding_layer)

dense_layer_1 = layers.Dense(10, activation='relu')(input_2)
dense_layer_2 = layers.Dense(10, activation='relu')(dense_layer_1)

concat_layer =  layers.concatenate([BiLSTM_Layer_1, dense_layer_2])
layer_drop = layers.Dropout(0.5)(concat_layer) 
dense_layer_3 = layers.Dense(64, activation='relu', kernel_initializer=keras.initializers.HeNormal(seed=None))(layer_drop)
output = layers.Dense(1, activation='sigmoid')(dense_layer_3)
model = keras.Model(inputs=[input_1, input_2], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam',
                   metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC()])
model.summary()



keras.utils.plot_model(model,show_shapes=True)



history = model.fit(x=[X1_train,X2_train], y=y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)


score = model.evaluate(x=[X1_test, X2_test], y=y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
print("Test Precision:", score[2])
print("Test Recall:", score[3])
print("Test AUC:", score[4])
