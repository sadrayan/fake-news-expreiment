
# coding: utf-8

# In[1]:


"""Defines a Keras model and input function for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import numpy as np
from tqdm import tqdm
import re

import tensorflow as tf
from tensorflow.keras import backend as K

#load embeddings
import os, re, csv, math, codecs
from tensorflow import keras
from tensorflow.keras import layers


# Initialize session
sess = tf.Session()

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)


# In[2]:


#training params
batch_size = 256 
num_epochs = 4 

#model parameters
num_filters = 64 
embed_dim = 300 
weight_decay = 1e-4


# In[6]:



def input_fn(features, labels, shuffle, num_epochs, batch_size):
    """Generates an input function to be used for model training.
    Args:
    features: numpy array of features used for training or inference
    labels: numpy array of labels for each example
    shuffle: boolean for whether to shuffle the data or not (set True for
      training, False for evaluation)
    num_epochs: number of epochs to provide the data for
    batch_size: batch size for training
    Returns:
    A tf.data.Dataset that can provide data to the Keras model for training or
      evaluation
    """
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


# In[4]:


def get_embedding():
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open('/home/sonic/.keras/datasets/wiki-news-300d-1M.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))

    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []

    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("sample words not found: ", np.random.choice(words_not_found, 10))
    return embedding_matrix


# In[5]:



def create_keras_model(nb_words, embedding_matrix,  embed_dim = 300):
    """Creates Keras Model for  Classification.
    Args:
    nb_words: How many features the input has
    embed_dim: embedding dimension = 300
    embedding_matrix: embedding weights
    Returns:
    The compiled Keras model (still needs to be trained)
    """

    claim_input   = keras.Input(shape=(None,), name='claim')    # Variable-length sequence of ints
    article_input = keras.Input(shape=(None,), name='article')  # Variable-length sequence of ints

    # Embed each word into vector
    claim_features   = layers.Embedding(nb_words, embed_dim, weights=[embedding_matrix], 
                                      input_length=max_seq_len, trainable=False, name='claim_emb')(claim_input)
    # Embed each word into vector
    article_features = layers.Embedding(nb_words, embed_dim, weights=[embedding_matrix], 
                                     input_length=max_seq_len, trainable=False, name='article_emb')(article_input)

    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    claim_features = layers.LSTM(128)(claim_features)
    # Reduce sequence of embedded words in the body into a single 32-dimensional vector
    article_features = layers.LSTM(128)(article_features)

    # Merge all available features into a single large vector via concatenation
    x = layers.concatenate([claim_features, article_features], name='concat-layer')

    # Stick a logistic regression for priority prediction on top of the features
    priority_pred = layers.Dense(num_classes, activation='softmax', name='priority')(x)
    # Stick a department classifier on top of the features
    # department_pred = layers.Dense(num_departments, activation='softmax', name='department')(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(inputs=[claim_input, article_input], outputs=[priority_pred], name='fake-model')

    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    keras.utils.plot_model(model, 'multi_input_model.png', show_shapes=True)

    return model

