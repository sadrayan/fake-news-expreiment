
# coding: utf-8

# In[5]:


"""Utilities to download and preprocess the Census data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tempfile
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


import numpy as np
import pandas as pd
import tensorflow as tf


# In[8]:


def download (BUCKET_NAME):
        # Storage directory
    import tempfile
    DATA_DIR = os.path.join(tempfile.gettempdir(), 'fakenews_data')

    # Download options.
    # https://storage.cloud.google.com/[BUCKET_NAME]/[OBJECT_NAME]
    DATA_URL = ('gs://%s/data' % BUCKET_NAME)
    TRAINING_FILE = 'full_train.cvs'
    TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)

    print(TRAINING_URL)

    if (tf.io.gfile.exists(TRAINING_URL)):
        print(TRAINING_URL)

#     tf.io.gfile.copy(
#         TRAINING_URL,
#         'data/',
#         overwrite=True
#     )

#     tf.gfile.GFile.r

    with tf.gfile.GFile(TRAINING_URL, 'rb') as file:
        train_df = pd.read_csv(StringIO( file.read().decode("utf-8") ))
        
    return train_df

def load_data(BUCKET_NAME):
    train_df = download(BUCKET_NAME)

    train_df = train_df.sample(frac=.10).reset_index(drop=True)

    X = train_df[['claim', 'article']]
    y = train_df['label']

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print("num train: ", X_train.shape)
    print("num test: ", y_train.shape)
    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test = load_data(BUCKET_NAME)
    
# print(X_train.shape)
# print(y_train.shape)
# X_train.tail()


# In[3]:


# GCloud
# ! export GOOGLE_APPLICATION_CREDENTIALS="/home/sonic/leadersprize/fakenews-40cea3fac9e2.json"
# # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/sonic/leadersprize/fakenews-40cea3fac9e2.json'
# # %env GOOGLE_APPLICATION_CREDENTIALS '/home/sonic/leadersprize/fakenews-40cea3fac9e2.json'

# PROJECT_ID = "fakenews-259222" #@param {type:"string"}
# BUCKET_NAME =  PROJECT_ID + "-model" #@param {type:"string"}
# REGION = "us-central1" #@param {type:"string"

# # Only if your bucket doesn't already exist: Run the following cell to create your Cloud Storage bucket.
# ! gsutil mb -l $REGION gs://$BUCKET_NAME
# # ! gsutil acl set -R project-private gs://$BUCKET_NAME
# ! gsutil ls -al gs://$BUCKET_NAME
# # Update datasets
# ! gsutil -m cp -r ../data/full_train.cvs gs://$BUCKET_NAME/data/

