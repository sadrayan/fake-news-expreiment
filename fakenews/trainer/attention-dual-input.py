
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm
import re

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import tensorflow as tf
from tensorflow.keras import backend as K

# Initialize session
sess = tf.Session()

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)



# In[2]:



labels = {0:'false', 1:'partly true', 2:'true'}


# In[4]:



train_df = pd.read_csv('../data/full_train.cvs')
train_df = train_df.sample(frac=.25).reset_index(drop=True)

train_df.tail()


# In[5]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


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

X_train.tail()


# In[7]:


from nltk.corpus import stopwords as sw
from keras.preprocessing.text import Tokenizer
import string
from keras.preprocessing.sequence import pad_sequences


MAX_NB_WORDS = 100000
max_seq_len = 300
label_names = ["true", "almost", "false"]

num_classes = len(label_names)

print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)

tokenizer.fit_on_texts(X_train['claim'].tolist() + X_train['article'].tolist() )

train_claim_seq   = tokenizer.texts_to_sequences(X_train['claim'].tolist())
train_article_seq = tokenizer.texts_to_sequences(X_train['article'].tolist())

test_claim_seq   = tokenizer.texts_to_sequences(X_test['claim'].tolist())
text_article_seq = tokenizer.texts_to_sequences(X_test['article'].tolist())

word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

#pad sequences
train_claim_seq   = pad_sequences(train_claim_seq, maxlen=max_seq_len)
train_article_seq = pad_sequences(train_article_seq, maxlen=max_seq_len)

test_claim_seq    = pad_sequences(test_claim_seq, maxlen=max_seq_len)
test_article_seq  = pad_sequences(text_article_seq, maxlen=max_seq_len)


# In[8]:


#training params
batch_size = 256 
num_epochs = 4 

#model parameters
num_filters = 64 
embed_dim = 300 
weight_decay = 1e-4


# In[9]:


#load embeddings
import os, re, csv, math, codecs

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


# In[10]:


from tensorflow import keras
from tensorflow.keras import layers


# num_tags = 12  # Number of unique issue tags
# num_words = 10000  # Size of vocabulary obtained when preprocessing text data
# num_departments = 4  # Number of departments for predictions

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

model.summary()


# In[11]:


#model training

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)
    
# Instantiate variables
initialize_vars(sess)


hist = model.fit({'claim': train_claim_seq, 'article': train_article_seq},
          y_train,
          epochs=num_epochs, validation_split=0.1,verbose=1,
          batch_size=batch_size)


# In[12]:


def plot_history(hist):
    #generate plots
    plt.figure()
    plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
    plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(loc='upper right')
    plt.show()

#     plt.figure()
#     plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')
#     plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(loc='upper left')
#     plt.show()

plot_history(hist)


# In[13]:


y_pred = model.predict({'claim': test_claim_seq, 'article': test_article_seq})
y_pred


# In[14]:


# Results
from sklearn.metrics import f1_score, classification_report

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Results
print('sklearn Macro-F1-Score:',      f1_score(y_test, y_pred, average='macro'))
print('sklearn Micro-F1-Score:',      f1_score(y_test, y_pred, average='micro'))  
print('sklearn weighted-F1-Score:',   f1_score(y_test, y_pred, average='weighted'))  
print('sklearn no average-F1-Score:', f1_score(y_test, y_pred, average=None))

print(classification_report(y_test, y_pred, target_names=labels.values()))


# # Gcloud

# In[15]:


# Export the model to a local SavedModel directory 
export_path = tf.contrib.saved_model.save_keras_model(model, 'keras_export')
print("Model exported to: ", export_path)


# In[16]:


# GCloud
get_ipython().system(u' export GOOGLE_APPLICATION_CREDENTIALS="/home/sonic/leadersprize/fakenews-40cea3fac9e2.json"')
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/sonic/leadersprize/fakenews-40cea3fac9e2.json'
# %env GOOGLE_APPLICATION_CREDENTIALS '/home/sonic/leadersprize/fakenews-40cea3fac9e2.json'

PROJECT_ID = "fakenews-259222" #@param {type:"string"}
BUCKET_NAME =  PROJECT_ID + "-model" #@param {type:"string"}
REGION = "us-central1" #@param {type:"string"

JOB_NAME = 'my_first_keras_job'
JOB_DIR = 'gs://' + BUCKET_NAME + '/keras-job-dir'

get_ipython().system(u' gcloud config set project $PROJECT_ID')
# Explicitly tell `gcloud ai-platform local train` to use Python 3 
get_ipython().system(u' gcloud config set ml_engine/local_python $(which python3)')


# In[17]:


# Only if your bucket doesn't already exist: Run the following cell to create your Cloud Storage bucket.
get_ipython().system(u' gsutil mb -l $REGION gs://$BUCKET_NAME')
# ! gsutil acl set -R project-private gs://$BUCKET_NAME
get_ipython().system(u' gsutil ls -al gs://$BUCKET_NAME')


# In[19]:


get_ipython().run_cell_magic(u'time', u'', u'\n# Export the model to a SavedModel directory in Cloud Storage\nexport_path = tf.contrib.saved_model.save_keras_model(model, JOB_DIR + \'/keras_export\')\nprint("Model exported to: ", export_path)')


# In[22]:


MODEL_NAME = "my_first_keras_model"
MODEL_VERSION = "v1"

get_ipython().system(u' gcloud ai-platform models create $MODEL_NAME   --regions $REGION')


# In[ ]:


# Get a list of directories in the `keras_export` parent directory
KERAS_EXPORT_DIRS = get_ipython().getoutput(u'gsutil ls')
print(KERAS_EXPORT_DIRS)


# In[23]:


# Get a list of directories in the `keras_export` parent directory
KERAS_EXPORT_DIRS = export_path.decode("utf-8") 
print (KERAS_EXPORT_DIRS)

# Pick the directory with the latest tiAmestamp, in case you've trained
# multiple times
SAVED_MODEL_PATH = KERAS_EXPORT_DIRS

print('saved model path', SAVED_MODEL_PATH)

# Create model version based on that SavedModel directory
get_ipython().system(u' gcloud ai-platform versions create $MODEL_VERSION   --model $MODEL_NAME   --runtime-version 1.13   --python-version 3.5   --framework tensorflow   --origin $SAVED_MODEL_PATH')


# In[58]:


import json
prediction_json = list(zip(test_claim_seq[:2].tolist(), test_article_seq[:2].tolist()))
prediction_json = list(zip(test_claim_seq[:2].tolist(), test_article_seq[:2].tolist()))

# prediction_json = []
# prediction_df = pd.DataFrame(prediction_json)
# prediction_df.to_json('prediction_input.json')

# with open('prediction_input.json', 'w') as json_file:
#     for row in prediction_df.values.tolist():
#         json.dump(row, json_file)
#         json_file.write('\n')

# with open("prediction_input.json", "w") as write_file:
#     json.dump(prediction_json, write_file)


# predict_instance_json = "prediction_input.json"

# with open(predict_instance_json, "wb") as fp:
#         fp.write(json.dumps(prediction_json).encode())
    
i = 1
with open('prediction_input.json', 'w') as json_file:
    for row in prediction_json:
        json.dump({"values": row, "key": i}, json_file)
        json_file.write('\n')
        i +=1
    
get_ipython().system(u' cat prediction_input.json')


# In[61]:


get_ipython().system(u' gcloud ai-platform predict   --model $MODEL_NAME   --version $MODEL_VERSION   --json-instances prediction_input.json')


# In[60]:


get_ipython().system(u' gcloud components update')


# # Cleanup

# In[62]:



# Delete bucket
get_ipython().system(u' gsutil rm -r gs://$BUCKET_NAME')
    
# Delete model version resource
get_ipython().system(u' gcloud ai-platform versions delete $MODEL_VERSION --quiet --model $MODEL_NAME ')

# Delete model resource
get_ipython().system(u' gcloud ai-platform models delete $MODEL_NAME --quiet')

# Delete Cloud Storage objects that were created
get_ipython().system(u' gsutil -m rm -r $JOB_DIR')

# If the training job is still running, cancel it
get_ipython().system(u' gcloud ai-platform jobs cancel $JOB_NAME --quiet --verbosity critical')

