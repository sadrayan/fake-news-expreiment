
# coding: utf-8

# In[4]:


"""Utilities to download and preprocess the FakeNews data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf


# In[ ]:


#! ipython nbconvert --to=python config_template.ipynb


# In[ ]:




# Storage directory
DATA_DIR = os.path.join(tempfile.gettempdir(), 'fakenews_data')

# Download options.
DATA_URL = ('https://storage.googleapis.com/cloud-samples-data/ai-platform/census/data')
TRAINING_FILE = 'adult.data.csv'
EVAL_FILE = 'adult.test.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)


def download(data_dir):
    """Downloads census data if it is not already present.

    Args:
    data_dir: directory where we will access/save the census data
    """
    tf.gfile.MakeDirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.gfile.Exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.gfile.Exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)

    return training_file_path, eval_file_path


def preprocess(dataframe):
  """Converts categorical features to numeric. Removes unused columns.

  Args:
    dataframe: Pandas dataframe with raw data

  Returns:
    Dataframe with preprocessed data
  """
  dataframe = dataframe.drop(columns=UNUSED_COLUMNS)

  # Convert integer valued (numeric) columns to floating point
  numeric_columns = dataframe.select_dtypes(['int64']).columns
  dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')

  # Convert categorical columns to numeric
  cat_columns = dataframe.select_dtypes(['object']).columns
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.astype(
    _CATEGORICAL_TYPES[x.name]))
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
  return dataframe



def load_data():
  """Loads data into preprocessed (train_x, train_y, eval_y, eval_y) dataframes.

  Returns:
    A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
    Pandas dataframes with features for training and train_y and eval_y are
    numpy arrays with the corresponding labels.
  """
  # Download Census dataset: Training and eval csv files.
  training_file_path, eval_file_path = download(DATA_DIR)

  # This census data uses the value '?' for missing entries. We use na_values to
  # find ? and set it to NaN.
  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
  train_df = pd.read_csv(training_file_path, names=_CSV_COLUMNS, na_values='?')
  eval_df = pd.read_csv(eval_file_path, names=_CSV_COLUMNS, na_values='?')

  train_df = preprocess(train_df)
  eval_df = preprocess(eval_df)

  # Split train and eval data with labels. The pop method copies and removes
  # the label column from the dataframe.
  train_x, train_y = train_df, train_df.pop(_LABEL_COLUMN)
  eval_x, eval_y = eval_df, eval_df.pop(_LABEL_COLUMN)

  # Join train_x and eval_x to normalize on overall means and standard
  # deviations. Then separate them again.
  all_x = pd.concat([train_x, eval_x], keys=['train', 'eval'])
  all_x = standardize(all_x)
  train_x, eval_x = all_x.xs('train'), all_x.xs('eval')

  # Reshape label columns for use with tf.data.Dataset
  train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
  eval_y = np.asarray(eval_y).astype('float32').reshape((-1, 1))

  return train_x, train_y, eval_x, eval_y


# In[2]:


import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K

# Initialize session
sess = tf.Session()


# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


# # Data
# 
# First, we load the sample data IMDB data

# In[3]:


def download ():
        # Storage directory
    import tempfile
    DATA_DIR = os.path.join(tempfile.gettempdir(), 'fakenews_data')

    # Download options.
    # https://storage.cloud.google.com/[BUCKET_NAME]/[OBJECT_NAME]
    DATA_URL = ('gs://%s/data' % BUCKET_NAME)
    TRAINING_FILE = 'train.json'
    TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)

    print(TRAINING_URL)

    if (tf.io.gfile.exists(TRAINING_URL)):
        print(TRAINING_URL)

    tf.io.gfile.copy(
        TRAINING_URL,
        'datadd/',
        overwrite=True
    )

    with tf.io.gfile.GFile(TRAINING_URL, "rb") as file:
        train_df = pd.read_json (file.read())
        
    return train_df

def get_train_test_def():
    train_df = download()

    train_df = train_df.sample(frac=.10).reset_index(drop=True)

    labels = {0:'false', 1:'partly true', 2:'true'}

    def label(x):
        return labels[x]

    train_df['labelCode'] = train_df.label.apply(label)

    print(train_df.labelCode.value_counts())
    train_df.labelCode.value_counts().plot(kind='bar')

    train_df.rename(columns={"claim": "sentence", "label": "polarity"}, inplace=True)

    train_df.shape
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(train_df, test_size = 0.2, random_state = 0)
    return train_df, test_df

train_df, test_df = get_train_test_def()
    
print(train_df.shape)
print(test_df.shape)
train_df.tail()


# In[3]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

max_seq_length = 256

# encode class values as integers
encoder = LabelEncoder()

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()


train_label = train_df['polarity'].tolist()
encoder.fit(train_label)

    
def 

    # Create datasets (Only take up to max_seq_length words for memory)
    train_text = train_df['sentence'].tolist()
    train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_df['polarity'].tolist()
    print(train_text[0], train_text.shape)
    print(train_label[0])

    train_label = encoder.fit_transform(train_label)
    train_label = np_utils.to_categorical(train_label)
    print(train_label.shape, train_label[0])

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, train_label)

    # Convert to features
    (train_input_ids, train_input_masks, 
     train_segment_ids, train_labels) = convert_examples_to_features(tokenizer, train_examples, 
                                                                     max_seq_length=max_seq_length)

    
    print('train_input_ids', train_input_ids.shape)
    print('train_input_masks', train_input_masks.shape)
    print('train_segment_ids', train_segment_ids.shape)
    print('train_labels', train_labels.shape)

def get_test_inputs():
    test_text = test_df['sentence'].tolist()
    test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = test_df['polarity'].tolist()
    print(test_text.shape)

    test_label = encoder.fit_transform(test_label)
    test_label = np_utils.to_categorical(test_label)
    print(test_label.shape, test_label[0])

    # Convert data to InputExample format
    test_examples = convert_text_to_examples(test_text, test_label)

    
    (test_input_ids, test_input_masks, 
     test_segment_ids, test_labels) = convert_examples_to_features(tokenizer, test_examples, 
                                                                   max_seq_length=max_seq_length)

    print('test_input_ids', test_input_ids.shape)
    print('test_input_masks', test_input_masks.shape)
    print('test_segment_ids', test_segment_ids.shape)
    print('test_labels', test_labels.shape)
    
    return 


# # Tokenize
# 
# Next, tokenize our text to create `input_ids`, `input_masks`, and `segment_ids`

# In[4]:


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids   = [0] * max_seq_length
        input_mask  = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label       = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
#         np.array(labels).reshape(-1, 1),
        np.array(labels),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples



# In[5]:





# In[6]:


print('train_input_ids', train_input_ids[1])
print('train_input_masks', train_input_masks[1])
print('train_segment_ids', train_segment_ids[1])
print('train_labels', train_labels[1])

