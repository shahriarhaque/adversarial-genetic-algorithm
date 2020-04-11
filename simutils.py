# Similarity tests
import textdistance

import json
import keras
import pandas as pd
import numpy as np
import pickle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras import initializers
from keras import optimizers
from keras import backend as K

from keras.layers import Embedding,Reshape, Activation, RepeatVector, Permute, Lambda, GlobalMaxPool1D
from keras.layers import Dense, Conv1D, MaxPooling1D, Input, Flatten, Dropout, LSTM, Bidirectional, GRU
from keras.layers import Concatenate
from keras.layers import Multiply
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model

# Numpy Seed
from numpy.random import seed
seed(123)

# TensorFlow Seed
import tensorflow as tf
tf.random.set_seed(2345)

# Random Seed
import random
random.seed(3567)

# Python Hash Seed
import os
os.environ['PYTHONHASHSEED'] = '0'

KERAS_INIT_SEED = 1


MODEL_DIRECTORY = '/Users/shahriar/Documents/Research/Code/themis/models/'
MODEL_FILE = MODEL_DIRECTORY + 'themis-small.h5'

TOKEN_HW_FILE = MODEL_DIRECTORY + 'tokenhw.pkl'
TOKEN_HC_FILE = MODEL_DIRECTORY + 'tokenhc.pkl'
TOKEN_BW_FILE = MODEL_DIRECTORY + 'tokenbw.pkl'
TOKEN_BC_FILE = MODEL_DIRECTORY + 'tokenbc.pkl'

MAX_TOKENS_HEADER_WORD = 50
MAX_TOKENS_HEADER_CHAR = 100
MAX_TOKENS_BODY_WORD = 150
MAX_TOKENS_BODY_CHAR = 300
EMBEDDING_DIM=256

def read_common_words():
    file = '/Users/shahriar/Documents/Research/Code/skunkworks/src/common-words.txt'
    all_lines = []
    with open(file) as fp:
        for line in fp:
            all_lines.append(line.strip())
    fp.close()
    all_lines.append(' ')
    return all_lines

def read_sample_phish():
    file = '/Users/shahriar/Documents/Research/Code/skunkworks/src/sample-phish.txt'
    all_lines = []
    with open(file) as fp:
        for line in fp:
            all_lines.append(line)
    fp.close()
    return all_lines[0]

def individual_to_text(individual):
    words = [COMMON_WORDS[index] for index in individual]
    text = ' '.join(words)
    return text.strip()


def load_tokenizers():
    tokenhw = pickle.load(open(TOKEN_HW_FILE, 'rb'))
    tokenhc = pickle.load(open(TOKEN_HC_FILE, 'rb'))
    tokenbw = pickle.load(open(TOKEN_BW_FILE, 'rb'))
    tokenbc = pickle.load(open(TOKEN_BC_FILE, 'rb'))

    tokenizers = (tokenhw, tokenhc, tokenbw, tokenbc)
    return tokenizers

def load_model_and_tokenizer():
    print('Using loaded model and tokenizers')
    model = load_model(MODEL_FILE)
    tokenizers = load_tokenizers()
    return (model, tokenizers)

def encode_sequences(tokenizers, train_header, train_body):
    tokenhw, tokenhc, tokenbw, tokenbc = tokenizers

    max_hw = MAX_TOKENS_HEADER_WORD
    max_hc = MAX_TOKENS_HEADER_CHAR
    max_bw = MAX_TOKENS_BODY_WORD
    max_bc = MAX_TOKENS_BODY_CHAR

    vocab_size_hw = len(tokenhw.word_index)+1
    encoded_hw= tokenhw.texts_to_sequences(train_header)
    hw_sequence_input=pad_sequences(encoded_hw, maxlen=max_hw,padding='post')

    vocab_size_hc = len(tokenhc.word_index)+1
    encoded_hc=tokenhc.texts_to_sequences(train_header)
    hc_sequence_input=pad_sequences(encoded_hc, maxlen=max_hc,padding='post')

    vocab_size_bw = len(tokenbw.word_index)+1
    encoded_bw= tokenbw.texts_to_sequences(train_body)
    bw_sequence_input=pad_sequences(encoded_bw, maxlen=max_bw,padding='post')

    vocab_size_bc = len(tokenbc.word_index)+1
    encoded_bc=tokenhc.texts_to_sequences(train_body)
    bc_sequence_input=pad_sequences(encoded_bc, maxlen=max_bc,padding='post')

    sequences = (hw_sequence_input, hc_sequence_input, bw_sequence_input, bc_sequence_input)

    return sequences

def predict(model, test_df, tokenizers):
    test_header, test_body, test_labels = df_to_header_body_label(test_df)
    sequences = encode_sequences(tokenizers, test_header, test_body)
    hw_sequence_input, hc_sequence_input, bw_sequence_input, bc_sequence_input = sequences
    predicted_prob = model.predict([hc_sequence_input, hw_sequence_input, bc_sequence_input, bw_sequence_input])

    return predicted_prob

def to_df(header, body, label):
    df = pd.DataFrame(list(zip(header, body, label)), columns = ['header', 'body', 'phishy'])
    return df

def fitness_text(text):
    body = [text]
    header = ['this is an email']
    label = [1]
    df = to_df(header, body, label)
    prob = predict(model, df, tokenizers)

    return prob[0][0]

def df_to_header_body_label(train_df):
    train_header = train_df['header'].tolist()
    train_body = train_df['body'].tolist()
    train_labels = train_df['phishy'].tolist()

    return (train_header, train_body, train_labels)

def fitness(individual):
    text = individual_to_text(individual)
    return fitness_text(text)

def run():
    # individual = [random.randint(0,NUM_COMMON_WORDS-1) for i in range(MAX_WORDS)]
    # body = individual_to_text(individual)
    body = SAMPLE_PHISH
    print(fitness_text(body))











SAMPLE_PHISH = read_sample_phish()
COMMON_WORDS = read_common_words()
NUM_COMMON_WORDS = len(COMMON_WORDS)
MAX_WORDS = 10
MAX_LENGTH = 1000
model, tokenizers = load_model_and_tokenizer()
run()



# run()
