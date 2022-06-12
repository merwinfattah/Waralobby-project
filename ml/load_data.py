import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import *
import io
import json
import random

def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a CSV file
    
    Args:
        filename (string): path to the CSV file
    
    Returns:
        sentences, labels (list of string, list of string): tuple containing lists of sentences and labels
    """
    sentences = []
    sentiment = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            sentiment.append(int(row[3]))
            sentence = row[1]
            sentences.append(sentence)

    return sentences, sentiment

def train_val_test_split(sentences, sentiment, training_split):    
    # Compute the number of sentences that will be used for training (should be an integer)
    train_size = int(len(sentences) * training_split)
    # val_size = train_size + int(len(sentences) * val_split)

    # Split the sentences and labels into train/validation splits
    train_sentences = sentences[:train_size]
    train_sentiment = sentiment[:train_size]

    validation_sentences = sentences[train_size:]
    validation_sentiment = sentiment[train_size:]

    # test_sentences = sentences[val_size:]
    # test_sentiment = sentiment[val_size:]
    
    return train_sentences, validation_sentences, train_sentiment, validation_sentiment

def fit_tokenizer(train_sentences, num_words, oov_token):
  tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
  tokenizer.fit_on_texts(train_sentences)

  return tokenizer

def seq_and_pad(sentences, tokenizer, padding, maxlen):
    """
    Generates an array of token sequences and pads them to the same length
    
    Args:
        sentences (list of string): list of sentences to tokenize and pad
        tokenizer (object): Tokenizer instance containing the word-index dictionary
        padding (string): type of padding to use
        maxlen (int): maximum length of the token sequence
    
    Returns:
        padded_sequences (array of int): tokenized sentences padded to the same length
    """    
       
    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    
    # Pad the sequences using the correct padding and maxlen
    padded_sequences = pad_sequences(sequences, padding=padding, maxlen=maxlen)

    return padded_sequences

def load_data():
    """
    Loads the data from the CSV files and splits them into train/validation/test sets
    
    Returns:
        train_sentences, validation_sentences, train_sentiment, validation_sentiment
    """
    # Load the data
    sentences, sentiment = parse_data_from_file('ml/dataset/preprocessed/processed_review_binary.csv')

    # Bundle the two lists into a single one
    sentences_and_sentiment = list(zip(sentences, sentiment))

    # Perform random sampling
    random.seed(42)
    sentences_and_sentiment = random.sample(sentences_and_sentiment, len(sentences))

    # Unpack back into separate lists
    sentences, sentiment = zip(*sentences_and_sentiment)

    # Split the data into train/validation/test sets
    train_sentences, validation_sentences, train_sentiment, validation_sentiment = train_val_test_split(sentences, sentiment, 0.8)

    # fit tokenizer
    tokenizer = fit_tokenizer(train_sentences, num_words=NUM_WORDS, oov_token=OOV_TOKEN)
    word_index = tokenizer.word_index
     # save tokenizer vocabulary to file
    tokenizer_json = tokenizer.to_json()
    with io.open('ml/tokenizer/tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    # sequencing and padding sentences
    train_padded_sentences = seq_and_pad(train_sentences, tokenizer, PADDING, MAXLEN)
    validation_padded_sentences = seq_and_pad(validation_sentences, tokenizer, PADDING, MAXLEN)
    # test_padded_sentences = seq_and_pad(test_sentences, tokenizer, PADDING, MAXLEN)

    # initializing vocab_size
    vocab_size = len(word_index) + 1

    # converting sentences into numpy arrays
    train_sentences = np.array(train_padded_sentences)
    validation_sentences = np.array(validation_padded_sentences)
    # test_sentences = np.array(test_padded_sentences)

    # converting sentiment into numpy arrays
    train_sentiment = np.array(train_sentiment)
    validation_sentiment = np.array(validation_sentiment)
    # test_sentiment = np.array(test_sentiment)

    print("Finished loading data\n")

    return train_sentences, validation_sentences, train_sentiment, validation_sentiment, vocab_size, word_index