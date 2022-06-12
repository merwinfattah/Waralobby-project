import numpy as np
import tensorflow as tf
from load_data import load_data
import io

def create_model(vocab_size, max_length, embedding_dim, word_index):
    
    lstm1_dim = 64
    lstm2_dim = 32
    dense_dim = 32

    embeddings_index = {}
    with io.open('ml/models/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            curr_word = values[0]
            coefs = np.asarray(values[1:], dtype='float64')
            embeddings_index[curr_word] = coefs

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1_dim, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2_dim)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['auc'])
    return model