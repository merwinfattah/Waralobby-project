import argparse
from keras.models import load_model
from load_data import load_data
from config import MAXLEN, PADD_TYPE, TRUNC_TYPE
import json
from utils import process
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences



model = load_model('ml/models/sentimentmodel1_final.h5')

# x_train, x_val, x_test, y_train, y_val, y_test, vocab_size, word_index = load_data()

def parse_argument():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--review', help='Review of the product')
    return parser.parse_args()


def load_tokenizer():
    with open('ml/tokenizer/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

def predict():
    args = parse_argument()
    review = args.review
    
    tokenizer = load_tokenizer()
    
    processed_review = process(review)
    encoded_review = tokenizer.texts_to_sequences([processed_review])[0]
    encoded_review = pad_sequences([encoded_review], maxlen=MAXLEN, padding=PADD_TYPE, truncating=TRUNC_TYPE)
    pred = model.predict(encoded_review)

    if pred[0][0] > 0.6:
        print('Positive with {}%'.format(pred[0][0]*100))
    else:
        print('Negative with {}%'.format(100-pred[0][0]*100))

if __name__ == "__main__":
    predict()

