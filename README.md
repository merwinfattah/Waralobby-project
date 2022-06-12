# Waralobby Sentiment Model

A sentiment analysis model using Tensorflow Keras

## Description

This is an API used to derive sentiment from a review made inside Waralobby app. It used Recurrent Neural Network (RNN) model with 2 LSTM layers and a pre trained GloVe100D word embeddings. The dataset used is Amazon Fine Food Review [here] from this link . Using a POST request, user can use this API and receive a sentiment of a review, and the confidence of that sentiment.

[here]: https://www.kaggle.com/snap/amazon-fine-food-reviews

[You can download the model here](./ml/models/sentimentmodel1_final.h5)

## Running the API

This API is built on top of Python, so what you'll need is Python v.3.8 and above

### Dependencies 

The `requirements.txt` file contains all the dependencies needed to run this API. You can install all the dependencies using pip inside the main directory `mfcc-extractor`.

```
pip install -r requirements.txt
```

## Running the API
To run the API, first go to the main directory `Waralobby-project` and run the `main.py` file using this command

```
uvicorn main:app --reload
```
The API server will run on http://127.0.0.1:8000

## Access the API Endpoints

FastAPI already have a GUI to easily try and access the API Endpoints. You can access the API endpoint through http://127.0.0.1:8000/docs#/

The API endpoint is http://127.0.0.1:8000/predict

### Parameters
The parameters needed in this API endpoint is
- review: A string containing review of a franchise

### Using the API
- Press the 'Try it out' button on the API method
- Insert the parameters mentioned above.
- Press 'Execute' and take a look at the response body

### Response

The response of the API is in JSON format (content-type header 'application/json') as follows

```
{
  "review": <YOUR REVIEW>,
  "sentiment": <Positive/Negative>,
  "confidence": <VALUE OF CONFIDENCE, 0-100>
}
```

## Contributors
Ida Bagus Raditya Avanindra Mahaputra\
M2002F0054\
Kanisius Sosrodimardito\
M2002G0037\
Bangkit Machine Learning Path