from typing import List
from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import ml.utils as utils
from ml.config import *
from uvicorn import run
import os
from fastapi.logger import logger
from pydantic import BaseSettings


# class Settings(BaseSettings):
#     # ... The rest of our FastAPI settings

#     BASE_URL = "http://localhost:8000"
#     USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"


# settings = Settings()

app = FastAPI()

# if settings.USE_NGROK:
#     # pyngrok should only ever be installed or initialized in a dev environment when this flag is set
#     from pyngrok import ngrok

#     # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
#     # when starting the server
#     port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 8000

#     # Open a ngrok tunnel to the dev server
#     public_url = ngrok.connect(port).public_url
#     logger.info("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

#     # Update any base URLs or webhooks to use the public ngrok URL
#     settings.BASE_URL = public_url
#     init_webhooks(public_url)


model_path = "ml/models/sentimentmodel1_final.h5"
tokenizer_path = "ml/tokenizer/tokenizer.json"

def load_tokenizer():
    with open(tokenizer_path) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

def predict(model, tokenizer, review):
    
    processed_review = utils.process(review)
    encoded_review = tokenizer.texts_to_sequences([processed_review])[0]
    encoded_review = pad_sequences([encoded_review], maxlen=MAXLEN, padding=PADD_TYPE, truncating=TRUNC_TYPE)
    pred = model.predict([encoded_review])

    if pred[0][0] > 0.6:
        sentiment = "Positive"
        print('Positive with {}%'.format(pred[0][0]*100))
        confidence = pred[0][0]*100
    else:
        sentiment = "Negative"
        print('Negative with {}%'.format(100-pred[0][0]*100))
        confidence = 100-pred[0][0]*100
    
    return sentiment, round(confidence, 2) 

model = load_model(model_path)
tokenizer = load_tokenizer()

@app.post("/predict")
async def get_review_sentiment(review: str = ""):
    if review == "":
        raise HTTPException(status_code=400, detail="No review provided")
    # get sentiment
    class_sentiment, confidence = predict(model, tokenizer, review)
    
    return {"review": review, "sentiment": class_sentiment, "confidence": confidence}