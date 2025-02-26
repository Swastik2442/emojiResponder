"AWS Lambda code for returning Emoji Prediction for the given Text"

import json
from urllib.parse import unquote_plus
from typing import Dict, List

import numpy as np
from tensorflow.keras.models import load_model # type: ignore

import boto3
from aws_lambda_powertools.utilities.data_classes import LambdaFunctionUrlEvent
from aws_lambda_powertools.utilities.typing import LambdaContext

S3_BUCKET = "awsfords"
WRITE_DIR = "/tmp/"
EMOJI_MODEL_FILE = "emoji_model.keras"
EMOJI_MAPPING_FILE = "emoji_mapping.json"
EMOJI_MODEL_PATH = WRITE_DIR + EMOJI_MODEL_FILE
EMOJI_MAPPING_PATH = WRITE_DIR + EMOJI_MAPPING_FILE

def response(data, status=200):
    "Creates a simple JSON Response for the Lambda Function"
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "success" if status < 400 else "error",
            "data": data,
        }),
        "isBase64Encoded": False,
    }

# Download Emoji Mapping and Model
s3 = boto3.client("s3")
s3.download_file(S3_BUCKET, EMOJI_MODEL_FILE, EMOJI_MODEL_PATH)
s3.download_file(S3_BUCKET, EMOJI_MAPPING_FILE, EMOJI_MAPPING_PATH)

# Create Emoji Mapping Dictionary
with open(EMOJI_MAPPING_PATH, "r", encoding="utf8") as file:
    emoji_dict: Dict[str, List[str]] = json.load(file)

# Emojis corresponding to smaller model
REDO = {'0': '473', '1': '462', '2': '1137', '3': '692', '4': '1119'}
def to_emoji(x):
    "Returns the Emoji corresponding to the ID"
    return emoji_dict[REDO[str(x)]][0]

# Create Text Embedding Utils
embedding_matrix = {}
with open("glove.6B.50d.txt", "r", encoding="utf8") as file:
    for line in file:
        values = line.split()
        embedding_matrix[values[0]] = np.array(values[1:], dtype="float")

def get_text_embeddings(text: str):
    "Converts Text to Numbers based on Embedding Matrix"

    embedding_data = np.zeros((1, 30, 50))

    for idx, word in enumerate(text.split()):
        word = word.lower()
        if embedding_matrix.get(word) is not None:
            embedding_data[0][idx] = embedding_matrix[word]

    return embedding_data

model = load_model(EMOJI_MODEL_PATH)

def handler(event: LambdaFunctionUrlEvent, _context: LambdaContext):
    "Runs when Lambda is invoked using the Function URL"

    try:
        params = event.get("queryStringParameters")
        if params is None:
            return response("Invalid Trigger Parameters", 400)
        text = params.get("text")
        if text is None:
            return response("Please provide a 'text' Query Parameter", 400)

        text = unquote_plus(text).strip()[:30]
        data = get_text_embeddings(text)
        pred = model.predict(data)
        probs = np.argmax(pred, axis=1)

        return response(to_emoji(probs[0]))
    except Exception as e:
        print("[ERROR]:", e)
        return response("Internal Server Error", 500)
