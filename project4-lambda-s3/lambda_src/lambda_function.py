# We should be using loggers instead of print statements
# https://stackoverflow.com/questions/37703609/using-python-logging-with-aws-lambda
import base64
import json
import logging
import os
import sys
import uuid
from urllib.parse import unquote_plus

import boto3
import numpy as np
import onnxruntime as ort
from PIL import Image

# from datetime import datetime
# from io import BytesIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
# _bucket_name = "lambda-mlops"


class NumberEngine:
    def __init__(self, model_path="model.onnx"):
        # self.inf_sess = ort.InferenceSession("../model.onnx")
        self.inf_sess = ort.InferenceSession(model_path)

    def predict(self, x):
        y = self.inf_sess.run(None, {"x": x})
        return y[0]  # first output

    def save_preds(self, inp, output):
        img = Image.open(inp).resize((28, 28))
        img = np.array(img, dtype=np.float32)[:, :, 3] / 255
        img = (img - 0.1307) / 0.3081

        input_img = img.reshape((1, 1, 28, 28))

        y = self.predict(input_img)
        logger.info(f"Classified as {int(np.argmax(y))}   {type(y)=}")
        logger.info(f"logits {len(y)=} {y}")
        #  logits len(y)=1 [...

        with open(output, "w") as f:
            json.dump(
                {
                    "output": int(np.argmax(y)),
                    "logits": y[0].tolist(),
                    # "details": repr(y),
                },
                f,
            )


def lambda_handler(event, context):
    # Log input event and context
    print("Event: %s", event)
    print("Context: %s", context)

    model = NumberEngine()

    r = []
    # https://docs.aws.amazon.com/lambda/latest/dg/with-s3-tutorial.html
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        tmpkey = key.replace("/", "")
        download_path = "/tmp/{}{}".format(uuid.uuid4(), tmpkey)
        upload_path = "/tmp/resized-{}".format(tmpkey)
        s3_client.download_file(bucket, key, download_path)
        logger.info(f"Downloaded {bucket} file {key} to {download_path}")

        model.save_preds(download_path, upload_path)

        # '{}-resized'.format(bucket),  # tutorial uses 2 separate buckets
        pred_key = "{}-preds.json".format(key)
        s3_client.upload_file(upload_path, bucket, pred_key)
        logger.info(f"Uploaded {pred_key} to {upload_path}")
        r.append(pred_key)
    return {
        "statusCode": 200,
        "body": f"Hello, World! Saved {len(r)} prediction(s)",
        "details": r,
    }

    # s3_client.put_object(...


if __name__ == "__main__":
    result = lambda_handler(None, None)
    print(f"Result: {result}")
