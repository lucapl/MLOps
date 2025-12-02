import litserve as ls
from starlette.middleware.cors import CORSMiddleware
import base64
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime as ort


class NumberEngine(ls.LitAPI):
    def setup(self, device):
        self.inf_sess = ort.InferenceSession("../model.onnx")

    def decode_request(self, request, **kwargs):
        img = request.get("input")
        img = Image.open(BytesIO(base64.b64decode(img.split(',')[1]))).resize((28, 28))
        img = np.array(img, dtype=np.float32)[:, :, 3]/255
        img = (img - 0.1307)/0.3081

        return img.reshape((1,1,28,28))

    def predict(self, x):
        y = self.inf_sess.run(None, {"x": x})
        return y

    def encode_response(self, y):
        return {"output": int(np.argmax(y))}


if __name__ == "__main__":
    cors_middleware = (
        CORSMiddleware,
        {
            "allow_origins": ["*"],  # Allows all origins
            "allow_methods": ["GET", "POST"],  # Allows GET and POST methods
            "allow_headers": ["*"],  # Allows all headers
        }
    )
    server = ls.LitServer(NumberEngine(), middlewares=[cors_middleware])
    server.run(port=8000)
