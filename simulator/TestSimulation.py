import cv2
from PIL import Image
from io import BytesIO
import base64
from flask import Flask
from tensorflow.keras.models import load_model
import numpy as np
import eventlet
import socketio
import os
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sio = socketio.Server()
app = Flask(__name__)  # __main__
maxSpeed = 10


def preProcessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255

    return img


@sio.on('telemetry')
def telemetry(sid, data):
    if not data or 'image' not in data:
        sendControl(0, 0)
        return

    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcessing(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = max(0.20, 1.0 - speed / maxSpeed)
    print(f'{throttle}, {steering}, {speed}')
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == "__main__":
    model = load_model('./models/model.h5', compile=False)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
