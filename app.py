from flask import Flask, request, jsonify, send_from_directory
import requests
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# 自动找 yolov8n.pt，防止路径错误
model_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
yolo_model = YOLO(model_path)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message')
    resp = requests.post('http://localhost:11434/api/chat', json={
        'model': 'qwen2.5:0.5b',
        'messages': [{'role': 'user', 'content': msg}],
        'stream': False
    })
    return jsonify({'reply': resp.json()['message']['content']})

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = yolo_model(img)
    annotated = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated)
    return jsonify({'image': base64.b64encode(buffer).decode()})

if __name__ == '__main__':
    app.run(port=5000)