from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

model = CNN()
try:
    model.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu')))
    print("CNN model loaded successfully!")
except:
    print("Using random initialized CNN model")
model.eval()

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream)
        img_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'probabilities': probabilities[0].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting CNN-based Handwritten Digit Recognition App...")
    app.run(debug=False, host='0.0.0.0', port=8080)