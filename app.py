from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import base64
import torch
from torchvision import transforms
from transformers import ViTForImageClassification

app = Flask(__name__)

# Load the Vision Transformer model and its state dictionary
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=33, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

print(model.classifier.weight.shape)  # Should output torch.Size([3, 768])
print(model.classifier.bias.shape)    # Should output torch.Size([3])

# Define image transformations for the ViT model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class names for sign language (23 letters + 10 digits)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'X', 'Y', 'Z']

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Remove the base64 prefix (e.g., "data:image/jpeg;base64,") and decode
        img_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Image decoding failed', 'details': str(e)}), 400
    
    # Apply transformations and add batch dimension
    image = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        logits = output.logits  # Access logits from the model's output
        prediction = logits.argmax(dim=1).item()
    
    # Map prediction to class name
    predicted_class = class_names[prediction]
    return jsonify({'predicted_class': predicted_class})

# Serve the HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)