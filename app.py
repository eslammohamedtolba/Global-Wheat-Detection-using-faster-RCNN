# Load dependencies
from flask import Flask, request, render_template
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import base64

# Define the model architecture
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the saved state dictionary with map_location to CPU
model.load_state_dict(torch.load('Prepare model/wheat_detection_model.pth', map_location=torch.device('cpu')))
# Set the model to evaluation mode
model.eval()

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')

# Create predict page route
@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        
        # Get the uploaded image
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = T.ToTensor()(image).unsqueeze(0)

        batch_prediction = model(batch)  
        image_prediction = batch_prediction[0]  # Get image prediction

        # Extract prediction details
        boxes = image_prediction['boxes'].detach().cpu().numpy()
        scores = image_prediction['scores'].detach().cpu().numpy()

        # Filter out low-confidence predictions with threshold = 0.8
        confidence_threshold = 0.5
        high_conf_indices = scores >= confidence_threshold
        boxes = boxes[high_conf_indices]
        scores = scores[high_conf_indices]

        # Generate random colors for the bounding boxes (in BGR format)
        colors = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(len(boxes))]
        
        # Draw bounding boxes on the image
        for box, color in zip(boxes, colors):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Convert the image back to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encode the image to base64 to be shown in index.html file
        buffer = cv2.imencode('.png', image_rgb)[1]
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template('index.html', image = image_base64)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)