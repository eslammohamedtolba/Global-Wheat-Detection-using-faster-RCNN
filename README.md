 # Global Wheat Detection using Faster R-CNN

The Global Wheat Detection project leverages the Faster R-CNN (Region-based Convolutional Neural Network) architecture to detect wheat heads in images. 
This project aims to provide an efficient and accurate solution for wheat head detection, which is crucial for agricultural research and crop management.

![Image about the final project](<Global Wheat Detection using Faster RCNN.png>)

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features
- **Wheat Detection**: Uses Faster R-CNN with a custom-trained model to detect wheat in images.
- **Interactive Web Application**: Built using Flask to upload images and display predictions.
- **Bounding Box Visualization**: Shows predicted bounding boxes around detected wheat with confidence scores.

## Requirements
- Python 3.7+
- Flask
- PyTorch
- torchvision
- OpenCV
- Pandas
- Numpy
- Matplotlib
- Pillow
- scikit-learn
- Kaggle API (for dataset download)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/eslammohamedtolba/Global-Wheat-Detection-using-faster-RCNN.git
   cd Global-Wheat-Detection-using-faster-RCNN
   
2. Install the required dependencies:
    ```bash
    pip install Flask torch torchvision opencv-python pandas numpy matplotlib scikit-learn
    ```

## Usage

1. Run the Flask app:

    ```bash
    python app.py
    ```

2. Access the web application at `http://127.0.0.1:5000/`.

3. Upload an image to detect wheat. The image will be processed, and bounding boxes will be drawn around detected wheat.

## Project Structure

- `app.py`: Main Flask application file.
- `Prepare model/Global Wheat Detection using faster RCNN.ipynb`: Jupyter notebook for training and evaluation.
- `Prepare model/wheat_detection_model.pth`: Saved state dictionary of the trained model.
- `templates/index.html`: HTML template for the web interface.
- `static/style.css`: CSS file for styling the web interface.

## Contributing

Contributions are welcome! Here’s how you can contribute:

1. Fork the repository.

2. Create a new branch (git checkout -b feature-branch).

3. Make your changes and commit them (git commit -m 'Add some feature').

4. Push to the branch (git push origin feature-branch).

5. Open a pull request.




