import torch
import os
import json
import subprocess

model_path = 'retrainedModels/yolov5s.pt'
# model_path = 'retrainedModels/best.pt'
model = None

def load_model():
    global model
    if os.path.exists(model_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

def get_model():
    return model

def update_model(object_id, label):
    """Update the model with a new label and retrain it."""
    with open('datasets/customLabels.json', 'r') as f:
        labels = json.load(f)
    
    labels[object_id] = label
    with open('datasets/customLabels.json', 'w') as f:
        json.dump(labels, f)
    
    retrain_model()

def retrain_model():
    """Retrain the model with the updated dataset."""
    command = [
        'python', 'train.py', 
        '--data', 'datasets/data.yaml', 
        '--cfg', 'models/yolov5s.yaml', 
        '--weights', '',  
        '--batch-size', '16'
    ]
    try:
        subprocess.run(command, check=True)
        if not os.path.exists('retrainedModels'):
            os.makedirs('retrainedModels')
        subprocess.run(['mv', 'runs/train/exp/weights/best.pt', model_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during model retraining: {e}")
