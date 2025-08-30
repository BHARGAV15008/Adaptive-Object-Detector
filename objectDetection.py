import torch
import cv2
import json
import Model

# Load custom labels
with open('datasets/customeLabels.json') as f:
    customLabels = json.load(f)

def detect_objects(frame):
    model = Model.get_model()
    results = model(frame)

    detected_objects = []
    for result in results.pred[0]:
        x1, y1, x2, y2, conf, class_id = result.tolist()
        class_id = int(class_id)
        label = customLabels.get(class_id, 'Unknown')
        detected_objects.append({
            'class_id': class_id,
            'label': label,
            'bbox': (x1, y1, x2, y2)
        })

        # Ask user for label if unknown
        if label == 'Unknown':
            # Logic for user interaction to get label
            pass

    # Draw bounding boxes
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        label = obj['label']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, detected_objects

def display_counts(frame, counts):
    y = 30
    for class_id, count in counts.items():
        label = customLabels.get(class_id, 'Unknown')
        cv2.putText(frame, f'{label}: {count}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y += 30
    return frame
