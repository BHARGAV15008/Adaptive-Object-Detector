from flask import Flask, render_template, request, jsonify, Response
import cv2
import json
import torch
from Model import load_model, get_model, update_model
from appUtils import load_labels, save_labels


app = Flask(__name__)

# Load the model and labels
load_model()
labels = load_labels()

# Object count dictionary
object_count = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/object_count')
def object_count_route():
    return jsonify(object_count)

@app.route('/label_object', methods=['POST'])
def label_object():
    data = request.json
    object_id = int(data['object_id'])
    label = data['label']
    labels[object_id] = label
    save_labels(labels)
    update_model(object_id, label)
    return jsonify({'status': 'success'})

def generate_frames():
    global labels
    global object_count

    cap = cv2.VideoCapture(0)  # Use camera or change to video file path

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Prepare frame for detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = get_model()(img_rgb)  # Run inference

        # Reset object count
        object_count.clear()

        # Process detections
        for det in results.xyxy[0]:  # detections
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls.item())
            label = labels.get(cls, 'unknown')

            if label == 'unknown':
                # Draw text on frame to ask for user label
                cv2.putText(frame, "Label this object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Update the object ID for labeling
                object_id = cls
                cv2.putText(frame, f'Object ID: {object_id}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                # Draw bounding box and label
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update object count
            if label in object_count:
                object_count[label] += 1
            else:
                object_count[label] = 1

        # Encode frame and yield for video streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)