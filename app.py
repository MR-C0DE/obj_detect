from flask import Flask, render_template, Response, jsonify, request, send_from_directory, url_for
import cv2 as cv
import numpy as np
import time
import threading
import os

app = Flask(__name__)

# Load the COCO class names
with open('object_detection_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# Get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the DNN model
model = cv.dnn.readNet(model='frozen_inference_graph.pb',
                       config='ssd_mobilenet_v2_coco.txt', framework='TensorFlow')

# Variables to manage the state
stop_sign_detected = False
stop_sign_detected_time = None
stop_sign_detection_timeout = 3
state = True

# Function to perform object detection and update the state


def Target_Detection(image):
    global stop_sign_detected, stop_sign_detected_time, state
    image_height, image_width, _ = image.shape
    blob = cv.dnn.blobFromImage(image=image, size=(
        300, 300), mean=(104, 117, 123), swapRB=True)
    model.setInput(blob)
    output = model.forward()
    detected_objects = []
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .4:
            class_id = detection[1]
            class_name = class_names[int(class_id) - 1]
            color = COLORS[int(class_id)]
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            if class_name == 'stop sign':
                stop_sign_detected = True
            cv.rectangle(image, (int(box_x), int(box_y)),
                         (int(box_width), int(box_height)), color, thickness=2)
            cv.putText(image, class_name, (int(box_x), int(box_y - 5)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            detected_objects.append(class_name)
    return image, detected_objects

# Function to capture video frames and process them


def capture_frames():
    global state, stop_sign_detected, stop_sign_detected_time
    capture = cv.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frame, detected_objects = Target_Detection(frame)
        if state:
            state_message = "Car moving"
            if stop_sign_detected:
                state = False
                stop_sign_detected_time = time.time()
        else:
            state_message = "Car Stopped"
            if stop_sign_detected_time is not None and (time.time() - stop_sign_detected_time) >= stop_sign_detection_timeout:
                state = True
                stop_sign_detected_time = None
        fps = capture.get(cv.CAP_PROP_FPS)
        detected_info = f"Objects detected: {', '.join(detected_objects)}"
        with open('status.txt', 'w') as f:
            f.write(f"FPS: {fps}\n")
            f.write(f"{state_message}\n")
            f.write(f"{detected_info}\n")
        # Write the frame to a temporary file for displaying on the webpage
        cv.imwrite('static/frame.jpg', frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()

# Route to serve the status information


@app.route('/status')
def status():
    if os.path.exists('status.txt'):
        with open('status.txt', 'r') as f:
            status_info = f.read().split('\n')
        return jsonify({
            'fps': status_info[0].replace('FPS: ', ''),
            'state': status_info[1],
            'objects': status_info[2].replace('Objects detected: ', '').split(', ')
        })
    else:
        return jsonify({
            'fps': '0',
            'state': 'No Data',
            'objects': []
        })

# Route to reset the console


@app.route('/reset', methods=['POST'])
def reset():
    if os.path.exists('status.txt'):
        os.remove('status.txt')
    return jsonify({'status': 'Console reset'})


def generate_frames():
    capture = cv.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frame, detected_objects = Target_Detection(frame)
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


# Start the frame capture in a separate thread
thread = threading.Thread(target=capture_frames)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    app.run(debug=True)
