from flask import Flask, render_template, Response, request, flash, redirect, url_for, jsonify
import cv2
import numpy as np
import os
import logging
import mediapipe as mp
from tensorflow.keras.models import load_model
import threading
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_123'

# --- Global Variables & Locks ---
model = None
all_labels = {}
current_language = 'English'
language_lock = threading.Lock()
prediction_lock = threading.Lock()
latest_prediction_data = { "prediction": "Ready...", "confidence": 0.0, "language": "English" }

# --- MediaPipe & Font Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cap = None
try:
    font = ImageFont.truetype("fonts/NotoSansDevanagari-Regular.ttf", 32)
except IOError:
    font = ImageFont.load_default()

def load_model_and_labels():
    global model, all_labels
    try:
        model = load_model('model/model.h5')
        all_labels = np.load('model/labels.npy', allow_pickle=True).item()
        logger.info("✅ Model and Labels loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading model/labels: {e}")

def initialize_camera():
    global cap
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.info(f"✅ Camera initialized on index {idx}.")
            return True
    cap = None
    return False

def generate_frames():
    global latest_prediction_data
    prediction_history = []
    
    while True:
        if cap is None or not cap.isOpened():
            if not initialize_camera():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Not Found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        current_prediction = "..."
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - x_[0])
                    data_aux.append(hand_landmarks.landmark[i].y - y_[0])
                
                input_data = np.array(data_aux).reshape(1, -1)
                prediction_result = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(prediction_result)
                confidence = float(np.max(prediction_result))
                
                english_labels = all_labels.get('English', {})
                if confidence > 0.85:
                    current_prediction = english_labels.get(predicted_index, "Unknown")
        
        prediction_history.append(current_prediction)
        if len(prediction_history) > 10: prediction_history.pop(0)
        
        stable_prediction = "..."
        try:
            most_common = max(set(prediction_history[-5:]), key=prediction_history[-5:].count)
            if most_common != "...": 
                stable_prediction = most_common
        except ValueError: pass

        with language_lock: lang_display = current_language
        
        # <<< FIX: Draw prediction text directly on the video frame >>>
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([(0, 0), (frame.shape[1], 50)], fill=(0, 0, 0, 128))
        draw.text((10, 5), f"Prediction: {stable_prediction}", font=font, fill=(52, 211, 153)) # Using success color
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Update shared prediction data for the sidebar API
        with prediction_lock:
            labels_for_lang = all_labels.get(lang_display, {})
            latest_prediction_data["prediction"] = labels_for_lang.get(predicted_index, stable_prediction) if 'predicted_index' in locals() else stable_prediction
            latest_prediction_data["confidence"] = confidence
            latest_prediction_data["language"] = lang_display

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- FLASK ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    global current_language
    if request.method == 'POST':
        with language_lock:
            current_language = request.form.get('language', 'English')
        flash(f'Language changed to {current_language}', 'success')
        return redirect(url_for('index'))
    
    with language_lock: lang = current_language
    available_langs = list(all_labels.keys()) if all_labels else ['English']
    
    return render_template('index.html', language=lang, available_languages=available_langs)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/prediction')
def api_prediction():
    with prediction_lock:
        data = latest_prediction_data.copy()
    return jsonify(data)

if __name__ == '__main__':
    load_model_and_labels()
    initialize_camera()
    logger.info("🚀 Starting ASL App...")
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)