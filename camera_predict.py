import cv2
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class VideoCamera:
    def __init__(self):
        self.model = None
        self.labels = {}
        self.language = 'English'
        self.cap = None
        self.is_active = False
        self.last_prediction = "Ready..."
        self.confidence = 0.0
        self.prediction_history = []
        self.stable_prediction = "Ready..."
        
        # Initialize camera
        self.initialize_camera()
        
        # Load model and labels
        self.load_model_and_labels()
    
    def initialize_camera(self):
        """Initialize camera with error handling"""
        try:
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    # Set camera properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.is_active = True
                    logger.info(f"Camera initialized successfully on index {camera_index}")
                    break
                else:
                    self.cap.release()
            
            if not self.is_active:
                logger.error("Could not initialize camera")
                
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            self.is_active = False

    def load_model_and_labels(self):
        """Load the trained model and labels"""
        try:
            # Check if model exists
            model_path = 'model/model.h5'
            labels_path = 'model/labels.npy'
            
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning("Model file not found. Please train the model first.")
                self.model = None
            
            if os.path.exists(labels_path):
                self.labels = np.load(labels_path, allow_pickle=True).item()
                logger.info("Labels loaded successfully")
            else:
                logger.warning("Labels file not found. Using default labels.")
                # Default labels if file doesn't exist
                self.create_default_labels()
                
        except Exception as e:
            logger.error(f"Error loading model/labels: {str(e)}")
            self.model = None
            self.create_default_labels()

    def create_default_labels(self):
        """Create default labels structure"""
        self.labels = {
            'English': {i: chr(65 + i) for i in range(26)},
            'Hindi': {i: f"अक्षर{i+1}" for i in range(26)},
            'Kannada': {i: f"ಅಕ್ಷರ{i+1}" for i in range(26)}
        }
        # Add special classes
        for lang in self.labels:
            self.labels[lang][26] = "SPACE" if lang == 'English' else "स्थान" if lang == 'Hindi' else "ಸ್ಥಳ"
            self.labels[lang][27] = "DELETE" if lang == 'English' else "मिटाएं" if lang == 'Hindi' else "ಅಳಿಸು"
            self.labels[lang][28] = "NOTHING" if lang == 'English' else "कुछ नहीं" if lang == 'Hindi' else "ಏನೂ ಇಲ್ಲ"

    def __del__(self):
        """Cleanup camera resources"""
        if self.cap:
            self.cap.release()

    def set_language(self, language):
        """Set the display language"""
        if language in self.labels:
            self.language = language
            logger.info(f"Language changed to {language}")
        else:
            logger.warning(f"Language {language} not supported")

    def get_stable_prediction(self, current_prediction):
        """Get stable prediction by tracking history"""
        self.prediction_history.append(current_prediction)
        
        # Keep only last 5 predictions
        if len(self.prediction_history) > 5:
            self.prediction_history.pop(0)
        
        # If we have enough predictions, check for stability
        if len(self.prediction_history) >= 3:
            # Check if last 3 predictions are the same
            if all(pred == current_prediction for pred in self.prediction_history[-3:]):
                self.stable_prediction = current_prediction
        
        return self.stable_prediction

    def preprocess_roi(self, roi):
        """Preprocess the region of interest for prediction"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input size
            resized = cv2.resize(gray, (64, 64))
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            # Reshape for model input
            reshaped = normalized.reshape(1, 64, 64, 1)
            
            return reshaped
            
        except Exception as e:
            logger.error(f"Error preprocessing ROI: {str(e)}")
            return None

    def predict_sign(self, roi):
        """Predict sign from region of interest"""
        if self.model is None:
            return "Model not loaded", 0.0
        
        try:
            # Preprocess the ROI
            processed_roi = self.preprocess_roi(roi)
            if processed_roi is None:
                return "Processing error", 0.0
            
            # Make prediction
            prediction = self.model.predict(processed_roi, verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            # Get label for current language
            label = self.labels.get(self.language, {}).get(predicted_index, "Unknown")
            
            # Only return prediction if confidence is above threshold
            if confidence > 0.7:  # 70% confidence threshold
                return label, confidence
            else:
                return "Low confidence", confidence
                
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return "Prediction error", 0.0

    def get_frame(self):
        """Get current frame with predictions"""
        if not self.is_active or self.cap is None:
            # Return a black frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        try:
            success, frame = self.cap.read()
            if not success:
                # Try to reinitialize camera
                self.initialize_camera()
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera error - reconnecting...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Define ROI coordinates (adjustable)
                x1, y1, x2, y2 = 100, 100, 300, 300
                
                # Extract ROI and predict
                roi = frame[y1:y2, x1:x2]
                prediction, confidence = self.predict_sign(roi)
                
                # Get stable prediction
                stable_pred = self.get_stable_prediction(prediction)
                
                # Draw ROI rectangle with color based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 0, 255)  # Red - low confidence
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Add corner markers for better visibility
                corner_length = 20
                # Top-left corner
                cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 5)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 5)
                # Top-right corner
                cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 5)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 5)
                # Bottom-left corner
                cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 5)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 5)
                # Bottom-right corner
                cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 5)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 5)
                
                # Create text background
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (550, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Add prediction text
                pred_text = f"Prediction: {stable_pred}"
                conf_text = f"Confidence: {confidence:.2f} ({confidence*100:.1f}%)"
                lang_text = f"Language: {self.language}"
                
                # Draw text with better formatting
                cv2.putText(frame, pred_text, (15, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, conf_text, (15, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, lang_text, (15, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add instructions
                instruction_text = "Place hand in the rectangle"
                cv2.putText(frame, instruction_text, (x1, y1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add status indicators
                status_text = "● LIVE" if self.is_active else "● OFFLINE"
                status_color = (0, 255, 0) if self.is_active else (0, 0, 255)
                cv2.putText(frame, status_text, (500, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Store current prediction for API
                self.last_prediction = stable_pred
                self.confidence = confidence

            # Encode frame as JPEG with good quality
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return jpeg.tobytes()
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            # Return error frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Error: {str(e)}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

    def release_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.is_active = False
            logger.info("Camera released")

    def get_camera_info(self):
        """Get camera information"""
        if not self.cap:
            return {"status": "Not initialized"}
        
        return {
            "status": "Active" if self.is_active else "Inactive",
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS))
        }