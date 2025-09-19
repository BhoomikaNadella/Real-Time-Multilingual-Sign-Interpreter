import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
DATASET_PATH = 'dataset/asl_alphabet_train/asl_alphabet_train'
MODEL_SAVE_PATH = 'model/model.h5'
LANDMARK_DATA_PATH = 'model/landmark_data.pickle'
MODEL_DIR = 'model'

def extract_landmarks(image_path, hands):
    """Processes an image and extracts normalized hand landmarks."""
    data_aux = []
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # Assuming only one hand is in the image
        hand_landmarks = results.multi_hand_landmarks[0]
        
        x_ = [landmark.x for landmark in hand_landmarks.landmark]
        y_ = [landmark.y for landmark in hand_landmarks.landmark]

        # Normalize landmarks to be relative to the wrist (point 0)
        # This makes the model robust to hand position and scale
        for i in range(len(hand_landmarks.landmark)):
            data_aux.append(hand_landmarks.landmark[i].x - x_[0])
            data_aux.append(hand_landmarks.landmark[i].y - y_[0])
            
        return data_aux
    return None

def create_landmark_dataset():
    """Creates a dataset of landmarks from the image folders and saves it."""
    print("🚀 Starting landmark extraction... This may take a while.")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    data = []
    labels = []
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        return

    class_dirs = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    # Create a consistent mapping from class name (folder name) to integer
    class_map = {name: i for i, name in enumerate(class_dirs)}
    
    for dir_name in class_dirs:
        class_index = class_map[dir_name]
        print(f"Processing class '{dir_name}' ({class_index+1}/{len(class_dirs)})...")
        class_path = os.path.join(DATASET_PATH, dir_name)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            landmarks = extract_landmarks(img_path, hands)
            
            if landmarks:
                data.append(landmarks)
                labels.append(class_index)

    print("✅ Landmark extraction complete.")
    
    # Ensure the model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Save the processed landmark data to a file
    with open(LANDMARK_DATA_PATH, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    print(f"💾 Landmark data saved to {LANDMARK_DATA_PATH}")

def train_new_model():
    """Trains a new MLP model on the extracted landmark data."""
    print("🧠 Starting model training on landmark data...")
    
    # Load the landmark data from the file
    with open(LANDMARK_DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)

    data = np.asarray(dataset['data'])
    labels = np.asarray(dataset['labels'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    num_classes = len(np.unique(labels))
    # The input shape is the number of features (21 landmarks * 2 coordinates = 42)
    input_shape = (data.shape[1],) 

    # Define the new, simple, and powerful model
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Callback to stop training when performance stops improving
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Train the model
    print("\n🏋️ Training model... (This will be very fast)")
    model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping]
    )

    # Evaluate the final model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Training complete. Final accuracy on test data: {accuracy*100:.2f}%")

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    # First, check if the landmark data file exists.
    # If not, create it by processing all the images.
    if not os.path.exists(LANDMARK_DATA_PATH):
        create_landmark_dataset()
    
    # Once the landmark data exists, train the model on it.
    train_new_model()