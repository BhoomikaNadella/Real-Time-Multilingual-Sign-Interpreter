import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
MODEL_PATH = 'model/model.h5'
LANDMARK_DATA_PATH = 'model/landmark_data.pickle'
LABELS_PATH = 'model/labels.npy'

def main():
    """Main function to evaluate the landmark-based model."""
    print("🚀 Starting Model Evaluation on Landmark Data...")

    # --- Load Data and Model ---
    try:
        with open(LANDMARK_DATA_PATH, 'rb') as f:
            dataset = pickle.load(f)
        model = load_model(MODEL_PATH)
        all_labels = np.load(LABELS_PATH, allow_pickle=True).item()
        # Use the English labels for the report
        class_names = list(all_labels['English'].values())
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found. {e}")
        print("Please run train_model.py first to create the data and model.")
        return

    data = np.asarray(dataset['data'])
    labels = np.asarray(dataset['labels'])

    # --- Make Predictions ---
    print("🤖 Making predictions on the landmark dataset...")
    predictions = model.predict(data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = labels

    # --- Calculate and Display Metrics ---
    print("\n" + "="*50)
    print("📊 Evaluation Report")
    print("="*50)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Ensure class_names list matches the number of classes in the model
    num_model_classes = model.output_shape[1]
    if len(class_names) != num_model_classes:
        print(f"⚠️ Warning: Number of labels ({len(class_names)}) does not match model outputs ({num_model_classes}). Adjusting for report.")
        # Create a generic list if mismatch is too large
        if len(class_names) < num_model_classes:
             class_names.extend([f"class_{i}" for i in range(len(class_names), num_model_classes)])
        else:
            class_names = class_names[:num_model_classes]

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # --- Generate and Save the Confusion Matrix ---
    print("\n📈 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(22, 22))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Actual Label', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    report_path = 'model/evaluation_report.png'
    plt.savefig(report_path)
    print(f"\n🖼️ Confusion matrix has been saved to '{report_path}'")
    
if __name__ == "__main__":
    main()