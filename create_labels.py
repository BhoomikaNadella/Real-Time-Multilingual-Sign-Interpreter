import numpy as np
import os

def create_multilingual_labels():
    """Create and save detailed multilingual labels for ASL alphabet."""
    
    MODEL_DIR = 'model'
    LABELS_PATH = os.path.join(MODEL_DIR, 'labels.npy')
    
    print("🌐 Creating detailed multilingual labels for the alphabet...")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"✅ Created directory: {MODEL_DIR}")

    # --- English Labels ---
    english_labels = {i: chr(65 + i) for i in range(26)}
    english_labels.update({26: "space", 27: "del", 28: "nothing"})
    
    # --- Hindi Labels (Devanagari script) ---
    hindi_labels = {
        0: "अ", 1: "ब", 2: "क", 3: "द", 4: "इ", 5: "फ", 6: "ग", 7: "ह",
        8: "आई", 9: "ज", 10: "के", 11: "ल", 12: "म", 13: "न", 14: "ओ",
        15: "प", 16: "क्यू", 17: "र", 18: "स", 19: "ट", 20: "उ", 21: "व",
        22: "डब्ल्यू", 23: "एक्स", 24: "वाई", 25: "जेड",
        26: "स्थान", 27: "मिटाएं", 28: "कुछ नहीं"
    }
    
    # --- Kannada Labels (Kannada script) ---
    kannada_labels = {
        0: "ಅ", 1: "ಬ", 2: "ಸಿ", 3: "ಡಿ", 4: "ಇ", 5: "ಎಫ್", 6: "ಜಿ", 7: "ಎಚ್",
        8: "ಐ", 9: "ಜೆ", 10: "ಕೆ", 11: "ಎಲ್", 12: "ಎಂ", 13: "ಎನ್", 14: "ಓ",
        15: "ಪಿ", 16: "ಕ್ಯೂ", 17: "ಆರ್", 18: "ಎಸ್", 19: "ಟಿ", 20: "ಯು",
        21: "ವಿ", 22: "ಡಬ್ಲ್ಯೂ", 23: "ಎಕ್ಸ್", 24: "ವೈ", 25: "ಝೆಡ್",
        26: "ಸ್ಥಳ", 27: "ಅಳಿಸು", 28: "ಏನೂ ಇಲ್ಲ"
    }

    # --- Marathi Labels (Devanagari script) ---
    marathi_labels = {
        0: "अ", 1: "ब", 2: "क", 3: "ड", 4: "इ", 5: "फ", 6: "ग", 7: "ह",
        8: "आय", 9: "ज", 10: "के", 11: "ल", 12: "म", 13: "न", 14: "ओ",
        15: "प", 16: "क्यू", 17: "र", 18: "स", 19: "ट", 20: "उ", 21: "व्ह",
        22: "डब्ल्यू", 23: "एक्स", 24: "वाय", 25: "झेड",
        26: "जागा", 27: "काढा", 28: "काहीही नाही"
    }

    # Combine all languages into a single dictionary
    all_labels = {
        'English': english_labels,
        'Hindi': hindi_labels,
        'Kannada': kannada_labels,
        'Marathi': marathi_labels,
    }

    try:
        np.save(LABELS_PATH, all_labels)
        print(f"✅ Labels successfully saved to: {LABELS_PATH}")
        print(f"Total classes created: {len(english_labels)}")
        print(f"Available languages: {list(all_labels.keys())}")
    except Exception as e:
        print(f"❌ Error saving labels: {str(e)}")

if __name__ == "__main__":
    create_multilingual_labels()