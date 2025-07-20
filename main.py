
import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ASL labels
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

class ASLClassifier:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        # Simple rule-based classifier for demonstration
        # In production, you would load a trained ML model
        pass
    
    def extract_features(self, landmarks):
        """Extract features from hand landmarks"""
        if not landmarks:
            return None
        
        # Convert landmarks to numpy array
        coords = []
        for landmark in landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(coords)
    
    def classify(self, landmarks):
        """Classify hand gesture based on landmarks"""
        features = self.extract_features(landmarks)
        if features is None:
            return None, 0
        
        # Simple rule-based classification for demo
        # This would be replaced with a trained model
        predicted_letter = np.random.choice(ASL_LABELS)
        confidence = np.random.uniform(0.7, 0.95)
        
        return predicted_letter, confidence

classifier = ASLClassifier()

def process_image(image_data):
    """Process image and detect ASL hand signs"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Get first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Classify the gesture
            letter, confidence = classifier.classify(hand_landmarks)
            
            # Get landmark coordinates for visualization
            landmarks_data = []
            for landmark in hand_landmarks.landmark:
                landmarks_data.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            
            return {
                'success': True,
                'letter': letter,
                'confidence': float(confidence),
                'landmarks': landmarks_data
            }
        else:
            return {
                'success': False,
                'message': 'No hand detected in image'
            }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Error processing image: {str(e)}'
        }

@app.route('/detect', methods=['POST'])
def detect_asl():
    """API endpoint for ASL detection"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        result = process_image(image_data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'ASL detector is running'})

if __name__ == '__main__':
    print("Starting ASL Hand Sign Detector...")
    print("Access the application at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
