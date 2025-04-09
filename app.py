from flask import Flask, render_template, request, jsonify, session
import pickle
import pandas as pd
import webbrowser
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import logging
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Create data directory if it doesn't exist
        nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK data
        for package in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                nltk.download(package, download_dir=nltk_data_dir)
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        # Continue execution even if download fails
        pass

download_nltk_data()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Enhanced follow-up questions with more detailed categories
FOLLOW_UP_QUESTIONS = {
    'pain': [
        "Where exactly is the pain located?",
        "How would you rate the pain on a scale of 1-10?",
        "Is the pain constant or does it come and go?",
        "What makes the pain better or worse?",
        "When did the pain start?",
        "Does the pain radiate to other areas?",
        "Have you taken any pain medication?",
        "Does the pain affect your daily activities?"
    ],
    'nausea': [
        "How long have you been experiencing nausea?",
        "Have you vomited? If yes, how many times?",
        "Are you able to keep food/water down?",
        "Does any particular food/smell trigger the nausea?",
        "Do you have any abdominal pain?",
        "Have you noticed any changes in appetite?",
        "Are you experiencing any dizziness?",
        "Have you had any recent changes in diet?"
    ],
    'respiratory': [
        "Is your cough dry or productive (with mucus)?",
        "Do you have shortness of breath when lying down or during activity?",
        "Have you been exposed to anyone with similar symptoms recently?",
        "Do you have any chest pain or tightness?",
        "How long have you had these symptoms?",
        "Does the cough worsen at night or in the morning?",
        "Have you noticed any change in the color of mucus?",
        "Do you have any allergies or asthma history?"
    ],
    'headache': [
        "Is the pain on one side or both sides of your head?",
        "Does light or sound make the pain worse?",
        "How long does the pain typically last?",
        "Do you experience any visual disturbances before the pain?",
        "Is the pain throbbing or constant?",
        "Do you have any neck stiffness?",
        "Have you had any recent head injury?",
        "Does the pain worsen with physical activity?"
    ],
    'gastrointestinal': [
        "When did the symptoms start?",
        "Have you eaten anything unusual recently?",
        "Is the pain constant or does it come and go?",
        "Do you have any fever or chills?",
        "Have you noticed any blood in your stool?",
        "Are you experiencing any bloating?",
        "Have you lost weight unintentionally?",
        "Does eating make the symptoms better or worse?"
    ],
    'cardiac': [
        "Is the chest pain constant or intermittent?",
        "Does the pain radiate to your arm, jaw, or back?",
        "Do you have any shortness of breath?",
        "Are you experiencing any dizziness or lightheadedness?",
        "Do you have any history of heart problems?",
        "Does the pain worsen with exertion?",
        "Have you noticed any irregular heartbeats?",
        "Do you have any swelling in your legs?"
    ],
    'skin': [
        "When did the rash first appear?",
        "Is the rash itchy or painful?",
        "Have you been exposed to any new products or foods?",
        "Does the rash spread or change in appearance?",
        "Is the affected area warm to touch?",
        "Have you noticed any blisters or scaling?",
        "Does the rash come and go?",
        "Have you used any medications for it?"
    ],
    'neurological': [
        "Are you experiencing any numbness or tingling?",
        "Do you have any difficulty with balance or coordination?",
        "Have you noticed any changes in vision or speech?",
        "Do you have any memory problems?",
        "Are you experiencing any tremors?",
        "Do you have any history of seizures?",
        "Have you had any recent head trauma?",
        "Do symptoms worsen with certain activities?"
    ],
    'musculoskeletal': [
        "Where exactly is the pain located?",
        "Does the pain worsen with movement?",
        "Have you had any recent injuries?",
        "Is there any swelling or redness?",
        "Does rest improve the symptoms?",
        "Is the pain sharp or dull?",
        "Do you have any joint stiffness?",
        "Have you noticed any muscle weakness?"
    ],
    'psychological': [
        "How long have you been feeling this way?",
        "Are your symptoms affecting your sleep?",
        "Have you noticed changes in your appetite?",
        "Do you feel more anxious or depressed?",
        "Are you able to concentrate on tasks?",
        "Have you had any panic attacks?",
        "Are your symptoms affecting your daily activities?",
        "Have you experienced any recent major life changes?"
    ]
}

def preprocess_text(text):
    try:
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        # Expanded medical variations dictionary
        medical_variations = {
            # Fever related
            'fever': 'elevated_temperature',
            'high temperature': 'elevated_temperature',
            'hot': 'elevated_temperature',
            'chills': 'fever_chills',
            'shivering': 'fever_chills',
            
            # Dengue specific
            'bone pain': 'severe_joint_pain',
            'eye pain': 'retro_orbital_pain',
            'muscle pain': 'myalgia',
            'joint pain': 'arthralgia',
            'rash': 'skin_rash',
            'bleeding': 'hemorrhage',
            
            # Other conditions
            'tired': 'fatigue',
            'exhausted': 'fatigue',
            'weak': 'weakness',
            'dizzy': 'dizziness',
            'headache': 'cephalgia',
            'throat pain': 'sore_throat',
            'runny nose': 'rhinorrhea',
            'stuffy nose': 'nasal_congestion',
            'coughing': 'cough',
            'short of breath': 'dyspnea',
            'stomach pain': 'abdominal_pain',
            'diarrhea': 'loose_stools',
            'vomiting': 'emesis',
            'nausea': 'nausea',
            'body ache': 'myalgia'
        }
        
        # Apply variations
        for old, new in medical_variations.items():
            text = re.sub(r'\b' + old + r'\b', new, text)
        
        # Handle measurements and numbers
        text = re.sub(r'(\d+)([°℃℉])', r'\1 degrees', text)
        text = re.sub(r'(\d+)\s*(days?|weeks?|months?|years?)', r'\1 time', text)
        text = re.sub(r'(\d+)\s*(hrs?|hours?|minutes?)', r'\1 duration', text)
        
        # Keep important medical punctuation
        text = text.replace(',', ' and ')
        text = text.replace('/', ' or ')
        text = text.replace('+', ' and ')
        
        # Remove other special characters but keep underscores
        text = re.sub(r'[^a-zA-Z_\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Expanded medical stopwords to keep
        medical_stopwords = {
            'pain', 'ache', 'fever', 'cough', 'cold', 'sore', 'swelling', 
            'rash', 'nausea', 'vomit', 'diarrhea', 'chest', 'head',
            'stomach', 'back', 'joint', 'muscle', 'skin', 'throat',
            'nose', 'ear', 'eye', 'mouth', 'tongue', 'neck', 'arm',
            'leg', 'foot', 'hand', 'blood', 'heart', 'lung', 'liver',
            'kidney', 'brain', 'severe', 'mild', 'moderate', 'acute',
            'chronic', 'sudden', 'gradual', 'constant', 'intermittent'
        }
        
        # Remove stopwords but keep medical terms
        stop_words = set(stopwords.words('english')) - medical_stopwords
        tokens = [token for token in tokens if token not in stop_words]
        
        # Double lemmatization with different POS tags
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
        tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        return ""

def extract_medical_features(text):
    """Extract comprehensive medical features from text"""
    features = {}
    
    # Severity patterns
    severity_patterns = {
        'mild': r'\b(mild|slight|minor)\b',
        'moderate': r'\b(moderate|medium)\b',
        'severe': r'\b(severe|intense|extreme|serious)\b',
        'critical': r'\b(critical|emergency|life.?threatening)\b'
    }
    
    # Duration patterns
    duration_patterns = {
        'acute': r'\b(acute|sudden|recent)\b',
        'chronic': r'\b(chronic|long.?term|ongoing)\b',
        'recurring': r'\b(recurring|intermittent|periodic)\b',
        'persistent': r'\b(persistent|constant|continuous)\b'
    }
    
    # Progression patterns
    progression_patterns = {
        'worsening': r'\b(worsening|deteriorating|getting worse)\b',
        'improving': r'\b(improving|getting better|resolving)\b',
        'stable': r'\b(stable|unchanged|consistent)\b',
        'fluctuating': r'\b(fluctuating|varying|changing)\b'
    }
    
    # Location patterns
    location_patterns = {
        'left': r'\b(left|left.?sided)\b',
        'right': r'\b(right|right.?sided)\b',
        'bilateral': r'\b(bilateral|both sides|both)\b',
        'upper': r'\b(upper|above)\b',
        'lower': r'\b(lower|below)\b'
    }
    
    # Add all pattern matches to features
    for category in [severity_patterns, duration_patterns, progression_patterns, location_patterns]:
        for name, pattern in category.items():
            features[name] = 1 if re.search(pattern, text, re.IGNORECASE) is not None else 0
    
    # Additional binary features
    features.update({
        'has_pain': 1 if re.search(r'\b(pain|ache|sore|hurt)', text, re.IGNORECASE) is not None else 0,
        'has_fever': 1 if re.search(r'\b(fever|temperature|chills)', text, re.IGNORECASE) is not None else 0,
        'has_breathing_issues': 1 if re.search(r'\b(breath|dyspnea|wheez)', text, re.IGNORECASE) is not None else 0,
        'has_gi_symptoms': 1 if re.search(r'\b(nausea|vomit|diarrhea)', text, re.IGNORECASE) is not None else 0,
        'has_skin_symptoms': 1 if re.search(r'\b(rash|itch|skin)', text, re.IGNORECASE) is not None else 0,
        'has_neuro_symptoms': 1 if re.search(r'\b(dizzy|headache|confusion)', text, re.IGNORECASE) is not None else 0
    })
    
    return features

def load_model_files():
    try:
        model_path = "model.pkl"
        vectorizer_path = "vectorizer.pkl"
        data_path = "expanded_data.pkl"
        
        if not all(os.path.exists(path) for path in [model_path, vectorizer_path, data_path]):
            raise FileNotFoundError("Required model files are missing. Please run train_model.py first.")
        
        model = pickle.load(open(model_path, "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))
        expanded_df = pickle.load(open(data_path, "rb"))
        
        return model, vectorizer, expanded_df
    except Exception as e:
        logger.error(f"Error loading model files: {str(e)}")
        raise

def get_symptom_category(symptoms):
    """Enhanced function to determine symptom categories with more specific conditions"""
    symptoms = symptoms.lower()
    
    # Enhanced categories with more specific symptoms and conditions
    categories = {
        'pain': [
            'pain', 'ache', 'hurt', 'sore', 'discomfort', 'burning', 'stinging',
            'cramping', 'throbbing', 'sharp', 'dull', 'tender'
        ],
        'nausea': [
            'nausea', 'vomit', 'sick to stomach', 'queasy', 'throwing up',
            'gagging', 'retching', 'stomach upset'
        ],
        'fever': [
            'fever', 'temperature', 'chills', 'shivering', 'hot', 'cold',
            'sweating', 'night sweats', 'fever_chills', 'elevated_temperature'
        ],
        'respiratory': [
            'cough', 'breath', 'chest', 'wheezing', 'pneumonia', 'congestion',
            'sputum', 'phlegm', 'sneeze', 'bronchitis', 'asthma', 'respiratory'
        ],
        'gastrointestinal': [
            'stomach', 'abdominal', 'bowel', 'digestive', 'acid reflux', 
            'bloating', 'gas', 'indigestion', 'diarrhea', 'constipation'
        ],
        'dengue_specific': [
            'joint pain', 'bone pain', 'eye pain', 'rash', 'bleeding',
            'petechiae', 'severe_joint_pain', 'retro_orbital_pain', 'hemorrhage'
        ],
        'cardiac': [
            'chest pain', 'heart', 'cardiac', 'palpitation', 'shortness of breath',
            'angina', 'arrhythmia', 'tachycardia', 'cardiovascular'
        ],
        'skin': [
            'rash', 'itching', 'swelling', 'hives', 'eczema', 'dermatitis',
            'skin lesion', 'blister', 'psoriasis', 'acne', 'dermis'
        ],
        'neurological': [
            'numbness', 'tingling', 'seizure', 'tremor', 'confusion', 'dizziness',
            'balance', 'coordination', 'memory', 'neural', 'brain'
        ],
        'musculoskeletal': [
            'muscle', 'joint', 'bone', 'arthritis', 'sprain', 'strain',
            'back pain', 'neck pain', 'fracture', 'tendon', 'ligament'
        ],
        'psychological': [
            'anxiety', 'depression', 'stress', 'mood', 'panic', 'mental',
            'emotional', 'psychological', 'behavioral', 'psychiatric'
        ]
    }
    
    # Symptom combinations that suggest specific conditions
    symptom_combinations = {
        'dengue_fever': {
            'required': ['fever'],
            'optional': ['joint pain', 'eye pain', 'rash', 'bleeding', 'headache'],
            'min_optional': 2
        },
        'migraine': {
            'required': ['headache'],
            'optional': ['nausea', 'light sensitivity', 'sound sensitivity', 'visual disturbance'],
            'min_optional': 1
        },
        'food_poisoning': {
            'required': ['nausea'],
            'optional': ['vomiting', 'diarrhea', 'stomach pain', 'fever'],
            'min_optional': 1
        }
    }
    
    # Check for symptom combinations first
    matched_conditions = []
    for condition, criteria in symptom_combinations.items():
        required_match = all(any(req in symptoms for req in criteria['required']))
        optional_count = sum(1 for opt in criteria['optional'] if any(o in symptoms for o in opt.split('|')))
        
        if required_match and optional_count >= criteria['min_optional']:
            matched_conditions.append(condition)
    
    # Then check individual categories
    matched_categories = []
    for category, keywords in categories.items():
        if any(keyword in symptoms for keyword in keywords):
            matched_categories.append(category)
    
    # Combine and prioritize results
    all_matches = matched_conditions + matched_categories
    return all_matches if all_matches else None

def get_follow_up_questions(symptoms):
    """Enhanced function to get more specific and relevant follow-up questions"""
    categories = get_symptom_category(symptoms)
    if not categories:
        return []
    
    all_questions = []
    seen_questions = set()
    
    # Process each category and its questions
    for category in categories:
        if category in FOLLOW_UP_QUESTIONS:
            category_questions = FOLLOW_UP_QUESTIONS[category]
            
            # Add questions that haven't been asked yet
            for question in category_questions:
                # Create a normalized version of the question for comparison
                normalized_q = question.lower().strip()
                if normalized_q not in seen_questions:
                    all_questions.append(question)
                    seen_questions.add(normalized_q)
    
    # Limit to most relevant questions (max 8)
    return all_questions[:8]

# Load trained model, vectorizer, and expanded data
try:
    model, vectorizer, expanded_df = load_model_files()
except Exception as e:
    logger.error(f"Failed to load model files: {str(e)}")
    model = None
    vectorizer = None
    expanded_df = None

@app.route("/")
def home():
    session.clear()  # Clear any existing session data
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model is loaded
        if model is None or vectorizer is None or expanded_df is None:
            logger.error("Model files not loaded properly")
            return render_template("error.html", error="Model not properly initialized")

        # Get symptoms from form
        symptoms = request.form.get("symptoms", "").strip()
        logger.info(f"Received symptoms: {symptoms}")

        if not symptoms:
            return render_template("error.html", error="Please enter your symptoms")

        # Process symptoms
        processed_symptoms = preprocess_text(symptoms)
        logger.info(f"Processed symptoms: {processed_symptoms}")

        # Create features
        tfidf_features = vectorizer.transform([processed_symptoms])
        medical_features = extract_medical_features(processed_symptoms)
        medical_features_df = pd.DataFrame([medical_features])
        
        # Combine features
        X = hstack([tfidf_features, medical_features_df.values])
        
        # Get predictions
        probabilities = model.predict_proba(X)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        predictions = [
            {
                'disease': model.classes_[idx],
                'probability': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        # Get main prediction and details
        main_prediction = predictions[0]
        disease_details = expanded_df[expanded_df['Disease'] == main_prediction['disease']].iloc[0]

        logger.info(f"Prediction made: {main_prediction['disease']}")
        
        return render_template(
            "result.html",
            disease=main_prediction['disease'],
            confidence=f"{main_prediction['probability']*100:.2f}%",
            description=disease_details.get('Description', ''),
            precautions=disease_details.get('Precautions', ''),
            medications=disease_details.get('Medications', ''),
            workouts=disease_details.get('Workouts', ''),
            diets=disease_details.get('Diets', ''),
            all_predictions=predictions
        )

    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return render_template("error.html", error="Unable to process request. Please try again.")

if __name__ == "__main__":
    # Initialize NLTK data
    download_nltk_data()
    
    # Get port from environment variable or default to 10000
    port = int(os.environ.get("PORT", 10000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port)

