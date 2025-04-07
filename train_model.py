import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import os
import logging
from collections import Counter
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    try:
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Expanded medical variations dictionary
        medical_variations = {
            # Common symptom variations
            'ache': 'pain',
            'painful': 'pain',
            'hurts': 'pain',
            'sore': 'pain',
            'discomfort': 'pain',
            
            # Respiratory symptoms
            'short of breath': 'dyspnea',
            'breathless': 'dyspnea',
            'cant breathe': 'dyspnea',
            'breathing difficulty': 'dyspnea',
            'wheezing': 'respiratory_distress',
            
            # Gastrointestinal symptoms
            'stomach ache': 'abdominal_pain',
            'belly pain': 'abdominal_pain',
            'throwing up': 'vomiting',
            'feeling sick': 'nausea',
            
            # Neurological symptoms
            'dizzy': 'dizziness',
            'lightheaded': 'dizziness',
            'vertigo': 'dizziness',
            'migraine': 'severe_headache',
            
            # Cardiovascular symptoms
            'heart racing': 'palpitations',
            'rapid heartbeat': 'palpitations',
            'chest pain': 'chest_discomfort',
            
            # Temperature related
            'high temperature': 'fever',
            'low temperature': 'hypothermia',
            'chills': 'fever',
            
            # Duration indicators
            'constant': 'persistent',
            'continuous': 'persistent',
            'intermittent': 'recurring',
            'occasional': 'recurring',
            
            # Severity indicators
            'intense': 'severe',
            'extreme': 'severe',
            'mild': 'slight',
            'minor': 'slight'
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
            'chronic', 'sudden', 'gradual', 'constant', 'intermittent',
            'persistent', 'recurring', 'worsening', 'improving', 'stable',
            'left', 'right', 'upper', 'lower', 'both', 'either'
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

def create_symptom_combinations(symptoms):
    """Create more focused symptom combinations"""
    try:
        # Split by both comma and semicolon
        symptoms_list = [s.strip() for s in re.split('[,;]', symptoms) if s.strip()]
        
        if not symptoms_list:
            return [symptoms]
            
        combinations = []
        
        # Add individual symptoms with relevant modifiers only
        for symptom in symptoms_list:
            combinations.append(symptom)
            if any(word in symptom.lower() for word in ['pain', 'ache', 'discomfort']):
                combinations.extend([f"severe {symptom}", f"mild {symptom}"])
            if any(word in symptom.lower() for word in ['cough', 'fever', 'headache']):
                combinations.extend([f"acute {symptom}", f"chronic {symptom}"])
        
        # Create meaningful pairs only (limit to 2 symptoms)
        for i in range(len(symptoms_list)):
            for j in range(i + 1, min(i + 2, len(symptoms_list))):
                combinations.append(f"{symptoms_list[i]} with {symptoms_list[j]}")
        
        return list(set(combinations))  # Remove duplicates
    except Exception as e:
        logger.error(f"Error creating symptom combinations: {str(e)}")
        return [symptoms]

def save_model_files(model, vectorizer, expanded_df):
    """Save model files with error handling"""
    try:
        files = {
            "model.pkl": model,
            "vectorizer.pkl": vectorizer,
            "expanded_data.pkl": expanded_df
        }
        
        for filename, data in files.items():
            with open(filename, "wb") as f:
                pickle.dump(data, f)
        logger.info("Model files saved successfully")
    except Exception as e:
        logger.error(f"Error saving model files: {str(e)}")
        raise

def filter_classes_with_min_samples(df, min_samples=1):
    """Filter out classes with fewer than min_samples samples"""
    class_counts = df['Disease'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    filtered_df = df[df['Disease'].isin(valid_classes)]
    logger.info(f"Removed {len(df) - len(filtered_df)} samples from classes with fewer than {min_samples} samples")
    logger.info(f"Remaining classes: {len(valid_classes)}")
    logger.info(f"Class distribution: {dict(class_counts[valid_classes])}")
    return filtered_df

def create_balanced_dataset(df, target_size=5):
    """Create a balanced dataset by duplicating samples"""
    balanced_data = []
    class_counts = df['Disease'].value_counts()
    
    for disease, count in class_counts.items():
        disease_samples = df[df['Disease'] == disease]
        # Calculate how many times to duplicate each sample
        multiplier = target_size // count
        remainder = target_size % count
        
        # Duplicate samples
        for _ in range(multiplier):
            balanced_data.extend(disease_samples.to_dict('records'))
        
        # Add remaining samples if needed
        if remainder > 0:
            balanced_data.extend(disease_samples.head(remainder).to_dict('records'))
    
    return pd.DataFrame(balanced_data)

def create_ensemble_model():
    """Create an improved ensemble model"""
    # Random Forest with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost with optimized parameters
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Replace LinearSVC with LogisticRegression
    lr = LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=3000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )
    
    # Naive Bayes with optimized parameters
    nb = MultinomialNB(
        alpha=1.0,
        fit_prior=True
    )
    
    # Create voting ensemble with soft voting
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('lr', lr),
            ('nb', nb)
        ],
        voting='soft',
        weights=[2, 2, 1, 1],
        n_jobs=-1
    )
    
    return ensemble

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

def create_feature_matrix(symptoms_series):
    """Create additional feature matrix"""
    feature_dicts = symptoms_series.apply(extract_medical_features)
    return pd.DataFrame(feature_dicts.tolist())

def validate_dataset(df):
    """Validate and clean the dataset with less strict filtering"""
    logger.info(f"Initial dataset size: {len(df)} rows")
    
    # Instead of dropping duplicates, combine similar entries
    df_grouped = df.groupby(['Disease']).agg({
        'Symptoms': lambda x: ' ; '.join(set(x)),
        'Description': 'first',
        'Precautions': 'first',
        'Medications': 'first',
        'Workouts': 'first',
        'Diets': 'first'
    }).reset_index()
    
    logger.info(f"After combining similar entries: {len(df_grouped)} rows")
    
    # Remove rows with missing values
    df_clean = df_grouped.dropna(subset=['Symptoms', 'Disease'])
    logger.info(f"After removing null values: {len(df_clean)} rows")
    
    return df_clean

def main():
    try:
        # Load dataset
        df = pd.read_csv("dataset.csv")
        logger.info(f"Loaded dataset with {len(df)} rows")
        
        # Validate and clean dataset
        df = validate_dataset(df)
        if len(df) == 0:
            raise ValueError("Dataset is empty after validation. Please check validation criteria.")
        
        # Balance the dataset
        df = create_balanced_dataset(df, target_size=10)  # Ensure at least 10 samples per class
        logger.info(f"Dataset after balancing: {len(df)} rows")
        
        # Preprocess symptoms
        df['Processed_Symptoms'] = df['Symptoms'].apply(preprocess_text)
        
        # Create expanded dataset with combinations
        expanded_data = []
        for _, row in df.iterrows():
            combinations = create_symptom_combinations(row['Processed_Symptoms'])
            for combo in combinations:
                expanded_data.append({
                    'Processed_Symptoms': combo,
                    'Disease': row['Disease'],
                    'Description': row.get('Description', ''),
                    'Precautions': row.get('Precautions', ''),
                    'Medications': row.get('Medications', ''),
                    'Workouts': row.get('Workouts', ''),
                    'Diets': row.get('Diets', '')
                })
        
        expanded_df = pd.DataFrame(expanded_data)
        logger.info(f"Created expanded dataset with {len(expanded_df)} rows")
        
        # Initialize TF-IDF vectorizer with optimized parameters
        vectorizer = TfidfVectorizer(
            max_features=2000,    # Reduced from 3000
            ngram_range=(1, 3),   # Reduced from (1, 5)
            min_df=1,            # Changed from 2
            max_df=0.9,          # Reduced from 0.95
            stop_words='english',
            analyzer='word',
            token_pattern=r'[a-zA-Z_]+',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # Create features
        tfidf_features = vectorizer.fit_transform(expanded_df["Processed_Symptoms"])
        medical_features = create_feature_matrix(expanded_df["Processed_Symptoms"])
        
        # Combine features
        X = hstack([tfidf_features, medical_features.values])
        y = expanded_df["Disease"]
        
        # Split data with larger test size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train ensemble model
        model = create_ensemble_model()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model files
        save_model_files(model, vectorizer, expanded_df)
        print("\nModel training complete. Files saved!")
        
    except Exception as e:
        logger.error(f"Error in main training process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
