from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('models', exist_ok=True)

class AdClickPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = ['age', 'daily_time_spent', 'area_income', 'daily_internet_usage', 'gender', 'device_type', 'ad_topic']
        self.model_trained = False
        
    def generate_sample_data(self, n_samples=1000):
        """Generate realistic advertising data"""
        np.random.seed(42)
        
        # Generate base features
        age = np.random.normal(35, 12, n_samples).astype(int)
        age = np.clip(age, 18, 65)
        
        daily_time_spent = np.random.normal(65, 15, n_samples)
        daily_time_spent = np.clip(daily_time_spent, 10, 120)
        
        area_income = np.random.normal(55000, 15000, n_samples)
        area_income = np.clip(area_income, 25000, 100000)
        
        daily_internet_usage = np.random.normal(180, 45, n_samples)
        daily_internet_usage = np.clip(daily_internet_usage, 30, 300)
        
        gender = np.random.choice(['Male', 'Female'], n_samples)
        device_type = np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples, p=[0.6, 0.3, 0.1])
        ad_topic = np.random.choice(['Technology', 'Fashion', 'Health', 'Travel', 'Finance', 'Food'], n_samples)
        
        # Create realistic click patterns
        click_probability = (
            0.3 +
            0.2 * (age < 30).astype(int) +
            0.15 * (daily_time_spent > 70).astype(int) +
            0.1 * (area_income > 60000).astype(int) +
            0.1 * (daily_internet_usage > 200).astype(int) +
            0.05 * (gender == 'Female').astype(int) +
            0.1 * (device_type == 'Mobile').astype(int) +
            0.05 * np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        )
        
        clicked = np.random.binomial(1, np.clip(click_probability, 0, 1), n_samples)
        
        df = pd.DataFrame({
            'age': age,
            'daily_time_spent': daily_time_spent,
            'area_income': area_income,
            'daily_internet_usage': daily_internet_usage,
            'gender': gender,
            'device_type': device_type,
            'ad_topic': ad_topic,
            'clicked': clicked
        })
        
        return df
    
    def preprocess_data(self, df, fit_encoders=True):
        """Preprocess the data for training/prediction"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'device_type', 'ad_topic']
        
        for col in categorical_cols:
            if fit_encoders:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    try:
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col])
                    except ValueError as e:
                        # If unseen category, assign to most frequent class
                        most_frequent = self.label_encoders[col].classes_[0]
                        df_processed[col] = df_processed[col].fillna(most_frequent)
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def train_model(self):
        """Train the logistic regression model"""
        try:
            # Generate sample data
            df = self.generate_sample_data(2000)
            
            # Preprocess data
            df_processed = self.preprocess_data(df, fit_encoders=True)
            
            # Split features and target
            X = df_processed[self.feature_names]
            y = df_processed['clicked']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_trained = True
            
            # Save model and preprocessors
            self.save_model()
            
            # Get classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            return {
                'accuracy': float(accuracy),
                'classification_report': class_report,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'feature_importance': self.model.coef_[0].tolist()
            }
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def predict_click_probability(self, user_data):
        """Predict click probability for a user"""
        try:
            if not self.model_trained:
                self.load_model()
            
            # Validate input data
            required_fields = ['age', 'daily_time_spent', 'area_income', 'daily_internet_usage', 'gender', 'device_type', 'ad_topic']
            for field in required_fields:
                if field not in user_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create DataFrame
            df = pd.DataFrame([user_data])
            
            # Preprocess
            df_processed = self.preprocess_data(df, fit_encoders=False)
            
            # Scale features
            X_scaled = self.scaler.transform(df_processed[self.feature_names])
            
            # Predict
            probability = self.model.predict_proba(X_scaled)[0][1]
            prediction = self.model.predict(X_scaled)[0]
            
            return {
                'probability': float(probability),
                'prediction': int(prediction),
                'confidence': float(max(self.model.predict_proba(X_scaled)[0]))
            }
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        try:
            joblib.dump(self.model, 'models/ad_click_model.pkl')
            joblib.dump(self.scaler, 'models/scaler.pkl')
            joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
            
            # Save metadata
            metadata = {
                'model_trained': self.model_trained,
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat()
            }
            with open('models/metadata.json', 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            if (os.path.exists('models/ad_click_model.pkl') and 
                os.path.exists('models/scaler.pkl') and 
                os.path.exists('models/label_encoders.pkl')):
                
                self.model = joblib.load('models/ad_click_model.pkl')
                self.scaler = joblib.load('models/scaler.pkl')
                self.label_encoders = joblib.load('models/label_encoders.pkl')
                self.model_trained = True
            else:
                # If model doesn't exist, train it
                print("Model not found. Training new model...")
                self.train_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}. Training new model...")
            self.train_model()

# Initialize predictor
predictor = AdClickPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model"""
    try:
        results = predictor.train_model()
        return jsonify({
            'success': True,
            'results': results,
            'message': 'Model trained successfully!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate input
        required_fields = ['age', 'daily_time_spent', 'area_income', 'daily_internet_usage', 'gender', 'device_type', 'ad_topic']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing field: {field}'
                }), 400
        
        # Convert numeric fields
        try:
            data['age'] = int(data['age'])
            data['daily_time_spent'] = float(data['daily_time_spent'])
            data['area_income'] = float(data['area_income'])
            data['daily_internet_usage'] = float(data['daily_internet_usage'])
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid numeric values'
            }), 400
        
        # Validate ranges
        if not (18 <= data['age'] <= 65):
            return jsonify({
                'success': False,
                'error': 'Age must be between 18 and 65'
            }), 400
        
        if not (10 <= data['daily_time_spent'] <= 120):
            return jsonify({
                'success': False,
                'error': 'Daily time spent must be between 10 and 120 minutes'
            }), 400
        
        if not (25000 <= data['area_income'] <= 100000):
            return jsonify({
                'success': False,
                'error': 'Area income must be between $25,000 and $100,000'
            }), 400
        
        if not (30 <= data['daily_internet_usage'] <= 300):
            return jsonify({
                'success': False,
                'error': 'Daily internet usage must be between 30 and 300 minutes'
            }), 400
        
        # Make prediction
        result = predictor.predict_click_probability(data)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analytics')
def analytics():
    """Get analytics data"""
    try:
        # Generate sample analytics data with some randomization
        base_time = datetime.now().timestamp()
        
        analytics_data = {
            'click_rates_by_age': {
                '18-25': round(0.35 + np.random.normal(0, 0.02), 3),
                '26-35': round(0.42 + np.random.normal(0, 0.02), 3),
                '36-45': round(0.28 + np.random.normal(0, 0.02), 3),
                '46-55': round(0.22 + np.random.normal(0, 0.02), 3),
                '56-65': round(0.18 + np.random.normal(0, 0.02), 3)
            },
            'click_rates_by_device': {
                'Mobile': round(0.38 + np.random.normal(0, 0.02), 3),
                'Desktop': round(0.25 + np.random.normal(0, 0.02), 3),
                'Tablet': round(0.20 + np.random.normal(0, 0.02), 3)
            },
            'click_rates_by_topic': {
                'Technology': round(0.45 + np.random.normal(0, 0.02), 3),
                'Fashion': round(0.38 + np.random.normal(0, 0.02), 3),
                'Health': round(0.32 + np.random.normal(0, 0.02), 3),
                'Travel': round(0.28 + np.random.normal(0, 0.02), 3),
                'Finance': round(0.22 + np.random.normal(0, 0.02), 3),
                'Food': round(0.35 + np.random.normal(0, 0.02), 3)
            },
            'daily_trends': [
                {'day': 'Monday', 'clicks': int(1250 + np.random.normal(0, 50))},
                {'day': 'Tuesday', 'clicks': int(1380 + np.random.normal(0, 50))},
                {'day': 'Wednesday', 'clicks': int(1420 + np.random.normal(0, 50))},
                {'day': 'Thursday', 'clicks': int(1390 + np.random.normal(0, 50))},
                {'day': 'Friday', 'clicks': int(1600 + np.random.normal(0, 50))},
                {'day': 'Saturday', 'clicks': int(1800 + np.random.normal(0, 50))},
                {'day': 'Sunday', 'clicks': int(1650 + np.random.normal(0, 50))}
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': predictor.model_trained,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Try to load existing model on startup
    try:
        predictor.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Could not load model: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)