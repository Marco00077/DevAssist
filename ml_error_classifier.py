"""
Machine Learning Error Classifier for Python Code Analysis
Custom ML models trained on Python error datasets - No external APIs
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import ast
from pathlib import Path
import json

class MLErrorClassifier:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.is_trained = False
        
        # Create models directory
        self.models_dir = Path("ml_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize ML models
        self._initialize_models()
        
        # Load or create training data
        self.training_data = self._create_training_dataset()
        
        print("🤖 ML Error Classifier initialized")
    
    def _initialize_models(self):
        """Initialize multiple ML models for ensemble learning"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=0.1),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        # Text vectorizers for different features
        self.vectorizers = {
            'tfidf': TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            ),
            'count': CountVectorizer(
                max_features=500,
                ngram_range=(1, 2)
            )
        }
    
    def _create_training_dataset(self):
        """Create comprehensive training dataset for Python errors"""
        training_data = {
            'code_snippets': [],
            'error_types': [],
            'error_messages': [],
            'suggestions': [],
            'confidence_scores': []
        }
        
        # Syntax Error Examples
        syntax_errors = [
            # Missing colon errors
            ("if x > 0\n    print('positive')", "SyntaxError", "Missing colon after if statement", "Add colon (:) after if condition", 0.9),
            ("for i in range(10)\n    print(i)", "SyntaxError", "Missing colon after for statement", "Add colon (:) after for loop", 0.9),
            ("def my_function()\n    return 5", "SyntaxError", "Missing colon after function definition", "Add colon (:) after function definition", 0.9),
            ("while True\n    break", "SyntaxError", "Missing colon after while statement", "Add colon (:) after while condition", 0.9),
            ("class MyClass\n    pass", "SyntaxError", "Missing colon after class definition", "Add colon (:) after class definition", 0.9),
            
            # Indentation errors
            ("if x > 0:\nprint('positive')", "IndentationError", "Expected indented block", "Indent the code block properly", 0.95),
            ("def func():\nreturn 5", "IndentationError", "Expected indented block after function", "Indent the function body", 0.95),
            ("for i in range(5):\ni += 1", "IndentationError", "Expected indented block in loop", "Indent the loop body", 0.95),
            
            # Parentheses errors
            ("print('hello'", "SyntaxError", "Unmatched parentheses", "Add closing parenthesis", 0.85),
            ("if (x > 0:\n    print('positive')", "SyntaxError", "Unmatched parentheses in condition", "Check parentheses matching", 0.85),
            ("my_list = [1, 2, 3", "SyntaxError", "Unmatched brackets", "Add closing bracket", 0.85),
            
            # Quote errors
            ("print('hello)", "SyntaxError", "Unmatched quotes", "Match opening and closing quotes", 0.85),
            ("name = 'John", "SyntaxError", "Unterminated string", "Add closing quote", 0.85),
        ]
        
        # Name Error Examples
        name_errors = [
            ("print(undefined_var)", "NameError", "Variable not defined", "Define the variable before using it", 0.9),
            ("result = x + y", "NameError", "Variable 'x' not defined", "Define variable 'x' before using", 0.9),
            ("my_function()", "NameError", "Function not defined", "Define the function or check spelling", 0.85),
            ("import math\nprint(pi)", "NameError", "Name 'pi' not defined", "Use math.pi instead of pi", 0.8),
        ]
        
        # Type Error Examples
        type_errors = [
            ("'hello' + 5", "TypeError", "Cannot concatenate string and integer", "Convert integer to string: 'hello' + str(5)", 0.9),
            ("len(5)", "TypeError", "Object has no len()", "Use len() only with sequences like strings or lists", 0.85),
            ("'hello'[1.5]", "TypeError", "String indices must be integers", "Use integer index: 'hello'[1]", 0.85),
            ("5()", "TypeError", "Integer object is not callable", "Check if you meant to call a function", 0.8),
        ]
        
        # OCR-specific typos
        ocr_typos = [
            ("num = it(input('Enter number: '))", "OCRTypo", "it should be int", "Change 'it' to 'int' for integer conversion", 0.95),
            ("pirnt('Hello World')", "OCRTypo", "pirnt should be print", "Change 'pirnt' to 'print'", 0.95),
            ("imoprt math", "OCRTypo", "imoprt should be import", "Change 'imoprt' to 'import'", 0.95),
            ("retrun 5", "OCRTypo", "retrun should be return", "Change 'retrun' to 'return'", 0.95),
            ("if nu > 0:", "OCRTypo", "nu should be num", "Change 'nu' to 'num'", 0.95),
            ("retrn x + y", "OCRTypo", "retrn should be return", "Change 'retrn' to 'return'", 0.95),
            ("inpt('Enter value: ')", "OCRTypo", "inpt should be input", "Change 'inpt' to 'input'", 0.9),
            ("if x > @:", "OCRTypo", "@ should be 0", "Change '@' symbol to '0' (zero)", 0.9),
        ]
        
        # Logic Error Examples
        logic_errors = [
            ("if x = 5:", "LogicError", "Assignment in condition", "Use == for comparison, not =", 0.85),
            ("for i in range(10):\n    print(j)", "LogicError", "Wrong variable in loop", "Use loop variable 'i' instead of 'j'", 0.8),
            ("def func():\n    print('hello')", "LogicError", "Function missing return", "Add return statement if needed", 0.6),
        ]
        
        # Combine all error types
        all_errors = syntax_errors + name_errors + type_errors + ocr_typos + logic_errors
        
        # Add to training data
        for code, error_type, message, suggestion, confidence in all_errors:
            training_data['code_snippets'].append(code)
            training_data['error_types'].append(error_type)
            training_data['error_messages'].append(message)
            training_data['suggestions'].append(suggestion)
            training_data['confidence_scores'].append(confidence)
        
        # Add some correct code examples
        correct_examples = [
            ("if x > 0:\n    print('positive')", "NoError", "Code is correct", "No changes needed", 1.0),
            ("for i in range(10):\n    print(i)", "NoError", "Code is correct", "No changes needed", 1.0),
            ("def my_func():\n    return 5", "NoError", "Code is correct", "No changes needed", 1.0),
            ("import math\nprint(math.pi)", "NoError", "Code is correct", "No changes needed", 1.0),
            ("name = 'John'\nprint(name)", "NoError", "Code is correct", "No changes needed", 1.0),
        ]
        
        for code, error_type, message, suggestion, confidence in correct_examples:
            training_data['code_snippets'].append(code)
            training_data['error_types'].append(error_type)
            training_data['error_messages'].append(message)
            training_data['suggestions'].append(suggestion)
            training_data['confidence_scores'].append(confidence)
        
        print(f"📊 Created training dataset with {len(training_data['code_snippets'])} examples")
        return training_data
    
    def _extract_features(self, code_snippet):
        """Extract features from code snippet for ML training"""
        features = {}
        
        # Basic code statistics
        features['line_count'] = len(code_snippet.split('\n'))
        features['char_count'] = len(code_snippet)
        features['word_count'] = len(code_snippet.split())
        
        # Syntax features
        features['has_colon'] = ':' in code_snippet
        features['has_parentheses'] = '(' in code_snippet and ')' in code_snippet
        features['has_quotes'] = '"' in code_snippet or "'" in code_snippet
        features['has_brackets'] = '[' in code_snippet and ']' in code_snippet
        features['has_braces'] = '{' in code_snippet and '}' in code_snippet
        
        # Python keyword features
        python_keywords = ['if', 'else', 'elif', 'for', 'while', 'def', 'class', 'import', 'from', 'return', 'print', 'input']
        for keyword in python_keywords:
            features[f'has_{keyword}'] = keyword in code_snippet.lower()
        
        # Indentation features
        lines = code_snippet.split('\n')
        features['has_indentation'] = any(line.startswith('    ') or line.startswith('\t') for line in lines)
        features['mixed_indentation'] = any('\t' in line and '    ' in line for line in lines)
        
        # Common typo patterns
        typo_patterns = ['it(', 'pirnt', 'imoprt', 'retrun', 'retrn', 'nu ', '@']
        for pattern in typo_patterns:
            features[f'has_typo_{pattern.replace("(", "").replace(" ", "").replace("@", "at")}'] = pattern in code_snippet.lower()
        
        return features
    
    def train_models(self):
        """Train all ML models on the dataset"""
        print("🚀 Starting ML model training...")
        
        # Prepare features
        X_text = self.training_data['code_snippets']
        X_features = []
        
        for code in X_text:
            features = self._extract_features(code)
            X_features.append(list(features.values()))
        
        X_features = np.array(X_features)
        
        # Prepare labels
        y = self.training_data['error_types']
        
        # Encode labels
        self.label_encoders['error_type'] = LabelEncoder()
        y_encoded = self.label_encoders['error_type'].fit_transform(y)
        
        # Vectorize text
        X_tfidf = self.vectorizers['tfidf'].fit_transform(X_text)
        X_count = self.vectorizers['count'].fit_transform(X_text)
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_count, X_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train models
        model_scores = {}
        for name, model in self.models.items():
            print(f"🔧 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[name] = accuracy
            
            print(f"✅ {name} accuracy: {accuracy:.3f}")
        
        # Find best model
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (accuracy: {model_scores[best_model_name]:.3f})")
        
        self.is_trained = True
        self._save_models()
    
    def predict_error(self, code_snippet):
        """Predict error type and provide suggestions"""
        if not self.is_trained:
            self.train_models()
        
        # Extract features
        features = self._extract_features(code_snippet)
        X_features = np.array([list(features.values())])
        
        # Vectorize text
        X_tfidf = self.vectorizers['tfidf'].transform([code_snippet])
        X_count = self.vectorizers['count'].transform([code_snippet])
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_count, X_features])
        
        # Predict with ensemble
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_combined)[0]
            if hasattr(model, 'predict_proba'):
                conf = np.max(model.predict_proba(X_combined)[0])
            else:
                conf = 0.8  # Default confidence for models without probability
            
            predictions[name] = pred
            confidences[name] = conf
        
        # Ensemble prediction (majority vote with confidence weighting)
        weighted_votes = {}
        for name, pred in predictions.items():
            if pred not in weighted_votes:
                weighted_votes[pred] = 0
            weighted_votes[pred] += confidences[name]
        
        # Get final prediction
        final_pred_encoded = max(weighted_votes, key=weighted_votes.get)
        final_confidence = weighted_votes[final_pred_encoded] / len(predictions)
        
        # Decode prediction
        final_pred = self.label_encoders['error_type'].inverse_transform([final_pred_encoded])[0]
        
        # Get suggestion
        suggestion = self._get_suggestion_for_error(final_pred, code_snippet)
        
        return {
            'error_type': final_pred,
            'confidence': final_confidence,
            'suggestion': suggestion,
            'individual_predictions': {
                name: self.label_encoders['error_type'].inverse_transform([pred])[0] 
                for name, pred in predictions.items()
            }
        }
    
    def _get_suggestion_for_error(self, error_type, code_snippet):
        """Get specific suggestion based on error type and code"""
        suggestions = {
            'SyntaxError': self._analyze_syntax_error(code_snippet),
            'IndentationError': "Fix your indentation - use consistent spaces or tabs",
            'NameError': "Define the variable or function before using it",
            'TypeError': "Check data types - you might be mixing incompatible types",
            'OCRTypo': self._analyze_ocr_typo(code_snippet),
            'LogicError': "Review your logic - check conditions and variable usage",
            'NoError': "Your code looks good!"
        }
        
        return suggestions.get(error_type, "Check your Python syntax and logic")
    
    def _analyze_syntax_error(self, code):
        """Analyze specific syntax errors"""
        if ':' not in code and any(keyword in code.lower() for keyword in ['if', 'for', 'while', 'def', 'class']):
            return "Add a colon (:) after your control statement"
        elif '(' in code and ')' not in code:
            return "Add missing closing parenthesis"
        elif '"' in code and code.count('"') % 2 != 0:
            return "Add missing closing quote"
        else:
            return "Check for missing colons, parentheses, or quotes"
    
    def _analyze_ocr_typo(self, code):
        """Analyze OCR-specific typos"""
        code_lower = code.lower()
        if 'it(' in code_lower:
            return "Change 'it' to 'int' for integer conversion"
        elif 'pirnt' in code_lower:
            return "Change 'pirnt' to 'print'"
        elif 'imoprt' in code_lower or 'improt' in code_lower:
            return "Fix the import statement spelling"
        elif 'retrun' in code_lower or 'retrn' in code_lower:
            return "Change to 'return'"
        elif 'nu ' in code_lower:
            return "Change 'nu' to 'num'"
        else:
            return "Check for common OCR typos in your code"
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save models
            for name, model in self.models.items():
                joblib.dump(model, self.models_dir / f"{name}_model.pkl")
            
            # Save vectorizers
            for name, vectorizer in self.vectorizers.items():
                joblib.dump(vectorizer, self.models_dir / f"{name}_vectorizer.pkl")
            
            # Save label encoders
            for name, encoder in self.label_encoders.items():
                joblib.dump(encoder, self.models_dir / f"{name}_encoder.pkl")
            
            print("💾 Models saved successfully")
        except Exception as e:
            print(f"⚠️ Error saving models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load models
            for name in self.models.keys():
                model_path = self.models_dir / f"{name}_model.pkl"
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
            
            # Load vectorizers
            for name in self.vectorizers.keys():
                vec_path = self.models_dir / f"{name}_vectorizer.pkl"
                if vec_path.exists():
                    self.vectorizers[name] = joblib.load(vec_path)
            
            # Load label encoders
            encoder_path = self.models_dir / "error_type_encoder.pkl"
            if encoder_path.exists():
                self.label_encoders['error_type'] = joblib.load(encoder_path)
                self.is_trained = True
                print("📂 Models loaded successfully")
                return True
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
        
        return False