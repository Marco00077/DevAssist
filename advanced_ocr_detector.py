"""
Advanced OCR Error Detection System
Uses multiple OCR engines and comprehensive Python error datasets
"""

import cv2
import numpy as np
import pyautogui
from PIL import Image
import re
import ast
import sys
import io
import json
from pathlib import Path
import friendly_traceback
from ml_error_classifier import MLErrorClassifier

# Multiple OCR engines for better accuracy
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCR not available. Install with: pip install easyocr")

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠️ PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")

import pytesseract

class AdvancedOCRErrorDetector:
    def __init__(self):
        self.screenshots_folder = "screenshots"
        Path(self.screenshots_folder).mkdir(exist_ok=True)
        
        # Initialize OCR engines
        self.ocr_engines = []
        self._initialize_ocr_engines()
        
        # Load comprehensive Python error dataset
        self.error_patterns = self._load_python_error_dataset()
        
        # Configure friendly-traceback
        friendly_traceback.install()
        friendly_traceback.set_lang('en')
        
        # Initialize ML Error Classifier
        self.ml_classifier = MLErrorClassifier()
        
        # Try to load pre-trained models, otherwise train new ones
        if not self.ml_classifier.load_models():
            print("🤖 Training ML models for the first time...")
            self.ml_classifier.train_models()
        
        print(f"🚀 Initialized with {len(self.ocr_engines)} OCR engines and ML classifier")
    
    def _initialize_ocr_engines(self):
        """Initialize multiple OCR engines for better accuracy"""
        
        # EasyOCR - Best for general text recognition
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                self.ocr_engines.append(('EasyOCR', self._extract_with_easyocr))
                print("✅ EasyOCR initialized")
            except Exception as e:
                print(f"⚠️ EasyOCR initialization failed: {e}")
        
        # PaddleOCR - Good for structured text
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddle_ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                self.ocr_engines.append(('PaddleOCR', self._extract_with_paddle))
                print("✅ PaddleOCR initialized")
            except Exception as e:
                print(f"⚠️ PaddleOCR initialization failed: {e}")
        
        # Only use Tesseract if no other engines are available
        if len(self.ocr_engines) == 0:
            self.ocr_engines.append(('Tesseract', self._extract_with_tesseract))
            print("✅ Tesseract initialized as fallback")
        else:
            print("⚠️ Tesseract skipped - better OCR engines available")
    
    def _extract_with_easyocr(self, image_path):
        """Extract text using EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image_path, detail=0, paragraph=True)
            return ' '.join(results)
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def _extract_with_paddle(self, image_path):
        """Extract text using PaddleOCR"""
        try:
            results = self.paddle_ocr.ocr(image_path, cls=True)
            text_lines = []
            for line in results[0]:
                if line:
                    text_lines.append(line[1][0])
            return '\n'.join(text_lines)
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return ""
    
    def _extract_with_tesseract(self, image_path):
        """Extract text using Tesseract with optimized settings"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Advanced preprocessing for code
            processed = self._preprocess_for_code(gray)
            
            # Multiple Tesseract configurations
            configs = [
                '--oem 3 --psm 6',  # Uniform block of text
                '--oem 3 --psm 8',  # Single word
                '--oem 3 --psm 13', # Raw line
            ]
            
            best_result = ""
            best_length = 0
            
            for config in configs:
                try:
                    result = pytesseract.image_to_string(processed, config=config)
                    if len(result) > best_length:
                        best_result = result
                        best_length = len(result)
                except:
                    continue
            
            return best_result
        except Exception as e:
            print(f"Tesseract error: {e}")
            return ""
    
    def _preprocess_for_code(self, gray_img):
        """Advanced preprocessing specifically for code recognition"""
        # Check if it's a dark theme (low average brightness)
        avg_brightness = np.mean(gray_img)
        
        if avg_brightness < 100:  # Dark theme detected
            # Invert colors for dark themes
            gray_img = cv2.bitwise_not(gray_img)
        
        # Resize for better recognition (3x instead of 2x)
        height, width = gray_img.shape
        resized = cv2.resize(gray_img, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast more aggressively
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(resized)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # Use OTSU thresholding instead of adaptive
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Dilation to make text thicker
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        final = cv2.dilate(cleaned, kernel2, iterations=1)
        
        return final
    
    def _load_python_error_dataset(self):
        """Load comprehensive Python error patterns and solutions"""
        return {
            # Syntax Errors
            'syntax_errors': {
                'missing_colon': {
                    'patterns': [r'if\s+[^:]+[^:]$', r'for\s+[^:]+[^:]$', r'while\s+[^:]+[^:]$', r'def\s+[^:]+[^:]$', r'class\s+[^:]+[^:]$'],
                    'message': 'Missing colon after control statement',
                    'fix': 'Add a colon (:) at the end of the line',
                    'voice': 'You are missing a colon. Add a colon at the end of your if, for, while, def, or class statement.'
                },
                'unmatched_parentheses': {
                    'patterns': [r'\([^)]*$', r'^[^(]*\)'],
                    'message': 'Unmatched parentheses',
                    'fix': 'Check for missing opening or closing parentheses',
                    'voice': 'You have unmatched parentheses. Check for missing opening or closing parentheses.'
                },
                'invalid_syntax': {
                    'patterns': [r'=\s*=\s*=', r'!\s*=\s*='],
                    'message': 'Invalid operator syntax',
                    'fix': 'Use == for comparison, = for assignment',
                    'voice': 'You have invalid syntax. Use double equals for comparison and single equals for assignment.'
                }
            },
            
            # Common Typos (OCR-specific)
            'ocr_typos': {
                'it_int': {
                    'patterns': [r'\bit\s*\(', r'=\s*it\s*\('],
                    'correct': 'int',
                    'message': 'Typo: "it" should be "int"',
                    'voice': 'You wrote "it" but it should be "int" for integer conversion.'
                },
                'nu_num': {
                    'patterns': [r'\bnu\s+[><=]', r'if\s+nu\s+'],
                    'correct': 'num',
                    'message': 'Typo: "nu" should be "num"',
                    'voice': 'You wrote "nu" but it should be "num" with an "m" at the end.'
                },
                'at_zero': {
                    'patterns': [r'>\s*@', r'<\s*@', r'==\s*@', r':\s*@'],
                    'correct': '0',
                    'message': 'OCR error: "@" should be "0"',
                    'voice': 'The "@" symbol should be the number zero.'
                },
                'pirnt_print': {
                    'patterns': [r'\bpirnt\s*\('],
                    'correct': 'print',
                    'message': 'Typo: "pirnt" should be "print"',
                    'voice': 'You wrote "pirnt" but it should be "print".'
                },
                'imoprt_import': {
                    'patterns': [r'\bimoprt\s+', r'\bimprot\s+'],
                    'correct': 'import',
                    'message': 'Typo: should be "import"',
                    'voice': 'You have a typo in your import statement. It should be "import".'
                },
                'retrun_return': {
                    'patterns': [r'\bretrun\s+', r'\breturn\s+'],
                    'correct': 'return',
                    'message': 'Typo: "retrun" should be "return"',
                    'voice': 'You wrote "retrun" but it should be "return".'
                },
                'funciton_function': {
                    'patterns': [r'\bfunciton\s+', r'\bfuntion\s+'],
                    'correct': 'function',
                    'message': 'Typo: should be "function"',
                    'voice': 'You have a typo. It should be "function".'
                }
            },
            
            # Name Errors
            'name_errors': {
                'undefined_variable': {
                    'message': 'Variable not defined',
                    'fix': 'Define the variable before using it',
                    'voice': 'You are using a variable that has not been defined yet.'
                }
            },
            
            # Type Errors
            'type_errors': {
                'string_int_concat': {
                    'message': 'Cannot concatenate string and integer',
                    'fix': 'Convert integer to string using str()',
                    'voice': 'You cannot combine text and numbers directly. Use str() to convert numbers to text.'
                }
            }
        }
    
    def analyze_screen(self):
        """Advanced screen analysis with multiple OCR engines"""
        try:
            # Capture screen
            screenshot = self._capture_screen()
            if not screenshot:
                return self._create_result(False, 'Could not capture screen')
            
            # Try multiple OCR engines
            ocr_results = []
            for engine_name, extract_func in self.ocr_engines:
                print(f"🔍 Trying {engine_name}...")
                text = extract_func(screenshot)
                if text.strip():
                    ocr_results.append((engine_name, text, len(text)))
                    print(f"✅ {engine_name}: {len(text)} characters")
                else:
                    print(f"❌ {engine_name}: No text extracted")
            
            if not ocr_results:
                return self._create_result(False, 'No text extracted by any OCR engine')
            
            # Choose best result based on quality, not just length
            best_engine, best_text = self._select_best_ocr_result(ocr_results)
            print(f"🏆 Best result from {best_engine}")
            print(f"📄 Extracted text: {best_text[:200]}...")
            
            # Check if it's Python code
            if not self._is_python_code(best_text):
                return self._create_result(False, 'No Python code detected')
            
            # Analyze for errors
            errors = self._comprehensive_error_analysis(best_text)
            
            if not errors:
                return self._create_result(True, 'No errors found', [], 
                    'Excellent! Your code looks perfect. No errors detected.')
            
            return self._create_result(True, f'Found {len(errors)} issues', 
                errors, self._create_voice_summary(errors))
                
        except Exception as e:
            return self._create_result(False, f'Analysis error: {str(e)}')
    
    def _capture_screen(self):
        """Capture and save screenshot"""
        try:
            from datetime import datetime
            screenshot = pyautogui.screenshot()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(self.screenshots_folder) / f"screenshot_{timestamp}.png"
            screenshot.save(filepath)
            print(f"📸 Screenshot saved: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"Screenshot error: {e}")
            return None
    
    def _is_python_code(self, text):
        """Advanced Python code detection"""
        python_keywords = ['def', 'class', 'import', 'from', 'if', 'else', 'elif', 
                          'for', 'while', 'try', 'except', 'print', 'input', 'return',
                          'int', 'str', 'float', 'len', 'range', 'True', 'False']
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in python_keywords if keyword in text_lower)
        
        # Look for Python syntax patterns
        patterns = [
            r'import\s+\w+', r'from\s+\w+\s+import', r'def\s+\w+\s*\(',
            r'if\s+.*:', r'for\s+.*:', r'while\s+.*:', r'print\s*\(',
            r'input\s*\(', r'=\s*\w+\s*\(', r'#.*'
        ]
        
        pattern_matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
        score = keyword_count + (pattern_matches * 2)
        
        print(f"🐍 Python detection: {keyword_count} keywords, {pattern_matches} patterns, score: {score}")
        return score >= 4
    
    def _comprehensive_error_analysis(self, text):
        """Comprehensive error analysis using ML and multiple approaches"""
        errors = []
        
        # Clean the text for analysis
        cleaned_code = self._clean_code_text(text)
        
        # 1. ML-based error prediction (highest priority)
        if cleaned_code.strip():
            ml_result = self.ml_classifier.predict_error(cleaned_code)
            
            if ml_result['error_type'] != 'NoError' and ml_result['confidence'] > 0.6:
                errors.append({
                    'type': 'ml_prediction',
                    'error': f"ML Detected: {ml_result['error_type']}",
                    'fix': ml_result['suggestion'],
                    'voice': f"My AI analysis detected a {ml_result['error_type'].lower()}. {ml_result['suggestion']}",
                    'confidence': ml_result['confidence'],
                    'ml_details': ml_result['individual_predictions']
                })
                print(f"🤖 ML prediction: {ml_result['error_type']} (confidence: {ml_result['confidence']:.3f})")
        
        # 2. OCR-specific typo detection (high priority)
        ocr_errors = self._detect_ocr_typos(text)
        errors.extend(ocr_errors)
        
        # 3. If no high-confidence errors, try syntax analysis
        if not any(e.get('confidence', 0) > 0.8 for e in errors):
            syntax_errors = self._analyze_syntax_with_friendly_traceback(cleaned_code)
            errors.extend(syntax_errors)
        
        # 4. Pattern-based error detection (lowest priority)
        if not errors:
            pattern_errors = self._detect_pattern_errors(text)
            errors.extend(pattern_errors)
        
        # Sort by confidence
        errors.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Return top 3 errors to avoid overwhelming
        return errors[:3]
    
    def _detect_ocr_typos(self, text):
        """Detect OCR-specific typos using the dataset"""
        errors = []
        text_lower = text.lower()
        
        print(f"🔍 Checking for OCR typos in: {text_lower[:100]}...")
        
        # Comprehensive OCR typo dataset - ordered by priority
        direct_typos = [
            # Print function typos (HIGHEST PRIORITY - most common OCR error)
            ('prnt ', 'print(', 'You wrote "prnt" but it should be "print".'),
            ('prnt(', 'print(', 'You wrote "prnt" but it should be "print".'),
            ('pirnt(', 'print(', 'You wrote "pirnt" but it should be "print".'),
            ('pirnt ', 'print(', 'You wrote "pirnt" but it should be "print".'),
            ('pritn(', 'print(', 'You wrote "pritn" but it should be "print".'),
            ('pint(', 'print(', 'You wrote "pint" but it should be "print".'),
            
            # Variable name typos (HIGH PRIORITY)
            ('if nu ', 'if num ', 'You wrote "nu" but it should be "num" with an "m" at the end.'),
            ('if nu>', 'if num>', 'You wrote "nu" but it should be "num" with an "m" at the end.'),
            ('if nu<', 'if num<', 'You wrote "nu" but it should be "num" with an "m" at the end.'),
            ('if nu=', 'if num=', 'You wrote "nu" but it should be "num" with an "m" at the end.'),
            ('nu ', 'num ', 'You wrote "nu" but it should be "num" with an "m" at the end.'),
            ('nu>', 'num>', 'You wrote "nu" but it should be "num" with an "m" at the end.'),
            ('nu<', 'num<', 'You wrote "nu" but it should be "num" with an "m" at the end.'),
            ('nu=', 'num=', 'You wrote "nu" but it should be "num" with an "m" at the end.'),
            ('numb ', 'num ', 'You wrote "numb" but it should be "num".'),
            
            # Return statement typos (HIGH PRIORITY)
            ('retrun', 'return', 'You wrote "retrun" but it should be "return".'),
            ('retrn', 'return', 'You wrote "retrn" but it should be "return".'),
            ('retum', 'return', 'You wrote "retum" but it should be "return".'),
            ('retur', 'return', 'You wrote "retur" but it should be "return".'),
            ('retun', 'return', 'You wrote "retun" but it should be "return".'),
            
            # Import statement typos
            ('imoprt', 'import', 'You have a typo in your import statement. It should be "import".'),
            ('improt', 'import', 'You have a typo in your import statement. It should be "import".'),
            ('imort', 'import', 'You have a typo in your import statement. It should be "import".'),
            ('imprt', 'import', 'You have a typo in your import statement. It should be "import".'),
            
            # Integer conversion typos (LOWER PRIORITY - check after variable names)
            ('it(', 'int(', 'You wrote "it" but it should be "int" for integer conversion.'),
            ('mt(', 'int(', 'You wrote "mt" but it should be "int" for integer conversion.'),
            ('nt(', 'int(', 'You wrote "nt" but it should be "int" for integer conversion.'),
            
            # Input function typos
            ('inpt(', 'input(', 'You wrote "inpt" but it should be "input".'),
            ('imput(', 'input(', 'You wrote "imput" but it should be "input".'),
            ('iput(', 'input(', 'You wrote "iput" but it should be "input".'),
            
            # If statement typos (be more specific to avoid false positives)
            ('fi num', 'if num', 'You wrote "fi" but it should be "if".'),
            ('lf num', 'if num', 'You wrote "lf" but it should be "if".'),
            
            # Else statement typos
            ('esle:', 'else:', 'You wrote "esle" but it should be "else".'),
            ('elese:', 'else:', 'You wrote "elese" but it should be "else".'),
            
            # For loop typos
            ('fro ', 'for ', 'You wrote "fro" but it should be "for".'),
            ('fr ', 'for ', 'You wrote "fr" but it should be "for".'),
            
            # While loop typos
            ('whle ', 'while ', 'You wrote "whle" but it should be "while".'),
            ('wile ', 'while ', 'You wrote "wile" but it should be "while".'),
            
            # Function definition typos
            ('def ', 'def ', 'You have a typo in your function definition.'),
            ('deff ', 'def ', 'You wrote "deff" but it should be "def".'),
            
            # Class definition typos
            ('calss ', 'class ', 'You wrote "calss" but it should be "class".'),
            ('clas ', 'class ', 'You wrote "clas" but it should be "class".'),
            
            # Common OCR symbol errors
            ('> @', '> 0', 'The "@" symbol should be the number zero.'),
            ('< @', '< 0', 'The "@" symbol should be the number zero.'),
            ('== @', '== 0', 'The "@" symbol should be the number zero.'),
            ('!= @', '!= 0', 'The "@" symbol should be the number zero.'),
            

            
            # String/length typos
            ('lenght(', 'length(', 'You wrote "lenght" but it should be "length".'),
            ('len9th(', 'length(', 'You wrote "len9th" but it should be "length".'),
            ('strign', 'string', 'You wrote "strign" but it should be "string".'),
            
            # Boolean typos
            ('Tme', 'True', 'You wrote "Tme" but it should be "True".'),
            ('Tue', 'True', 'You wrote "Tue" but it should be "True".'),
            ('Flase', 'False', 'You wrote "Flase" but it should be "False".'),
            ('Fasle', 'False', 'You wrote "Fasle" but it should be "False".'),
        ]
        
        for typo, correct, voice_msg in direct_typos:
            if typo in text_lower:
                # Enhanced false positive prevention
                if typo == 'it(' and 'int(' in text_lower:
                    continue  # Skip if correct version also exists
                if typo == 'nt(' and 'int(' in text_lower:
                    continue  # Skip if correct version also exists
                if typo.startswith('nu') and 'num' in text_lower and text_lower.count('num') > text_lower.count(typo.strip()):
                    continue  # Skip if correct version is more frequent
                
                # Special check for "int" related errors - make sure it's actually wrong
                if 'int' in correct and 'int(' in text_lower:
                    continue  # If we already have correct "int(" in the text, don't report it as error
                
                # Skip "if" related errors if they seem like false positives
                if typo in ['fi ', 'lf '] and 'if ' in text_lower:
                    continue  # Skip if correct "if" already exists
                
                errors.append({
                    'type': 'ocr_typo',
                    'error': f'Typo: "{typo.rstrip("( <>!=")}" should be "{correct.rstrip("( <>!=")}"',
                    'fix': f'Change "{typo.rstrip("( <>!=")}" to "{correct.rstrip("( <>!=")}"',
                    'voice': voice_msg,
                    'confidence': 0.95
                })
                print(f"🎯 Direct OCR typo found: {typo} -> {correct}")
                return errors
        
        # Fallback to pattern matching
        for typo_name, typo_data in self.error_patterns['ocr_typos'].items():
            for pattern in typo_data['patterns']:
                if re.search(pattern, text_lower):
                    errors.append({
                        'type': 'ocr_typo',
                        'error': typo_data['message'],
                        'fix': f"Change to '{typo_data['correct']}'",
                        'voice': typo_data['voice'],
                        'confidence': 0.95
                    })
                    print(f"🎯 Pattern OCR typo detected: {typo_name}")
                    return errors
        
        print("🔍 No OCR typos found")
        return errors
    
    def _analyze_syntax_with_friendly_traceback(self, text):
        """Use friendly-traceback for syntax analysis"""
        errors = []
        
        # Clean the text for Python analysis
        cleaned_code = self._clean_code_text(text)
        
        try:
            # Try to compile the code
            compile(cleaned_code, '<string>', 'exec')
            print("✅ Code compiles successfully")
        except SyntaxError as e:
            print(f"🔍 Syntax error: {e}")
            
            # Get friendly explanation
            friendly_msg = self._get_friendly_explanation(str(e))
            
            errors.append({
                'type': 'syntax_error',
                'error': f'Syntax Error: {str(e)}',
                'fix': friendly_msg['fix'],
                'voice': friendly_msg['voice'],
                'confidence': 0.9
            })
        
        return errors
    
    def _clean_code_text(self, text):
        """Clean extracted text to make it more like Python code"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip UI elements
            if any(skip in line.lower() for skip in ['file', 'edit', 'view', 'run', 'class python', '@', 'x |']):
                continue
            
            # Clean line
            line = re.sub(r'[{}@#$%^&*]', '', line)  # Remove artifacts
            line = re.sub(r'\s+', ' ', line)  # Normalize whitespace
            line = line.strip()
            
            if line and len(line) > 2 and any(c.isalpha() for c in line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _detect_pattern_errors(self, text):
        """Detect errors using pattern matching"""
        errors = []
        
        # Only check for missing colon if we have clear evidence
        lines = text.split('\n')
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # More specific colon detection - only if line clearly needs one
            if (re.match(r'^\s*(if|for|while|def|class)\s+.+[^:]$', line_clean, re.IGNORECASE) and 
                len(line_clean) > 5 and  # Must be substantial line
                not any(skip in line_clean.lower() for skip in ['print', 'input', '()', '[]', '{}'])):  # Skip function calls
                
                errors.append({
                    'type': 'pattern_error',
                    'error': 'Missing colon after control statement',
                    'fix': 'Add a colon (:) at the end of the line',
                    'voice': 'You are missing a colon. Add a colon at the end of your control statement.',
                    'confidence': 0.7  # Lower confidence to avoid false positives
                })
                print(f"🔍 Potential missing colon in: {line_clean}")
                break  # Only report first one
        
        return errors
    
    def _get_friendly_explanation(self, error_msg):
        """Get user-friendly explanation for errors"""
        if 'invalid syntax' in error_msg.lower():
            return {
                'fix': 'Check for missing colons, incorrect operators, or typos',
                'voice': 'There is invalid syntax in your code. Check for missing colons, incorrect operators, or typos.'
            }
        elif 'unmatched' in error_msg.lower():
            return {
                'fix': 'Check for unmatched parentheses, brackets, or quotes',
                'voice': 'You have unmatched parentheses, brackets, or quotes in your code.'
            }
        else:
            return {
                'fix': 'Check your Python syntax',
                'voice': f'There is a syntax error in your code. {error_msg}'
            }
    
    def _create_result(self, success, message, errors=None, voice=None):
        """Create standardized result"""
        return {
            'success': success,
            'message': message,
            'errors': errors or [],
            'voice': voice or ('Sorry, I encountered an error.' if not success else 'Analysis complete.')
        }
    
    def _create_voice_summary(self, errors):
        """Create voice summary prioritizing highest confidence errors"""
        if not errors:
            return 'No errors found.'
        
        if len(errors) == 1:
            return errors[0]['voice']
        
        # Multiple errors - speak the highest confidence one
        highest_error = errors[0]  # Already sorted by confidence
        voice_parts = [f"I found {len(errors)} issues. The main issue is:"]
        voice_parts.append(highest_error['voice'])
        
        if len(errors) > 1:
            voice_parts.append("Check the console for other issues.")
        
        return ' '.join(voice_parts)
    
    def _select_best_ocr_result(self, ocr_results):
        """Select the best OCR result based on quality metrics, not just length"""
        if not ocr_results:
            return None, ""
        
        # Calculate scores for all results
        scored_results = []
        for engine_name, text, length in ocr_results:
            score = self._calculate_ocr_quality_score(text, engine_name)
            scored_results.append((engine_name, text, score))
            print(f"📊 {engine_name} quality score: {score:.2f}")
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        # Additional safety check: if top result has very low score, prefer EasyOCR
        best_engine, best_text, best_score = scored_results[0]
        
        if best_score < 20:  # Very low score threshold
            # Look for EasyOCR as fallback
            for engine_name, text, score in scored_results:
                if engine_name == 'EasyOCR' and score > 10:
                    print(f"🔄 Overriding low score result, using EasyOCR instead")
                    return engine_name, text
        
        return best_engine, best_text
    
    def _calculate_ocr_quality_score(self, text, engine_name):
        """Calculate quality score for OCR result with strict garbage detection"""
        if not text.strip():
            return 0
        
        score = 0
        text_lower = text.lower()
        
        # STRICT garbage detection - immediate disqualification
        garbage_indicators = [
            'bduwmwuw', 'sepnam', 'xru', 'qlpy', 'wm:', 'opy', 'jy =', 'ay ', 'dy ',
            '¢', '©', '®', '™', '°', '§', '¶', '†', '‡', '•', '…', '‰', '‹', '›', '€'
        ]
        
        garbage_count = sum(1 for indicator in garbage_indicators if indicator in text_lower)
        if garbage_count > 2:  # If more than 2 garbage indicators, heavily penalize
            score -= 1000  # Massive penalty for garbage text
            print(f"⚠️ {engine_name} has {garbage_count} garbage indicators - heavily penalized")
        
        # Check for excessive random uppercase sequences (like BDUWMWUW)
        random_caps = re.findall(r'[A-Z]{4,}', text)
        if len(random_caps) > 1:
            score -= 500  # Heavy penalty for random caps
            print(f"⚠️ {engine_name} has random caps: {random_caps}")
        
        # Check for excessive special character sequences
        weird_sequences = re.findall(r'[^a-zA-Z0-9\s]{3,}', text)
        if len(weird_sequences) > 2:
            score -= 300  # Penalty for weird character sequences
            print(f"⚠️ {engine_name} has weird sequences: {weird_sequences[:3]}")
        
        # Positive indicators (good OCR) - but only if not garbage
        if garbage_count <= 2:
            python_keywords = ['import', 'def', 'class', 'if', 'else', 'for', 'while', 'print', 'input', 'return', 'int', 'str']
            keyword_count = sum(1 for keyword in python_keywords if keyword in text_lower)
            score += keyword_count * 8  # Reduced from 10 to 8
            
            # Good patterns - only count if they look realistic
            good_patterns = [
                r'import\s+\w+',
                r'def\s+\w+\s*\(',
                r'if\s+\w+\s*[><=]',
                r'print\s*\(',
                r'input\s*\(',
                r'=\s*int\s*\(',
                r'num\s*=',
                r':\s*$'
            ]
            
            for pattern in good_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                # Only count if the matches look reasonable
                clean_matches = [m for m in matches if not any(garbage in str(m).lower() for garbage in garbage_indicators)]
                score += len(clean_matches) * 5
        
        # Readability check - much stricter
        words = text.split()
        if words:
            # Count truly readable words (no garbage)
            readable_words = []
            for word in words:
                if (len(word) >= 2 and 
                    word.isalpha() and 
                    not any(garbage in word.lower() for garbage in garbage_indicators) and
                    not re.match(r'^[A-Z]{3,}$', word)):  # Not all caps random
                    readable_words.append(word)
            
            if len(words) > 0:
                readability_ratio = len(readable_words) / len(words)
                score += readability_ratio * 20  # Increased bonus for truly readable words
                print(f"📖 {engine_name} readability: {len(readable_words)}/{len(words)} = {readability_ratio:.2f}")
        
        # Strong preference for EasyOCR if Tesseract is producing garbage
        if engine_name == 'EasyOCR':
            score += 15  # Increased preference
        
        # Length penalty for very short results
        if len(text) < 50:
            score -= 10
        
        print(f"🔍 {engine_name} final score calculation: {score}")
        return max(0, score)  # Don't return negative scores