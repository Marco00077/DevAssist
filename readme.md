# 🤖 AI Voice Assistant with OCR Python Error Detection

An intelligent voice-controlled assistant that can analyze your Python code on screen using advanced OCR technology and provide real-time error detection and suggestions.

## ✨ Features

### 🗣️ Voice Control
- **Natural Language Commands**: "check my code", "where is the error", "debug my code"
- **Text-to-Speech Feedback**: Clear voice explanations of errors and solutions
- **Hands-Free Operation**: No need to type or click anything

### 🔍 Advanced OCR Technology
- **Multiple OCR Engines**: EasyOCR, PaddleOCR, and Tesseract for maximum accuracy
- **Code-Optimized Processing**: Specialized image preprocessing for code recognition
- **Dark Theme Support**: Automatically detects and handles dark VS Code themes

### 🐍 Intelligent Python Error Detection
- **OCR-Specific Typos**: Detects common OCR errors like `it` → `int`, `nu` → `num`, `@` → `0`
- **Syntax Error Analysis**: Uses friendly-traceback for comprehensive error explanations
- **Pattern Recognition**: Identifies missing colons, unmatched parentheses, and more
- **Confidence Scoring**: Prioritizes high-confidence errors to avoid false positives

### 📸 Screenshot Management
- **Automatic Capture**: Takes screenshots when analyzing code
- **Processed Images**: Saves both original and processed images for debugging
- **Organized Storage**: All screenshots saved in `screenshots/` folder with timestamps

## 🚀 Quick Start

### Prerequisites
1. **Python 3.8+** installed on your system
2. **Tesseract OCR** installed:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

### Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For Windows users (PyAudio fix)**:
   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```

### Usage

1. **Start the voice assistant**:
   ```bash
   python va.py
   ```

2. **Use voice commands**:
   - "check my code"
   - "where is the error"
   - "debug my code"
   - "find error"
   - "analyze screen"

3. **Get instant feedback**:
   - Voice explanation of errors
   - Console output with detailed analysis
   - Screenshots saved for reference

## 🎯 Voice Commands

| Command | Description |
|---------|-------------|
| "check my code" | Analyze visible Python code for errors |
| "where is the error" | Find and explain errors in your code |
| "debug my code" | Comprehensive code analysis |
| "find error" | Quick error detection |
| "analyze screen" | Full screen code analysis |

## 🔧 How It Works

### 1. Voice Recognition
- Listens for trigger phrases using Google Speech Recognition
- Processes natural language commands
- Provides voice feedback using text-to-speech

### 2. Screen Capture & OCR
- Captures current screen content
- Uses multiple OCR engines for best accuracy:
  - **EasyOCR**: Neural network-based, excellent for code
  - **PaddleOCR**: High accuracy with angle correction
  - **Tesseract**: Fallback with advanced preprocessing

### 3. Code Analysis
- Detects Python code patterns
- Identifies common OCR-induced typos
- Analyzes syntax using Python AST and friendly-traceback
- Provides confidence-scored results

### 4. Error Reporting
- Prioritizes errors by confidence level
- Speaks the most important error first
- Saves detailed analysis to console
- Stores screenshots for debugging

## 📁 Project Structure

```
ai-voice-assistant/
├── va.py                      # Main voice assistant application
├── advanced_ocr_detector.py   # Advanced OCR and error detection engine
├── requirements.txt           # Python dependencies
├── screenshots/              # Auto-generated screenshot storage
└── README.md                 # This file
```

## 🛠️ Technical Details

### OCR Engines Used
- **EasyOCR**: Deep learning-based OCR with 80+ language support
- **PaddleOCR**: Ultra-lightweight OCR with high accuracy
- **Tesseract**: Google's OCR engine with custom preprocessing

### Error Detection Categories
1. **OCR-Specific Typos**: `it`→`int`, `nu`→`num`, `@`→`0`, `pirnt`→`print`
2. **Syntax Errors**: Missing colons, unmatched parentheses, invalid operators
3. **Pattern Errors**: Common Python mistakes and anti-patterns

### Voice Feedback Examples
- **Typo Detection**: "I found a typo. You wrote 'nu' but it should be 'num' with an 'm' at the end."
- **Syntax Error**: "You are missing a colon after your if statement. Add a colon at the end of the line."
- **Success**: "Excellent! Your code looks perfect. No errors detected."

## 🎨 Supported Code Editors

The system works with any code editor or IDE:
- ✅ **VS Code** (Dark/Light themes)
- ✅ **PyCharm**
- ✅ **Sublime Text**
- ✅ **Atom**
- ✅ **Notepad++**
- ✅ **Any text editor**

## 🔍 Common Error Types Detected

### OCR-Induced Typos
- `it(input())` → Should be `int(input())`
- `if nu > 0:` → Should be `if num > 0:`
- `if x > @:` → Should be `if x > 0:`
- `pirnt("hello")` → Should be `print("hello")`

### Syntax Errors
- Missing colons after `if`, `for`, `while`, `def`, `class`
- Unmatched parentheses, brackets, or braces
- Invalid operators (`===` instead of `==`)

### Logic Errors
- Undefined variables
- Type mismatches
- Incorrect indentation

## 🚨 Troubleshooting

### Common Issues

**1. "Tesseract not found"**
- Install Tesseract OCR from the official website
- Add Tesseract to your system PATH
- Restart your terminal/command prompt

**2. "PyAudio installation fails"**
- Windows: Use `pipwin install pyaudio`
- macOS: Install portaudio first: `brew install portaudio`
- Linux: Install system dependencies: `sudo apt-get install python3-pyaudio`

**3. "No text extracted by any OCR engine"**
- Ensure your code is clearly visible on screen
- Try increasing font size in your code editor
- Use high contrast themes (dark text on light background works best)

**4. "Voice recognition not working"**
- Check microphone permissions
- Ensure stable internet connection (uses Google Speech API)
- Speak clearly and avoid background noise

### Improving Accuracy

**For Better OCR Results:**
- Use larger font sizes (14pt or higher)
- High contrast themes work best
- Ensure good lighting
- Clean, uncluttered screen

**For Better Voice Recognition:**
- Speak clearly and at normal pace
- Use the exact trigger phrases
- Minimize background noise
- Ensure microphone is working

## 🔄 Dependencies

### Core Dependencies
- `pyttsx3`: Text-to-speech engine
- `SpeechRecognition`: Voice recognition
- `opencv-python`: Image processing
- `Pillow`: Image manipulation
- `numpy`: Numerical operations
- `friendly-traceback`: Python error explanations

### Advanced OCR (Optional but Recommended)
- `easyocr`: Neural network OCR
- `paddleocr`: High-accuracy OCR
- `paddlepaddle`: PaddleOCR backend

### Voice Recognition
- `pyaudio`: Audio input/output
- `wikipedia`: Wikipedia integration
- `requests`: HTTP requests

## 🎯 Use Cases

### For Students
- **Learning Python**: Get instant feedback on syntax errors
- **Homework Help**: Understand what's wrong with your code
- **Exam Preparation**: Practice error identification

### For Developers
- **Code Review**: Quick error checking during development
- **Debugging**: Voice-controlled error analysis
- **Accessibility**: Hands-free code analysis

### For Educators
- **Teaching Tool**: Demonstrate common Python errors
- **Code Review**: Help students identify mistakes
- **Accessibility**: Assist visually impaired students

## 🚀 Future Enhancements

- [ ] Support for more programming languages (JavaScript, Java, C++)
- [ ] Integration with popular IDEs via plugins
- [ ] Real-time code analysis (continuous monitoring)
- [ ] Custom error pattern training
- [ ] Multi-language voice support
- [ ] Code suggestion and auto-fix capabilities

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional OCR engines
- More error patterns
- Better voice recognition
- Support for other programming languages
- UI improvements

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **friendly-traceback**: Excellent Python error explanations
- **EasyOCR**: State-of-the-art OCR technology
- **PaddleOCR**: High-performance OCR engine
- **Tesseract**: Reliable OCR foundation
- **Google Speech Recognition**: Voice recognition API

---

**Made with ❤️ for Python developers who want smarter debugging tools**