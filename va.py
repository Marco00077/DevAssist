import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import logging
from advanced_ocr_detector import AdvancedOCRErrorDetector

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Initialize the advanced OCR error detector
error_detector = AdvancedOCRErrorDetector()

def search_wikipedia(query):
    for phrase in ["according to", "tell me about", "who is", "what is", "on", "of"]:
        query = query.replace(phrase, "")
    query = query.strip()

    try:
        results = wikipedia.search(query, results=3)
        if results:
            page = results[0]
            summary = wikipedia.summary(page, sentences=2)
            return summary
        else:
            return "Sorry, I couldn't find anything on Wikipedia."
    except Exception as e:
        return "Sorry, there was an error while searching Wikipedia."

def wishMe():
    hour = datetime.datetime.now().hour
    if 0 <= hour < 12:
        speak("Good Morning!")
    elif 12 <= hour < 18:
        speak("Good Afternoon!")   
    else:
        speak("Good Evening!")  
    speak("Hii their. Please tell me how may I help you today")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        # Faster voice recognition settings
        r.pause_threshold = 0.8  # Reduced from 1 to 0.5
        r.energy_threshold = 3500  # Reduced from 4000 to 3000
        r.dynamic_energy_threshold = True
        r.timeout = 3  # Add timeout to prevent hanging
        r.phrase_time_limit = 5  # Limit phrase length

        try:
            audio = r.listen(source, timeout=3, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("Listening timeout - try again...")
            return "None"
    
    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
    except Exception as e:
        print("Say that again please...")  
        return "None"
    return query.lower()

if __name__ == "__main__":
    wishMe()
    while True:
        query = takeCommand()

        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            summary = search_wikipedia(query)
            print(summary)
            speak(summary)

        elif 'open youtube' in query:
            webbrowser.open("https://www.youtube.com/")
            speak("As you wish")
        
        elif 'open google' in query:
            webbrowser.open("https://www.google.com")
            speak("As you wish")

        elif 'open stackoverflow' in query:
            webbrowser.open("https://stackoverflow.com")

        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            speak(f"the time is {strTime}")
            
        elif 'play looser' in query:
            query = query.replace("play","")
            speak(f"Playing {query}")
            webbrowser.open('https://youtu.be/s3a4OQR-10M')
            speak("As you wish")
        
        elif 'open discord' in query:
            path = "C:\\Users\\marco\\OneDrive\\Desktop\\Discord.lnk"
            os.startfile(path)
            speak("As you wish")
        
        elif 'open code' in query:
            codePath = "C:\\Users\\marco\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
            os.startfile(codePath)
        
        elif any(phrase in query for phrase in ['check my code', 'where is the error', 'check my screen', 'find error', 'debug my code', 'what is wrong', 'analyze screen']):
            speak("Let me check your screen for Python errors...")
            print("🚀 Starting advanced OCR error analysis...")
            
            try:
                result = error_detector.analyze_screen()
                print(f"🔍 Analysis result: {result}")
                
                if result['success']:
                    if result['errors']:
                        # Print detailed results to console
                        print(f"\n🚨 Found {len(result['errors'])} issue(s):")
                        for i, error in enumerate(result['errors'], 1):
                            print(f"{i}. {error['error']}")
                            print(f"   Fix: {error['fix']}")
                            if 'line' in error:
                                print(f"   Line: {error['line']}")
                            print()
                        
                        # Speak the voice summary
                        print(f"🗣️ About to speak: {result['voice']}")
                        speak(result['voice'])
                    else:
                        print("✅ No errors found!")
                        speak(result['voice'])
                else:
                    print(f"❌ {result['message']}")
                    speak(result['voice'])
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                print(error_msg)
                speak(error_msg)
        
        elif 'shutdown' in query or 'turn off' in query:
            speak("Ok, I'll now shut down")
            break
