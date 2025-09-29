import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import logging

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

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
        r.pause_threshold = 1
        r.energy_threshold = 4000
        r.dynamic_energy_threshold = True

        audio = r.listen(source)
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
            speak("What should I search on YouTube?")
            yt_query = takeCommand()

            if yt_query == "None" or yt_query.strip() == "":
                speak("Sorry, I didn’t catch that. Please say it again.")
            else:
                webbrowser.open(f"https://www.youtube.com/results?search_query={yt_query}")
                speak(f"Here are the YouTube results for {yt_query}")
        
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
        
        elif 'shutdown' in query or 'turn off' in query:
            speak("Ok, I'll now shut down")
            break
