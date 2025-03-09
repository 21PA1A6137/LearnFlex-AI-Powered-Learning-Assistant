import streamlit as st
import pyttsx3
from io import BytesIO
import io

# # Initialize pyttsx3 TTS engine
def text_to_speech_pyttsx3(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  
    engine.setProperty('volume', 0.9)

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    text_with_pause = text.replace('Answer:', 'Answer: . . . ') 
    audio_file = BytesIO()
    engine.save_to_file(text, "output.mp3")
    engine.runAndWait()

    with open("output.mp3", "rb") as file:
        audio_file.write(file.read())
    audio_file.seek(0)
    return audio_file

