# import torch
# from transformers import pipeline
# import io
# import librosa

# def convert_bytes_to_array(audio_bytes):
#     audio_bytes = io.BytesIO(audio_bytes)
#     audio, sample_rate = librosa.load(audio_bytes)
#     print(sample_rate)
#     return audio

# def transcribe_audio(audio_bytes):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"

#     pipe = pipeline(
#         task = "automatic-speech-recognition",
#         model="openai/whisper-medium.en",
#         chunk_length_s=30,
#         device=device,
#     )
#     audio_array = convert_bytes_to_array(audio_bytes)
#     prediction = pipe(audio_array, batch_size=1)["text"]
#     return prediction

import torch
import io
import librosa
from transformers import pipeline
from pydub import AudioSegment


AudioSegment.converter = r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe"
def convert_bytes_to_array(audio_bytes):
    """Converts audio bytes to WAV format and then loads as numpy array."""
    audio_io = io.BytesIO(audio_bytes)

    # ✅ Convert to WAV using pydub
    audio = AudioSegment.from_file(audio_io)  
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    # ✅ Load into numpy array using librosa
    audio_array, sample_rate = librosa.load(wav_io, sr=16000)
    return audio_array

def transcribe_audio(audio_bytes):
    """Transcribes speech using an open-source Whisper model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-medium.en",  # Free and fast model
        chunk_length_s=30,  
        device=device,
    )

    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array)["text"]
    return prediction
