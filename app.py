import os
import streamlit as st
from transformers import pipeline
import pandas as pd

@st.cache_resource
def load_model():
    model = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    model.model.config.forced_decoder_ids = model.tokenizer.get_decoder_prompt_ids(language="de", task="transcribe")
    return model

model = load_model()

AUDIO_FOLDER = "audio_files"
TRANSCRIPTIONS_FOLDER = "transcriptions"
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

def save_audio(file, folder):
    file_path = os.path.join(folder, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def transcribe_audio(audio_path):
    transcription = model(audio_path, return_timestamps=True)["text"]
    transcription_path = os.path.join(TRANSCRIPTIONS_FOLDER, f"{os.path.basename(audio_path)}.txt")
    with open(transcription_path, "w") as f:
        f.write(transcription)
    return transcription, transcription_path


st.title("German Audio Transcription App")
st.write("Upload German audio files to transcribe them to German text.")

uploaded_file = st.file_uploader("Choose a German audio file...", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.write(f"Processing file: {uploaded_file.name}")
    audio_path = save_audio(uploaded_file, AUDIO_FOLDER)
    transcription, transcription_path = transcribe_audio(audio_path)
    st.write("Transcription complete!")
    st.audio(audio_path, format="audio/wav")
    st.write(transcription)

st.header("Transcriptions")
search_query = st.text_input("Search transcriptions by filename or content")

def load_transcriptions():
    transcriptions = []
    for file in os.listdir(TRANSCRIPTIONS_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TRANSCRIPTIONS_FOLDER, file), "r") as f:
                transcriptions.append({"filename": file, "transcription": f.read()})
    return pd.DataFrame(transcriptions)

transcriptions_df = load_transcriptions()

if search_query:
    results = transcriptions_df[
        transcriptions_df["filename"].str.contains(search_query, case=False) |
        transcriptions_df["transcription"].str.contains(search_query, case=False)
    ]
else:
    results = transcriptions_df

for idx, row in results.iterrows():
    st.subheader(row["filename"])
    audio_file = os.path.join(AUDIO_FOLDER, row["filename"].replace(".txt", ""))
    st.audio(audio_file, format="audio/wav")
    st.write(row["transcription"])
