from dotenv import load_dotenv
import os
from PIL import Image
import matplotlib.pyplot as plt
from llama_index.core.multi_modal_llms import MultiModalLLM
import nest_asyncio

from IPython.display import display, Markdown
from llama_index.multi_modal_llms.mistralai import MistralAIMultiModal

import pyaudio
import audioop
import requests
import io
import wave
import speech_recognition as sr
import pyttsx3
from bleak import BleakScanner

import os
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.core.agent import FunctionCallingAgent

import cv2 # pip install opencv-python
import speech_recognition as sr # pip install SpeechRecognition


nest_asyncio.apply()
# -----------------------------------------------------------------------------#
################################# PARAMETRI ####################################
# -----------------------------------------------------------------------------#
stop_command = ["stop", "fine", "exit"]
assistant_command = ["assistenza", "assistente", "aiuto", "dubbio"]
image_command = ["immagine"]

API_URL = "https://api.openai.com/v1/audio/transcriptions"
SILENCE_THRESHOLD = 1000  # Silence threshold in RMS (root mean square)
SILENCE_DURATION = 3  # Duration in seconds for which silence is detected before stopping the recording

'''
directory = "C:/Users/filow/OneDrive/Desktop/Paper_InfraDIAG/Test_banale_ChatGPT"
file_path = "C:/Users/filow/OneDrive/Desktop/Paper_InfraDIAG/Test_banale_ChatGPT/IMG_2.jpg"
'''

directory = "C:/Users/filow/OneDrive/Desktop/Paper_InfraDIAG/Test_codici"
file_path = "C:/Users/filow/OneDrive/Desktop/Paper_InfraDIAG/Test_codici/captured_image.jpg"


# -----------------------------------------------------------------------------#
################################# AUDIO ########################################
# -----------------------------------------------------------------------------#
def text_to_audio(text):
    """Convert text to audio output."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def get_llm_response(user_prompt):
    messages = [
        ChatMessage(role="system", content="You are an assistant"),
        ChatMessage(role="user", content=user_prompt),
    ]
    # Get the response from the chat
    resp = llm.chat(messages)
    return resp


def record_audio(mic_index):
    """
    Record audio from the selected microphone and return it as raw audio bytes.
    Stops recording on silence.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=mic_index,  # Use the selected mic
                    frames_per_buffer=1024)

    print(f"Recording from {p.get_device_info_by_index(mic_index)['name']}...")
    frames = []
    silence_count = 0
    previous_rms = 0

    while True:
        data = stream.read(1024)
        frames.append(data)
        rms = audioop.rms(data, 2)  # Compute RMS value of the audio chunk

        # If RMS is below the threshold for a certain number of times, stop recording
        if rms < SILENCE_THRESHOLD:
            silence_count += 1
        else:
            silence_count = 0

        if silence_count > (SILENCE_DURATION * 16000 / 1024):  # Check if silence duration is met
            print("Silence detected. Stopping recording.")
            break

        previous_rms = rms

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Return the raw audio data (in-memory)
    audio_data = b"".join(frames)
    return audio_data


def convert_to_wav(audio_data):
    """
    Convert raw audio data to a valid WAV file-like object.
    """
    # Create a BytesIO stream to store the WAV data
    wav_buffer = io.BytesIO()

    # Set the parameters for the WAV file (1 channel, 16-bit, 16000 Hz)
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for 16-bit audio
        wf.setframerate(16000)
        wf.writeframes(audio_data)

    # Seek back to the beginning of the BytesIO buffer
    wav_buffer.seek(0)
    return wav_buffer


def transcribe_audio_with_whisper(audio_data):
    """
    Send raw audio data to Whisper API for transcription (without saving it to a file).
    """
    headers = {
        "Authorization": f"Bearer {os.getenv("API_KEY")}",
    }

    # Convert the raw audio data to WAV format in memory
    wav_audio = convert_to_wav(audio_data)

    # Send the audio file to Whisper for transcription
    response = requests.post(
        API_URL,
        headers=headers,
        files={"file": ("audio.wav", wav_audio, "audio/wav")},
        data={"model": "whisper-1", "language": "it"}
    )

    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def audio_setup():
    # List all available microphones
    p = pyaudio.PyAudio()
    print("Available microphones:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxInputChannels"] > 0:  # Only list input devices
            print(f"Index {i}: {device_info['name']}")

    # Ask user to select a microphone by index
    mic_index = int(input("Enter the index of the microphone you want to use: "))
    return mic_index


def audio_transcription(mic_index):
    # Step 1: Record audio using the selected microphone
    audio_data = record_audio(mic_index)
    
    # Step 2: Transcribe audio using Whisper API
    print("Transcribing audio using Whisper...")
    transcription = transcribe_audio_with_whisper(audio_data)
    if transcription:
        print("You said:", transcription)
        return transcription



# -----------------------------------------------------------------------------#
################### MODEL CONFIGURATION AND LOADING ############################
# -----------------------------------------------------------------------------#
def model_load(api_key = []):
    print("""
    ===============================
    Caricamento modelli in corso...
    ===============================
        """)
    
    mistralai_mm_llm = MistralAIMultiModal(
        model="pixtral-large-latest",
        #max_new_tokens=10000, 
        api_key=api_key[0]
    )
    
    mistralai_llm = MistralAI(
        model="mistral-large-latest",
        api_key=api_key[0])
    
    query_llm = MistralAI(
        model="mistral-medium",
        api_key=api_key[0])
    
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key[1])
    
    
    print(f"""
    ==============================
    Modelli correttamente caricati
     --> {mistralai_mm_llm.model} <--
     --> {mistralai_llm.model} <--
    --> text-embedding-3-small <--
    ==============================
        """)
    return mistralai_mm_llm, mistralai_llm, query_llm, embed_model


def api_configure():
    print(f"""
    ==============================
    Configurazione API in corso...
    ==============================
        """)
    load_dotenv()
    
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if mistral_api_key and openai_api_key:
        os.environ["MISTRAL_API_KEY"] = mistral_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        print("OpenAI API key loaded successfully.")
        print("Mistral API key loaded successfully.")
        
        return mistral_api_key, openai_api_key
    else:
        print("API key not found in .env file.")
        

def image_load(directory, image_path, mic_index):
    required_exts = [".jpg", ".jpeg", ".png"]

    while True:
        foto_capture()
        
        try:
            image_documents = SimpleDirectoryReader(
                directory,
                input_files=[image_path],
                required_exts=required_exts
            ).load_data()

            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            print("🤖 L'immagine va bene o vuoi scattarne un'altra?\n")
            text_to_audio("L'immagine va bene o vuoi scattarne un'altra?") 

        except Exception as e:
            print(f"⚠️ Errore nel caricamento dell'immagine: {str(e)}\n")
            text_to_audio("Si è verificato un errore nel caricamento dell'immagine. Riprovo.")
            continue  # Riprova a scattare una nuova foto

        # Ascolta la risposta vocale dell'utente
        while True:
            user_response = audio_transcription(mic_index)

            if not user_response:
                print("🔄 Non ho capito, puoi ripetere?\n")
                text_to_audio("Non ho capito, puoi ripetere?")
                continue  

            user_response = user_response.strip(".").lower()

            if "sì" in user_response or "va bene" in user_response:
                print("✅ Immagine accettata.\n")
                text_to_audio("Perfetto, l'immagine è stata accettata.")
                return image_documents  # Esce dal loop restituendo il documento

            elif "no" in user_response or "un'altra" in user_response:
                print("🔄 Scattiamo una nuova immagine...\n")
                text_to_audio("Scattiamo una nuova immagine.")
                break  # Esce dal loop interno e ripete il ciclo principale per scattare di nuovo

            else:
                print("⚠️ Risposta non riconosciuta. Puoi rispondere 'sì' o 'no'.\n")
                text_to_audio("Risposta non riconosciuta. Puoi rispondere 'sì' o 'no'.")



# -----------------------------------------------------------------------------#
############################### RAG AGENTS #####################################
# -----------------------------------------------------------------------------#
def textual_RAG(agent, user_input):
    response = agent.chat(user_input
    ) 
    return response

def image_RAG(llm):
    image_documents = image_load(directory, file_path, 0)
    #llm = model_load()
    response = llm.complete(
        prompt=audio_transcription(0), # richiesta da parte dell'utente con la voce
        image_documents=image_documents,
    )
    return response

def documents_agent_configuration(llm, query_llm, embed_model):

    docs = SimpleDirectoryReader(
        "C:/Users/filow/OneDrive/Desktop",
        input_files=["C:/Users/filow/OneDrive/Desktop/Tesi_Traversi.pdf"]
    ).load_data()

    index = VectorStoreIndex.from_documents(
        docs, embed_model=embed_model
    )
    engine = index.as_query_engine(similarity_top_k=3, llm=query_llm)
    query_engine_tool = QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name="Processo_assemblaggio",
            
            description=(
                "Provides Information about an experimental assembly process"
                "This assembly process regards a box with 24 different boxes filled with several objexts"
                "Each objects has a unique code, as the boxes"
                "You must help the user to assembly the final product with the correct objects in the correct boxes."
            ),
        ),
    )
    
    agent = FunctionCallingAgent.from_tools(
    [query_engine_tool], llm=llm, verbose=True
    )
    return agent



# -----------------------------------------------------------------------------#
################################## FOTO ########################################
# -----------------------------------------------------------------------------#

def foto_capture():
    
    # Choose the camera by its index (0 is the default webcam)
    camera_index = 0  # Change this if you want to use a different webcam

    # Open the webcam
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("ERRORE: non è possibile aprire la webcam\n")
        exit()

    # Initialize the recognizer for speech recognition
    recognizer = sr.Recognizer()

    print("top")
    while True:
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("ERRORE: non è stato possibile catturare l'immagine\n")
            break

        # Display the frame in a window (optional)
        # cv2.imshow('Webcam', frame)

        # Wait for the microphone to listen for a command
        with sr.Microphone() as source: # pip install pyaudio
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            print("In attesa del comando vocale...\n")

            try:
                # Listen for audio and convert it to text
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio, language="it-IT").lower()
                print(f"Hai detto: {command}\n")
                
                if "foto" in command:
                    print("Cattura immagine in corso...\n")
                    cv2.imwrite('captured_image.jpg', frame)
                    print("Immagine salvata con successo!\n")
                    break
                
            except sr.UnknownValueError:
                # If the speech is not understood, just skip
                pass
            except sr.RequestError:
                # If there is an issue with the speech service
                print("Could not request results from Google Speech Recognition service\n")

        # Wait for a key press to exit the loop
        key = cv2.waitKey(1) & 0xFF

        # If 'q' key is pressed, exit the loop
        if key == ord('q'):
            print("Uscendo...\n")
            break

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()



# -----------------------------------------------------------------------------#
################################## CHAT ########################################
# -----------------------------------------------------------------------------#

def chat_vocale(agent, mm_llm, mic_index):
    print("\n🗣️ Chat vocale avviata! Parla liberamente. Dì 'stop' per uscire.\n")

    while True:
        user_input = audio_transcription(mic_index)  
        if not user_input:
            print("🔄 Non ho capito, puoi ripetere?\n")
            continue

        user_input = user_input.strip(".").lower()

        if "stop" in user_input:
            print("🛑 Chat vocale terminata. Alla prossima!\n")
            text_to_audio("Chat terminata. Alla prossima!")
            break

        print(f"👤 Utente: {user_input}")

        # Verifica se l'utente sta parlando di un'immagine
        if "immagine" in user_input or "foto" in user_input or "analisi immagine" in user_input:
            print("🖼️ Attivando analisi immagine...\n")
            image_response = image_RAG(mm_llm)  
            #display(Markdown(f"🤖 Assistente: {image_response}"))
            print(f"🤖 Assistente: {image_response}\n")
            text_to_audio(str(image_response))
        else:
            # Risposta testuale tramite il modello RAG
            textual_response = textual_RAG(agent, user_input)
            
            #display(Markdown(f"🤖 Assistente: {textual_response}"))
            print(f"🤖 Assistente: {textual_response}\n")
            text_to_audio(str(textual_response))
