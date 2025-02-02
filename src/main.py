from utils import *

def main():
    # Configurazione delle API
    mistral_api_key, openai_api_key = api_configure()    
    
    # Configurazione dei modelli
    mm_llm, llm, query_llm, embed_model = model_load([mistral_api_key, openai_api_key])

    # Configurazione dell'agente per i documenti
    agent = documents_agent_configuration(llm, query_llm, embed_model)
    
    # Configurazione audio
    try:
        mic_index = audio_setup()
    except Exception as e:
        print(f"⚠️ Errore durante la configurazione del microfono:, {str(e)}\n")
        return 

    print("\n🎙️ Assistente avviato! Puoi parlare con me.")
    print("➡️ Per chiudere dire: 'stop', 'chiudi' o 'exit'.")
    print("➡️ Per chiedere assistenza dire: 'assistenza'.\n")

    while True:
        user_input = audio_transcription(mic_index)  

        if not user_input:
            print("🔄 Non ho capito, puoi ripetere?\n")
            continue  

        user_input = user_input.strip('.').lower()

        if "stop" in user_input:
            print("👋 Chiusura dell'assistente. Alla prossima!\n")
            text_to_audio("Chiusura dell'assistente. Alla prossima!")
            break  

        elif "assistenza" in user_input:
            chat_vocale(agent, mm_llm, mic_index)
            print("\n🔹 Hai richiesto assistenza. Che tipo di supporto vuoi?\n")
           
        else:
            print("⚠️ Comando non riconosciuto, riprova.\n")ì


if __name__ == "__main__":
    main()

