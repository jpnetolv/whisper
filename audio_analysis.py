import whisper
import subprocess
import ollama
import os

def transcrever_audio(caminho_audio: str, modelo: str = "base") -> str:
    print("[Whisper] Carregando modelo...")
    model = whisper.load_model(modelo)
    
    print(f"[Whisper] Transcrevendo: {caminho_audio}")
    resultado = model.transcribe(caminho_audio)
    texto = resultado["text"]

    print("[Whisper] Transcrição concluída.")
    return texto

def extrair_pontos_chave(texto: str, modelo_ollama: str = "llama2") -> str:
    prompt = (
        "Resumo do texto a seguir com os principais pontos-chave:\n\n"
        f"{texto}\n\n"
        "Liste os pontos-chave de forma objetiva e clara."
    )

    print("[Ollama] Enviando prompt para modelo...")
    resposta = ollama.chat(model=modelo_ollama, messages=[
        {"role": "user", "content": prompt}
    ])

    return resposta["message"]["content"]

def main():
    caminho_audio = "C:/Users/Tioh_/Documents/Projetos/whisper/whisper/data/audio.mp3"
    modelo_whisper = "base"  

    if not os.path.exists(caminho_audio):
        print(f"Erro: arquivo '{caminho_audio}' não encontrado.")
        return

    # Etapa 1: Transcrição
    texto_transcrito = transcrever_audio(caminho_audio, modelo_whisper)
    print("\n--- TRANSCRIÇÃO COMPLETA ---\n")
    print(texto_transcrito)

    # Etapa 2: Extração de pontos-chave
    resumo = extrair_pontos_chave(texto_transcrito)
    print("\n--- PONTOS-CHAVE ---\n")
    print(resumo)

if __name__ == "__main__":
    main()
