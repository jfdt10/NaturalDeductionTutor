from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests

app = FastAPI()

# Carregar variáveis de ambiente
load_dotenv()

# Verificar variáveis de ambiente
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGING_FACE_API_TOKEN")

if not GOOGLE_API_KEY:
    raise ValueError("A variável de ambiente GOOGLE_API_KEY não foi encontrada no arquivo .env.")
if not HUGGINGFACE_API_KEY:
    raise ValueError("A variável de ambiente HUGGINGFACE_API_KEY não foi encontrada no arquivo .env.")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar a API do Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

class LogicRequest(BaseModel):
    prompt: str
    data: dict

class LogicResponse(BaseModel):
    result: str
    modelInfo: str
    error: Optional[str] = None

# Configurações dos modelos
MODEL_CONFIGS = {
    "gemini": {
        "model": "gemini-1.5-flash",
        "provider": "google"
    },
    "qwen": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "provider": "huggingface",
        "api_url": "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
    },
    "mixtral": {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "provider": "huggingface",
        "api_url": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    }
}

# Prompts detalhados para cada tarefa
TASK_PROMPTS = {
    "translation": """
    Traduza a seguinte frase em linguagem natural para lógica proposicional. Siga o formato abaixo:
    - Identifique as proposições atômicas e atribua a elas variáveis lógicas (por exemplo, P, Q, R, A, B, C, etc.).
    - Traduza conectivos lógicos como "e", "ou", "se...então", "se e somente se" para os símbolos correspondentes (∧, ∨, →, ↔).
    - Forneça a tradução final em lógica proposicional, incluindo premissas e conclusão.

    Exemplo:
    Frase: "Se estudo e faço exercícios, então serei aprovado. Eu estudo e faço exercícios."
    Tradução em lógica proposicional:
    Premissas:
    1. (A ∧ B) → C
    2. A ∧ B
    Conclusão: C

    Formato final:
    (A ∧ B) → C, A ∧ B ⊢ C

    Onde:
    - A: Eu estudo.
    - B: Eu faço exercícios.
    - C: Eu serei aprovado.

    Agora, traduza a seguinte frase:
    """,
    "deduction": """
    Forneça uma prova dedutiva usando dedução natural para a seguinte afirmação. Siga estas etapas:
    - Liste as premissas.
    - Aplique regras de inferência (como Modus Ponens, Modus Tollens, Introdução/Eliminação de conectivos) para derivar a conclusão.
    - Formate a resposta apenas com os passos da dedução, indicando a regra aplicada e a referência.
    - Você pode usar quaisquer símbolos para as proposições atômicas (P, Q, R, A, B, C, etc.) e aplicar outras regras de inferência ou equivalências lógicas, se necessário.

    Exemplo:
    Premissas e Conclusão:
    A → B, B → C, A ⊢ C

    Passos da Dedução:
    1. A → B (Hipótese)
    2. B → C (Hipótese)
    3. A (Hipótese)
    4. B MP(1,3)
    5. C MP(2,4)

    Agora, prove a seguinte afirmação:
    """,
    "validation": """
    Valide os seguintes passos de dedução. Verifique se:
    - As premissas são verdadeiras.
    - As regras de inferência foram aplicadas corretamente.
    - A conclusão segue logicamente das premissas.
    - Para cada passo, explique a regra aplicada e valide se está correta.
    - Você pode usar quaisquer símbolos para as proposições atômicas (P, Q, R, A, B, C, etc.) e aplicar outras regras de inferência ou equivalências lógicas, se necessário.

    Exemplo:
    Premissas e Conclusão:
    A → B, B → C, A ⊢ C v D

    Passos da Dedução:
    1. A → B (Hipótese)
    2. B → C (Hipótese)
    3. A (Hipótese)
    4. B MP(1,3)
    5. C MP(2,4)
    6. C v D AD(5)

    Validação:
    - Passo 4: B foi derivado corretamente usando Modus Ponens (MP) das linhas 1 e 3.
    - Passo 5: C foi derivado corretamente usando Modus Ponens (MP) das linhas 2 e 4.
    - Passo 6: C v D foi derivado corretamente usando Adição (AD) da linha 5.
    - Conclusão 'C v D' foi deduzida corretamente.

    Agora, valide os seguintes passos:
    """
}

@app.post("/api/ai_completion")
async def ai_completion(request: LogicRequest):
    try:
        model_name = request.data.get("model", "gemini")
        task = request.data.get("task", "translation")
        
        if model_name not in MODEL_CONFIGS:
            raise HTTPException(status_code=400, detail="Modelo não suportado")

        model_config = MODEL_CONFIGS[model_name]
        
        # Preparar o prompt baseado na tarefa
        if task not in TASK_PROMPTS:
            raise HTTPException(status_code=400, detail="Tarefa não suportada")
        
        formatted_prompt = f"{TASK_PROMPTS[task]}\n{request.data['input']}"
        
        # Fazer a chamada ao modelo
        if model_config["provider"] == "google":
            # Usar o Google Gemini
            model = genai.GenerativeModel(model_config["model"])
            response = model.generate_content(formatted_prompt)
            result = response.text
        else:
            # Usar o Hugging Face
            headers = {
                "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_tokens": 1024
                }
            }
            response = requests.post(model_config["api_url"], headers=headers, json=payload)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Erro na chamada ao Hugging Face")
            result = response.json()[0]["generated_text"]
        
        return LogicResponse(
            result=result,
            modelInfo=f"{model_name.upper()} via {model_config['provider'].upper()}"
        )
        
    except Exception as e:
        return LogicResponse(
            result="",
            modelInfo="",
            error=f"Erro ao processar: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
