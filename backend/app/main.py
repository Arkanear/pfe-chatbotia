"""Point d'entrée de l'API du chatbot Backbone."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm.ollama_client import OllamaError, check_ollama, generate_response


app = FastAPI(
    title="Chatbot IA Backbone",
    description="API locale du chatbot documentaire Backbone.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    """Question envoyée au chatbot."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=4_000,
        description="Question posée par l'utilisateur.",
    )


class ChatResponse(BaseModel):
    """Réponse générée par le chatbot."""

    answer: str
    model_source: str = "ollama"


@app.get("/")
async def root() -> dict[str, str]:
    """Retourne les informations générales de l'API."""

    return {
        "application": "Chatbot IA Backbone",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Vérifie l'état de l'API et d'Ollama."""

    ollama_available = await check_ollama()

    return {
        "api": "available",
        "ollama": "available" if ollama_available else "unavailable",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Envoie une question au modèle local."""

    system_instruction = (
        "Tu es un assistant technique destiné aux équipes Backbone. "
        "Réponds en français de manière claire, précise et professionnelle. "
        "Pour le moment, indique clairement lorsque tu n'es pas certain "
        "d'une information.\n\n"
    )

    prompt = system_instruction + f"Question : {request.question}"

    try:
        answer = await generate_response(prompt)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(answer=answer)