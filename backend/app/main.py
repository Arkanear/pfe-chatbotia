"""Point d'entrée de l'API du chatbot Backbone."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm.ollama_client import OllamaError, check_ollama, generate_response

from app.rag.retriever import SearchResult, search_documents

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

class SearchRequest(BaseModel):
    """Paramètres d'une recherche documentaire."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=4_000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    category: str | None = None


class SearchResultResponse(BaseModel):
    """Fragment documentaire retourné par l'API."""

    content: str
    source: str
    filename: str
    category: str
    subcategory: str
    extension: str
    chunk_index: int
    distance: float

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

@app.post(
    "/search",
    response_model=list[SearchResultResponse],
)
async def search(
    request: SearchRequest,
) -> list[SearchResultResponse]:
    """Recherche des passages pertinents dans la documentation."""

    try:
        results = await search_documents(
            question=request.question,
            top_k=request.top_k,
            category=request.category,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur est survenue pendant "
                "la recherche documentaire."
            ),
        ) from exc

    return [
        SearchResultResponse(
            content=result.content,
            source=result.source,
            filename=result.filename,
            category=result.category,
            subcategory=result.subcategory,
            extension=result.extension,
            chunk_index=result.chunk_index,
            distance=result.distance,
        )
        for result in results
    ]


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