"""Génération d'embeddings avec l'API locale Ollama."""

import httpx

from app.config import get_settings


class EmbeddingError(RuntimeError):
    """Erreur liée à la génération d'embeddings."""


async def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Transforme une liste de textes en vecteurs."""

    if not texts:
        return []

    settings = get_settings()

    payload = {
        "model": settings.ollama_embedding_model,
        "input": texts,
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/embed",
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise EmbeddingError(
            "Impossible de générer les embeddings avec Ollama."
        ) from exc

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings:
        raise EmbeddingError(
            "Aucun vecteur n'a été retourné par Ollama."
        )

    if len(embeddings) != len(texts):
        raise EmbeddingError(
            "Le nombre d'embeddings ne correspond pas "
            "au nombre de textes envoyés."
        )

    return embeddings