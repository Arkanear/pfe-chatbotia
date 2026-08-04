"""Client HTTP permettant d'interroger Ollama."""

from typing import Any

import httpx

from app.config import get_settings


class OllamaError(RuntimeError):
    """Erreur rencontrée lors d'un appel à Ollama."""


async def generate_response(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.1,
) -> str:
    """Envoie un prompt au modèle Ollama et retourne sa réponse."""

    settings = get_settings()

    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    if system:
        payload["system"] = system

    endpoint = f"{settings.ollama_base_url}/api/generate"

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                endpoint,
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(
            "Impossible de communiquer avec Ollama. "
            "Vérifiez que le service est lancé."
        ) from exc

    data = response.json()
    generated_text = data.get("response")

    if not generated_text:
        raise OllamaError(
            "Ollama n'a retourné aucune réponse exploitable."
        )

    return generated_text.strip()


async def check_ollama() -> bool:
    """Vérifie que l'API locale Ollama est accessible."""

    settings = get_settings()
    endpoint = f"{settings.ollama_base_url}/api/tags"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
    except httpx.HTTPError:
        return False

    return True