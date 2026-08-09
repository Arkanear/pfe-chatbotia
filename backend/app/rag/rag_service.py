"""Construction des réponses documentaires du chatbot RAG."""

from dataclasses import dataclass

from app.config import get_settings
from app.llm.ollama_client import generate_response
from app.rag.retriever import (
    SearchResult,
    detect_category,
    search_documents,
)


@dataclass
class RagSource:
    """Source documentaire utilisée pour produire une réponse."""

    source: str
    filename: str
    category: str
    subcategory: str
    chunk_index: int
    distance: float
    content: str


@dataclass
class RagAnswer:
    """Réponse du chatbot accompagnée de ses sources."""
    answer: str
    sources: list[RagSource]
    category: str | None


def build_document_context(results: list[SearchResult]) -> str:
    """Transforme les fragments trouvés en contexte numéroté."""

    context_parts: list[str] = []

    for position, result in enumerate(results, start=1):
        context_parts.append(
            "\n".join(
                [
                    f"[Source {position}]",
                    f"Fichier : {result.filename}",
                    f"Chemin : {result.source}",
                    f"Catégorie : {result.category}",
                    f"Sous-catégorie : {result.subcategory}",
                    f"Fragment : {result.chunk_index}",
                    "",
                    result.content,
                ]
            )
        )

    separator = "\n\n" + "-" * 80 + "\n\n"
    return separator.join(context_parts)


async def answer_with_rag(
    question: str,
    top_k: int = 5,
    category: str | None = None,
) -> RagAnswer:
    """Recherche les documents puis génère une réponse fondée dessus."""

    cleaned_question = question.strip()

    detected_category = category

    if not detected_category:
        detected_category = await detect_category(
            question=cleaned_question
            )

    results = await search_documents(
        question=cleaned_question,
        top_k=top_k,
        category=detected_category,
    )

    if not cleaned_question:
        raise ValueError("La question ne peut pas être vide.")

    settings = get_settings()
     

    if not results:
        return RagAnswer(
            answer=(
                "Je n'ai trouvé aucun document permettant "
                "de répondre à cette question."
            ),
            sources=[],
            category=detected_category,
        )

    relevant_results = [
        result
        for result in results
        if result.distance <= settings.rag_max_distance
    ]

    if not relevant_results:
        return RagAnswer(
            answer=(
                "Les documents actuellement indexés ne contiennent "
                "pas d'information suffisamment pertinente pour répondre "
                "de manière fiable à cette question."
            ),
            sources=[],
            category=detected_category,
        )

    context = build_document_context(relevant_results)

    system_prompt = """
Tu es un assistant technique destiné aux équipes de la direction Backbone.

Tu dois répondre uniquement à partir des extraits documentaires fournis.

Règles obligatoires :
1. N'utilise pas tes connaissances générales pour compléter la réponse.
2. Si les documents ne permettent pas de répondre précisément, indique-le.
3. Ne crée aucune commande, adresse IP, procédure ou valeur absente des sources.
4. Cite les extraits utilisés avec la notation [Source 1], [Source 2], etc.
5. Réponds en français, de manière claire, structurée et professionnelle.
6. Lorsque plusieurs types d'équipements ou procédures apparaissent, distingue-les.
7. Si les extraits ne répondent pas directement à la question, indique uniquement
   que l'information n'est pas disponible, sans résumer les extraits hors sujet.
""".strip()

    user_prompt = f"""
QUESTION DE L'UTILISATEUR

{cleaned_question}

EXTRAITS DOCUMENTAIRES DISPONIBLES

{context}

CONSIGNE FINALE

Réponds uniquement à partir des extraits ci-dessus.
Cite chaque information importante avec la source correspondante.
Si les extraits sont insuffisants, indique clairement que l'information
n'est pas disponible dans la documentation fournie.
""".strip()

    generated_answer = await generate_response(
        prompt=user_prompt,
        system=system_prompt,
        temperature=0.1,
    )

    sources = [
        RagSource(
            source=result.source,
            filename=result.filename,
            category=result.category,
            subcategory=result.subcategory,
            chunk_index=result.chunk_index,
            distance=result.distance,
            content=result.content,
        )
        for result in relevant_results
    ]

    return RagAnswer(
        answer=generated_answer,
        sources=sources,
        category=detected_category,
    )