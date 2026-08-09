"""Recherche sémantique dans la base documentaire ChromaDB."""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import get_settings
from app.rag.embeddings import create_embeddings
from collections import Counter

async def detect_category(
    question: str,
    candidate_count: int = 8,
) -> str | None:
    """
    Détecte automatiquement la catégorie documentaire
    la plus pertinente pour une question.
    """

    results = await search_documents(
        question=question,
        top_k=candidate_count,
        category=None,
    )

    if not results:
        return None

    # On ne conserve que les résultats suffisamment pertinents
    settings = get_settings()

    relevant_results = [
        result
        for result in results
        if result.distance <= settings.rag_max_distance
    ]

    if not relevant_results:
        return None

    categories = [
        result.category
        for result in relevant_results
        if result.category
    ]

    if not categories:
        return None

    counts = Counter(categories)

    most_common_category, _ = counts.most_common(1)[0]

    return most_common_category

@dataclass
class SearchResult:
    """Fragment documentaire retourné par la recherche."""

    content: str
    source: str
    filename: str
    category: str
    subcategory: str
    extension: str
    chunk_index: int
    distance: float


def get_collection() -> chromadb.Collection:
    """Ouvre la collection documentaire persistante."""

    settings = get_settings()

    project_root = Path(__file__).resolve().parents[3]
    vector_db_path = project_root / settings.vector_db_path

    client = chromadb.PersistentClient(
        path=str(vector_db_path)
    )

    return client.get_collection(
        name=settings.rag_collection_name
    )


async def search_documents(
    question: str,
    top_k: int | None = None,
    category: str | None = None,
) -> list[SearchResult]:
    """Recherche les fragments les plus proches d'une question."""

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("La question ne peut pas être vide.")

    settings = get_settings()
    result_count = top_k or settings.rag_top_k

    query_embeddings = await create_embeddings(
        [cleaned_question]
    )

    collection = get_collection()

    where_filter = None

    if category:
        where_filter = {
            "category": category
        }

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=result_count,
        where=where_filter,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not documents or not documents[0]:
        return []

    search_results: list[SearchResult] = []

    for document, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        if document is None or metadata is None:
            continue

        search_results.append(
            SearchResult(
                content=document,
                source=str(
                    metadata.get("source", "Source inconnue")
                ),
                filename=str(
                    metadata.get("filename", "Fichier inconnu")
                ),
                category=str(
                    metadata.get("category", "Non classé")
                ),
                subcategory=str(
                    metadata.get("subcategory", "Racine")
                ),
                extension=str(
                    metadata.get("extension", "")
                ),
                chunk_index=int(
                    metadata.get("chunk_index", 0)
                ),
                distance=float(distance),
            )
        )

    return search_results