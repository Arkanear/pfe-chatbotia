"""Indexation locale des documents dans ChromaDB."""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_PATH))

from app.config import get_settings
from app.rag.chunker import split_text
from app.rag.document_loader import find_documents, load_document
from app.rag.embeddings import create_embeddings


def build_chunk_id(
    relative_path: str,
    chunk_index: int,
    content: str,
) -> str:
    """Construit un identifiant stable pour un fragment."""

    raw_value = f"{relative_path}:{chunk_index}:{content}"
    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()


async def index_documents(
    limit: int | None,
    category_filter: str | None,
    reset: bool,
) -> None:
    """Indexe les documents compatibles dans ChromaDB."""

    settings = get_settings()

    documents_root = PROJECT_ROOT / settings.documents_path
    vector_db_path = PROJECT_ROOT / settings.vector_db_path

    if not documents_root.exists():
        raise FileNotFoundError(
            f"Dossier documentaire introuvable : {documents_root}"
        )

    files = find_documents(documents_root)

    if category_filter:
        normalized_filter = category_filter.lower()

        files = [
            path
            for path in files
            if path.relative_to(documents_root).parts[0].lower()
            == normalized_filter
        ]

    if limit is not None:
        files = files[:limit]

    if not files:
        print("Aucun document compatible trouvé.")
        return

    client = chromadb.PersistentClient(
        path=str(vector_db_path)
    )

    if reset:
        try:
            client.delete_collection(
                settings.rag_collection_name
            )
            print("Ancienne collection supprimée.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.rag_collection_name,
        metadata={
            "description": (
                "Documentation technique Backbone"
            ),
        },
    )

    indexed_files = 0
    indexed_chunks = 0
    failed_files = 0

    for file_number, file_path in enumerate(files, start=1):
        relative_path = str(
            file_path.relative_to(PROJECT_ROOT)
        )

        print(
            f"[{file_number}/{len(files)}] "
            f"Lecture : {relative_path}"
        )

        try:
            loaded_document = load_document(
                path=file_path,
                documents_root=documents_root,
            )

            chunks = split_text(
                text=loaded_document.text,
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            )

            if not chunks:
                print("  Aucun texte exploitable.")
                failed_files += 1
                continue

            embeddings = await create_embeddings(chunks)

            ids: list[str] = []
            metadatas: list[dict[str, str | int]] = []

            for chunk_index, chunk in enumerate(chunks):
                ids.append(
                    build_chunk_id(
                        relative_path=relative_path,
                        chunk_index=chunk_index,
                        content=chunk,
                    )
                )

                metadatas.append(
                    {
                        "source": relative_path,
                        "filename": file_path.name,
                        "category": loaded_document.category,
                        "subcategory": loaded_document.subcategory,
                        "extension": loaded_document.extension,
                        "chunk_index": chunk_index,
                    }
                )

            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            indexed_files += 1
            indexed_chunks += len(chunks)

            print(f"  {len(chunks)} fragments indexés.")

        except Exception as exc:
            failed_files += 1
            print(f"  ERREUR : {exc}")

    print("\n--- Résultat de l'indexation ---")
    print(f"Fichiers traités : {indexed_files}")
    print(f"Fichiers en erreur : {failed_files}")
    print(f"Fragments indexés : {indexed_chunks}")
    print(f"Collection : {settings.rag_collection_name}")
    print(f"Base locale : {vector_db_path}")


def parse_arguments() -> argparse.Namespace:
    """Lit les arguments de la ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Indexation des documents Backbone."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximal de fichiers à indexer.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "Catégorie à indexer : "
            "FTTH, DSL, DCN ou Collecte-Fixe."
        ),
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprime la collection avant l'indexation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    asyncio.run(
        index_documents(
            limit=arguments.limit,
            category_filter=arguments.category,
            reset=arguments.reset,
        )
    )