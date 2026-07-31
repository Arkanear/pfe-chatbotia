"""Test en ligne de commande de la recherche documentaire."""

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_PATH))

from app.rag.retriever import search_documents


def display_result(
    position: int,
    result,
) -> None:
    """Affiche un résultat de recherche lisiblement."""

    print("\n" + "=" * 80)
    print(f"RÉSULTAT {position}")
    print("=" * 80)

    print(f"Distance       : {result.distance:.4f}")
    print(f"Catégorie      : {result.category}")
    print(f"Sous-catégorie : {result.subcategory}")
    print(f"Fichier        : {result.filename}")
    print(f"Source         : {result.source}")
    print(f"Fragment       : {result.chunk_index}")

    print("\nContenu :")
    print("-" * 80)
    print(result.content)
    print("-" * 80)


async def main(
    question: str,
    top_k: int,
    category: str | None,
) -> None:
    """Lance une recherche et affiche les résultats."""

    results = await search_documents(
        question=question,
        top_k=top_k,
        category=category,
    )

    if not results:
        print("Aucun fragment documentaire trouvé.")
        return

    print(f"\nQuestion : {question}")
    print(f"Résultats retournés : {len(results)}")

    for position, result in enumerate(
        results,
        start=1,
    ):
        display_result(position, result)


def parse_arguments() -> argparse.Namespace:
    """Lit les paramètres de ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Recherche sémantique dans la documentation Backbone."
        )
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question à rechercher dans les documents.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de fragments retournés.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "Filtre facultatif : "
            "FTTH, DSL, DCN ou Collecte-Fixe."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    asyncio.run(
        main(
            question=arguments.question,
            top_k=arguments.top_k,
            category=arguments.category,
        )
    )