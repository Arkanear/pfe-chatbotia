"""Découpage des documents en fragments textuels."""

import re


def clean_text(text: str) -> str:
    """Nettoie les espaces inutiles sans supprimer les paragraphes."""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Découpe un texte en fragments avec chevauchement."""

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être supérieur à zéro.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap ne peut pas être négatif.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap doit être inférieur à chunk_size."
        )

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in cleaned_text.split("\n\n")
        if paragraph.strip()
    ]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = (
            f"{current_chunk}\n\n{paragraph}".strip()
            if current_chunk
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if len(paragraph) <= chunk_size:
            current_chunk = paragraph
            continue

        start = 0

        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            chunk = paragraph[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == len(paragraph):
                break

            start = end - chunk_overlap

        current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks