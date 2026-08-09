import time
import uuid

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/rag/chat"


# -------------------------------------------------------------------
# Configuration de la page
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Chatbot IA Backbone",
    page_icon="💬",
    layout="wide",
)


# -------------------------------------------------------------------
# Initialisation de l'état de session
# -------------------------------------------------------------------

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_conversation_id" not in st.session_state:
    conversation_id = str(uuid.uuid4())

    st.session_state.conversations[conversation_id] = {
        "title": "Nouvelle conversation",
        "messages": [],
    }

    st.session_state.current_conversation_id = conversation_id


# -------------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------------

def create_new_conversation() -> None:
    """Crée une nouvelle conversation vide."""

    conversation_id = str(uuid.uuid4())

    st.session_state.conversations[conversation_id] = {
        "title": "Nouvelle conversation",
        "messages": [],
    }

    st.session_state.current_conversation_id = conversation_id


def generate_conversation_title(question: str) -> str:
    """
    Génère un titre court à partir de la première question.

    Pour l'instant, on utilise une règle simple sans faire appel au LLM.
    """

    cleaned_question = question.strip()

    if len(cleaned_question) <= 42:
        return cleaned_question

    return cleaned_question[:39].rstrip() + "..."


def count_unique_documents(sources: list[dict]) -> int:
    """Compte le nombre de fichiers distincts utilisés."""

    filenames = {
        source.get("filename")
        for source in sources
        if source.get("filename")
    }

    return len(filenames)


def distance_to_relevance(distance: float) -> int:
    """
    Convertit approximativement une distance en score lisible.

    Ce score est uniquement indicatif pour l'interface.
    """

    score = max(
        0.0,
        min(
            1.0,
            1.0 - distance,
        ),
    )

    return round(score * 100)


def display_sources(sources: list[dict]) -> None:
    """Affiche les sources ainsi que les extraits réellement utilisés par le RAG."""

    if not sources:
        st.caption(
            "Aucune source documentaire utilisée."
        )
        return

    unique_documents = {
        source.get("filename")
        for source in sources
        if source.get("filename")
    }

    with st.expander(
        (
            f"📚 Sources utilisées "
            f"({len(unique_documents)} document(s), "
            f"{len(sources)} fragment(s))"
        ),
        expanded=False,
    ):

        for index, source in enumerate(
            sources,
            start=1,
        ):

            filename = source.get(
                "filename",
                "Fichier inconnu",
            )

            category = source.get(
                "category",
                "Non classé",
            )

            subcategory = source.get(
                "subcategory",
                "Racine",
            )

            chunk_index = source.get(
                "chunk_index",
                "?",
            )

            distance = float(
                source.get(
                    "distance",
                    1.0,
                )
            )

            content = source.get(
                "content",
                "Extrait indisponible.",
            )

            st.markdown(
                f"### 📄 Source {index} — {filename}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.caption("Domaine")
                st.code(
                    category,
                    language=None,
                )

            with col2:
                st.caption("Sous-domaine")
                st.code(
                    subcategory,
                    language=None,
                )

            with col3:
                st.caption("Fragment")
                st.code(
                    str(chunk_index),
                    language=None,
                )

            st.caption(
                f"Distance vectorielle : {distance:.3f}"
            )

            # Extrait documentaire réellement fourni à Gemma
            with st.expander(
                "🔎 Afficher l'extrait utilisé",
                expanded=False,
            ):
                st.markdown(
                    "**Contenu envoyé au modèle :**"
                )

                st.text(
                    content
                )

                st.caption(
                    "Cet extrait provient directement "
                    "du fragment récupéré dans ChromaDB."
                )

            # Chemin du fichier
            with st.expander(
                "📁 Informations sur le document",
                expanded=False,
            ):
                st.markdown(
                    f"**Fichier :** `{filename}`"
                )

                st.markdown(
                    f"**Chemin :** `{source.get('source', 'Inconnu')}`"
                )

                st.markdown(
                    f"**Catégorie :** `{category}`"
                )

                st.markdown(
                    f"**Sous-catégorie :** `{subcategory}`"
                )

                st.markdown(
                    f"**Fragment :** `{chunk_index}`"
                )

            if index < len(sources):
                st.divider()


def display_assistant_metadata(
    detected_category: str | None,
    sources: list[dict],
    response_time: float | None,
) -> None:
    """Affiche les principales informations de génération."""

    document_count = count_unique_documents(
        sources
    )

    fragment_count = len(sources)

    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "Domaine",
            detected_category
            if detected_category
            else "Non détecté",
        )

    with columns[1]:
        st.metric(
            "Documents",
            document_count,
        )

    with columns[2]:
        st.metric(
            "Fragments",
            fragment_count,
        )

    with columns[3]:
        if response_time is None:
            st.metric(
                "Temps",
                "-",
            )
        else:
            st.metric(
                "Temps",
                f"{response_time:.1f} s",
            )


def display_message(message: dict) -> None:
    """Affiche un message de l'historique."""

    role = message["role"]

    with st.chat_message(role):

        if role == "assistant":
            display_assistant_metadata(
                detected_category=message.get(
                    "detected_category"
                ),
                sources=message.get(
                    "sources",
                    [],
                ),
                response_time=message.get(
                    "response_time"
                ),
            )

        st.markdown(
            message["content"]
        )

        if role == "assistant":
            display_sources(
                message.get(
                    "sources",
                    [],
                )
            )


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:

    st.title("💬 Backbone IA")

    st.caption(
        "Assistant documentaire technique"
    )

    if st.button(
        "➕ Nouvelle conversation",
        use_container_width=True,
        type="primary",
    ):
        create_new_conversation()
        st.rerun()

    st.divider()

    st.subheader("Conversations")

    conversation_items = list(
        st.session_state.conversations.items()
    )

    for conversation_id, conversation in reversed(
        conversation_items
    ):

        title = conversation["title"]

        is_current = (
            conversation_id
            == st.session_state.current_conversation_id
        )

        button_label = (
            f"▶ {title}"
            if is_current
            else title
        )

        if st.button(
            button_label,
            key=f"conversation_{conversation_id}",
            use_container_width=True,
        ):
            st.session_state.current_conversation_id = (
                conversation_id
            )
            st.rerun()

    st.divider()

    st.caption(
        "PFE — Assistant IA Backbone"
    )


# -------------------------------------------------------------------
# Conversation active
# -------------------------------------------------------------------

current_conversation_id = (
    st.session_state.current_conversation_id
)

current_conversation = (
    st.session_state.conversations[
        current_conversation_id
    ]
)


# -------------------------------------------------------------------
# Interface principale
# -------------------------------------------------------------------

st.title("Chatbot IA Backbone")

st.caption(
    "Assistant documentaire basé sur les procédures techniques internes."
)

st.divider()


# -------------------------------------------------------------------
# Affichage de l'historique de la conversation
# -------------------------------------------------------------------

for message in current_conversation[
    "messages"
]:
    display_message(message)


# -------------------------------------------------------------------
# Champ de saisie
# -------------------------------------------------------------------

question = st.chat_input(
    "Posez une question sur la documentation Backbone..."
)


# -------------------------------------------------------------------
# Traitement d'une nouvelle question
# -------------------------------------------------------------------

if question:

    # Première question = titre de conversation
    if (
        current_conversation["title"]
        == "Nouvelle conversation"
    ):
        current_conversation["title"] = (
            generate_conversation_title(
                question
            )
        )

    user_message = {
        "role": "user",
        "content": question,
    }

    current_conversation[
        "messages"
    ].append(
        user_message
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Recherche dans la documentation..."
        ):

            payload = {
                "question": question,
                "top_k": 5,
                "category": None,
            }

            start_time = time.perf_counter()

            try:

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=300,
                )

                response.raise_for_status()

                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                data = response.json()

                answer = data.get(
                    "answer",
                    "Aucune réponse reçue.",
                )

                sources = data.get(
                    "sources",
                    [],
                )

                detected_category = data.get(
                    "detected_category"
                )

                display_assistant_metadata(
                    detected_category=(
                        detected_category
                    ),
                    sources=sources,
                    response_time=(
                        elapsed_time
                    ),
                )

                st.markdown(answer)

                display_sources(
                    sources
                )

                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "detected_category": (
                        detected_category
                    ),
                    "response_time": (
                        elapsed_time
                    ),
                }

                current_conversation[
                    "messages"
                ].append(
                    assistant_message
                )

            except requests.exceptions.RequestException:

                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                error_message = (
                    "Impossible de contacter le "
                    "backend du chatbot."
                )

                st.error(
                    error_message
                )

                current_conversation[
                    "messages"
                ].append(
                    {
                        "role": "assistant",
                        "content": (
                            error_message
                        ),
                        "sources": [],
                        "detected_category": None,
                        "response_time": (
                            elapsed_time
                        ),
                    }
                )

            except ValueError:

                error_message = (
                    "Le backend a retourné une "
                    "réponse invalide."
                )

                st.error(
                    error_message
                )

                current_conversation[
                    "messages"
                ].append(
                    {
                        "role": "assistant",
                        "content": (
                            error_message
                        ),
                        "sources": [],
                        "detected_category": None,
                        "response_time": None,
                    }
                )

    st.rerun()