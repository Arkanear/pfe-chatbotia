# PFE Chatbot IA Backbone

Assistant conversationnel basé sur une architecture **RAG (Retrieval-Augmented Generation)** permettant d'interroger en langage naturel une base documentaire technique.

Ce projet a été développé dans le cadre d'un **Projet de Fin d'Études (PFE)**. Son objectif est de faciliter l'accès aux procédures et connaissances techniques en proposant une interface conversationnelle capable de rechercher les informations pertinentes dans un corpus documentaire local et de générer des réponses sourcées.

## Fonctionnalités

* Interface conversationnelle Web
* Recherche documentaire en langage naturel
* Architecture RAG
* Génération locale avec Ollama
* Base vectorielle ChromaDB
* Indexation de documents PDF, DOCX et TXT
* Découpage automatique des documents en fragments
* Recherche sémantique par embeddings
* Détection automatique du domaine documentaire
* Réponses générées uniquement à partir des documents retrouvés
* Affichage des sources utilisées
* Consultation des extraits documentaires transmis au modèle
* Gestion de plusieurs conversations dans l'interface
* Refus de répondre lorsque la documentation disponible est insuffisante
* API REST avec FastAPI
* Orchestration possible avec n8n

## Architecture

Le fonctionnement général du système est le suivant :

```text
Documents locaux
(PDF / DOCX / TXT)
        │
        ▼
Extraction du texte
        │
        ▼
Découpage en fragments
        │
        ▼
Génération des embeddings
        │
        ▼
ChromaDB
        │
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             │
Question utilisateur                  │
        │                             │
        ▼                             │
Interface Streamlit                   │
        │                             │
        ▼                             │
FastAPI                               │
        │                             │
        ▼                             │
Détection du domaine                  │
        │                             │
        ▼                             │
Recherche vectorielle ◄───────────────┘
        │
        ▼
Fragments pertinents
        │
        ▼
LLM local via Ollama
        │
        ▼
Réponse + sources
        │
        ▼
Interface utilisateur
```

## Technologies utilisées

| Technologie | Rôle                                |
| ----------- | ----------------------------------- |
| Python      | Langage principal                   |
| FastAPI     | API backend                         |
| Streamlit   | Interface utilisateur               |
| Ollama      | Exécution locale des modèles        |
| ChromaDB    | Base de données vectorielle         |
| n8n         | Orchestration des workflows         |
| Docker      | Exécution de services conteneurisés |
| Git         | Gestion des versions                |

## Organisation du projet

```text
pfe-chatbotia/
│
├── backend/
│   └── app/
│       ├── rag/
│       ├── llm/
│       ├── config.py
│       └── main.py
│
├── frontend/
│   └── app.py
│
├── scripts/
│   └── index_documents.py
│
├── workflows/
│
├── documents/
│   ├── FTTH/
│   ├── DSL/
│   ├── DCN/
│   └── Collecte-Fixe/
│
├── vector_db/
│
├── .gitignore
└── README.md
```

> Les documents métiers et la base vectorielle locale ne sont pas versionnés dans le dépôt Git.

## Fonctionnement du RAG

### 1. Indexation

Les documents sont analysés puis découpés en fragments.

Chaque fragment est transformé en représentation vectorielle grâce au modèle d'embeddings exécuté localement avec Ollama.

Les vecteurs et leurs métadonnées sont ensuite enregistrés dans ChromaDB.

### 2. Question utilisateur

L'utilisateur pose une question depuis l'interface Streamlit.

Exemple :

```text
Comment configurer un LAG sur un DSLAM Huawei ?
```

### 3. Détection du domaine

Le système recherche automatiquement le domaine documentaire le plus pertinent parmi les catégories disponibles, par exemple :

```text
FTTH
DSL
DCN
Collecte-Fixe
```

L'utilisateur n'a donc pas besoin de sélectionner manuellement la catégorie correspondant à sa question.

### 4. Recherche documentaire

La question est convertie en embedding puis comparée aux fragments enregistrés dans ChromaDB.

Les fragments les plus pertinents sont récupérés.

### 5. Génération

La question et les extraits documentaires sélectionnés sont transmis au LLM local.

Le modèle reçoit comme instruction de répondre uniquement à partir des informations présentes dans ces extraits.

Lorsque la documentation retrouvée n'est pas suffisamment pertinente, le système refuse de produire une réponse non documentée.

### 6. Sources

La réponse contient les références aux documents utilisés.

Depuis l'interface, l'utilisateur peut également consulter les fragments documentaires ayant servi à générer la réponse.

Cette fonctionnalité permet de vérifier la provenance de l'information et facilite l'analyse d'une éventuelle erreur de recherche ou de génération.

## Installation

### Prérequis

* Python 3.11
* Git
* Ollama
* Docker Desktop, si n8n est utilisé

### Création de l'environnement Python

Depuis la racine du projet :

```powershell
python -m venv .venv
```

Activation sous Windows :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installation des dépendances :

```powershell
python -m pip install -r backend\requirements.txt
```

## Préparation des documents

Les documents doivent être placés dans le dossier :

```text
documents/
```

et organisés par domaine.

Exemple :

```text
documents/
├── FTTH/
├── DSL/
├── DCN/
└── Collecte-Fixe/
```

Les formats actuellement pris en charge sont :

```text
.pdf
.docx
.txt
```

Les dossiers identifiés comme archives, sauvegardes ou anciennes versions peuvent être exclus de l'indexation.

## Indexation

Exemple d'indexation d'un domaine :

```powershell
python scripts\index_documents.py --category FTTH --limit 30 --reset
```

Le paramètre `--reset` supprime la collection existante. Il doit donc être utilisé uniquement lorsqu'une reconstruction volontaire de la base est souhaitée.

Les autres domaines peuvent ensuite être ajoutés sans réinitialiser la collection :

```powershell
python scripts\index_documents.py --category DSL --limit 30
python scripts\index_documents.py --category DCN --limit 20
python scripts\index_documents.py --category Collecte-Fixe --limit 30
```

## Lancement

### 1. Démarrer Ollama

Vérifier que les modèles nécessaires sont disponibles localement.

### 2. Démarrer FastAPI

```powershell
python -m uvicorn app.main:app `
    --app-dir backend `
    --host 0.0.0.0 `
    --port 8000 `
    --reload
```

L'API est alors accessible localement sur le port `8000`.

La documentation Swagger est disponible sur :

```text
http://127.0.0.1:8000/docs
```

### 3. Démarrer l'interface

Dans un second terminal :

```powershell
streamlit run frontend\app.py
```

L'interface est alors disponible par défaut sur :

```text
http://localhost:8501
```

## Mise à jour de la documentation

Dans la configuration actuelle, la documentation est volontairement gérée localement.

Les mises à jour du corpus sont effectuées manuellement afin de conserver un contrôle sur les documents intégrés au système.

Une mise à jour périodique peut être réalisée, ainsi qu'une mise à jour immédiate lorsqu'une nouvelle consigne importante doit être prise en compte.

Le processus général est :

```text
Nouveaux documents
        ↓
Validation
        ↓
Ajout dans documents/
        ↓
Indexation
        ↓
Mise à jour de ChromaDB
        ↓
Documents disponibles pour le chatbot
```

## Confidentialité

L'architecture a été pensée pour permettre un fonctionnement local.

Les composants principaux du chatbot, la base vectorielle et les modèles peuvent être exécutés sur la machine hébergeant la solution.

Les documents internes ne doivent pas être ajoutés au dépôt Git.

Le fichier `.gitignore` doit notamment exclure les éléments sensibles ou générés localement tels que :

```text
.env
.venv/
documents/
vector_db/
__pycache__/
```

## Limites actuelles

* Le corpus documentaire est mis à jour manuellement.
* L'accès direct au SharePoint n'est pas implémenté.
* La qualité des réponses dépend de la qualité et de la couverture des documents indexés.
* Certains documents complexes ou très volumineux peuvent nécessiter un traitement spécifique.
* L'historique des conversations de l'interface n'est pas persistant après la fin de la session.
* Le système reste un outil d'assistance : les procédures critiques doivent être vérifiées dans leur documentation de référence.

## Perspectives

Plusieurs évolutions peuvent être envisagées :

* indexation incrémentale des nouveaux documents ;
* meilleure gestion des versions documentaires ;
* recherche hybride sémantique et lexicale ;
* reranking des résultats ;
* persistance de l'historique ;
* amélioration de l'évaluation automatique du RAG ;
* authentification et gestion des droits ;
* déploiement sur une infrastructure interne ;
* intégration à des outils collaboratifs sous réserve des autorisations nécessaires.

## Statut

**Prototype fonctionnel — Projet de Fin d'Études**

Le système permet actuellement d'indexer une documentation technique locale, d'effectuer des recherches sémantiques multi-domaines et de produire des réponses sourcées depuis une interface conversationnelle.
