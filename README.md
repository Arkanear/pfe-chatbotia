# PFE – Chatbot IA Backbone

Projet de fin d'études réalisé dans le cadre de la formation d'ingénieur
à IMT Nord Europe.

## Objectif

Développer un chatbot IA destiné à faciliter l'accès à la documentation
technique de la direction Backbone.

Le chatbot devra notamment :

- répondre à des questions métiers ;
- rechercher des informations dans une base documentaire ;
- fournir des réponses fondées sur les documents disponibles ;
- indiquer clairement lorsqu'une information n'est pas trouvée ;
- limiter les hallucinations du modèle ;
- fonctionner dans un environnement local et sécurisé.

## Architecture prévisionnelle

```text
Utilisateur
    |
    v
Interface / n8n
    |
    v
API Python
    |
    +------> RAG et base vectorielle
    |
    v
Ollama
    |
    v
Gemma

# Technologies
Python
FastAPI
Ollama
Gemma
Docker
n8n
base vectorielle à définir
État du projet
 Préparation du poste de développement
 Installation de Git
 Installation de Docker
 Installation d'Ollama
 Test de Gemma sur GPU
 Création de l'API Python
 Connexion de n8n à l'API
 Création du pipeline RAG
 Intégration de la base documentaire
 Tests utilisateurs
Confidentialité

Les documents internes utilisés par le chatbot ne doivent pas être publiés
dans ce dépôt sans autorisation préalable.

Auteur

Islem Zouaoui
