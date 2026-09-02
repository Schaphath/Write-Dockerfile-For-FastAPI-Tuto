# 🩺 API de Détection du Cancer du Sein — FastAPI & Docker

Ce projet montre comment déployer un modèle de machine learning entraîné (issu d'un notebook) sous forme d'une **API REST avec FastAPI**, puis comment **conteneuriser** cette API avec **Docker** pour la rendre facilement déployable.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128.0-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Sommaire

- [Description](#-description)
- [Fonctionnalités](#-fonctionnalités)
- [Structure du projet](#-structure-du-projet)
- [Prérequis](#-prérequis)
- [Installation & lancement en local](#-installation--lancement-en-local)
- [Lancement avec Docker](#-lancement-avec-docker)
- [Utilisation de l'API](#-utilisation-de-lapi)
- [Avertissement](#-avertissement)
- [Auteur](#-auteur)

---

## 📖 Description

Cette API permet de prédire si une tumeur mammaire est **bénigne (B)** ou **maligne (M)**, à partir de **10 caractéristiques** sélectionnées lors de l'entraînement du modèle.

Plusieurs algorithmes ont été testés et comparés ; le modèle retenu pour ce tutoriel est un **`HistGradientBoostingClassifier`**.

### Performances du modèle

| Métrique  | Score |
|-----------|-------|
| Recall    | ~98%  |
| Precision | ~97%  |
| Accuracy  | ~97%  |

> ⚠️ Ces résultats sont à considérer avec du recul : ce projet a un but pédagogique et démonstratif, pas un usage clinique.

---

## ⚙️ Fonctionnalités

L'API expose trois endpoints :

| Endpoint    | Méthode | Description                                              |
|-------------|---------|-----------------------------------------------------------|
| `/infos`    | `GET`   | Informations générales sur l'API (version, description…) |
| `/health`   | `GET`   | Statut de santé de l'API et de ses dépendances (modèle, scaler) |
| `/predict`  | `POST`  | Réalise une prédiction à partir des caractéristiques fournies |

---

## 🗂 Structure du projet

```
.
├── app.py                  # Point d'entrée de l'API FastAPI
├── model/                  # Modèle entraîné (.pkl) et scaler
├── requirements-prod.txt   # Dépendances Python pour la production
├── Dockerfile              # Instructions de build de l'image Docker
└── README.md
```

*(Adaptez cette arborescence selon l'organisation réelle de votre projet.)*

---

## ✅ Prérequis

- [Python 3.12+](https://www.python.org/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

Dépendances Python principales (voir `requirements-prod.txt`) :

```
fastapi==0.128.0
uvicorn==0.40.0
numpy==2.4.0
scikit-learn==1.8.0
pydantic==2.12.5
```

---

## 💻 Installation & lancement en local

### 1. Installer les dépendances

```bash
pip install -r requirements-prod.txt
```

### 2. Lancer l'API

```bash
uvicorn app:app --reload
```

L'API est alors accessible sur [http://localhost:8000](http://localhost:8000), et sa documentation interactive sur [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🐳 Lancement avec Docker

### 1. Construire l'image

```bash
docker build -t api-cancer:v1.0.0 .
```

### 2. Lancer le conteneur

```bash
docker run -d -p 8000:8000 --name api_cancer api-cancer:v1.0.0
```

### 3. Gérer le conteneur

```bash
# Voir les logs en direct
docker logs -f api_cancer

# Arrêter le conteneur
docker stop api_cancer

# Redémarrer le conteneur
docker start api_cancer

# Supprimer le conteneur
docker rm api_cancer
```

---

## 🚀 Utilisation de l'API

Une fois l'API lancée (en local ou via Docker), voici quelques exemples de requêtes et réponses attendues.

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "model_version": "1.0.0"
}
```

### `POST /predict`

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "area_worst": 850,
  "compactness_worst": 0.28,
  "concave_points_worst": 0.15,
  "concavity_worst": 0.32,
  "fractal_dimension_worst": 0.3,
  "perimeter_worst": 110.2,
  "radius_worst": 16.5,
  "smoothness_worst": 0.14,
  "symmetry_worst": 0.29,
  "texture_worst": 25.3
}'
```

```json
{
  "prediction": "M",
  "label": "Risque de cancer",
  "probability": 0.979,
  "model_version": "1.0.0"
}
```

> ℹ️ Les noms exacts des 10 caractéristiques attendues sont documentés dans le schéma Pydantic de l'API, consultable via `/docs`.

---

## ⚠️ Avertissement

Cette API a un **but purement démonstratif** : illustrer comment passer d'un modèle entraîné dans un notebook à une API packagée dans une image Docker.

Elle **ne doit pas être utilisée à des fins de diagnostic médical réel**. Vous êtes libre de reprendre, modifier et renforcer le code pour vos propres besoins.

---

## 👤 Auteur

**@Madiba**