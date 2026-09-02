
# Auteur : @Madiba


#==================================================#
#  ÉTAPE 1 : Build & Installation des dépendances  #
#==================================================#
FROM python:3.12-slim AS builder

WORKDIR /build

# Copie uniquement le fichier de dépendances 
COPY requirements-prod.txt .

# Installation des dépendances dans /install avec cache pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --prefix=/install --no-warn-script-location -r requirements-prod.txt


#=========================================#
#  ÉTAPE 2 : Image finale de production   #
#=========================================#
FROM python:3.12-slim AS runner

WORKDIR /app

# Création du groupe et de l'utilisateur non-root
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin appuser

# Copie des dépendances depuis le builder vers le répertoire système
COPY --from=builder /install /usr/local

# Copie de TOUS les fichiers du projet (api.py, models/, etc.)
COPY --chown=appuser:appgroup . /app

# Bascule sur l'utilisateur non-root
USER 10001

# Vérification de l'état de santé du conteneur via l'endpoint /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose le port d'écoute
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]