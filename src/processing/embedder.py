import logging
import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

logger = logging.getLogger(__name__)


def generate_embedding(text: str) -> list[float] | None:
    """Sendet einen Text-Chunk an Ollama und gibt den 768-dimensionalen Vektor zurück."""
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }

    try:
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if isinstance(embedding, list) and len(embedding) == 768:
            return embedding
        else:
            logger.error("Unerwartete Vektor-Dimension erhalten")
            return None
    except requests.exceptions.RequestException:
        logger.exception("Fehler beim Generieren des Embeddings via Ollama")
        return None