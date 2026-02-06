# MeshWiki

Accès Wikipedia hors-ligne via radio Meshtastic LoRa.

Les utilisateurs envoient une question par radio, MeshWiki cherche dans une base Wikipedia locale (recherche sémantique), génère une réponse via un LLM local, et la renvoie par radio — le tout sans connexion internet.

**Cas d'usage** : accès fiable à l'information pendant les coupures internet (cyclones à La Réunion).

## Architecture

```
[Radio Meshtastic] ←→ [Bridge] → [Rate limiter] → [RAG] → [LLM] → [Chunker] → [Radio]
                                                     ↓
                                              [ChromaDB / embeddings]
```

1. Un message `?Question` arrive via Meshtastic
2. La question est encodée et cherchée sémantiquement dans ChromaDB
3. Les extraits Wikipedia pertinents sont envoyés au LLM avec la question
4. La réponse est découpée en chunks de 220 octets max et renvoyée par radio

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.ai/) avec un modèle tiré (`ollama pull phi4-mini:3.8b`)
- Un module Meshtastic connecté en USB ou BLE
- ~4 Go d'espace disque pour le dump Wikipedia FR

## Installation

```bash
pip install -e .
```

## Lancement

```bash
python -m meshwiki
```

Sous Windows, double-cliquer sur `run.bat`.

Au premier lancement, MeshWiki télécharge automatiquement le dump Wikipedia français depuis Kiwix (~3.5 Go) et l'indexe dans ChromaDB. L'indexation prend 2-5h selon le GPU. Pendant ce temps, les questions reçoivent des réponses du LLM sans source Wikipedia.

## Configuration

Tout se configure dans `config.yaml` :

```yaml
meshtastic_timezone: "Indian/Reunion"  # Timezone pour les heures affichées aux utilisateurs

meshtastic:
  connection: "serial"       # "serial" ou "ble"
  # port: "/dev/ttyUSB0"     # Optionnel, auto-détecté si absent
  trigger_prefix: "?"        # Préfixe pour déclencher une question
  response_delay: 2.5        # Délai entre chunks (secondes)

ollama:
  model: "phi4-mini:3.8b"    # Modèle Ollama à utiliser
  temperature: 0.1           # Bas = réponses factuelles
  max_tokens: 300            # Court pour la radio

rate_limiting:
  max_requests: 10           # Requêtes par fenêtre par noeud
  window_seconds: 600        # Fenêtre de 10 minutes

updater:
  enabled: true
  interval_days: 30          # Vérification tous les 30 jours
  allowed_hours: [2, 6]      # Téléchargement entre 2h et 6h du matin
```

## Tests

```bash
pytest tests/ -v
```

## Utilisation radio

Voir [GUIDE_UTILISATEUR.md](GUIDE_UTILISATEUR.md).

## Fonctionnement hors-ligne

Après le premier lancement (qui nécessite internet pour télécharger Wikipedia), MeshWiki fonctionne **100% hors-ligne**. Le module de mise à jour vérifie périodiquement si un nouveau dump est disponible, mais c'est optionnel.

## Modules

| Module | Rôle |
|--------|------|
| `main.py` | Point d'entrée, vérifications, orchestration |
| `meshtastic_bridge.py` | Écoute et envoi des messages radio |
| `rag.py` | Recherche sémantique + construction du prompt LLM |
| `llm.py` | Interface Ollama |
| `chunker.py` | Découpage des réponses en paquets radio (220 octets max) |
| `rate_limiter.py` | Limite de débit par noeud Meshtastic |
| `wikipedia_indexer.py` | Indexation ZIM → ChromaDB avec checkpoints |
| `wikipedia_updater.py` | Téléchargement et mise à jour automatique |
| `progress.py` | Affichage progression en console |
