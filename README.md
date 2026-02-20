# MeshWiki

Accès Wikipedia hors-ligne via radio Meshtastic LoRa.

Les utilisateurs envoient une question par radio, MeshWiki cherche dans une base Wikipedia locale (recherche sémantique), génère une réponse via un LLM local, et la renvoie par radio — le tout sans connexion internet.

**Cas d'usage** : accès fiable à l'information pendant les coupures internet (cyclones à La Réunion).

## Architecture

```
[Radio Meshtastic] ←→ [Bridge] → [Rate limiter] → [RAG] → [LLM] → [Chunker] → [Radio]
                                                     ↓
                                              [ChromaDB / embeddings]
                                              [Kiwix / recherche ZIM]
```

1. Un message `?Question` arrive via Meshtastic
2. Si la recherche Kiwix assistée par LLM est activée : le LLM génère des expressions de recherche, qui sont cherchées dans le(s) fichier(s) ZIM via Kiwix (stratégie Phase 1 / Phase 2 avec déduplication)
3. Si aucune réponse Kiwix : la question est encodée et cherchée sémantiquement dans ChromaDB
4. Les extraits Wikipedia pertinents sont envoyés au LLM avec la question
5. La réponse est découpée en chunks de 220 octets max et renvoyée par radio

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

rag:
  max_distance: 0.60           # Seuil de distance cosinus max
  use_llm_for_kiwix_keywords: true  # Recherche Kiwix assistée par LLM

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
| `rag.py` | Pipeline RAG : recherche Kiwix assistée LLM, recherche sémantique ChromaDB, construction du prompt |
| `llm.py` | Interface Ollama (paramètre `max_tokens` configurable par appel) |
| `chunker.py` | Découpage des réponses en paquets radio (220 octets max) |
| `rate_limiter.py` | Limite de débit par noeud Meshtastic |
| `kiwix_search.py` | Recherche plein-texte dans les fichiers ZIM (suggestions + Xapian), multi-ZIM |
| `wikipedia_indexer.py` | Indexation ZIM → ChromaDB avec checkpoints |
| `wikipedia_updater.py` | Téléchargement et mise à jour automatique |
| `collection_state.py` | Gestion des slots ChromaDB (swap A/B sans downtime) |
| `config.py` | Chargement de `config.yaml` + `config.local.yaml` avec merge |
| `zim_discovery.py` | Détection automatique des fichiers ZIM dans le dossier data |
| `export_android.py` | Export de la base vectorielle au format Android |
| `build_zim.py` | Construction de ZIM personnalisés à partir de catégories Wikipedia |
| `remote_embeddings.py` | Encodage distant via API OpenAI/Ollama (pour indexation GPU) |
| `progress.py` | Affichage progression en console |
