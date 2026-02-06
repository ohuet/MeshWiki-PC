# MeshWiki — Assistant Wikipedia offline via Meshtastic

## Contexte et objectif

Créer une application Python qui reçoit des messages texte via un module Meshtastic (Bluetooth/USB), les transmet à un LLM local (Ollama) qui s'appuie sur une base Wikipedia France locale pour fournir des réponses factuelles, puis renvoie la réponse via Meshtastic.

**Cas d'usage** : Accès à des informations fiables en cas de coupure internet (cyclones à La Réunion) via un réseau mesh LoRa.

**Contraintes** :
- 100% offline une fois installé (sauf pour la mise à jour périodique de Wikipedia)
- Messages Meshtastic limités à ~228 octets par paquet
- Le LLM doit répondre de façon concise et factuelle
- Latence acceptable : quelques secondes à une minute

---

## Architecture

```
[Utilisateur Meshtastic]
    ↕ (radio LoRa)
[Module Meshtastic USB/BT]
    ↕ (série/BLE)
[meshwiki-bridge] ← module Python, écoute les messages entrants
    ↓
[RAG Pipeline]
    ├── Recherche sémantique dans Wikipedia (ChromaDB)
    └── Prompt + contexte → Ollama (LLM local)
    ↓
[Réponse découpée en chunks ≤228 octets]
    ↕
[Module Meshtastic] → réponse radio
```

---

## Structure du projet

```
meshwiki/
├── pyproject.toml
├── README.md
├── config.yaml             # configuration utilisateur
├── meshwiki/
│   ├── __init__.py
│   ├── main.py             # point d'entrée, orchestre tout
│   ├── meshtastic_bridge.py # communication avec le module Meshtastic
│   ├── rag.py              # recherche dans Wikipedia + construction du prompt
│   ├── llm.py              # appel à Ollama
│   ├── chunker.py          # découpage des réponses longues
│   ├── rate_limiter.py     # rate limiting par utilisateur (fenêtre glissante)
│   ├── wikipedia_indexer.py # indexation du dump Kiwix dans ChromaDB
│   └── wikipedia_updater.py # téléchargement auto + ré-indexation
├── data/
│   ├── chroma_db/          # base vectorielle (générée par l'indexer)
│   ├── tmp/                # fichiers de téléchargement temporaires
│   └── last_update.json    # suivi de la dernière mise à jour
└── tests/
    ├── test_rag.py
    ├── test_chunker.py
    ├── test_bridge.py
    ├── test_rate_limiter.py
    └── test_updater.py
```

**Dépendances principales** :
- `meshtastic` — communication avec le module
- `chromadb` — base vectorielle
- `sentence-transformers` — embeddings (modèle : `paraphrase-multilingual-MiniLM-L12-v2`)
- `requests` — API Ollama + téléchargement Kiwix
- `libzim` (paquet PyPI : `libzim`) — lecture des fichiers .zim
- `pyyaml` — configuration
- `schedule` ou `apscheduler` — planification des mises à jour
- `beautifulsoup4` — parsing HTML pour extraire la liste des fichiers Kiwix

---

## Phase 1 — Découpage des réponses (chunker.py)

**Fichier** : `meshwiki/chunker.py`

**Fonction** : `split_message(text: str, max_bytes: int = 220) -> list[str]`

Logique :
- Max 220 octets par chunk (marge de sécurité sous les 228 de Meshtastic)
- Numéroter les messages : `[1/3] début du texte...`
- Couper sur les espaces ou la ponctuation, jamais au milieu d'un mot
- Encoder en UTF-8 pour calculer la taille réelle en octets (les caractères accentués français font 2 octets)
- Limiter à 5 chunks maximum. Si la réponse est trop longue, tronquer avec `... [tronqué]`
- Si le texte tient dans un seul message, ne pas ajouter de numérotation

---

## Phase 2 — Rate Limiter (rate_limiter.py)

**Fichier** : `meshwiki/rate_limiter.py`

**Objectif** : Limiter le nombre de requêtes par utilisateur Meshtastic sur une fenêtre de temps glissante. Tous les paramètres sont configurables.

**Classe** : `RateLimiter`

`__init__(max_requests: int, window_seconds: int)` :
- `max_requests` : nombre max de requêtes autorisées par fenêtre (défaut : 10)
- `window_seconds` : durée de la fenêtre en secondes (défaut : 600, soit 10 minutes)
- Stockage interne : `dict[str, list[float]]` — pour chaque user_id, la liste des timestamps de ses requêtes récentes

`check(user_id: str) -> tuple[bool, str | None]` :
- Nettoyer d'abord les timestamps plus anciens que `window_seconds` pour cet utilisateur
- Compter les requêtes restantes dans la fenêtre
- Si le nombre de requêtes < `max_requests` :
  - Enregistrer le timestamp actuel
  - Retourner `(True, None)` — requête autorisée
- Si le nombre de requêtes >= `max_requests` :
  - Calculer le temps restant : `window_seconds` - (maintenant - timestamp de la première requête dans la fenêtre)
  - Formater le temps restant de façon lisible :
    - Si >= 60 secondes : afficher en minutes (arrondi supérieur), ex: `"3 min"`
    - Si < 60 secondes : afficher en secondes, ex: `"45 sec"`
  - Construire le message de refus dynamiquement à partir des paramètres :
    `f"Limite atteinte ({max_requests} requêtes par {window_minutes} min). Réessayez dans {temps_restant}."`
    Exemple avec les valeurs par défaut : `"Limite atteinte (10 requêtes par 10 min). Réessayez dans 3 min."`
  - Retourner `(False, message_de_refus)`

`cleanup()` :
- Méthode optionnelle appelable périodiquement pour purger les entrées d'utilisateurs inactifs depuis longtemps (> 1 heure) afin d'éviter une fuite mémoire lente

**Points importants** :
- Le rate limiter doit être thread-safe (utiliser un `threading.Lock`) car les messages Meshtastic arrivent de façon asynchrone
- L'identifiant utilisateur est le node ID Meshtastic de l'expéditeur
- Le message de refus doit tenir dans un seul message Meshtastic (< 220 octets), ce qui est garanti vu sa brièveté

---

## Phase 3 — Interface avec Ollama (llm.py)

**Fichier** : `meshwiki/llm.py`

**Fonction** : `generate(system_prompt: str, user_prompt: str) -> str`

Implémentation :
- Appel HTTP POST à `http://localhost:11434/api/generate`
- Paramètres :
  - `model` : depuis config.yaml (ex: `mistral`)
  - `prompt` : le prompt construit par rag.py
  - `system` : le system prompt
  - `stream` : false
  - `options.temperature` : 0.1 (réponses factuelles)
  - `options.num_predict` : 300 (limiter la longueur)
- Timeout : 120 secondes
- Gestion d'erreur : si Ollama ne répond pas, retourner un message d'erreur clair destiné à l'utilisateur Meshtastic

---

## Phase 4 — Indexation de Wikipedia (wikipedia_indexer.py)

**Fichier** : `meshwiki/wikipedia_indexer.py`

**Objectif** : Lire un fichier .zim Kiwix et créer un index vectoriel ChromaDB.

**Fonction principale** : `index_zim(zim_path: Path, collection_name: str = "wikipedia") -> dict`

Étapes :
1. Ouvrir le fichier .zim avec `libzim`
2. Pour chaque article :
   - Extraire le titre et le contenu HTML
   - Nettoyer le HTML pour obtenir du texte brut
   - Ignorer les redirections, pages de catégorie, pages méta
   - Découper en chunks de ~500 tokens avec chevauchement de 50 tokens
   - Chaque chunk garde comme métadonnée : titre de l'article, position dans l'article
3. Générer les embeddings avec `sentence-transformers` (modèle `paraphrase-multilingual-MiniLM-L12-v2`)
4. Stocker dans ChromaDB (persisté sur disque dans `data/chroma_db/`)
5. Afficher une barre de progression dans le terminal
6. Retourner des stats : `{"article_count": ..., "chunk_count": ...}`

L'indexation complète peut prendre plusieurs heures — c'est normal.

---

## Phase 5 — Pipeline RAG (rag.py)

**Fichier** : `meshwiki/rag.py`

**Fonction principale** : `query(question: str) -> str`

Étapes :
1. Encoder la question avec le même modèle d'embeddings
2. Chercher les 3 à 5 chunks les plus pertinents dans ChromaDB
3. Construire le prompt pour le LLM :

```
SYSTEM_PROMPT = """Tu es un assistant encyclopédique offline.
Tu réponds UNIQUEMENT à partir des extraits Wikipedia fournis.
Tes réponses doivent être :
- Concises (max 400 caractères si possible, car transmises par radio)
- Factuelles et précises
- En français
Si les extraits ne contiennent pas la réponse, dis-le clairement.
Ne fabrique JAMAIS d'information."""

USER_PROMPT = f"""Extraits Wikipedia pertinents :
---
{contexte_chunks}
---

Question : {question}

Réponds de façon concise."""
```

4. Envoyer au LLM via `llm.py`
5. Retourner la réponse

---

## Phase 6 — Communication Meshtastic (meshtastic_bridge.py)

**Fichier** : `meshwiki/meshtastic_bridge.py`

Utiliser la bibliothèque Python `meshtastic` officielle.

**Classe** : `MeshtasticBridge`

`__init__(connection_type, port)` :
- `connection_type` : `"serial"` ou `"ble"`
- Se connecter via `meshtastic.serial_interface.SerialInterface` ou `meshtastic.ble_interface.BLEInterface`

`on_message_received(packet)` :
- Filtrer : ne traiter que les messages texte (`TEXT_MESSAGE_APP`)
- Ignorer les messages provenant de notre propre node
- Reconnaître le préfixe configuré (par défaut `?`) pour ne répondre qu'aux questions explicites
- Exemple : un utilisateur envoie `?Quelle est la capitale de Madagascar` → le `?` est retiré et la question est passée au pipeline RAG
- **Vérifier le rate limiter** : appeler `rate_limiter.check(sender_id)`
  - Si `(False, message)` → envoyer le message de refus à l'expéditeur et ne pas traiter la requête
  - Si `(True, None)` → continuer le traitement
- Passer le message au pipeline RAG

`send_response(destination_id, text)` :
- Si `len(text.encode('utf-8')) <= 220` : envoyer directement
- Sinon : utiliser `chunker.py` pour découper, envoyer séquentiellement avec un délai configurable entre chaque message (par défaut 2.5 secondes pour éviter la congestion du mesh)

S'abonner aux événements :
```python
pub.subscribe(on_message_received, "meshtastic.receive.text")
```

Gestion de la reconnexion : si le module se déconnecte, tenter une reconnexion automatique toutes les 30 secondes.

---

## Phase 7 — Téléchargement automatique de Wikipedia (wikipedia_updater.py)

**Fichier** : `meshwiki/wikipedia_updater.py`

**Classe** : `WikipediaUpdater`

### Détection du dernier dump disponible

**Fonction** : `get_latest_dump_url() -> tuple[str, str] | None`

- Faire un GET sur `https://download.kiwix.org/zim/wikipedia/?C=M;O=D` (fichiers triés par date décroissante)
- Parser le HTML de la page pour extraire la liste des liens
- Chercher le premier lien dont le nom de fichier correspond au pattern `wikipedia_fr_all_mini_*.zim` (c'est le dump le plus récent)
- Retourner `(url_complete, nom_fichier)` ou `None` si non trouvé

### Téléchargement

**Fonction** : `download_dump() -> Path | None`

- Appeler `get_latest_dump_url()`
- Comparer le nom du fichier distant avec celui stocké dans `data/last_update.json`. Si c'est le même → skip, pas de nouveau dump
- Télécharger dans `data/tmp/` avec :
  - Support de reprise (HTTP `Range` headers) car le fichier peut être volumineux
  - Affichage de la progression dans les logs
- En cas d'échec réseau → logger l'erreur, ne rien casser, réessayer au prochain cycle
- Retourner le chemin du fichier téléchargé, ou `None` si échec ou pas de mise à jour

### Ré-indexation sécurisée

**Fonction** : `reindex(zim_path: Path) -> bool`

L'objectif est de ne jamais casser l'index actif. Procédure :

1. Indexer le nouveau dump dans une **nouvelle collection ChromaDB temporaire** (`"wikipedia_new"`)
2. Si l'indexation réussit complètement :
   - Renommer l'ancienne collection en `"wikipedia_old"` (backup)
   - Renommer `"wikipedia_new"` en `"wikipedia"` (collection active)
   - Supprimer `"wikipedia_old"`
   - Mettre à jour `data/last_update.json`
3. Si l'indexation échoue :
   - Supprimer la collection temporaire
   - Logger l'erreur
   - L'ancien index reste intact et fonctionnel

→ L'application continue de répondre aux requêtes Meshtastic pendant toute l'indexation grâce à ce swap.

### Nettoyage

**Fonction** : `cleanup(zim_path: Path)`

- Supprimer le fichier .zim téléchargé
- Supprimer tout fichier partiel dans `data/tmp/`
- Logger l'espace disque récupéré

### Orchestrateur

**Fonction** : `run_update()`

1. Log `"Vérification des mises à jour Wikipedia..."`
2. `zim_path = download_dump()`
3. Si `zim_path is None` → return (pas de MAJ disponible ou échec téléchargement)
4. Log `"Nouveau dump disponible, ré-indexation en cours..."`
5. `success = reindex(zim_path)`
6. `cleanup(zim_path)` — toujours nettoyer, que ce soit un succès ou un échec
7. Si succès → log `"Base Wikipedia mise à jour avec succès"`
8. Sinon → log `"Échec de la mise à jour, ancien index conservé"`

### Fichier de suivi : data/last_update.json

```json
{
  "last_filename": "wikipedia_fr_all_mini_2025-01.zim",
  "last_update": "2025-01-15T10:30:00",
  "index_article_count": 2450000,
  "index_chunk_count": 12000000,
  "next_check": "2025-02-15T02:00:00"
}
```

---

## Phase 8 — Orchestration (main.py)

**Fichier** : `meshwiki/main.py`

**Point d'entrée** : `python -m meshwiki`

### Séquence de démarrage

1. Charger `config.yaml`
2. Vérifier qu'Ollama est accessible (`GET http://localhost:11434/`)
3. Vérifier si un index ChromaDB existe dans `data/chroma_db/`
4. **Si aucun index n'existe** (premier lancement) :
   - Afficher `"Aucune base Wikipedia trouvée. Téléchargement initial en cours..."`
   - Lancer `wikipedia_updater.run_update()` de manière **bloquante** (l'application ne peut pas fonctionner sans données)
   - Si le téléchargement/indexation échoue → afficher une erreur claire et quitter
5. **Si un index existe** :
   - Charger le modèle d'embeddings + ChromaDB
   - Vérifier si une mise à jour est due (comparer la date dans `last_update.json` avec l'intervalle configuré). Si oui, lancer `run_update()` en **arrière-plan** (thread séparé) pour ne pas bloquer le service
6. Se connecter au module Meshtastic
7. Initialiser le scheduler de mise à jour périodique (thread séparé)
8. Afficher `"MeshWiki opérationnel. En attente de messages..."`
9. Logger chaque requête/réponse dans le terminal
10. Boucle principale (attente d'événements Meshtastic)

### Scheduler de mise à jour

- Planifier `run_update()` selon l'intervalle configuré (par défaut 30 jours)
- Option : restreindre les heures de mise à jour (par défaut entre 2h et 6h du matin) pour ne pas impacter les performances en journée
- Le scheduler tourne dans un thread séparé
- Baisser la priorité du thread d'indexation (`nice`) pour préserver la réactivité des réponses Meshtastic

### Gestion des erreurs en fonctionnement

- Si Ollama est down → répondre `"Service LLM indisponible"`
- Si la recherche Wikipedia ne trouve rien → répondre `"Aucun article pertinent trouvé pour cette question"`
- Si le module Meshtastic se déconnecte → tenter une reconnexion automatique toutes les 30 secondes

---

## Phase 9 — Tests

### tests/test_chunker.py
- Tester le découpage avec des textes courts (un seul message, pas de numérotation)
- Tester avec des textes longs (vérifier la numérotation `[1/N]`)
- Tester avec des caractères accentués français (vérifier le calcul en octets UTF-8)
- Vérifier que chaque chunk ≤ 220 octets
- Vérifier la troncature au-delà de 5 chunks

### tests/test_rag.py
- Mocker ChromaDB avec quelques articles de test
- Vérifier que la recherche retourne des résultats pertinents
- Vérifier la construction correcte du prompt (system + user + contexte)

### tests/test_rate_limiter.py
- Tester qu'un utilisateur peut faire `max_requests` requêtes sans être bloqué
- Tester qu'à la requête `max_requests + 1` dans la fenêtre, le limiter retourne `(False, message)`
- Vérifier que le message de refus contient le bon nombre de requêtes et la bonne fenêtre (s'adapte aux paramètres)
- Vérifier que le temps restant affiché est correct (en minutes si >= 60s, en secondes sinon)
- Tester qu'après expiration de la fenêtre, l'utilisateur peut de nouveau faire des requêtes
- Tester que deux utilisateurs différents ont des compteurs indépendants
- Tester le cleanup des utilisateurs inactifs
- Tester la thread-safety avec des requêtes concurrentes

### tests/test_bridge.py
- Mocker l'interface Meshtastic
- Tester le filtrage des messages (préfixe `?`, ignorer ses propres messages)
- Tester l'envoi séquentiel avec délai entre les chunks

### tests/test_updater.py
- Mocker les requêtes HTTP vers download.kiwix.org
- Tester le parsing HTML pour trouver le bon fichier `wikipedia_fr_all_mini_*.zim`
- Tester la détection d'un nouveau dump vs pas de changement
- Tester le swap atomique d'index (ancienne collection préservée si échec)
- Tester le cleanup (vérifier que les fichiers temporaires sont supprimés)

---

## Configuration complète — config.yaml

```yaml
meshtastic:
  connection: "serial"        # "serial" ou "ble"
  port: "/dev/ttyUSB0"        # port série ou adresse MAC BLE
  trigger_prefix: "?"         # préfixe pour déclencher une requête
  response_delay: 2.5         # secondes entre chaque chunk de réponse
  max_response_chunks: 5      # nombre max de messages par réponse

rate_limiting:
  max_requests: 10              # nombre de requêtes autorisées par fenêtre
  window_seconds: 600           # durée de la fenêtre en secondes (600 = 10 min)

wikipedia:
  source_format: "zim"

embeddings:
  model: "paraphrase-multilingual-MiniLM-L12-v2"
  chunk_size: 500             # tokens par chunk d'article
  chunk_overlap: 50           # chevauchement entre chunks

vectordb:
  path: "./data/chroma_db"

ollama:
  base_url: "http://localhost:11434"
  model: "mistral"            # ou phi3:mini, llama3, mixtral...
  temperature: 0.1
  max_tokens: 300

updater:
  enabled: true
  interval_days: 30
  check_on_startup: true
  allowed_hours: [2, 6]       # fenêtre horaire pour DL/indexation (2h-6h)
  kiwix_url: "https://download.kiwix.org/zim/wikipedia/?C=M;O=D"
  dump_pattern: "wikipedia_fr_all_mini_"  # pattern du nom de fichier à chercher
  temp_dir: "./data/tmp"
  max_download_retries: 3
  download_timeout_hours: 24
```

---

## Ordre d'implémentation recommandé

Implémente chaque phase dans cet ordre. Après chaque phase, écris les tests correspondants et vérifie qu'ils passent.

1. **chunker.py** — le plus simple, testable immédiatement sans dépendance externe
2. **rate_limiter.py** — simple aussi, testable unitairement sans dépendance externe
3. **llm.py** — vérifiable si Ollama tourne localement
4. **wikipedia_indexer.py** — parsing ZIM + indexation ChromaDB
5. **rag.py** — recherche sémantique + construction du prompt
6. **wikipedia_updater.py** — téléchargement auto + ré-indexation sécurisée
7. **meshtastic_bridge.py** — communication avec le module (intègre le rate limiter)
8. **main.py** — assemblage final de tous les composants

---

## Notes techniques

- **Choix du modèle Ollama** : `mistral` ou `phi3:mini` sont de bons compromis vitesse/qualité en français si la machine n'est pas très puissante. Avec un bon GPU, `llama3` ou `mixtral` donneront de meilleures réponses.
- **Espace disque** : pendant une mise à jour, le dump ZIM + l'ancien index + le nouvel index coexistent temporairement. Prévoir suffisamment d'espace libre.
- **Premier lancement** : l'application télécharge et indexe automatiquement si aucune donnée n'est présente. C'est bloquant car elle ne peut pas fonctionner sans index.
- **Résilience** : l'application ne doit jamais crasher à cause d'une mise à jour échouée. L'ancien index est toujours préservé en cas de problème.
