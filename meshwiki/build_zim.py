"""Build a custom ZIM file from selected Wikipedia categories and articles.

Fetches full HTML articles from the French Wikipedia API, strips images,
and packages them into a ZIM file for offline use with MeshWiki.

Usage:
    python -m meshwiki.build_zim --init-config      # Generate default config
    python -m meshwiki.build_zim --dry-run           # Discovery only
    python -m meshwiki.build_zim                     # Full build
    python -m meshwiki.build_zim --resume            # Resume after interruption
    python -m meshwiki.build_zim -c custom.yaml      # Alternative config
"""

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.parse
from datetime import date
from pathlib import Path

import requests
import yaml
from libzim.writer import Creator, Hint, Item, StringProvider
from lxml import html as lxml_html

from meshwiki.progress import ProgressDisplay

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("zim_build_config.yaml")
STAGING_DIR = Path("data/zim_build_staging")
STAGING_INDEX = STAGING_DIR / "_index.json"


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "output": "data/meshwiki_reunion.zim",
    "wikipedia": {
        "api_url": "https://fr.wikipedia.org/w/api.php",
        "user_agent": "MeshWikiBot/0.1 (offline-wiki-for-mesh-radio)",
        "request_delay": 0.5,
        "max_retries": 3,
        "retry_delay": 5,
    },
    "categories": {
        "max_depth": 10,
        "blacklist": [
            "Catégorie:Portail:La Réunion/Articles liés",
            "Catégorie:Projet:La Réunion",
            "Catégorie:Théorie des nœuds",
        ],
        "list": [
            # La Réunion
            {"name": "La Réunion"},
            {"name": "Géographie de La Réunion"},
            {"name": "Histoire de La Réunion"},
            {"name": "Catastrophe à La Réunion"},
            {"name": "Santé à La Réunion"},
            {"name": "Environnement à La Réunion"},
            {"name": "Eau à La Réunion"},
            # Faune / Flore
            {"name": "Faune endémique de La Réunion"},
            {"name": "Flore endémique de La Réunion"},
            {"name": "Faune à La Réunion"},
            {"name": "Flore à La Réunion"},
            # Volcanologie
            {"name": "Piton de la Fournaise"},
            {"name": "Cratère volcanique à La Réunion"},
            {"name": "Cône volcanique à La Réunion"},
            {"name": "Volcanologie"},
            {"name": "Type d'éruption volcanique"},
            {"name": "Risque volcanique"},
            # Cyclones
            {"name": "Cyclone tropical"},
            {"name": "Cyclone tropical à La Réunion"},
            {"name": "Protection cyclonique"},
            {"name": "Prévision des cyclones tropicaux"},
            {"name": "Gestion des risques en météorologie"},
            # Santé / Secours
            {"name": "Premiers secours"},
            {"name": "Maladie tropicale"},
            {"name": "Maladie infectieuse tropicale"},
            {"name": "Dengue"},
            {"name": "Brûlure"},
            {"name": "Numéro d'urgence"},
            # Nœuds
            {"name": "Nœud"},
            # Urgences
            {"name": "Technique de survie"},
            {"name": "Plan d'urgence"},
            {"name": "Risque naturel"},
            {"name": "Inondation"},
            {"name": "Glissement de terrain"},
            # Médicaments
            {"name": "Médicament essentiel listé par l'OMS"},
            # Alimentation / Agriculture
            {"name": "Cuisine réunionnaise"},
            {"name": "Agriculture à La Réunion"},
            # Plantes médicinales
            {"name": "Plante médicinale"},
        ],
    },
    "articles": [
        # Échelles météo
        "Échelle de Saffir-Simpson",
        "Échelle de Beaufort",
        "Échelle de Fujita améliorée",
        # Premiers secours — article principal et gestes
        "Premiers secours (médecine)",
        "Gestes de premiers secours",
        "Secourisme",
        "Réanimation cardiopulmonaire",
        "Manœuvre de Heimlich",
        "Méthode de Mofenson",
        "Position latérale de sécurité",
        "Libération des voies aériennes",
        "Garrot",
        "Point de compression",
        "Pansement compressif",
        "Bandage",
        "Attelle",
        "Collier cervical",
        "Écharpe (médecine)",
        "Trousse de secours",
        "Défibrillateur automatisé externe",
        # Premiers secours — évaluation et protocoles
        "Alerte (premiers secours)",
        "Bilan (premiers secours)",
        "Malaise (premiers secours)",
        "Protocole GREC",
        # Premiers secours — pathologies
        "Arrêt cardiorespiratoire",
        "Hémorragie",
        "Arrêt d'une hémorragie",
        "Accident vasculaire cérébral",
        "Infarctus du myocarde",
        "Fibrillation ventriculaire",
        "Troubles du rythme cardiaque",
        "Fausse route",
        "Hyperglycémie",
        "Hypoglycémie diabétique",
        "Gelure",
        "Brûlure",
        "Hypothermie",
        "Coup de chaleur",
        "Noyade",
        "Sauvetage aquatique",
        "Morsure de serpent",
        "Méduse (animal)",
        # Maladies tropicales
        "Chikungunya",
        "Zika",
        "Leptospirose",
        "Paludisme",
        "Aedes albopictus",
        # Eau / Survie
        "Purification de l'eau",
        "Déshydratation (médecine)",
        "Soluté de réhydratation orale",
        "Eau potable",
        # Médicaments
        "Paracétamol",
        "Ibuprofène",
        "Aspirine",
        "Insectifuge",
        # Numéros d'urgence
        "Numéro d'appel d'urgence 112",
        "SAMU",
        # Radio
        "LoRa",
        "Meshtastic",
        "Radioamateur",
        # Survie / Autonomie
        "Techniques de production de feu",
        "Abri",
        "Signal de détresse",
        "Morse (alphabet)",
        "Conservation des aliments",
        "Panneau solaire",
        "Groupe électrogène",
        "Boussole",
        # Alimentation / Autonomie
        "Plante comestible",
        "Pêche (halieutique)",
        "Jardin potager",
        "Compostage",
        # Météo
        "Nuage",
        "Pression atmosphérique",
        "Marée",
        "Courant marin",
        # Santé mentale
        "Stress post-traumatique",
        "Deuil",
        # Communication
        "Alphabet phonétique de l'OTAN",
        # Réparation / Construction
        "Charpente",
        "Étanchéité (construction)",
        # Assainissement
        "Assainissement",
        "Latrine",
        # Soins des plaies / Traumatismes
        "Infection",
        "Fracture osseuse",
        "Entorse",
        "Luxation",
        "Désinfection",
        "Tétanos",
        "Suture (médecine)",
        # Risques naturels océan Indien
        "Tsunami",
        "Séisme",
        # Insectes / Nuisibles tropicaux
        "Moustique",
        "Termite",
        # Savoirs pratiques / Réparation
        "Nage",
        "Couture",
        "Cordage",
        "Électricité",
        "Pile électrique",
        # Éclairage / Cuisson sans électricité
        "Bougie (éclairage)",
        "Lampe à pétrole",
        "Réchaud",
        # Conservation alimentaire
        "Salaison",
        "Séchage (aliment)",
        "Fermentation",
        # Eau
        "Récupération d'eau de pluie",
        "Puits",
        # Post-cyclone
        "Moisissure",
        # Autosuffisance
        "Apiculture",
        "Élevage",
        # Orientation
        "Navigation astronomique",
        # Administratif post-crise
        "État de catastrophe naturelle",
        # Géographie / Culture
        "Créole réunionnais",
        "Cirques de La Réunion",
        "Piton des Neiges",
        "Piton de la Fournaise",
        # Cyclones spécifiques
        "Cyclone Belal",
        "Cyclone Batsirai",
        "Cyclone Dina",
        "Cyclone Firinga",
        "Cyclone Gamède",
        "Cyclone Hyacinthe",
        # Animaux dangereux
        "Requin bouledogue",
        "Requin tigre",
        "Attaque de requin",
        "Poisson-pierre",
        "Poisson-lion",
        "Cône (mollusque)",
        "Murène",
        "Scolopendre",
        # Médecine traditionnelle
        "Phytothérapie",
        "Tisaneur",
    ],
    "static_pages": [
        {
            "title": "Hébergements d'urgence à La Réunion",
            "file": "static/hebergements_urgence_reunion.html",
        },
    ],
    "metadata": {
        "name": "meshwiki_reunion",
        "title": "MeshWiki La Réunion",
        "description": "Articles Wikipedia sélectionnés pour l'usage hors-ligne à La Réunion",
        "language": "fra",
    },
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _strip_images(html_str: str) -> str:
    """Remove images, figures, infoboxes, and navboxes from HTML.

    Keeps text content, removes only visual/navigational clutter.
    """
    try:
        doc = lxml_html.fromstring(html_str)
    except Exception:
        return html_str

    # Remove <img> and <figure> tags
    for tag in ("img", "figure", "figcaption"):
        for el in doc.iter(tag):
            el.drop_tree()

    # Remove elements by CSS class
    remove_classes = {"thumb", "infobox", "navbox", "navbox-styles",
                      "metadata", "sistersitebox", "noprint",
                      "mw-empty-elt", "bandeau-portail", "catlinks"}
    for el in doc.iter():
        classes = set(el.get("class", "").split())
        if classes & remove_classes:
            el.drop_tree()

    # Remove style and link tags (CSS)
    for tag in ("style", "link"):
        for el in doc.iter(tag):
            el.drop_tree()

    from lxml import etree
    return etree.tostring(doc, encoding="unicode", method="html")


def _title_to_path(title: str) -> str:
    """Convert a Wikipedia title to a ZIM path."""
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="/:@!$&'()*+,;=-._~")
    return f"A/{encoded}"


def _safe_filename(title: str) -> str:
    """Return a SHA-256 hash prefix for safe staging filenames."""
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]


def _build_home_page(titles: list[str],
                     static_pages: list[dict] | None = None) -> str:
    """Build an HTML home page with an alphabetical index of articles.

    Args:
        titles: List of Wikipedia article titles.
        static_pages: List of dicts with 'title' keys for static pages.
    """
    # Static pages section
    # Accepts list of dicts ({"title": ...}) or tuples (title, path, html)
    static_section = ""
    if static_pages:
        static_links = []
        for page in static_pages:
            title = page["title"] if isinstance(page, dict) else page[0]
            path = _title_to_path(title)
            static_links.append(f'<li><a href="/{path}">{title}</a></li>')
        static_section = (
            "<h2>Pages locales</h2>\n<ul>\n"
            + "\n".join(static_links)
            + "\n</ul>\n"
        )

    # Articles section
    sorted_titles = sorted(titles)
    links = []
    for title in sorted_titles:
        path = _title_to_path(title)
        links.append(f'<li><a href="/{path}">{title}</a></li>')
    links_html = "\n".join(links)
    return f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><title>MeshWiki La Réunion</title></head>
<body>
<h1>MeshWiki La Réunion</h1>
<p>{len(titles)} articles Wikipedia</p>
{static_section}<h2>Articles Wikipedia</h2>
<ul>
{links_html}
</ul>
</body>
</html>"""


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 0:
        return "?"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}min"
    if minutes > 0:
        return f"{minutes}min{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Wikipedia API client
# ---------------------------------------------------------------------------

class WikipediaAPI:
    """Client for the Wikipedia API with rate limiting and retries."""

    def __init__(self, api_url: str, user_agent: str, request_delay: float = 0.5,
                 max_retries: int = 3, retry_delay: float = 5):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Wait if needed to respect the request delay."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, params: dict) -> dict:
        """Make a GET request with retries."""
        params["format"] = "json"
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                resp = self.session.get(self.api_url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Requête échouée (tentative %d/%d) : %s — retry dans %ds",
                        attempt + 1, self.max_retries, e, self.retry_delay,
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Requête échouée après %d tentatives : %s", self.max_retries, e)
                    raise

    def get_category_members(self, category: str, namespace: int = 0) -> list[str]:
        """List all members of a category (with pagination).

        Args:
            category: Category name (without 'Catégorie:' prefix).
            namespace: 0 for articles, 14 for subcategories.

        Returns list of page titles.
        """
        titles = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Catégorie:{category}",
            "cmnamespace": str(namespace),
            "cmlimit": "500",
        }
        while True:
            data = self._get(params)
            for member in data.get("query", {}).get("categorymembers", []):
                titles.append(member["title"])
            cont = data.get("continue", {}).get("cmcontinue")
            if not cont:
                break
            params["cmcontinue"] = cont
        return titles

    def get_category_articles_recursive(self, category: str, max_depth: int = 3,
                                        blacklist: list[str] | None = None,
                                        seen: set[str] | None = None,
                                        _depth: int = 0) -> set[str]:
        """Recursively collect article titles from a category and its subcategories.

        Args:
            category: Category name (without 'Catégorie:' prefix).
            max_depth: Maximum recursion depth.
            blacklist: Category names to skip (with 'Catégorie:' prefix).
            seen: Already-visited categories (to avoid cycles).

        Returns set of article titles.
        """
        if seen is None:
            seen = set()
        if blacklist is None:
            blacklist = []

        cat_full = f"Catégorie:{category}"
        if cat_full in seen or cat_full in blacklist:
            return set()
        seen.add(cat_full)

        articles = set()

        # Get articles (namespace 0)
        try:
            members = self.get_category_members(category, namespace=0)
            articles.update(members)
        except requests.RequestException:
            logger.warning("Impossible de lister la catégorie '%s' — résultats partiels", category)

        # Recurse into subcategories
        if _depth < max_depth:
            try:
                subcats = self.get_category_members(category, namespace=14)
            except requests.RequestException:
                subcats = []
            for subcat in subcats:
                # subcat is like "Catégorie:Foo"
                if subcat in blacklist or subcat in seen:
                    continue
                subcat_name = subcat.removeprefix("Catégorie:")
                sub_articles = self.get_category_articles_recursive(
                    subcat_name, max_depth, blacklist, seen, _depth + 1,
                )
                articles.update(sub_articles)

        return articles

    def get_article_html(self, title: str) -> str | None:
        """Fetch the parsed HTML for an article.

        Returns the HTML string, or None on failure.
        """
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "disableeditsection": "true",
            "disabletoc": "true",
        }
        try:
            data = self._get(params)
            return data["parse"]["text"]["*"]
        except (KeyError, TypeError):
            logger.warning("Pas de contenu HTML pour '%s'", title)
            return None
        except requests.RequestException:
            logger.warning("Échec de récupération de '%s'", title)
            return None


# ---------------------------------------------------------------------------
# libzim writer Item
# ---------------------------------------------------------------------------

class WikiArticleItem(Item):
    """A Wikipedia article item for libzim Creator."""

    def __init__(self, title: str, path: str, content: str):
        super().__init__()
        self.title = title
        self.path = path
        self.content = content

    def get_path(self):
        return self.path

    def get_title(self):
        return self.title

    def get_mimetype(self):
        return "text/html"

    def get_contentprovider(self):
        return StringProvider(self.content)

    def get_hints(self):
        return {Hint.FRONT_ARTICLE: True}


# ---------------------------------------------------------------------------
# Staging management
# ---------------------------------------------------------------------------

def _load_staging_index() -> dict[str, str]:
    """Load the staging index mapping title → hash."""
    if STAGING_INDEX.exists():
        try:
            return json.loads(STAGING_INDEX.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Index staging corrompu, reconstruction...")
    return {}


def _save_staging_index(index: dict[str, str]) -> None:
    """Save the staging index."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STAGING_INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STAGING_INDEX)


def _save_article_staging(title: str, html: str, index: dict[str, str]) -> None:
    """Save an article to staging."""
    h = _safe_filename(title)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    (STAGING_DIR / f"{h}.html").write_text(html, encoding="utf-8")
    (STAGING_DIR / f"{h}.json").write_text(
        json.dumps({"title": title}, ensure_ascii=False), encoding="utf-8",
    )
    index[title] = h


def _load_article_staging(title: str, index: dict[str, str]) -> str | None:
    """Load an article from staging."""
    h = index.get(title)
    if h is None:
        return None
    html_path = STAGING_DIR / f"{h}.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return None


def _clean_staging() -> None:
    """Remove the staging directory."""
    import shutil
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
        logger.info("Staging nettoyé")


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_zim(config_path: str | Path, resume: bool = False, dry_run: bool = False) -> dict:
    """Build a ZIM file from the given configuration.

    Args:
        config_path: Path to the YAML configuration file.
        resume: If True, reuse existing staging data.
        dry_run: If True, only run the discovery phase.

    Returns:
        Stats dict with keys: article_count, failed_count, output.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error("Fichier de configuration introuvable : %s", config_path)
        logger.error("Utilisez --init-config pour en générer un.")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wiki_cfg = cfg["wikipedia"]
    api = WikipediaAPI(
        api_url=wiki_cfg["api_url"],
        user_agent=wiki_cfg["user_agent"],
        request_delay=wiki_cfg.get("request_delay", 0.5),
        max_retries=wiki_cfg.get("max_retries", 3),
        retry_delay=wiki_cfg.get("retry_delay", 5),
    )

    cat_cfg = cfg["categories"]
    max_depth = cat_cfg.get("max_depth", 3)
    blacklist = cat_cfg.get("blacklist", [])

    # ------------------------------------------------------------------
    # Phase 1: Discovery
    # ------------------------------------------------------------------
    logger.info("=== Phase 1 : Découverte des articles ===")
    all_titles: set[str] = set()

    for cat_entry in cat_cfg.get("list", []):
        cat_name = cat_entry["name"]
        logger.info("Catégorie : %s", cat_name)
        articles = api.get_category_articles_recursive(cat_name, max_depth, blacklist)
        before = len(all_titles)
        all_titles.update(articles)
        logger.info("  → %d articles (%d nouveaux)", len(articles), len(all_titles) - before)

    # Add explicit articles
    explicit = cfg.get("articles", [])
    before = len(all_titles)
    all_titles.update(explicit)
    logger.info("Articles explicites : %d (%d nouveaux)", len(explicit), len(all_titles) - before)

    logger.info("Total articles uniques : %d", len(all_titles))

    if dry_run:
        sorted_titles = sorted(all_titles)
        for t in sorted_titles:
            print(f"  - {t}")
        return {"article_count": len(all_titles), "failed_count": 0, "output": None}

    # ------------------------------------------------------------------
    # Phase 2: Download
    # ------------------------------------------------------------------
    logger.info("=== Phase 2 : Téléchargement des articles ===")

    staging_index = _load_staging_index() if resume else {}
    if not resume:
        _clean_staging()
        staging_index = {}

    titles_list = sorted(all_titles)
    to_download = [t for t in titles_list if t not in staging_index]
    logger.info(
        "%d articles à télécharger (%d déjà en staging)",
        len(to_download), len(titles_list) - len(to_download),
    )

    progress = ProgressDisplay()
    progress.start()
    start_time = time.monotonic()
    failed_count = 0
    downloaded = 0

    try:
        for i, title in enumerate(to_download):
            progress.set_info(f"Téléchargement : {title[:60]}")

            html = api.get_article_html(title)
            if html is None:
                failed_count += 1
                logger.warning("Article ignoré (échec) : %s", title)
            else:
                cleaned = _strip_images(html)
                _save_article_staging(title, cleaned, staging_index)
                downloaded += 1

            # Checkpoint every 50 articles
            if (i + 1) % 50 == 0:
                _save_staging_index(staging_index)

            # Progress
            done = i + 1
            total = len(to_download)
            elapsed = time.monotonic() - start_time
            pct = done / total * 100
            if done > 0:
                remaining = elapsed / done * (total - done)
                eta_dur = _format_eta(remaining)
                eta_time = time.strftime("%H:%M", time.localtime(time.time() + remaining))
            else:
                eta_dur = "?"
                eta_time = "?"
            progress.set_progress(pct, downloaded, eta_dur, eta_time)
    except KeyboardInterrupt:
        logger.info("Interruption — sauvegarde du staging...")
    finally:
        _save_staging_index(staging_index)
        progress.stop()

    logger.info(
        "Téléchargement terminé : %d réussis, %d échoués",
        downloaded, failed_count,
    )

    # ------------------------------------------------------------------
    # Phase 3: Build ZIM
    # ------------------------------------------------------------------
    logger.info("=== Phase 3 : Construction du fichier ZIM ===")
    output_path = cfg["output"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    meta = cfg.get("metadata", {})

    # Collect articles from staging
    built_titles = []
    articles_data = []
    for title in titles_list:
        html = _load_article_staging(title, staging_index)
        if html is None:
            continue
        path = _title_to_path(title)
        articles_data.append((title, path, html))
        built_titles.append(title)

    logger.info("Articles à inclure dans le ZIM : %d", len(built_titles))

    # Load static pages
    static_pages = cfg.get("static_pages", [])
    static_data = []
    config_dir = config_path.parent
    for page in static_pages:
        file_path = config_dir / page["file"]
        if not file_path.exists():
            logger.warning("Page statique introuvable : %s", file_path)
            continue
        html = file_path.read_text(encoding="utf-8")
        path = _title_to_path(page["title"])
        static_data.append((page["title"], path, html))
        logger.info("Page statique : %s", page["title"])

    home_html = _build_home_page(built_titles, static_pages=static_data)

    with Creator(output_path).config_indexing(True, meta.get("language", "fra")) as creator:
        creator.set_mainpath("mainpage")

        # Add metadata
        for name, value in {
            "name": meta.get("name", "meshwiki"),
            "title": meta.get("title", "MeshWiki"),
            "description": meta.get("description", ""),
            "language": meta.get("language", "fra"),
            "publisher": "MeshWiki",
            "creator": "MeshWiki",
            "date": date.today().isoformat(),
        }.items():
            creator.add_metadata(name.title(), value)

        # Add home page
        home_item = WikiArticleItem("Accueil", "mainpage", home_html)
        creator.add_item(home_item)

        # Add static pages
        for title, path, html in static_data:
            item = WikiArticleItem(title, path, html)
            creator.add_item(item)

        # Add articles
        for title, path, html in articles_data:
            item = WikiArticleItem(title, path, html)
            creator.add_item(item)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        "ZIM créé : %s (%.1f Mo, %d articles)",
        output_path, file_size_mb, len(built_titles),
    )

    _clean_staging()

    return {
        "article_count": len(built_titles),
        "failed_count": failed_count,
        "output": output_path,
    }


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def init_config(path: Path = DEFAULT_CONFIG_PATH, force: bool = False) -> None:
    """Generate a default configuration file.

    Args:
        path: Destination file path.
        force: If True, overwrite without asking.
    """
    if path.exists() and not force:
        answer = input(f"Le fichier {path} existe déjà. Écraser ? [o/N] ").strip().lower()
        if answer not in ("o", "oui"):
            logger.info("Annulé.")
            return

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info("Configuration générée : %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Construit un fichier ZIM custom depuis Wikipedia"
    )
    parser.add_argument(
        "-c", "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Chemin du fichier de configuration (défaut: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Génère un fichier de configuration par défaut",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Découverte seule (pas de téléchargement ni construction)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend le téléchargement après une interruption",
    )
    args = parser.parse_args()

    if args.init_config:
        init_config(Path(args.config))
        return

    stats = build_zim(args.config, resume=args.resume, dry_run=args.dry_run)

    if stats["output"]:
        print(
            f"\n{stats['article_count']} articles — "
            f"{stats['failed_count']} échecs — "
            f"{stats['output']}"
        )
    else:
        print(f"\n{stats['article_count']} articles trouvés (dry-run)")


if __name__ == "__main__":
    main()
