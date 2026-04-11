#!/usr/bin/env python3
"""AI Content Factory — FastAPI backend."""

import os, sys, re, json, base64, io, zipfile, traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECTS_DIR = Path.home() / "smcc_projects"
PROJECTS_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCbEageR8y92Lj9niiwA74mgri7wEgRGwo")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Content Factory")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_slug(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:40] + "-" + datetime.now().strftime("%y%m%d")


def project_dir(slug: str) -> Path:
    d = PROJECTS_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Anthropic helper
# ---------------------------------------------------------------------------
import httpx

async def claude_chat(messages: list, system: str = "", model: str = "claude-sonnet-4-5", max_tokens: int = 4096) -> str:
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
        if r.status_code != 200:
            raise HTTPException(500, f"Claude API error: {r.text[:500]}")
        data = r.json()
        return data["content"][0]["text"]


# ---------------------------------------------------------------------------
# Gemini image helper
# ---------------------------------------------------------------------------
async def generate_gemini_image(prompt: str, images: list = None) -> Optional[str]:
    """Call Gemini to generate an image with optional reference images. Return base64 PNG or None.
    images: list of {"data": base64, "mime": "image/png"} dicts
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key={GEMINI_API_KEY}"

    parts = []
    for img in (images or []):
        if img.get("data"):
            parts.append({"inlineData": {"mimeType": img.get("mime", "image/png"), "data": img["data"]}})
    parts.append({"text": prompt})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    }
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(url, json=body)
            if r.status_code == 200:
                data = r.json()
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "inlineData" in part:
                            return part["inlineData"]["data"]
            else:
                print(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"Gemini error: {e}")
    return None


# ---------------------------------------------------------------------------
# SerpAPI + Jina research
# ---------------------------------------------------------------------------
async def research_keyword(keyword: str) -> dict:
    if SERPAPI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get("https://serpapi.com/search", params={
                    "q": keyword, "hl": "it", "gl": "it",
                    "api_key": SERPAPI_API_KEY, "num": 10
                })
                if r.status_code == 200:
                    data = r.json()
                    questions = [q.get("question", "") for q in data.get("related_questions", [])]
                    topics = [s.get("query", "") for s in data.get("related_searches", [])]
                    return {"questions": questions, "topics": topics}
        except Exception:
            pass
    # Jina fallback
    try:
        query = keyword.replace(" ", "+")
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"https://s.jina.ai/{query}",
                headers={"Accept": "application/json", "X-Return-Format": "json"}
            )
            if r.status_code == 200:
                data = r.json() if isinstance(r.text, str) and r.text.startswith("{") else {}
                return {"questions": [], "topics": [str(d) for d in data.get("data", [])[:5]]}
    except Exception:
        pass
    return {"questions": [], "topics": []}


# ---------------------------------------------------------------------------
# Jina URL fetch
# ---------------------------------------------------------------------------
async def fetch_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"https://r.jina.ai/{url}", headers={"Accept": "text/markdown"})
            return r.text[:15000] if r.is_success else ""
    except Exception:
        return ""


async def fetch_url_with_images(url: str) -> dict:
    """Fetch URL content via Jina and also extract image URLs + screenshot.

    Runs markdown fetch and screenshot fetch in parallel via asyncio.gather.
    """
    import asyncio

    content = ""
    images = []
    screenshot_b64 = ""

    async def _fetch_markdown(client):
        r = await client.get(f"https://r.jina.ai/{url}", headers={"Accept": "text/markdown"})
        if not r.is_success:
            return "", []
        text = r.text[:15000]
        imgs = []
        img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        for m in img_pattern.finditer(r.text):
            alt, img_url = m.group(1), m.group(2)
            if img_url and not img_url.startswith('data:'):
                imgs.append({"url": img_url, "alt": alt})
        src_pattern = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']')
        for m in src_pattern.finditer(r.text):
            img_url = m.group(1)
            if img_url and not img_url.startswith('data:') and not any(i['url'] == img_url for i in imgs):
                imgs.append({"url": img_url, "alt": ""})
        return text, imgs

    async def _fetch_screenshot(client):
        """Fetch screenshot URL from Jina, then download and base64-encode it."""
        rs = await client.get(f"https://r.jina.ai/{url}", headers={"x-respond-with": "screenshot"})
        if not rs.is_success:
            return ""
        scr_url = rs.text.strip()
        if not scr_url:
            return ""
        # Download the actual screenshot image
        try:
            r = await client.get(scr_url)
            if r.is_success and r.headers.get("content-type", "").startswith("image"):
                return base64.b64encode(r.content).decode()
        except Exception:
            pass
        return ""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            (content, images), screenshot_b64 = await asyncio.gather(
                _fetch_markdown(client),
                _fetch_screenshot(client),
            )
    except Exception:
        pass

    return {"content": content, "images": images[:20], "screenshot_b64": screenshot_b64}


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.post("/api/fetch-url")
async def fetch_url_endpoint(request: Request):
    """Fetch a URL via Jina and extract brand info with Claude."""
    data = await request.json()
    url = data.get("url", "").strip()
    if not url:
        raise HTTPException(400, "URL richiesta")

    # Fetch content + images + screenshot via Jina (parallel)
    fetched = await fetch_url_with_images(url)
    content = fetched["content"]
    site_images = fetched["images"]
    screenshot_b64 = fetched.get("screenshot_b64", "")
    if not content:
        raise HTTPException(400, "Impossibile leggere il contenuto della URL")

    system = """Sei un esperto di brand strategy. Analizza il contenuto di questa pagina web e estrai tutte le informazioni utili per un brand brief.
Rispondi SOLO con JSON valido, senza markdown fences.
Scrivi SEMPRE in italiano."""

    # Build message parts — include screenshot if available for visual color analysis
    user_parts = []
    if screenshot_b64:
        user_parts.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}})

    user_parts.append({"type": "text", "text": f"""Dalla seguente pagina web, estrai un brand brief strutturato.
{'Ho allegato uno screenshot del sito — usalo per estrarre i colori REALI della palette visiva.' if screenshot_b64 else ''}

Contenuto pagina:
{content[:12000]}

Restituisci JSON:
{{
  "niche": "descrizione della nicchia/prodotto/servizio",
  "usp": "unique selling proposition trovata o dedotta",
  "tone": "tono di voce del brand",
  "target": "target audience identificato",
  "keywords": ["keyword1", "keyword2", ...],
  "instagram": "@handle se trovato oppure stringa vuota",
  "notes": "altre info utili trovate (nome founder, brand name, ecc.)",
  "colors": [{{"hex": "#...", "role": "background|text|accent|secondary|dim"}}],
  "heading_font": "font heading usato o suggerito dal sito",
  "body_font": "font body usato o suggerito dal sito"
}}

IMPORTANTE per i colori:
- Restituisci 5-6 colori con i ruoli: background (sfondo principale), text (colore testo), accent (colore accento/primario), secondary (sfondo alternativo), dim (testo secondario/sottotitoli).
- {'Analizza lo SCREENSHOT allegato per estrarre i colori REALI visibili nel sito.' if screenshot_b64 else 'Deduci i colori dal branding descritto nel contenuto.'}
- Sii preciso con i valori hex. Se il sito ha sfondo scuro, il background deve essere scuro e il testo chiaro.
- Se non riesci a dedurli, restituisci un array vuoto."""})

    result = await claude_chat([{"role": "user", "content": user_parts}], system=system)
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"raw": result}

    # Attach extracted images
    parsed["site_images"] = site_images
    parsed["has_screenshot"] = bool(screenshot_b64)
    return parsed


@app.post("/api/parse-brief")
async def parse_brief(request: Request):
    """Parse an uploaded brand brief text/file and extract structured data."""
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        raise HTTPException(400, "Testo del brief richiesto")

    system = """Sei un esperto di brand strategy. Analizza questo brand brief e estrai informazioni strutturate.
Rispondi SOLO con JSON valido, senza markdown fences.
Scrivi SEMPRE in italiano."""

    prompt = f"""Dal seguente brand brief, estrai dati strutturati:

{text[:12000]}

Restituisci JSON:
{{
  "niche": "descrizione della nicchia/prodotto/servizio",
  "usp": "unique selling proposition",
  "tone": "tono di voce",
  "target": "target audience",
  "keywords": ["keyword1", "keyword2", ...],
  "instagram": "@handle se presente oppure stringa vuota",
  "notes": "altre info utili"
}}"""

    result = await claude_chat([{"role": "user", "content": prompt}], system=system)
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"raw": result}
    return parsed





@app.get("/api/projects")
async def list_projects():
    projects = []
    if PROJECTS_DIR.exists():
        for d in sorted(PROJECTS_DIR.iterdir()):
            if d.is_dir():
                brief = load_json(d / "brand-brief.json")
                cal = load_json(d / "calendar.json")
                posts_dir = d / "posts"
                post_count = len(list(posts_dir.iterdir())) if posts_dir.exists() else 0
                projects.append({
                    "slug": d.name,
                    "name": brief.get("niche", d.name)[:60] if brief else d.name,
                    "created_at": datetime.fromtimestamp(d.stat().st_ctime).isoformat(),
                    "post_count": len(cal.get("entries", [])) if cal else 0,
                    "completed_count": post_count,
                })
    return projects


@app.get("/api/project/{slug}")
async def get_project(slug: str):
    d = PROJECTS_DIR / slug
    if not d.exists():
        raise HTTPException(404, "Project not found")
    return {
        "brand_data": load_json(d / "brand-brief.json"),
        "pillars": load_json(d / "pillars.json"),
        "calendar": load_json(d / "calendar.json"),
        "posts": _list_posts(d),
    }


def _list_posts(project_path: Path) -> list:
    posts_dir = project_path / "posts"
    if not posts_dir.exists():
        return []
    result = []
    for p in sorted(posts_dir.iterdir()):
        if p.is_dir():
            post_data = load_json(p / "post.json")
            if post_data:
                result.append(post_data)
    return result


@app.post("/api/save-brand")
async def save_brand(request: Request):
    data = await request.json()
    niche = data.get("niche", "")
    if not niche:
        raise HTTPException(400, "La nicchia è obbligatoria")
    slug = data.get("slug") or make_slug(niche)
    d = project_dir(slug)
    data["slug"] = slug
    save_json(d / "brand-brief.json", data)
    return {"slug": slug, "status": "saved"}


@app.post("/api/analyze-brand")
async def analyze_brand(request: Request):
    data = await request.json()
    system = """Sei un esperto di brand strategy e social media marketing italiano.
Analizza i dati del brand e restituisci un'analisi strutturata in JSON.
Scrivi SEMPRE in italiano.
Non essere generico — sii specifico al settore e alla nicchia.
Rispondi SOLO con JSON valido, senza markdown fences."""

    prompt = f"""Analizza questi dati brand e restituisci un JSON con:
- brand_name: nome brand estratto/suggerito
- sector: settore codificato
- target_structured: {{age, situation, main_problem}}
- tone_synthesis: tono sintetizzato in 2-3 frasi
- suggested_keywords: array di 8-10 keyword suggerite

Dati brand:
Nicchia: {data.get('niche', '')}
USP: {data.get('usp', '')}
Tone of voice: {data.get('tone', '')}
Target: {data.get('target', '')}
Keywords: {', '.join(data.get('keywords', []))}
Note: {data.get('notes', '')}"""

    result = await claude_chat([{"role": "user", "content": prompt}], system=system)
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"raw": result}

    slug = data.get("slug", make_slug(data.get("niche", "brand")))
    d = project_dir(slug)
    data["analysis"] = parsed
    data["slug"] = slug
    save_json(d / "brand-brief.json", data)
    return {"slug": slug, "analysis": parsed}


@app.post("/api/research-keywords")
async def research_keywords(request: Request):
    data = await request.json()
    keywords = data.get("keywords", [])[:5]
    all_questions = []
    all_topics = []
    for kw in keywords:
        res = await research_keyword(kw)
        all_questions.extend(res.get("questions", []))
        all_topics.extend(res.get("topics", []))
    result = {
        "questions": list(set(all_questions))[:20],
        "trending_topics": list(set(all_topics))[:20],
        "competitor_angles": [],
    }
    return result


@app.post("/api/generate-pillars")
async def generate_pillars(request: Request):
    data = await request.json()
    brand_data = data.get("brand_data", {})
    research = data.get("research_results", {})

    system = """Sei un content strategist esperto di Instagram e TikTok italiano.
Generi pillar editoriali basati su ricerca reale delle keyword del settore.
Ogni pillar deve avere 15 angoli di contenuto diversi e concreti.
I pillar devono coprire: educativo, ispirazionale, social proof, prodotto, dietro le quinte, engagement.
Scrivi SEMPRE in italiano.
Rispondi SOLO con JSON valido, senza markdown fences."""

    prompt = f"""Genera 6 pillar editoriali per questo brand.

Brand: {brand_data.get('niche', '')}
USP: {brand_data.get('usp', '')}
Target: {brand_data.get('target', '')}
Tone: {brand_data.get('tone', '')}
Keywords: {', '.join(brand_data.get('keywords', []))}
Ricerca — domande frequenti: {'; '.join(research.get('questions', [])[:10])}
Ricerca — topic trending: {'; '.join(research.get('trending_topics', [])[:10])}

Restituisci JSON:
{{
  "pillars": [
    {{
      "id": 1,
      "name": "Nome Pillar",
      "subtitle": "Sottotitolo breve",
      "color": "#hex",
      "angles": ["Angolo 1", "Angolo 2", ...]
    }}
  ]
}}"""

    result = await claude_chat([{"role": "user", "content": prompt}], system=system)
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"pillars": [], "raw": result}

    slug = brand_data.get("slug", "")
    if slug:
        save_json(project_dir(slug) / "pillars.json", parsed)
    return parsed


@app.post("/api/generate-calendar")
async def generate_calendar(request: Request):
    data = await request.json()
    pillars = data.get("pillars", [])
    duration = data.get("duration", 7)
    frequency = data.get("frequency", {})
    mix_preset = data.get("mix_preset", "bilanciato")
    extra_notes = data.get("extra_notes", "")
    brand_data = data.get("brand_data", {})

    days_per_week = frequency.get("days_per_week", 5)
    posts_per_day = frequency.get("posts_per_day", 1)
    selected_days = frequency.get("selected_days", [1, 2, 3, 4, 5])

    pillar_names = [p.get("name", "") for p in pillars]

    system = """Sei un content strategist esperto di social media marketing italiano.
Genera un calendario editoriale dettagliato.
Scrivi SEMPRE in italiano.
Rispondi SOLO con JSON valido, senza markdown fences."""

    prompt = f"""Genera un calendario editoriale di {duration} giorni.

Pillar disponibili: {json.dumps(pillar_names, ensure_ascii=False)}
Frequenza: {days_per_week} giorni/settimana, {posts_per_day} post/giorno
Giorni attivi (0=dom, 1=lun...): {selected_days}
Mix contenuti: {mix_preset}
Note extra: {extra_notes}
Data inizio: {datetime.now().strftime('%Y-%m-%d')}

CATEGORIE CONTENUTI da usare:
- educativo: tutorial_how_to, errori_comuni, checklist, did_you_know, statistiche_shock, step_by_step
- storytelling: storia_trasformazione, dietro_le_quinte, day_in_the_life
- engagement: sondaggio, domanda_aperta, this_or_that, caption_contest
- social_proof: testimonianza, caso_studio, risultati_prima_dopo
- ispirazionale: quote_motivazionale, lezioni_di_vita, mindset_shift
- vendita: offerta_diretta, problema_soluzione, confronto_prodotto
- intrattenimento: meme_settore, trend_reinterpretato, hot_take
- autorità: previsioni_settore, analisi_trend, opinione_esperta
- community: user_generated, spotlight_follower, challenge

Per ogni entry usa formato Carosello o Post singolo.

Restituisci JSON:
{{
  "entries": [
    {{
      "date": "YYYY-MM-DD",
      "pillar": "Nome Pillar",
      "topic": "Argomento specifico",
      "format": "carousel|single",
      "content_type": "tipo_specifico",
      "content_category": "categoria",
      "objective": "obiettivo del post",
      "hook": "Hook iniziale",
      "cta": "Call to action"
    }}
  ]
}}"""

    result = await claude_chat([{"role": "user", "content": prompt}], system=system, max_tokens=8000)
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"entries": [], "raw": result}

    slug = brand_data.get("slug", "")
    if slug:
        save_json(project_dir(slug) / "calendar.json", parsed)
    return parsed


@app.post("/api/generate-copy")
async def generate_copy(request: Request):
    data = await request.json()
    entry = data.get("entry", {})
    brand_data = data.get("brand_data", {})
    pillars = data.get("pillars", [])

    fmt = entry.get("format", "carousel")
    if fmt == "mini":
        num_slides = 2
    elif fmt == "carousel":
        num_slides = 6
    else:
        num_slides = 1
    ig_handle = brand_data.get("instagram", "@brand")

    # Mini carousel special instructions
    mini_rules = ""
    if fmt == "mini":
        mini_rules = """
FORMATO MINI CAROSELLO (2 slide):
- SLIDE 1: Immagine impattante con headline breve e provocatoria che crea curiosità.
  Deve essere una DOMANDA, una PROVOCAZIONE o un DATO SHOCK che obbliga a swipare.
- SLIDE 2: La risposta/soluzione/spiegazione con CTA integrata.
  Testo più lungo (60-80 parole), con heading + body + CTA finale.
- L'obiettivo è creare un "cliffhanger" tra le due slide.
"""

    system = f"""Sei un copywriter esperto di social media italiano.
Scrivi copy per slide Instagram/TikTok seguendo queste regole:

SLIDE 1 (Hero): headline 20-30 parole max. Usa Title Case o TUTTO MAIUSCOLO.
  Formule: [Oggetto] + [Aggettivo Inaspettato]? | [N] [Soggetto] Che [Azione Sorprendente] |
  Come [Risultato] Senza [Barriera] | Il [Meccanismo] Che [ICP] Non Conosce
  MAI: "Scopri il segreto...", "Ti sei mai chiesto...", "X. Tu fai Y."

SLIDE 2..N-1 (Intermedie): 50-70 parole, testo educativo distribuito in blocchi.
  Heading 22-26px, grassetto su parole chiave.

ULTIMA SLIDE (CTA): 20-25 parole, heading + sub + bottone CTA.
{mini_rules}
REGOLE GENERALI:
- Scrivi come una persona intelligente che parla a un amico davanti a un caffè
- Frasi brevi alternate a frasi più lunghe — ritmo naturale
- MAI "Non X. Non Y. Non Z." — usa contrasto: "Eppure...", "A differenza di..."
- MAI domande retoriche a raffica
- MAI liste con emoji identiche ripetute
- Usa "tu", mai "voi"
- SEMPRE in italiano

Rispondi SOLO con JSON valido, senza markdown fences."""

    slide_schema = ""
    if fmt == "mini":
        slide_schema = """  "slides": [
    {"index": 1, "type": "hero", "heading": "...", "body": "..."},
    {"index": 2, "type": "cta", "heading": "...", "body": "..."}
  ]"""
    elif num_slides == 1:
        slide_schema = """  "slides": [
    {"index": 1, "type": "hero", "heading": "...", "body": "..."}
  ]"""
    else:
        slide_schema = f"""  "slides": [
    {{"index": 1, "type": "hero", "heading": "...", "body": "..."}},
    {{"index": 2, "type": "body", "heading": "...", "body": "..."}},
    ...
    {{"index": {num_slides}, "type": "cta", "heading": "...", "body": "..."}}
  ]"""

    prompt = f"""Genera il copy per questo post:

Topic: {entry.get('topic', '')}
Tipo: {entry.get('content_type', '')}
Categoria: {entry.get('content_category', '')}
Obiettivo: {entry.get('objective', '')}
Hook: {entry.get('hook', '')}
CTA: {entry.get('cta', '')}
Formato: {fmt} {'(mini carosello a 2 slide — cliffhanger + risposta)' if fmt == 'mini' else ''}
Numero slide: {num_slides}

Brand:
Nicchia: {brand_data.get('niche', '')}
USP: {brand_data.get('usp', '')}
Tono: {brand_data.get('tone', '')}
Target: {brand_data.get('target', '')}
Instagram: {ig_handle}

Restituisci JSON:
{{
{slide_schema},
  "caption": "caption IG con hashtag",
  "reel_script": "script narrativo reel max 200 parole",
  "layout": "suggerimento layout"
}}"""

    result = await claude_chat(
        [{"role": "user", "content": prompt}],
        system=system,
        model="claude-haiku-4-5-20251001",
        max_tokens=3000,
    )
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"slides": [], "raw": result}

    # Save to project
    slug = brand_data.get("slug", "")
    if slug and entry.get("date") and entry.get("topic"):
        topic_slug = re.sub(r"[^a-z0-9]+", "-", entry["topic"].lower())[:30].strip("-")
        post_dir = project_dir(slug) / "posts" / f"{entry['date']}-{topic_slug}"
        post_dir.mkdir(parents=True, exist_ok=True)
        post_data = {**entry, "copy": parsed, "status": "copy_generated"}
        save_json(post_dir / "post.json", post_data)
        if parsed.get("caption"):
            (post_dir / "caption.txt").write_text(parsed["caption"], encoding="utf-8")
        if parsed.get("reel_script"):
            (post_dir / "script_reel.txt").write_text(parsed["reel_script"], encoding="utf-8")

    return parsed


@app.post("/api/generate-reel-script")
async def generate_reel_script(request: Request):
    data = await request.json()
    slides = data.get("slides", [])
    caption = data.get("caption", "")
    brand_data = data.get("brand_data", {})

    system = """Sei un copywriter esperto di social media italiano.
Scrivi script narrativi per Reel Instagram/TikTok.
Scrivi come una storia — inizio, sviluppo, conclusione. Tono del brand.
Max 200 parole per un reel da 60 secondi.
Rispondi SOLO con JSON: {"script": "testo..."}"""

    slides_text = "\n".join([f"Slide {s['index']}: {s.get('heading', '')} — {s.get('body', '')}" for s in slides])
    prompt = f"""Converti questo contenuto carousel in uno script reel narrativo:

{slides_text}

Caption: {caption}
Tono brand: {brand_data.get('tone', 'professionale')}
Target: {brand_data.get('target', '')}"""

    result = await claude_chat(
        [{"role": "user", "content": prompt}],
        system=system,
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
    )
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"script": result}
    return parsed


@app.post("/api/generate-images")
async def generate_images(request: Request):
    data = await request.json()
    slides = data.get("slides", [])
    brand_data = data.get("brand_data", {})
    palette = data.get("palette", ["#FF4500", "#1a1a1a", "#ffffff"])
    logo_b64 = data.get("logo_b64")          # brand logo
    author_b64 = data.get("author_b64")      # author photo
    author_name = data.get("author_name", "")
    author_title = data.get("author_title", "")
    custom_instructions = data.get("custom_instructions", "")
    total_slides = data.get("total_slides", len(slides))
    visual_elements = data.get("visual_elements", {})

    palette_desc = ", ".join(palette[:5])
    mood = brand_data.get("tone", "professional")
    sector = brand_data.get("niche", "lifestyle")
    brand_name = brand_data.get("notes", "").split(",")[0][:40] if brand_data.get("notes") else ""

    # Build visual elements instruction string
    ve_parts = []
    if visual_elements.get("doodles"):
        ve_parts.append("Add hand-drawn style arrows, underlines, circles and doodle elements to highlight key points and create visual flow.")
    if visual_elements.get("icons"):
        ve_parts.append("Include thematic flat icons relevant to the content topic (e.g. lightbulb, target, chart, checkmark).")
    if visual_elements.get("stickers"):
        ve_parts.append("Add fun sticker-like visual elements — relevant and sometimes ironic/playful — to make the slide more engaging and shareable.")
    if visual_elements.get("swipe_visual"):
        ve_parts.append("Add a clear swipe indicator on the right edge (subtle chevron, gradient fade, or visual cue) to encourage swiping to next slide.")
    if visual_elements.get("shapes"):
        ve_parts.append("Use geometric shapes, patterns, and abstract forms as decorative background elements to fill empty space.")
    visual_instructions = "\n".join(ve_parts)
    hero_image_requested = visual_elements.get("hero_image", False)

    results = []
    for slide in slides:
        st = slide.get("type", "body")
        ct = slide.get("content_type", "")
        heading = slide.get("heading", "")
        body_text = slide.get("body", "")
        slide_idx = slide.get("index", 1)
        ref_b64 = slide.get("reference_png")

        # Build reference images list
        ref_images = []
        if ref_b64:
            ref_images.append({"data": ref_b64, "mime": "image/png"})
        if logo_b64:
            # Strip data URL prefix if present
            logo_data = logo_b64.split(",")[1] if "," in logo_b64 else logo_b64
            ref_images.append({"data": logo_data, "mime": "image/png"})
        if author_b64:
            author_data = author_b64.split(",")[1] if "," in author_b64 else author_b64
            ref_images.append({"data": author_data, "mime": "image/png"})

        # Custom instructions suffix
        extra = f"\n\nAdditional creative direction: {custom_instructions}" if custom_instructions else ""

        # --- Build rich prompt based on slide type and content category ---
        if ref_b64:
            # REFERENCE-BASED: enhance the HTML preview
            author_section = ""
            if author_b64 and author_name:
                author_section = (
                    f"Include the author's profile photo (provided as reference image) in a circular frame "
                    f"at the bottom-left with the name \"{author_name}\" and title \"{author_title}\" next to it. "
                )
            logo_section = ""
            if logo_b64:
                logo_section = "Place the brand logo (provided as reference image) in the top-left corner, small and elegant. "

            # Swipe arrow for carousel slides (not first, not last CTA)
            arrow_section = ""
            if total_slides > 1 and st != 'cta' and slide_idx > 1:
                arrow_section = "Add a subtle swipe-right arrow indicator on the right edge to invite swiping. "
            elif total_slides > 1 and slide_idx == 1:
                arrow_section = "Add a subtle swipe-right arrow indicator on the right edge to invite swiping to the next slide. "

            # Content-type specific visual enhancements
            visual_style = ""
            if ct in ["checklist", "step_by_step"]:
                visual_style = "Use checkmark icons (✓) or numbered circles next to each point. Make it look like an actionable checklist. "
            elif ct in ["tutorial_how_to", "errori_comuni"]:
                visual_style = "Use icon-based visual cues (warning icons, lightbulbs, arrows) to make the educational content scannable. "
            elif ct in ["did_you_know", "statistiche_shock"]:
                visual_style = "Incorporate bold data visualization elements: large numbers, percentage circles, bar charts, or infographic icons. Make the data POP visually. "
            elif ct in ["quote_motivazionale", "lezioni_di_vita"]:
                visual_style = 'Add elegant quotation marks ("") as decorative elements. Use a premium editorial feel. '
            elif ct in ["mappa_mentale", "framework"]:
                visual_style = "Create a visual mind-map or flowchart structure with connecting lines and nodes. Make it look like an infographic. "

            # Hero image overlay for slide 1
            hero_overlay = ""
            if hero_image_requested and slide_idx == 1:
                hero_overlay = (
                    "CRITICAL FOR SLIDE 1: Add a prominent, attention-grabbing visual element — "
                    "this could be a relevant icon, emoji-style illustration, a bold graphic symbol, "
                    "or a striking visual metaphor related to the topic. "
                    "This hero visual should occupy ~30% of the slide and be immediately eye-catching. "
                )

            prompt = (
                f"You are a world-class Instagram content designer. "
                f"I'm providing a reference layout image for slide {slide_idx}/{total_slides} of an Instagram carousel. "
                f"RECREATE this slide as a stunning, scroll-stopping, professionally designed Instagram image. "
                f"\n\nDESIGN REQUIREMENTS:\n"
                f"- Color palette: {palette_desc}\n"
                f"- Brand sector: {sector}\n"
                f"- Mood/tone: {mood}\n"
                f"- Format: vertical 4:5 (1080x1350px)\n"
                f"\nTEXT CONTENT (must be EXACTLY this, in Italian):\n"
                f"- Heading: \"{heading[:100]}\"\n"
                f"- Body: \"{body_text[:200]}\"\n"
                f"\nVISUAL DESIGN DIRECTION:\n"
                f"- Create a PREMIUM, ENGAGING social media graphic — NOT a plain text-on-background slide.\n"
                f"- Add visual depth: subtle gradients, geometric shapes, decorative lines, accent elements.\n"
                f"- Use bold typography hierarchy: heading large and impactful, body text clean and readable.\n"
                f"- Add decorative graphic elements that support the content (icons, shapes, dividers, highlights).\n"
                f"- The design should make people STOP scrolling and want to save/share.\n"
                f"- FILL EMPTY SPACE: Do not leave large blank areas — use decorative elements, subtle patterns, or visual accents to fill the composition.\n"
                f"{visual_style}"
                f"{hero_overlay}"
                f"{logo_section}"
                f"{author_section}"
                f"{arrow_section}"
                f"{visual_instructions + chr(10) if visual_instructions else ''}"
                f"\nIMPORTANT: Output ONE finished, ready-to-post image. No mockups, no phone frames."
                f"{extra}"
            )
        else:
            # NO REFERENCE — generate from scratch with rich prompts
            if st == "hero":
                hero_visual = ""
                if hero_image_requested:
                    hero_visual = (
                        "Add a PROMINENT, attention-grabbing visual element — "
                        "a relevant icon, bold graphic illustration, striking visual metaphor, "
                        "or eye-catching image related to the topic. Should occupy ~30% of the slide. "
                    )
                prompt = (
                    f"Create a stunning Instagram carousel COVER slide for a {sector} brand. "
                    f"Bold, scroll-stopping design with headline: \"{heading[:80]}\" "
                    f"and subtitle: \"{body_text[:120]}\" "
                    f"Color palette: {palette_desc}. Mood: {mood}. "
                    f"Premium graphic design with decorative elements, shapes, gradients. "
                    f"FILL ALL EMPTY SPACE with visual accents, patterns, or decorative elements. "
                    f"{hero_visual}"
                    f"{'Include the brand logo from the reference image in top-left corner. ' if logo_b64 else ''}"
                    f"{'Include the author photo from reference as a circular profile picture. ' if author_b64 else ''}"
                    f"{'Add a swipe arrow on the right edge. ' if total_slides > 1 else ''}"
                    f"{visual_instructions + ' ' if visual_instructions else ''}"
                    f"Vertical 4:5 format, 1080x1350px, ready to post."
                    f"{extra}"
                )
            elif ct in ["checklist", "step_by_step"]:
                prompt = (
                    f"Create an Instagram CHECKLIST/INFOGRAPHIC slide. "
                    f"Title: \"{heading[:80]}\". Content: \"{body_text[:200]}\" "
                    f"Design as an actionable checklist with checkmark icons, numbered steps, and visual hierarchy. "
                    f"Color palette: {palette_desc}. Sector: {sector}. "
                    f"Clean, professional infographic style that people want to SAVE and SHARE. "
                    f"FILL ALL EMPTY SPACE with visual accents. "
                    f"{visual_instructions + ' ' if visual_instructions else ''}"
                    f"Vertical 4:5 format, 1080x1350px."
                    f"{extra}"
                )
            elif ct in ["did_you_know", "statistiche_shock"]:
                prompt = (
                    f"Create an Instagram DATA/INFOGRAPHIC slide with bold statistics. "
                    f"Title: \"{heading[:80]}\". Data: \"{body_text[:200]}\" "
                    f"Include large numbers, percentage circles, bar charts, or data visualization icons. "
                    f"Make the numbers POP with oversized bold typography. "
                    f"FILL ALL EMPTY SPACE with visual accents. "
                    f"Color palette: {palette_desc}. Sector: {sector}. "
                    f"{visual_instructions + ' ' if visual_instructions else ''}"
                    f"Vertical 4:5 format, 1080x1350px."
                    f"{extra}"
                )
            elif ct in ["mappa_mentale", "framework"]:
                prompt = (
                    f"Create an Instagram MIND MAP / FRAMEWORK slide. "
                    f"Title: \"{heading[:80]}\". Content: \"{body_text[:200]}\" "
                    f"Design as a visual mind-map with connected nodes, flowchart arrows, and structured layout. "
                    f"Color palette: {palette_desc}. Sector: {sector}. "
                    f"Infographic style, clean and educational. FILL ALL EMPTY SPACE. "
                    f"{visual_instructions + ' ' if visual_instructions else ''}"
                    f"Vertical 4:5 format, 1080x1350px."
                    f"{extra}"
                )
            elif st == "cta":
                prompt = (
                    f"Create a CALL-TO-ACTION Instagram slide. "
                    f"Title: \"{heading[:80]}\". Message: \"{body_text[:150]}\" "
                    f"Include a prominent CTA button, follow prompt, and engagement elements. "
                    f"{'Include the author photo from reference as a circular profile picture with name and title. ' if author_b64 else ''}"
                    f"Color palette: {palette_desc}. Mood: {mood}. Sector: {sector}. "
                    f"FILL ALL EMPTY SPACE with decorative elements. "
                    f"{visual_instructions + ' ' if visual_instructions else ''}"
                    f"Vertical 4:5 format, 1080x1350px."
                    f"{extra}"
                )
            else:
                prompt = (
                    f"Create a professional Instagram carousel BODY slide. "
                    f"Title: \"{heading[:80]}\". Content: \"{body_text[:200]}\" "
                    f"Design with visual hierarchy: bold heading, clear body text, decorative elements. "
                    f"Add subtle graphics, icons, accent shapes that support the content. "
                    f"FILL ALL EMPTY SPACE — no large blank areas. Use decorative elements, patterns, visual accents. "
                    f"Color palette: {palette_desc}. Mood: {mood}. Sector: {sector}. "
                    f"{'Add swipe arrow indicator on right edge. ' if total_slides > 1 and st != 'cta' else ''}"
                    f"{visual_instructions + ' ' if visual_instructions else ''}"
                    f"Vertical 4:5 format, 1080x1350px."
                    f"{extra}"
                )

        img_b64 = await generate_gemini_image(prompt, images=ref_images)
        results.append({"index": slide.get("index", 0), "img_b64": img_b64, "prompt": prompt})

    return {"slides_with_images": results}


@app.post("/api/extract-palette")
async def extract_palette(request: Request):
    data = await request.json()
    image_b64 = data.get("image", "")

    system = """Analizza l'immagine e restituisci i 5-6 colori dominanti con ruoli assegnati.
Rispondi SOLO con JSON valido: {"colors": [{"hex": "#...", "role": "background|text|accent|secondary|dim"}]}
Ruoli: background = sfondo principale, text = colore testo heading, accent = colore accento/primario, secondary = sfondo alternativo, dim = testo secondario.
Se l'immagine ha sfondo scuro → background scuro, text chiaro. Se sfondo chiaro → background chiaro, text scuro."""

    result = await claude_chat(
        [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
            {"type": "text", "text": "Estrai la palette colori dominante da questa immagine con i ruoli corretti (background, text, accent, secondary, dim). Rispondi solo con JSON."}
        ]}],
        system=system,
    )
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"colors": []}
    return parsed


def _brand_brief_to_md(data: dict) -> str:
    """Convert brand brief JSON to a readable Markdown document."""
    lines = ["# Brand Brief\n"]
    if data.get("niche"):
        lines.append(f"## Nicchia / Prodotto\n{data['niche']}\n")
    if data.get("usp"):
        lines.append(f"## USP — Unique Selling Proposition\n{data['usp']}\n")
    if data.get("tone"):
        lines.append(f"## Tone of Voice\n{data['tone']}\n")
    if data.get("target"):
        lines.append(f"## Target Audience\n{data['target']}\n")
    if data.get("instagram"):
        lines.append(f"## Profilo Instagram\n{data['instagram']}\n")
    if data.get("keywords"):
        lines.append("## Keywords\n" + ", ".join(data["keywords"]) + "\n")
    if data.get("colors"):
        lines.append("## Color Palette\n" + " | ".join(data["colors"]) + "\n")
    if data.get("heading_font") or data.get("body_font"):
        lines.append(f"## Tipografia\n- Heading: {data.get('heading_font', 'N/A')}\n- Body: {data.get('body_font', 'N/A')}\n")
    if data.get("notes"):
        lines.append(f"## Note Aggiuntive\n{data['notes']}\n")
    if data.get("analysis"):
        a = data["analysis"]
        if a.get("brand_name"):
            lines.append(f"## Brand Name\n{a['brand_name']}\n")
        if a.get("sector"):
            lines.append(f"## Settore\n{a['sector']}\n")
        if a.get("tone_synthesis"):
            lines.append(f"## Sintesi Tono\n{a['tone_synthesis']}\n")
    return "\n".join(lines)


@app.post("/api/save-slides")
async def save_slides(request: Request):
    """Save slide HTML and images to disk for a post."""
    data = await request.json()
    slug = data.get("slug", "")
    entry = data.get("entry", {})
    slides = data.get("slides", [])
    caption = data.get("caption", "")
    reel_script = data.get("reel_script", "")

    if not slug or not entry.get("topic"):
        raise HTTPException(400, "slug and entry required")

    topic_slug = re.sub(r"[^a-z0-9]+", "-", entry.get("topic", "post").lower())[:30].strip("-")
    post_dir = project_dir(slug) / "posts" / f"{entry.get('date', 'nodate')}-{topic_slug}"
    slides_dir = post_dir / "slides"
    images_dir = post_dir / "images"
    slides_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for slide in slides:
        idx = slide.get("index", 0)
        img_b64 = slide.get("img_b64")
        preview_b64 = slide.get("preview_png")

        # Save Gemini-generated image as PNG (priority)
        if img_b64:
            img_data = base64.b64decode(img_b64)
            (images_dir / f"slide_{idx:02d}.png").write_bytes(img_data)
        elif preview_b64:
            # Fallback: save html2canvas preview PNG
            img_data = base64.b64decode(preview_b64)
            (images_dir / f"slide_{idx:02d}.png").write_bytes(img_data)

    # Save caption & script
    if caption:
        (post_dir / "caption.txt").write_text(caption, encoding="utf-8")
    if reel_script:
        (post_dir / "script_reel.txt").write_text(reel_script, encoding="utf-8")

    # Save post.json
    post_data = {**entry, "slides": [{"index": s.get("index"), "type": s.get("type"), "heading": s.get("heading"), "body": s.get("body")} for s in slides], "status": "completed"}
    save_json(post_dir / "post.json", post_data)

    return {"post_dir": str(post_dir), "slides_saved": len(slides)}


@app.post("/api/export-zip")
async def export_zip(request: Request):
    data = await request.json()
    slug = data.get("slug", "")
    post_slug = data.get("post_slug", "")

    if not slug:
        raise HTTPException(400, "slug required")

    project_root = PROJECTS_DIR / slug
    d = project_root / "posts"
    if not d.exists():
        raise HTTPException(404, "No posts found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add brand brief as markdown
        brief_data = load_json(project_root / "brand-brief.json")
        if brief_data:
            md = _brand_brief_to_md(brief_data)
            zf.writestr("brand-brief.md", md)

        if post_slug:
            dirs = [d / post_slug] if (d / post_slug).exists() else []
        else:
            dirs = sorted(d.iterdir())

        for post_dir in dirs:
            if not post_dir.is_dir():
                continue
            prefix = post_dir.name + "/"
            # Add slide PNGs in a "slides/" subfolder
            images_dir = post_dir / "images"
            if images_dir.exists():
                for f in sorted(images_dir.iterdir()):
                    if f.suffix.lower() == ".png":
                        zf.write(f, prefix + "slides/" + f.name)
            # Add caption
            for fname in ["caption.txt"]:
                fp = post_dir / fname
                if fp.exists():
                    zf.write(fp, prefix + fname)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={slug}-export.zip"},
    )


@app.post("/api/import-calendar")
async def import_calendar(file: UploadFile = File(...)):
    import csv
    content = (await file.read()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    entries = []
    for row in reader:
        entries.append({
            "date": row.get("date", ""),
            "pillar": row.get("pillar", ""),
            "topic": row.get("topic", ""),
            "format": row.get("format", "carousel"),
            "content_type": row.get("content_type", ""),
            "content_category": row.get("content_category", ""),
            "objective": row.get("objective", ""),
            "hook": row.get("hook", ""),
            "cta": row.get("cta", ""),
            "status": row.get("status", "pending"),
        })
    return {"entries": entries}


@app.post("/api/upload-asset")
async def upload_asset(slug: str = Form(...), asset_type: str = Form("reference"), file: UploadFile = File(...)):
    d = project_dir(slug) / "assets"
    if asset_type == "reference":
        d = d / "ref_images"
    d.mkdir(parents=True, exist_ok=True)
    fpath = d / file.filename
    fpath.write_bytes(await file.read())
    return {"path": str(fpath), "filename": file.filename}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 AI Content Factory running at http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
