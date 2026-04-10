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
async def generate_gemini_image(prompt: str) -> Optional[str]:
    """Call Gemini 2.0 Flash to generate an image, return base64 PNG or None."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "image/png"}
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=body)
            if r.status_code == 200:
                data = r.json()
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "inlineData" in part:
                            return part["inlineData"]["data"]
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


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
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

    num_slides = 6 if entry.get("format") == "carousel" else 1
    ig_handle = brand_data.get("instagram", "@brand")

    system = """Sei un copywriter esperto di social media italiano.
Scrivi copy per slide Instagram/TikTok seguendo queste regole:

SLIDE 1 (Hero): headline 20-30 parole max. Usa Title Case o TUTTO MAIUSCOLO.
  Formule: [Oggetto] + [Aggettivo Inaspettato]? | [N] [Soggetto] Che [Azione Sorprendente] |
  Come [Risultato] Senza [Barriera] | Il [Meccanismo] Che [ICP] Non Conosce
  MAI: "Scopri il segreto...", "Ti sei mai chiesto...", "X. Tu fai Y."

SLIDE 2..N-1 (Intermedie): 50-70 parole, testo educativo distribuito in blocchi.
  Heading 22-26px, grassetto su parole chiave.

ULTIMA SLIDE (CTA): 20-25 parole, heading + sub + bottone CTA.

REGOLE GENERALI:
- Scrivi come una persona intelligente che parla a un amico davanti a un caffè
- Frasi brevi alternate a frasi più lunghe — ritmo naturale
- MAI "Non X. Non Y. Non Z." — usa contrasto: "Eppure...", "A differenza di..."
- MAI domande retoriche a raffica
- MAI liste con emoji identiche ripetute
- Usa "tu", mai "voi"
- SEMPRE in italiano

Rispondi SOLO con JSON valido, senza markdown fences."""

    prompt = f"""Genera il copy per questo post:

Topic: {entry.get('topic', '')}
Tipo: {entry.get('content_type', '')}
Categoria: {entry.get('content_category', '')}
Obiettivo: {entry.get('objective', '')}
Hook: {entry.get('hook', '')}
CTA: {entry.get('cta', '')}
Formato: {entry.get('format', 'carousel')}
Numero slide: {num_slides}

Brand:
Nicchia: {brand_data.get('niche', '')}
USP: {brand_data.get('usp', '')}
Tono: {brand_data.get('tone', '')}
Target: {brand_data.get('target', '')}
Instagram: {ig_handle}

Restituisci JSON:
{{
  "slides": [
    {{"index": 1, "type": "hero", "heading": "...", "body": "..."}},
    {{"index": 2, "type": "body", "heading": "...", "body": "..."}},
    ...
    {{"index": {num_slides}, "type": "cta", "heading": "...", "body": "..."}}
  ],
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

    palette_desc = ", ".join(palette[:4])
    mood = brand_data.get("tone", "professional")
    sector = brand_data.get("niche", "lifestyle")

    results = []
    for slide in slides:
        st = slide.get("type", "body")
        ct = slide.get("content_type", "")
        heading = slide.get("heading", "")

        if st == "hero":
            prompt = f"Cinematic background for {sector} brand, {mood} mood, {palette_desc} color palette, no people, no text, no logos, vertical 4:5 format, photorealistic, premium quality"
        elif ct in ["tutorial_how_to", "errori_comuni", "checklist", "step_by_step"]:
            subject = heading[:50] if heading else sector
            prompt = f"Hand-drawn doodle illustration of {subject}, black ink on white background, sketch style, square composition, no text, clean minimal"
        elif ct in ["did_you_know", "statistiche_shock"]:
            prompt = f"Abstract data visualization concept, {palette_desc} colors, minimal flat design, no text, square composition, professional"
        elif ct in ["storia_trasformazione", "case_study"]:
            prompt = f"Lifestyle photo concept, {mood} atmosphere, warm light, no text, vertical 4:5, photorealistic"
        elif ct in ["quote_motivazionale", "lezioni_di_vita"]:
            prompt = f"Abstract minimalist background, {palette_desc}, soft gradient, no text, no people, vertical 4:5, artistic"
        else:
            prompt = f"Professional lifestyle image for {sector}, {mood} mood, {palette_desc} palette, no text, vertical 4:5"

        img_b64 = await generate_gemini_image(prompt)
        results.append({"index": slide.get("index", 0), "img_b64": img_b64, "prompt": prompt})

    return {"slides_with_images": results}


@app.post("/api/extract-palette")
async def extract_palette(request: Request):
    data = await request.json()
    image_b64 = data.get("image", "")

    system = "Analizza l'immagine e restituisci i 5 colori dominanti come JSON: {\"colors\": [{\"hex\": \"#...\", \"name\": \"nome colore\", \"role\": \"primary|secondary|accent|background|text\"}]}"

    result = await claude_chat(
        [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
            {"type": "text", "text": "Estrai la palette colori dominante da questa immagine. Rispondi solo con JSON."}
        ]}],
        system=system,
    )
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"colors": []}
    return parsed


@app.post("/api/export-zip")
async def export_zip(request: Request):
    data = await request.json()
    slug = data.get("slug", "")
    post_slug = data.get("post_slug", "")

    if not slug:
        raise HTTPException(400, "slug required")

    d = PROJECTS_DIR / slug / "posts"
    if not d.exists():
        raise HTTPException(404, "No posts found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if post_slug:
            dirs = [d / post_slug] if (d / post_slug).exists() else []
        else:
            dirs = sorted(d.iterdir())

        for post_dir in dirs:
            if not post_dir.is_dir():
                continue
            prefix = post_dir.name + "/"
            # Add slides
            slides_dir = post_dir / "slides"
            if slides_dir.exists():
                for f in sorted(slides_dir.iterdir()):
                    zf.write(f, prefix + "slides/" + f.name)
            # Add export PNGs
            export_dir = post_dir / "export"
            if export_dir.exists():
                for f in sorted(export_dir.iterdir()):
                    zf.write(f, prefix + "slides/" + f.name)
            # Add caption & script
            for fname in ["caption.txt", "script_reel.txt"]:
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
