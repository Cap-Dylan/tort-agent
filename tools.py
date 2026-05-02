"""
tools.py — Tort Agent Tool Registry
====================================
All callable tools for the Tort Agent. Designed for tower deployment
(RTX 4090 running `qwen3.6:35b-a3b` via Ollama), but model and host
are env-driven so it'll run anywhere Ollama runs.

Tools:
  1. morning_brief         → weather + canvas due dates + class refresher
  2. export_apple_notes    → Apple Notes folder → PDFs in iCloud staging
  3. convert_notes         → OCR handwritten iCloud notes → .md in vault
  4. develop_concepts      → extract atomic concepts from notes, link them
  5. weekly_summary        → per-class weekly summary of new concepts
  6. list_directory        → list files/folders at a path

Setup (env vars in .env — see .env.example):
  CANVAS_API_TOKEN        → your Canvas LMS API token
  CANVAS_BASE_URL         → e.g. https://your-school.instructure.com
  OBSIDIAN_VAULT          → path to your Obsidian vault
  ICLOUD_NOTES_DIR        → iCloud folder with handwritten note PDFs
  WEATHER_LOCATION        → city for the morning weather block
  OLLAMA_HOST             → Ollama base URL
  OLLAMA_MODEL            → model tag (e.g. qwen3.6:35b-a3b)

Dependencies (see requirements.txt):
  pip install -r requirements.txt

macOS OCR (for convert_notes):
  ocrmac wraps Apple Vision under the hood — great for handwriting.
  pymupdf rasterizes PDF pages → images so ocrmac can OCR them.
  These tools (and export_apple_notes) are macOS-only.
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

CANVAS_API_TOKEN = os.getenv("CANVAS_API_TOKEN", "")
CANVAS_BASE_URL  = os.getenv("CANVAS_BASE_URL", "https://your-school.instructure.com")
OBSIDIAN_VAULT   = os.getenv("OBSIDIAN_VAULT", str(Path.home() / "Obsidian" / "vault"))
ICLOUD_NOTES_DIR = os.getenv(
    "ICLOUD_NOTES_DIR",
    str(Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/HandwrittenNotes")
)
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "qwen3.6:35b-a3b")
WEATHER_LOCATION = os.getenv("WEATHER_LOCATION", "Denver")

# Canvas API headers — reused across calls
CANVAS_HEADERS = {
    "Authorization": f"Bearer {CANVAS_API_TOKEN}",
    "Accept": "application/json",
}


# ──────────────────────────────────────────────
# TOOL REGISTRY  (schema the agent reads)
# ──────────────────────────────────────────────
# Each entry tells the model what the tool does
# and what parameters it accepts. Keep descriptions
# short — the 9b context window isn't huge.

TOOL_REGISTRY = [
    {
        "name": "morning_brief",
        "description": (
            "Get the full morning briefing: today's weather for the configured "
            "location, Canvas assignments due today and tomorrow, and a short "
            "refresher paragraph from recent notes for each upcoming class."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "export_apple_notes",
        "description": (
            "Export handwritten Apple Notes from a folder as PDFs "
            "into the iCloud staging directory so convert_notes can OCR them. "
            "Run this BEFORE convert_notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Apple Notes folder name, e.g. 'ECON204' or 'CO300'.",
                },
            },
            "required": ["folder"],
        },
    },
    {
        "name": "convert_notes",
        "description": (
            "OCR handwritten note PDFs from iCloud and save as lecture .md files "
            "in 03-courses/{course}/ inside the Obsidian vault."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "course": {
                    "type": "string",
                    "description": "Course folder name in the vault, e.g. 'ECON' or 'CO300'.",
                },
            },
            "required": ["course"],
        },
    },
    {
        "name": "develop_concepts",
        "description": (
            "Read a lecture note from 03-courses/, extract atomic concepts, "
            "and create individual .md files in 01-concepts/ with proper "
            "frontmatter, backlinks, and source tags per CLAUDE.md conventions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note_path": {
                    "type": "string",
                    "description": "Path to the source note relative to vault root, e.g. '03-courses/ECON/lecture-05.md'.",
                },
            },
            "required": ["note_path"],
        },
    },
    {
        "name": "weekly_summary",
        "description": (
            "Generate a weekly summary .md in 05-logs/ for a course, "
            "covering all new concept notes in 01-concepts/ with the "
            "matching source tag created this week."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "course": {
                    "type": "string",
                    "description": "Course code for source tag filtering, e.g. 'econ204' or 'cs201'.",
                },
            },
            "required": ["course"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and folders at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or vault-relative path to list.",
                },
            },
            "required": ["path"],
        },
    },
]


# ──────────────────────────────────────────────
# HELPERS  (not exposed as tools)
# ──────────────────────────────────────────────

def _ollama_generate(prompt: str, system: str = "") -> str:
    """
    Call the local Ollama instance to generate text.
    Used by tools that need the LLM to write content
    (refresher paragraphs, concept notes, summaries).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.RequestException as e:
        return f"[Ollama error: {e}]"


def _canvas_get(endpoint: str, params: dict | None = None) -> list | dict:
    """
    Make a GET request to the Canvas LMS API.
    Returns parsed JSON or an error dict.
    """
    url = f"{CANVAS_BASE_URL}/api/v1/{endpoint.lstrip('/')}"
    try:
        r = requests.get(url, headers=CANVAS_HEADERS, params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def _ocr_file(file_path: str) -> str:
    """
    Run Apple Vision OCR on an image OR PDF file.

    For PDFs (your handwritten notes): uses pymupdf to rasterize
    each page into an image, then runs ocrmac on each page.

    For images: runs ocrmac directly.

    Install:  pip install ocrmac pymupdf
    """
    try:
        from ocrmac import ocrmac
    except ImportError:
        return (
            "[ocrmac not installed — run: pip install ocrmac]\n"
            "This package wraps Apple Vision for handwriting OCR on macOS."
        )

    path = Path(file_path)
    all_text = []

    try:
        if path.suffix.lower() == ".pdf":
            # Rasterize each PDF page → PNG in a temp dir, then OCR each
            import fitz  # pymupdf
            import tempfile

            doc = fitz.open(str(path))
            with tempfile.TemporaryDirectory() as tmp:
                for i, page in enumerate(doc):
                    # Render at 2x resolution for better OCR accuracy
                    pix = page.get_pixmap(dpi=200)
                    img_path = Path(tmp) / f"page_{i}.png"
                    pix.save(str(img_path))

                    results = ocrmac.OCR(str(img_path)).recognize()
                    lines = [text for text, _conf, _bbox in results]
                    if lines:
                        all_text.append(f"--- Page {i + 1} ---")
                        all_text.extend(lines)
            doc.close()
        else:
            # Direct image OCR
            results = ocrmac.OCR(file_path).recognize()
            all_text = [text for text, _conf, _bbox in results]

    except ImportError:
        return "[pymupdf not installed — run: pip install pymupdf]"
    except Exception as e:
        return f"[OCR error on {file_path}: {e}]"

    return "\n".join(all_text) if all_text else "[No text detected]"


def _read_vault_file(relative_path: str) -> str:
    """Read a file from the Obsidian vault. Returns contents or error string."""
    full = Path(OBSIDIAN_VAULT) / relative_path
    if not full.exists():
        return f"[File not found: {relative_path}]"
    return full.read_text(encoding="utf-8", errors="replace")


def _write_vault_file(relative_path: str, content: str) -> str:
    """Write (or overwrite) a file in the Obsidian vault. Creates parent dirs."""
    full = Path(OBSIDIAN_VAULT) / relative_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return f"Wrote {full}"


def _vault_files_modified_since(folder: str, since: datetime) -> list[Path]:
    """Return vault files in `folder` modified after `since`."""
    base = Path(OBSIDIAN_VAULT) / folder
    if not base.exists():
        return []
    results = []
    for f in base.rglob("*.md"):
        if datetime.fromtimestamp(f.stat().st_mtime) >= since:
            results.append(f)
    return sorted(results, key=lambda p: p.stat().st_mtime)


# ──────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ──────────────────────────────────────────────

def export_apple_notes(folder: str = "", **kwargs) -> str:
    """
    Export handwritten Apple Notes from a folder as PDFs.

    Uses AppleScript to tell Notes.app to export each note in the
    specified folder as a PDF, saved to ICLOUD_NOTES_DIR/{folder}/.

    Run this BEFORE convert_notes — it stages the PDFs for OCR.

    NOTE: You need to create a matching folder in Apple Notes
    (e.g. a folder called "ECON204" with your handwritten notes in it).
    """
    if not folder:
        return "Error: 'folder' parameter required (e.g. 'ECON204')."

    output_dir = Path(ICLOUD_NOTES_DIR) / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # AppleScript to export all notes in the folder as PDFs.
    # Notes.app doesn't have a native "export PDF" command, so we
    # use the print-to-PDF trick via System Events.
    # Fallback: a simpler script that extracts note names so you know what's there.

    applescript = f'''
    tell application "Notes"
        set noteFolder to folder "{folder}"
        set noteList to every note in noteFolder
        set exported to {{}}
        repeat with aNote in noteList
            set noteTitle to the name of aNote
            -- Sanitize the title for use as a filename
            set safeName to do shell script "echo " & quoted form of noteTitle & " | sed 's/[^a-zA-Z0-9._-]/-/g' | sed 's/--*/-/g'"
            set pdfPath to "{output_dir}/" & safeName & ".pdf"

            -- Skip if already exported
            try
                do shell script "test -f " & quoted form of pdfPath
                -- File exists, skip
            on error
                -- File doesn't exist, export it
                set noteBody to the body of aNote
                -- Write HTML body to temp file, convert to PDF via textutil
                set tmpHTML to "/tmp/tort_note_export.html"
                do shell script "echo " & quoted form of noteBody & " > " & tmpHTML
                do shell script "textutil -convert pdf " & tmpHTML & " -output " & quoted form of pdfPath
                set end of exported to safeName
            end try
        end repeat
        return exported
    end tell
    '''

    try:
        import subprocess
        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            # If AppleScript fails (common with handwritten notes since
            # their "body" is drawing data, not HTML), fall back to
            # listing what's in the folder so the user knows the state.
            return (
                f"AppleScript export had issues: {result.stderr.strip()}\n\n"
                f"For handwritten notes, AppleScript can't always extract the content.\n"
                f"Alternative: select notes in Notes.app → File → Export as PDF → save to:\n"
                f"  {output_dir}/\n\n"
                f"Or set up a Shortcut called 'Export Notes to PDF' and I can call it with:\n"
                f"  shortcuts run 'Export Notes to PDF'"
            )

        exported = result.stdout.strip()
        if exported:
            return f"Exported to {output_dir}/:\n{exported}"
        else:
            return f"No new notes to export (all already exist as PDFs in {output_dir}/)"

    except FileNotFoundError:
        return "[osascript not found — this tool only works on macOS]"
    except subprocess.TimeoutExpired:
        return "[Export timed out — try exporting fewer notes at once]"
    except Exception as e:
        return f"[Export error: {e}]"


def morning_brief(**kwargs) -> str:
    """
    Orchestrates the full morning briefing:
      1. Weather for WEATHER_LOCATION today
      2. Canvas assignments due today + tomorrow
      3. Per-class refresher paragraph from recent vault notes
    """
    sections = []
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    today_str = today.strftime("%A, %B %d")

    # ── 1. WEATHER ──────────────────────────────
    try:
        w = requests.get(f"https://wttr.in/{WEATHER_LOCATION}?format=j1", timeout=10)
        w.raise_for_status()
        data = w.json()
        current = data["current_condition"][0]
        forecast = data["weather"][0]

        weather_block = (
            f"🌤  Weather — {today_str} ({WEATHER_LOCATION})\n"
            f"   Now: {current['temp_F']}°F, {current['weatherDesc'][0]['value']}\n"
            f"   High/Low: {forecast['maxtempF']}°F / {forecast['mintempF']}°F\n"
            f"   Precip chance: {forecast['hourly'][4].get('chanceofrain', '?')}%\n"
        )
    except Exception as e:
        weather_block = f"🌤  Weather — could not fetch: {e}\n"

    sections.append(weather_block)

    # ── 2. CANVAS ASSIGNMENTS ───────────────────
    # Fetch all active courses first
    courses = _canvas_get("courses", {"enrollment_state": "active", "per_page": 50})

    if isinstance(courses, dict) and "error" in courses:
        sections.append(f"📚  Canvas — error: {courses['error']}\n")
    else:
        due_items = []
        course_names = {}  # id → name, for refresher step

        for c in courses:
            cid = c.get("id")
            cname = c.get("name", "Unknown Course")
            course_names[cid] = cname

            assignments = _canvas_get(
                f"courses/{cid}/assignments",
                {
                    "bucket": "upcoming",
                    "per_page": 50,
                    "order_by": "due_at",
                },
            )
            if isinstance(assignments, list):
                for a in assignments:
                    due = a.get("due_at")
                    if not due:
                        continue
                    # Canvas returns ISO 8601 UTC timestamps
                    due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                    due_local = due_dt.astimezone()  # convert to local tz
                    if due_local.date() <= tomorrow.date():
                        due_items.append({
                            "course": cname,
                            "name": a.get("name", "Untitled"),
                            "due": due_local.strftime("%a %I:%M %p"),
                            "points": a.get("points_possible", "—"),
                        })

        if due_items:
            canvas_lines = ["📚  Due Today / Tomorrow"]
            for item in due_items:
                canvas_lines.append(
                    f"   [{item['course']}] {item['name']}  "
                    f"— due {item['due']}  ({item['points']} pts)"
                )
            sections.append("\n".join(canvas_lines) + "\n")
        else:
            sections.append("📚  Nothing due today or tomorrow — nice.\n")

    # ── 3. CLASS REFRESHER ──────────────────────
    # Look in 03-courses/ for course folders (per vault structure).
    # Read the most recent note and ask Ollama for a quick refresher.
    courses_root = Path(OBSIDIAN_VAULT) / "03-courses"
    if courses_root.exists():
        course_folders = [
            d.name for d in courses_root.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        ]

        refresher_lines = ["📝  Class Refresher"]
        found_any = False

        for folder in course_folders:
            # grab the most recently modified .md in this course folder
            notes = sorted(
                (courses_root / folder).rglob("*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not notes:
                continue

            latest = notes[0]
            content = latest.read_text(encoding="utf-8", errors="replace")[:3000]

            refresher = _ollama_generate(
                prompt=(
                    f"Here are my most recent notes for {folder}:\n\n"
                    f"{content}\n\n"
                    "Write a 3-4 sentence refresher paragraph so I walk into "
                    "class primed on the latest material. Be concise and direct. "
                    "Talk like a classmate, not a textbook."
                ),
                system="You are a study assistant. Be concise, casual, no fluff.",
            )
            refresher_lines.append(f"\n   [{folder}]\n   {refresher}")
            found_any = True

        if found_any:
            sections.append("\n".join(refresher_lines) + "\n")
        else:
            sections.append("📝  No course notes found in 03-courses/ for refresher.\n")

    return "\n".join(sections)


def convert_notes(course: str = "", **kwargs) -> str:
    """
    Scan the iCloud handwritten-notes folder for PDFs (and images),
    OCR each one via Apple Vision, and save as lecture .md files
    in the vault's 03-courses/ structure per CLAUDE.md conventions.

    Expected iCloud folder structure:
      {ICLOUD_NOTES_DIR}/
        {course}/
          lecture-01.pdf
          lecture-02.pdf

    Output:
      {OBSIDIAN_VAULT}/03-courses/{course}/
        lecture-01.md
        lecture-02.md
    """
    if not course:
        return "Error: 'course' parameter is required (e.g. 'ECON' or 'CO300')."

    source_dir = Path(ICLOUD_NOTES_DIR) / course
    if not source_dir.exists():
        return (
            f"No folder found at {source_dir}\n"
            f"Expected iCloud notes at: {ICLOUD_NOTES_DIR}/{course}/\n"
            f"Check ICLOUD_NOTES_DIR env var and folder structure."
        )

    # PDFs first, but also grab any loose images
    supported_exts = {".pdf", ".jpg", ".jpeg", ".png", ".heic", ".tiff"}
    files = [f for f in source_dir.iterdir() if f.suffix.lower() in supported_exts]

    if not files:
        return f"No PDF or image files found in {source_dir}"

    # Output to 03-courses/{course}/ per vault structure
    output_dir = Path(OBSIDIAN_VAULT) / "03-courses" / course
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    converted = 0

    for src in sorted(files):
        md_name = src.stem + ".md"
        md_path = output_dir / md_name

        # Skip if already converted (don't re-OCR)
        if md_path.exists():
            results.append(f"  skip  {src.name} (already converted)")
            continue

        # Run OCR (handles both PDFs and images)
        raw_text = _ocr_file(str(src))

        if raw_text.startswith("["):
            results.append(f"  fail  {src.name}: {raw_text}")
            continue

        # Ask Ollama to clean up the OCR into lecture note format.
        # Per CLAUDE.md: lecture notes are summaries with [[wikilinks]]
        # to concepts, written casually like Dylan talks.
        cleaned = _ollama_generate(
            prompt=(
                f"Below is raw OCR text from my handwritten lecture notes for {course}.\n\n"
                f"Clean it up into well-structured markdown lecture notes.\n"
                f"- Fix obvious OCR errors\n"
                f"- Add headers where topics change\n"
                f"- Use [[wikilinks]] for key concepts (e.g. [[inflation]], [[gdp]])\n"
                f"- Keep my phrasing — don't make it sound like a textbook\n"
                f"- Keep all the content, don't summarize away detail\n\n"
                f"Raw OCR:\n{raw_text}"
            ),
            system=(
                "You are a note formatter for an Obsidian vault. "
                "Output clean markdown only. Use [[wikilinks]] for key terms. "
                "Write casually — these are personal notes, not publications. "
                "Do not add commentary — just the formatted notes."
            ),
        )

        # Frontmatter matching CLAUDE.md lecture note conventions
        today = datetime.now().strftime("%Y-%m-%d")
        frontmatter = (
            f"---\n"
            f"date: {today}\n"
            f"tags:\n"
            f"  - lecture\n"
            f"  - course/{course}\n"
            f"source: handwritten ({src.name})\n"
            f"---\n\n"
        )

        md_path.write_text(frontmatter + cleaned, encoding="utf-8")
        results.append(f"  done  {src.name} → 03-courses/{course}/{md_name}")
        converted += 1

    summary = f"Converted {converted}/{len(files)} notes for {course}\n"
    return summary + "\n".join(results)


def develop_concepts(note_path: str = "", **kwargs) -> str:
    """
    Read a lecture note from 03-courses/, extract atomic concepts,
    and create individual .md files in 01-concepts/ following
    CLAUDE.md conventions exactly.

    Per CLAUDE.md:
      - One idea per note ("atomic concepts")
      - Check 01-concepts/ for existing notes before creating
      - Proper frontmatter: date, tags, related, source, status
      - Write like Dylan talks — casual, direct, concrete
      - Leave a stub in the source note linking to extracted concepts
    """
    if not note_path:
        return "Error: 'note_path' required (relative to vault root, e.g. '03-courses/ECON/lecture-05.md')."

    # Read the source note
    note_content = _read_vault_file(note_path)
    if note_content.startswith("["):
        return note_content

    # Read vault conventions so the model knows the rules
    claude_md = _read_vault_file("CLAUDE.md")
    if claude_md.startswith("["):
        claude_md = ""

    # Figure out the course from the path (e.g. "03-courses/ECON/lecture-05.md" → "ECON")
    parts = note_path.split("/")
    if len(parts) >= 2 and parts[0] == "03-courses":
        course = parts[1]  # e.g. "ECON"
    elif len(parts) >= 1:
        course = parts[0]
    else:
        course = "general"

    # Map course folder → source tag (per CLAUDE.md graph color system)
    # Add new courses here as they come up
    SOURCE_TAG_MAP = {
        "CS201":  "source/cs201",
        "CS150B": "source/cs150b",
        "ECON":   "source/econ204",
        "CO300":  "source/co300",
        "STAT":   "source/stat",
    }
    source_tag = SOURCE_TAG_MAP.get(course, "source/homelab")

    # Check what already exists in 01-concepts/ so we don't duplicate
    concepts_dir = Path(OBSIDIAN_VAULT) / "01-concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    existing = [f.stem for f in concepts_dir.glob("*.md")]
    existing_list = ", ".join(existing[:50]) if existing else "(none yet)"

    # Build the extraction prompt with CLAUDE.md's exact format
    extraction_prompt = (
        f"I have lecture notes from my {course} class:\n\n"
        f"---\n{note_content[:4000]}\n---\n\n"
        f"Existing concept notes in the vault (DON'T create duplicates):\n"
        f"{existing_list}\n\n"
        f"Extract 3-7 atomic concepts. For each concept, provide:\n"
        f'- "filename": kebab-case slug (e.g. "marginal-cost.md")\n'
        f'- "title": concept name\n'
        f'- "domain_tag": one of: domain/ml, domain/iot, domain/networking, '
        f'domain/linux, domain/cv, domain/embedded, domain/programming, '
        f'domain/ethics, domain/economics, domain/rhetoric\n'
        f'- "explanation": one paragraph in casual language — explain it like '
        f'you\'d tell a classmate. Not a textbook definition.\n'
        f'- "how_it_works": the mechanism or technical detail, concrete.\n'
        f'- "where_it_shows_up": coursework connections, project connections, '
        f'real-world applications.\n'
        f'- "related": list of [[wikilinks]] to related concepts '
        f'(from the existing list above, or new ones being created).\n\n'
        f"If a concept already exists in the vault (check the list above), "
        f"skip it — don't create a duplicate.\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"concepts": [...]}}\n'
        f"No markdown fences. No commentary."
    )

    raw = _ollama_generate(
        extraction_prompt,
        system=(
            "You are a study assistant for an Obsidian vault. "
            "Output ONLY valid JSON. Write casually — like a college student "
            "explaining to a classmate, not a textbook. Never say 'utilizing'. "
            "Keep explanations concrete and direct."
        ),
    )

    # Parse JSON response
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return (
            f"Ollama returned unparseable JSON. Raw response:\n{raw[:500]}\n\n"
            "Try running again — small models sometimes need a second attempt."
        )

    concepts = data.get("concepts", [])
    if not concepts:
        return "No concepts extracted — the note might be too short or unclear."

    # Write each concept file to 01-concepts/ with proper frontmatter
    today = datetime.now().strftime("%Y-%m-%d")
    created = []
    skipped = []

    for c in concepts:
        filename = c.get("filename", "untitled.md")
        if not filename.endswith(".md"):
            filename += ".md"
        title = c.get("title", "Untitled")
        domain = c.get("domain_tag", "domain/programming")
        explanation = c.get("explanation", "")
        how = c.get("how_it_works", "")
        where = c.get("where_it_shows_up", "")
        related = c.get("related", [])

        # Skip if already exists (per CLAUDE.md rule 3: check before creating)
        out_path = concepts_dir / filename
        if out_path.exists():
            skipped.append(f"  skip  {filename} (already exists)")
            continue

        # Build related links string
        related_str = ", ".join(related) if related else ""

        # Assemble the note in CLAUDE.md's exact concept format
        note_body = (
            f"---\n"
            f"date: {today}\n"
            f"tags:\n"
            f"  - concept\n"
            f"  - {domain}\n"
            f"  - {source_tag}\n"
            f"related:\n"
        )
        for r in related:
            link = r if r.startswith("[[") else f"[[{r}]]"
            note_body += f'  - "{link}"\n'
        note_body += (
            f"source: \"[[{note_path}]]\"\n"
            f"status: seed\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"{explanation}\n\n"
            f"## How it works\n\n"
            f"{how}\n\n"
            f"## Where this shows up\n\n"
            f"{where}\n\n"
            f"## Links\n\n"
            f"- Related concepts: {related_str}\n"
            f"- Source: [[{note_path}]]\n"
        )

        out_path.write_text(note_body, encoding="utf-8")
        created.append(f"  + 01-concepts/{filename}")

    # Update the source lecture note with links to extracted concepts
    if created:
        concept_links = "\n".join(
            f"- [[{c.get('filename', '').replace('.md', '')}]]"
            for c in concepts
            if not (concepts_dir / c.get("filename", "")).exists()
               or c.get("filename", "").replace(".md", "") not in existing
        )
        if concept_links:
            stub = (
                f"\n\n## Extracted concepts\n\n"
                f"{concept_links}\n"
            )
            source_full = Path(OBSIDIAN_VAULT) / note_path
            if source_full.exists():
                with open(source_full, "a", encoding="utf-8") as f:
                    f.write(stub)
                created.append(f"  ↳ updated {note_path} with concept links")

    result_lines = []
    if created:
        result_lines.append(f"Created {len(created) - (1 if '↳' in created[-1] else 0)} concept notes:")
        result_lines.extend(created)
    if skipped:
        result_lines.append(f"Skipped {len(skipped)} (already exist):")
        result_lines.extend(skipped)

    return "\n".join(result_lines) if result_lines else "Nothing to do."


def weekly_summary(course: str = "", **kwargs) -> str:
    """
    Generate a weekly summary .md in 05-logs/ for a course,
    covering all concept notes in 01-concepts/ with the matching
    source tag that were created or modified in the past 7 days.
    """
    if not course:
        return "Error: 'course' parameter required (e.g. 'econ204' or 'cs201')."

    # The source tag to filter by (e.g. "source/econ204")
    source_tag = f"source/{course.lower()}"
    one_week_ago = datetime.now() - timedelta(days=7)

    # Scan 01-concepts/ for recently modified files with matching source tag
    concepts_dir = Path(OBSIDIAN_VAULT) / "01-concepts"
    if not concepts_dir.exists():
        return f"01-concepts/ directory not found in vault."

    recent = []
    for f in concepts_dir.glob("*.md"):
        if datetime.fromtimestamp(f.stat().st_mtime) < one_week_ago:
            continue
        # Check if the file contains the matching source tag
        content = f.read_text(encoding="utf-8", errors="replace")
        if source_tag in content:
            recent.append((f, content))

    if not recent:
        return f"No new concept notes with tag '{source_tag}' in the past 7 days."

    # Build content for the summary prompt
    note_contents = []
    for f, content in recent:
        note_contents.append(f"## [[{f.stem}]]\n{content[:1500]}")

    all_notes = "\n\n---\n\n".join(note_contents)

    summary = _ollama_generate(
        prompt=(
            f"Here are all the new concept notes for {course} from the past week:\n\n"
            f"{all_notes}\n\n"
            "Write a weekly summary that:\n"
            "1. Lists the key topics covered this week\n"
            "2. Explains how they connect to each other\n"
            "3. Highlights the 2-3 most important takeaways\n"
            "4. Includes [[wikilinks]] to each concept note\n"
            "Keep it to ~300 words. Write casually, like explaining to a classmate."
        ),
        system=(
            "You are a study assistant. Write concise Obsidian-flavored markdown. "
            "Use [[wikilinks]] for all concept references. Be direct, not textbooky."
        ),
    )

    # Save to 05-logs/ per vault structure
    week_label = datetime.now().strftime("week-%Y-%m-%d")
    summary_path = f"05-logs/{course}-{week_label}.md"
    today = datetime.now().strftime("%Y-%m-%d")

    frontmatter = (
        f"---\n"
        f"date: {today}\n"
        f"tags:\n"
        f"  - log\n"
        f"  - course/{course}\n"
        f"type: weekly-summary\n"
        f"concepts_covered: {len(recent)}\n"
        f"---\n\n"
        f"# {course.upper()} — Weekly Summary ({week_label})\n\n"
    )

    result = _write_vault_file(summary_path, frontmatter + summary)
    return f"Weekly summary for {course} ({len(recent)} concepts):\n{result}"


def list_directory(path: str = ".", **kwargs) -> str:
    """List files and directories at the given path."""
    target = Path(path).expanduser()
    if not target.exists():
        return f"Path not found: {target}"
    if not target.is_dir():
        return f"Not a directory: {target}"

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    lines = []
    for e in entries:
        prefix = "📁 " if e.is_dir() else "📄 "
        lines.append(f"{prefix}{e.name}")

    return f"{target}/\n" + "\n".join(lines) if lines else f"{target}/ (empty)"


# ──────────────────────────────────────────────
# DISPATCHER  (called by the agent's tool loop)
# ──────────────────────────────────────────────

# Map tool names → functions
_TOOL_MAP = {
    "morning_brief":      morning_brief,
    "export_apple_notes": export_apple_notes,
    "convert_notes":      convert_notes,
    "develop_concepts":   develop_concepts,
    "weekly_summary":     weekly_summary,
    "list_directory":     list_directory,
}


def execute_tool(name: str, arguments: dict | None = None) -> str:
    """
    Called by the agent's tool-use loop.

    Usage:
        from tool import execute_tool, TOOL_REGISTRY

        # Pass TOOL_REGISTRY to the model so it knows what's available.
        # When the model returns a tool_call, run:
        result = execute_tool(tool_call["name"], tool_call["arguments"])
    """
    func = _TOOL_MAP.get(name)
    if func is None:
        return f"Unknown tool: '{name}'. Available: {list(_TOOL_MAP.keys())}"

    try:
        return func(**(arguments or {}))
    except Exception as e:
        return f"Tool '{name}' failed: {type(e).__name__}: {e}"


# ──────────────────────────────────────────────
# QUICK TEST  (run this file directly to verify)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Tort Agent — Tool Smoke Test")
    print("=" * 50)

    # Test list_directory
    print("\n── list_directory ──")
    print(execute_tool("list_directory", {"path": OBSIDIAN_VAULT}))

    # Test morning_brief (requires Canvas token + network)
    print("\n── morning_brief ──")
    result = execute_tool("morning_brief")
    print(result[:1000] + ("..." if len(result) > 1000 else ""))

    print("\n✅ Smoke test complete.")