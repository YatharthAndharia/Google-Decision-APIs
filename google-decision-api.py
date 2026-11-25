import base64
import html
import json
import os
import threading
import time
from datetime import date, datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()


def setting(name, default=None):
    """Read a setting from Streamlit secrets (deployed) or the environment / .env (local)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # noqa: BLE001 — no secrets.toml when running locally
        pass
    return os.getenv(name, default)


API_KEY = setting("GEMINI_API_KEY")
MODEL = "gemini-decision-1"

# ---- Public-deployment guardrails (override any of these in Streamlit secrets) ----
MAX_RUNS_PER_SESSION = int(setting("MAX_RUNS_PER_SESSION", 10))
MAX_RUNS_PER_DAY = int(setting("MAX_RUNS_PER_DAY", 200))  # shared by all visitors
RUN_COOLDOWN_SECONDS = int(setting("RUN_COOLDOWN_SECONDS", 10))
MAX_UPLOAD_MB = int(setting("MAX_UPLOAD_MB", 20))
MAX_TEXT_CHARS = int(setting("MAX_TEXT_CHARS", 10_000))
MAX_QUESTIONS = int(setting("MAX_QUESTIONS", 10))
API_VERSION = "v1beta"

st.set_page_config(
    page_title="Gemini Decisions Playground",
    page_icon="⚡",
    layout="wide",
)

ss = st.session_state


# =========================================================
# MIME TYPES & LIMITS
# =========================================================

MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",

    "wav": "audio/wav",
    "mp3": "audio/mp3",
    "mpeg": "audio/mpeg",
    "aiff": "audio/aiff",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/m4a",
    "opus": "audio/opus",

    "mp4": "video/mp4",
    "mov": "video/mov",
    "avi": "video/avi",
    "wmv": "video/wmv",
    "flv": "video/x-flv",
    "3gp": "video/3gpp",

    "pdf": "application/pdf",
}

IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/webp",
    "image/heic", "image/heif", "image/gif",
    "image/bmp", "image/tiff",
}

AUDIO_TYPES = {
    "audio/wav", "audio/mp3", "audio/mpeg",
    "audio/aiff", "audio/aac", "audio/ogg",
    "audio/flac", "audio/m4a", "audio/opus",
}

VIDEO_TYPES = {
    "video/mp4", "video/mov", "video/avi",
    "video/wmv", "video/x-flv", "video/3gpp",
}

ALLOWED_EXTENSIONS = list(MIME_TYPES.keys())

MAX_REQUEST_SIZE = MAX_UPLOAD_MB * 1024 * 1024
MAX_PDF_SIZE = min(50, MAX_UPLOAD_MB) * 1024 * 1024

RESOLUTIONS = ["low", "medium", "high", "ultra_high"]

DECISION_TYPES = {
    "noul": {
        "label": "Yes / No",
        "icon": ":material/rule:",
        "color": "blue",
        "help": "A binary judgement with a probability of “yes”.",
    },
    "choice": {
        "label": "Multiple choice",
        "icon": ":material/checklist:",
        "color": "violet",
        "help": "Pick exactly one option from a list you define.",
    },
    "score": {
        "label": "Rubric score",
        "icon": ":material/leaderboard:",
        "color": "orange",
        "help": "Grade the input against ordered levels, lowest to highest.",
    },
}

TAB_INPUT = ":material/upload_file: Input & media"
TAB_QUESTIONS = ":material/quiz: Decision questions"
TAB_RESULTS = ":material/insights: Results"
TAB_GUIDES = ":material/play_circle: Video guides"

# Paste the YouTube links here after uploading the videos (unlisted is fine).
# An empty url shows a "coming soon" placeholder instead of the player.
VIDEO_GUIDES = {
    "overview": {
        "title": "How the Decision API works",
        "caption": "The three decision types, a photo example, and what confidence means.",
        "url": setting("VIDEO_URL_OVERVIEW", ""),
    },
    "choice": {
        "title": "Multiple choice, step by step",
        "caption": "A restaurant review, then a template and a photo example.",
        "url": setting("VIDEO_URL_CHOICE", ""),
    },
    "noul": {
        "title": "Yes / No, step by step",
        "caption": "A work email, then a template and a photo example.",
        "url": setting("VIDEO_URL_NOUL", ""),
    },
    "score": {
        "title": "Rubric score, step by step",
        "caption": "A hotel review, then a template and a photo example.",
        "url": setting("VIDEO_URL_SCORE", ""),
    },
}

QUESTION_FIELDS = ("key", "type", "instructions", "true", "false", "levels")


# =========================================================
# STYLES
# =========================================================

st.html(
    """
<style>
/* ---------- Layout polish ---------- */
.block-container { padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1360px; }

/* ---------- Hero ---------- */
.dp-hero {
  position: relative; overflow: hidden;
  border-radius: 18px; padding: 30px 34px 28px;
  background:
    radial-gradient(1200px 300px at 100% -40%, rgba(236,72,153,.35), transparent 60%),
    radial-gradient(900px 320px at -10% 140%, rgba(20,184,166,.30), transparent 60%),
    linear-gradient(120deg, #312E81 0%, #4F46E5 55%, #7C3AED 100%);
  color: #fff; box-shadow: 0 18px 40px -22px rgba(79,70,229,.65);
}
.dp-hero::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: radial-gradient(rgba(255,255,255,.14) 1px, transparent 1px);
  background-size: 18px 18px;
  mask-image: linear-gradient(90deg, transparent 35%, #000 100%);
  -webkit-mask-image: linear-gradient(90deg, transparent 35%, #000 100%);
}
.dp-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  opacity: .9;
}
.dp-logo {
  display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 8px;
  background: rgba(255,255,255,.18); font-size: .9rem;
}
.dp-hero h1 {
  color: #fff !important; margin: 12px 0 6px; padding: 0;
  font-size: 2.05rem; line-height: 1.15; font-weight: 800; letter-spacing: -.02em;
}
.dp-hero p { margin: 0; max-width: 760px; font-size: 1rem; line-height: 1.55; color: rgba(255,255,255,.86); }
.dp-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; position: relative; z-index: 1; }
.dp-chip {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 5px 12px; border-radius: 999px; font-size: .8rem; font-weight: 500;
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
  backdrop-filter: blur(6px);
}
.dp-chip code { background: transparent; color: #fff; font-size: .78rem; padding: 0; }
.dp-dot { width: 8px; height: 8px; border-radius: 50%; background: #34D399; box-shadow: 0 0 0 3px rgba(52,211,153,.3); }

/* ---------- Steps ---------- */
.dp-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0 6px; }
.dp-step {
  display: flex; gap: 12px; align-items: flex-start;
  padding: 12px 14px; border-radius: 12px;
  border: 1px solid rgba(128,128,128,.22); background: rgba(128,128,128,.04);
}
.dp-step-num {
  flex: none; width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center;
  font-weight: 700; font-size: .85rem; color: #fff; background: #6366F1;
}
.dp-step.done .dp-step-num { background: #10B981; }
.dp-step b { display: block; font-size: .92rem; }
.dp-step span { font-size: .8rem; opacity: .7; }
@media (max-width: 760px) { .dp-steps { grid-template-columns: 1fr; } .dp-hero { padding: 22px; } .dp-hero h1 { font-size: 1.6rem; } }

/* ---------- Section headers ---------- */
.dp-section { margin: 4px 0 2px; }
.dp-section h3 { margin: 0 !important; padding: 0 !important; font-size: 1.08rem !important; font-weight: 700; }
.dp-section p { margin: 2px 0 0; font-size: .86rem; opacity: .68; }

/* ---------- Run panel ---------- */
div[data-testid="stColumn"]:has(.st-key-run_panel) { position: sticky; top: 3.75rem; align-self: flex-start; }
.dp-check { display: flex; gap: 10px; align-items: flex-start; padding: 9px 0; border-bottom: 1px dashed rgba(128,128,128,.25); }
.dp-check:last-child { border-bottom: 0; }
.dp-check-icon {
  flex: none; width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
  font-size: .75rem; font-weight: 700; color: #fff;
}
.dp-check.ok .dp-check-icon { background: #10B981; }
.dp-check.todo .dp-check-icon { background: rgba(128,128,128,.45); }
.dp-check.bad .dp-check-icon { background: #EF4444; }
.dp-check b { display: block; font-size: .9rem; }
.dp-check span { font-size: .8rem; opacity: .72; }

/* ---------- Question card header ---------- */
.dp-qnum {
  display: inline-grid; place-items: center; width: 30px; height: 30px; border-radius: 9px;
  font-weight: 700; font-size: .88rem; color: #fff;
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
}

/* ---------- Media meter ---------- */
.dp-meter { margin-top: 6px; }
.dp-meter-top { display: flex; justify-content: space-between; font-size: .8rem; opacity: .75; margin-bottom: 5px; }
.dp-meter-track { height: 6px; border-radius: 99px; background: rgba(128,128,128,.18); overflow: hidden; }
.dp-meter-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #14B8A6, #6366F1); }

/* ---------- Result cards ---------- */
.dp-card {
  border-radius: 14px; padding: 18px 20px; height: 100%;
  border: 1px solid rgba(128,128,128,.22); background: rgba(128,128,128,.035);
}
.dp-card-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.dp-card-key { font-family: 'JetBrains Mono', monospace; font-size: .84rem; font-weight: 500; opacity: .8; word-break: break-all; }
.dp-type {
  flex: none; font-size: .7rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  padding: 3px 9px; border-radius: 999px;
}
.dp-type.noul { color: #2563EB; background: rgba(37,99,235,.12); }
.dp-type.choice { color: #7C3AED; background: rgba(124,58,237,.13); }
.dp-type.score { color: #D97706; background: rgba(217,119,6,.13); }
.dp-instr { font-size: .84rem; opacity: .68; margin: 6px 0 0; line-height: 1.45; }
.dp-verdict { font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em; margin: 12px 0 2px; line-height: 1.2; word-break: break-word; }
.dp-verdict.yes { color: #059669; }
.dp-verdict.no { color: #DC2626; }
.dp-verdict-sub { font-size: .85rem; opacity: .72; margin-bottom: 4px; }
.dp-conf { margin: 14px 0 4px; }
.dp-conf-top { display: flex; justify-content: space-between; font-size: .8rem; margin-bottom: 5px; }
.dp-conf-top span:first-child { opacity: .7; }
.dp-conf-top b.high { color: #059669; } .dp-conf-top b.mid { color: #D97706; } .dp-conf-top b.low { color: #DC2626; }
.dp-conf-track { height: 8px; border-radius: 99px; background: rgba(128,128,128,.18); overflow: hidden; }
.dp-conf-fill { height: 100%; border-radius: 99px; }
.dp-conf-fill.high { background: linear-gradient(90deg, #34D399, #059669); }
.dp-conf-fill.mid { background: linear-gradient(90deg, #FCD34D, #D97706); }
.dp-conf-fill.low { background: linear-gradient(90deg, #FCA5A5, #DC2626); }
.dp-probs { margin-top: 14px; }
.dp-probs-title { font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; opacity: .55; margin-bottom: 6px; }
.dp-pb { display: grid; grid-template-columns: minmax(80px, 38%) 1fr 54px; gap: 10px; align-items: center; padding: 4px 0; font-size: .84rem; }
.dp-pb-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; opacity: .85; }
.dp-pb.sel .dp-pb-label { font-weight: 700; opacity: 1; }
.dp-pb-track { height: 10px; border-radius: 99px; background: rgba(128,128,128,.16); overflow: hidden; }
.dp-pb-fill { height: 100%; border-radius: 99px; background: rgba(99,102,241,.45); }
.dp-pb.sel .dp-pb-fill { background: linear-gradient(90deg, #818CF8, #4F46E5); }
.dp-pb-val { text-align: right; font-variant-numeric: tabular-nums; opacity: .8; }
.dp-pb.sel .dp-pb-val { font-weight: 700; opacity: 1; }
.dp-scale { position: relative; margin: 16px 4px 6px; height: 8px; border-radius: 99px;
  background: linear-gradient(90deg, #F87171, #FBBF24, #34D399); }
.dp-scale-pin { position: absolute; top: 50%; width: 18px; height: 18px; border-radius: 50%;
  transform: translate(-50%, -50%); background: #fff; border: 4px solid #4F46E5; box-shadow: 0 2px 6px rgba(0,0,0,.25); }
.dp-scale-ends { display: flex; justify-content: space-between; font-size: .74rem; opacity: .6; }

/* ---------- Empty state ---------- */
.dp-empty { text-align: center; padding: 56px 20px; border-radius: 16px; border: 1.5px dashed rgba(128,128,128,.3); }
.dp-empty-icon { font-size: 2.4rem; margin-bottom: 8px; }
.dp-empty h4 { margin: 0 0 4px; padding: 0; font-size: 1.1rem; }
.dp-empty p { margin: 0 auto; max-width: 420px; font-size: .9rem; opacity: .7; }

/* ---------- Sidebar ---------- */
.dp-side-type { padding: 10px 12px; border-radius: 10px; margin-bottom: 8px;
  border: 1px solid rgba(128,128,128,.2); background: rgba(128,128,128,.04); }
.dp-side-type b { font-size: .88rem; }
.dp-side-type div { font-size: .8rem; opacity: .72; margin-top: 2px; }
</style>
"""
)


# =========================================================
# SESSION STATE
# =========================================================

def _default(key, value):
    if key not in ss:
        ss[key] = value


_default("question_ids", [])
_default("question_counter", 0)
_default("last_response", None)
_default("last_request", None)
_default("last_meta", None)
_default("last_error", None)
_default("pending_tab", None)
_default("user_text", "")
_default("resolution", "medium")
_default("uploader_version", 0)
_default("run_count", 0)
_default("last_run_at", 0.0)
_default("guide_type", "choice")


@st.cache_resource
def daily_usage():
    """Run counter shared by every visitor of this app instance (resets daily and on restart)."""
    return {"day": date.today(), "count": 0, "lock": threading.Lock()}


def try_reserve_run():
    """Check the guardrails and reserve one API call. Returns an error message or None."""
    if ss.run_count >= MAX_RUNS_PER_SESSION:
        return (
            f"You've reached the limit of {MAX_RUNS_PER_SESSION} runs for this session. "
            "Thanks for trying the playground!"
        )
    wait = RUN_COOLDOWN_SECONDS - (time.time() - ss.last_run_at)
    if wait > 0:
        return f"Please wait {int(wait) + 1} more second(s) before the next run."
    usage = daily_usage()
    with usage["lock"]:
        if usage["day"] != date.today():
            usage["day"], usage["count"] = date.today(), 0
        if usage["count"] >= MAX_RUNS_PER_DAY:
            return "The playground has reached today's usage limit. Please try again tomorrow."
        usage["count"] += 1
    ss.run_count += 1
    ss.last_run_at = time.time()
    return None


def blank_options():
    return pd.DataFrame(
        [{"Option": "", "Description": ""}, {"Option": "", "Description": ""}]
    )


def create_question(
    key=None,
    qtype="noul",
    instructions="",
    true="",
    false="",
    options=None,
    levels=None,
):
    ss.question_counter += 1
    qid = f"q{ss.question_counter}"

    ss[f"{qid}_key"] = key or f"question_{ss.question_counter}"
    ss[f"{qid}_type"] = qtype
    ss[f"{qid}_instructions"] = instructions
    ss[f"{qid}_true"] = true
    ss[f"{qid}_false"] = false
    ss[f"{qid}_levels"] = "\n".join(levels or [])
    ss[f"{qid}_options_init"] = (
        pd.DataFrame(options, columns=["Option", "Description"])
        if options
        else blank_options()
    )

    ss.question_ids.append(qid)
    return qid


def remove_question(qid):
    ss.question_ids.remove(qid)


def duplicate_question(qid):
    options_df = ss.get(f"{qid}_options_value", ss.get(f"{qid}_options_init"))
    options = None
    if isinstance(options_df, pd.DataFrame):
        options = options_df.fillna("").values.tolist()

    create_question(
        key=f"{ss.get(f'{qid}_key', 'question')}_copy",
        qtype=ss.get(f"{qid}_type", "noul"),
        instructions=ss.get(f"{qid}_instructions", ""),
        true=ss.get(f"{qid}_true", ""),
        false=ss.get(f"{qid}_false", ""),
        options=options,
        levels=[
            line for line in ss.get(f"{qid}_levels", "").splitlines() if line.strip()
        ],
    )


TEMPLATES = {
    "noul": dict(
        key="meets_policy",
        qtype="noul",
        instructions="Does the content comply with our community guidelines?",
        true="No harassment, hate speech, spam or explicit content.",
        false="Contains any prohibited content.",
    ),
    "choice": dict(
        key="category",
        qtype="choice",
        instructions="Which category best describes this input?",
        options=[
            ["billing", "Payments, invoices, refunds"],
            ["technical", "Bugs, errors, outages"],
            ["account", "Login, profile, access"],
            ["other", "Anything else"],
        ],
    ),
    "score": dict(
        key="quality",
        qtype="score",
        instructions="Rate the overall quality of the input.",
        levels=[
            "Poor: unclear, incomplete or incorrect",
            "Fair: understandable but with notable gaps",
            "Good: clear and mostly complete",
            "Excellent: clear, complete and well structured",
        ],
    ),
}


def add_from_template(template):
    if template:
        create_question(**TEMPLATES[template])
    else:
        create_question()


def load_example():
    ss.question_ids = []
    ss.user_text = (
        "Hi team — I was charged twice for my Pro subscription this month "
        "(invoice #48213 and #48214). I've already contacted support twice "
        "this week with no reply. I need this refunded before Friday or I'll "
        "have to dispute it with my bank. Really disappointed with the service."
    )
    create_question(
        key="is_urgent",
        qtype="noul",
        instructions="Does this ticket need to be handled within 24 hours?",
        true="Customer mentions a deadline, financial loss, or escalation.",
        false="Routine request with no time pressure.",
    )
    create_question(
        key="category",
        qtype="choice",
        instructions="Which team should handle this ticket?",
        options=TEMPLATES["choice"]["options"],
    )
    create_question(
        key="sentiment",
        qtype="score",
        instructions="How does the customer feel?",
        levels=[
            "Very negative: angry or threatening to churn",
            "Negative: frustrated or disappointed",
            "Neutral: matter-of-fact",
            "Positive: satisfied or appreciative",
        ],
    )
    ss.pending_tab = TAB_QUESTIONS
    st.toast("Example loaded — a support ticket with 3 questions.", icon="✨")


def open_guides():
    ss.pending_tab = TAB_GUIDES


def reset_all():
    for key in list(ss.keys()):
        del ss[key]


if not ss.question_ids:
    create_question()

# Keep values of fields that are temporarily hidden (e.g. when switching the
# decision type) so they are not lost by Streamlit's widget cleanup.
for _qid in ss.question_ids:
    for _field in QUESTION_FIELDS:
        _k = f"{_qid}_{_field}"
        if _k in ss:
            ss[_k] = ss[_k]

if ss.pending_tab:
    ss.active_tab = ss.pending_tab
    ss.pending_tab = None


# =========================================================
# HELPERS
# =========================================================

def esc(value):
    return html.escape(str(value))


def get_mime_type(filename):
    extension = filename.rsplit(".", 1)[-1].lower()
    return MIME_TYPES.get(extension)


def media_kind(mime_type):
    if mime_type in IMAGE_TYPES:
        return "image"
    if mime_type in AUDIO_TYPES:
        return "audio"
    if mime_type in VIDEO_TYPES:
        return "video"
    if mime_type == "application/pdf":
        return "document"
    return None


def fmt_size(num_bytes):
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def section(title, subtitle=None):
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.html(f'<div class="dp-section"><h3>{esc(title)}</h3>{sub}</div>')


def uploaded_files():
    return ss.get(f"media_files_{ss.uploader_version}") or []


def check_input():
    """Lightweight validation that runs on every rerun (no encoding)."""
    text = ss.get("user_text", "").strip()
    files = uploaded_files()
    errors = []
    total = 0

    for file in files:
        total += file.size
        mime_type = get_mime_type(file.name)
        if not media_kind(mime_type):
            errors.append(f"Unsupported file format: {file.name}")
        elif mime_type == "application/pdf" and file.size > MAX_PDF_SIZE:
            errors.append(f"{file.name} exceeds the {MAX_PDF_SIZE // (1024 * 1024)} MB PDF limit.")

    if total > MAX_REQUEST_SIZE:
        errors.append(f"Total uploaded media exceeds {MAX_UPLOAD_MB} MB.")

    if len(text) > MAX_TEXT_CHARS:
        errors.append(f"Context is {len(text):,} characters; the limit is {MAX_TEXT_CHARS:,}.")

    if not text and not files:
        errors.append("Add some text or upload at least one file.")

    return text, files, total, errors


def build_input(text, files, resolution):
    content = []

    if text.strip():
        content.append({"type": "text", "text": text.strip()})

    for file in files:
        mime_type = get_mime_type(file.name)
        block_type = media_kind(mime_type)

        if not block_type:
            raise ValueError(f"Unsupported file format: {file.name}")

        block = {
            "type": block_type,
            "data": base64.b64encode(file.getvalue()).decode("utf-8"),
            "mime_type": mime_type,
        }

        if block_type in ("image", "video"):
            block["resolution"] = resolution

        content.append(block)

    if not content:
        raise ValueError("Please enter text or upload at least one file.")

    return content


def collect_questions():
    """Build the questions payload and collect every validation problem."""
    questions = {}
    errors = []

    for index, qid in enumerate(ss.question_ids, start=1):
        label = f"Q{index}"
        key = ss.get(f"{qid}_key", "").strip()
        qtype = ss.get(f"{qid}_type") or "noul"
        instructions = ss.get(f"{qid}_instructions", "").strip()

        if not key:
            errors.append(f"{label}: a question ID is required.")
            continue
        if key in questions:
            errors.append(f"{label}: the ID “{key}” is already used.")
            continue
        if not instructions:
            errors.append(f"{label} ({key}): instructions are required.")

        question = {"type": qtype, "instructions": instructions}

        if qtype == "noul":
            criteria = {}
            true_desc = ss.get(f"{qid}_true", "").strip()
            false_desc = ss.get(f"{qid}_false", "").strip()
            if true_desc:
                criteria["true_description"] = true_desc
            if false_desc:
                criteria["false_description"] = false_desc
            if criteria:
                question["criteria"] = criteria

        elif qtype == "choice":
            options_df = ss.get(f"{qid}_options_value", ss.get(f"{qid}_options_init"))
            options = {}
            if isinstance(options_df, pd.DataFrame):
                for _, row in options_df.fillna("").iterrows():
                    option_key = str(row.get("Option", "")).strip()
                    description = str(row.get("Description", "")).strip()
                    if not option_key:
                        if description:
                            errors.append(
                                f"{label} ({key}): an option has a description but no key."
                            )
                        continue
                    if option_key in options:
                        errors.append(
                            f"{label} ({key}): duplicate option “{option_key}”."
                        )
                        continue
                    options[option_key] = description
            if not 2 <= len(options) <= 255:
                errors.append(f"{label} ({key}): add between 2 and 255 options.")
            question["criteria"] = options

        elif qtype == "score":
            levels = [
                line.strip()
                for line in ss.get(f"{qid}_levels", "").splitlines()
                if line.strip()
            ]
            if not 2 <= len(levels) <= 255:
                errors.append(f"{label} ({key}): add between 2 and 255 rubric levels.")
            question["criteria"] = levels

        questions[key] = question

    if not ss.question_ids:
        errors.append("Add at least one decision question.")
    elif len(ss.question_ids) > MAX_QUESTIONS:
        errors.append(f"Use at most {MAX_QUESTIONS} questions per request.")

    return questions, errors


# =========================================================
# SIDEBAR — GUIDE
# =========================================================

with st.sidebar:
    st.markdown("### :material/menu_book: Quick guide")
    st.caption("Everything you need to get a decision in under a minute.")

    st.markdown(
        """
1. **Add context**: type text and/or upload media.
2. **Ask questions**: each one is evaluated independently.
3. **Run**: all questions go in a single API request.
"""
    )

    st.button(
        "Try an example",
        icon=":material/auto_awesome:",
        on_click=load_example,
        width="stretch",
    )
    st.button(
        "Watch video guides",
        icon=":material/play_circle:",
        on_click=open_guides,
        width="stretch",
    )

    st.divider()
    st.markdown("##### Decision types")
    for meta in DECISION_TYPES.values():
        st.html(
            f'<div class="dp-side-type"><b>{esc(meta["label"])}</b>'
            f'<div>{esc(meta["help"])}</div></div>'
        )

    st.divider()
    st.markdown("##### Supported input")
    st.markdown(
        """
- :material/image: **Images**: png, jpg, webp, heic, gif…
- :material/graphic_eq: **Audio**: wav, mp3, aac, flac, ogg…
- :material/movie: **Video**: mp4, mov, avi, wmv, 3gp…
- :material/picture_as_pdf: **PDF** files
"""
    )
    st.caption(
        f"Public demo limits: {MAX_UPLOAD_MB} MB of media per request, "
        f"{MAX_RUNS_PER_SESSION} runs per session."
    )

    st.divider()
    st.button(
        "Reset playground",
        icon=":material/restart_alt:",
        on_click=reset_all,
        type="tertiary",
    )


# =========================================================
# HERO
# =========================================================

key_chip = (
    '<span class="dp-chip"><span class="dp-dot"></span>API key loaded</span>'
    if API_KEY
    else '<span class="dp-chip"><span class="dp-dot" style="background:#F87171;'
    'box-shadow:0 0 0 3px rgba(248,113,113,.3)"></span>API key missing</span>'
)

st.html(
    f"""
<div class="dp-hero">
  <div class="dp-eyebrow"><span class="dp-logo">⚡</span>Gemini Decisions · Playground</div>
  <h1>Turn any input into structured decisions</h1>
  <p>Provide text, images, audio, video or PDFs, define the questions you care about,
     and get calibrated answers with confidence scores, all in one request.</p>
  <div class="dp-chips">
    {key_chip}
    <span class="dp-chip">Model&nbsp;<code>{esc(MODEL)}</code></span>
    <span class="dp-chip">API&nbsp;<code>{esc(API_VERSION)}</code></span>
  </div>
</div>
"""
)

if not API_KEY:
    st.error(
        "**GEMINI_API_KEY was not found.** Locally, add it to a `.env` file next to this app. "
        "On Streamlit Community Cloud, add it under **App settings → Secrets**. Then restart the app.",
        icon=":material/key_off:",
    )
    st.stop()


# Validation runs before rendering so the step indicators are accurate.
# Widget values from the previous interaction are already in session state.
text_value, files_value, total_size, input_errors = check_input()
questions_value, question_errors = collect_questions()
input_ready = not input_errors
questions_ready = not question_errors
has_results = ss.last_response is not None


def step(num, title, subtitle, done):
    state = "done" if done else ""
    mark = "✓" if done else num
    return (
        f'<div class="dp-step {state}"><div class="dp-step-num">{mark}</div>'
        f"<div><b>{esc(title)}</b><span>{esc(subtitle)}</span></div></div>"
    )


st.html(
    '<div class="dp-steps">'
    + step(1, "Provide context", "Text and/or media to evaluate", input_ready)
    + step(2, "Define questions", f"{len(ss.question_ids)} configured", questions_ready)
    + step(3, "Review results", "Answers, confidence, usage", has_results)
    + "</div>"
)


# =========================================================
# MAIN LAYOUT
# =========================================================

main_col, side_col = st.columns([2.35, 1], gap="large")

with main_col:
    tab_input, tab_questions, tab_results, tab_guides = st.tabs(
        [TAB_INPUT, TAB_QUESTIONS, TAB_RESULTS, TAB_GUIDES],
        key="active_tab",
        on_change="rerun",
    )


# ---------------------------------------------------------
# INPUT TAB
# ---------------------------------------------------------

with tab_input:
    st.space("small")
    section("Context", "The text you want Gemini to evaluate. Optional if you upload media.")

    st.text_area(
        "Context",
        key="user_text",
        height=170,
        placeholder="Paste a support ticket, a product review, a transcript, a policy…",
        label_visibility="collapsed",
    )

    chars = len(ss.get("user_text", ""))
    words = len(ss.get("user_text", "").split())
    st.caption(f"{chars:,} characters · {words:,} words")

    st.space("small")
    section("Media", "Images, audio, video and PDFs are sent alongside the text.")

    st.file_uploader(
        "Upload media",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        key=f"media_files_{ss.uploader_version}",
        label_visibility="collapsed",
    )

    files = uploaded_files()

    if files:
        pct = min(total_size / MAX_REQUEST_SIZE, 1) * 100
        st.html(
            f"""
<div class="dp-meter">
  <div class="dp-meter-top"><span>{len(files)} file(s) attached</span>
  <span>{fmt_size(total_size)} of {MAX_UPLOAD_MB} MB</span></div>
  <div class="dp-meter-track"><div class="dp-meter-fill" style="width:{pct:.1f}%"></div></div>
</div>"""
        )
        st.space("small")

        kind_badges = {
            "image": ("Image", "blue", ":material/image:"),
            "audio": ("Audio", "green", ":material/graphic_eq:"),
            "video": ("Video", "violet", ":material/movie:"),
            "document": ("PDF", "red", ":material/picture_as_pdf:"),
        }

        grid = st.columns(3)
        for index, file in enumerate(files):
            kind = media_kind(get_mime_type(file.name))
            with grid[index % 3]:
                with st.container(border=True):
                    if kind == "image":
                        st.image(file, width="stretch")
                    elif kind == "audio":
                        st.audio(file)
                    elif kind == "video":
                        st.video(file)
                    elif kind == "document":
                        st.html(
                            '<div style="display:grid;place-items:center;height:110px;'
                            'border-radius:10px;background:rgba(239,68,68,.08);'
                            'font-size:2.2rem">📄</div>'
                        )

                    if kind:
                        label, color, icon = kind_badges[kind]
                        st.badge(label, color=color, icon=icon)
                    else:
                        st.badge("Unsupported", color="red")
                    st.markdown(f"**{esc(file.name)}**")
                    st.caption(fmt_size(file.size))

        if st.button("Remove all files", icon=":material/delete_sweep:", type="tertiary"):
            ss.uploader_version += 1
            st.rerun()


# ---------------------------------------------------------
# QUESTIONS TAB
# ---------------------------------------------------------

with tab_questions:
    st.space("small")

    with st.container(horizontal=True, vertical_alignment="bottom"):
        section(
            "Decision questions",
            "Every question is answered independently against the same input.",
        )
        st.space("stretch")
        st.button(
            "Add question",
            icon=":material/add:",
            type="primary",
            on_click=add_from_template,
            args=(None,),
        )
        with st.popover("Templates", icon=":material/library_add:"):
            st.caption("Start from a ready-made question")
            for tkey, meta in DECISION_TYPES.items():
                st.button(
                    meta["label"],
                    icon=meta["icon"],
                    key=f"tpl_{tkey}",
                    help=meta["help"],
                    on_click=add_from_template,
                    args=(tkey,),
                    width="stretch",
                )

    st.space("small")

    for index, qid in enumerate(ss.question_ids, start=1):
        qtype = ss.get(f"{qid}_type") or "noul"
        meta = DECISION_TYPES[qtype]

        with st.container(border=True, key=f"qcard_{qid}"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.html(f'<span class="dp-qnum">{index}</span>', width="content")
                st.markdown(f"**Question {index}**", width="content")
                st.badge(meta["label"], color=meta["color"], icon=meta["icon"])
                st.space("stretch")
                st.button(
                    "",
                    icon=":material/content_copy:",
                    key=f"dup_{qid}",
                    help="Duplicate this question",
                    type="tertiary",
                    on_click=duplicate_question,
                    args=(qid,),
                )
                st.button(
                    "",
                    icon=":material/delete:",
                    key=f"remove_{qid}",
                    help="Remove this question",
                    type="tertiary",
                    on_click=remove_question,
                    args=(qid,),
                    disabled=len(ss.question_ids) == 1,
                )

            id_col, type_col = st.columns([1, 1.4])
            with id_col:
                st.text_input(
                    "Question ID",
                    key=f"{qid}_key",
                    help="The key you will find this answer under in the response.",
                    placeholder="e.g. is_urgent",
                )
            with type_col:
                st.segmented_control(
                    "Decision type",
                    options=list(DECISION_TYPES),
                    format_func=lambda x: DECISION_TYPES[x]["label"],
                    key=f"{qid}_type",
                    required=True,
                    width="stretch",
                )

            st.text_area(
                "Instructions",
                key=f"{qid}_instructions",
                height=80,
                placeholder="What should Gemini decide? e.g. “Does this ticket need a reply within 24 hours?”",
            )

            if qtype == "noul":
                st.caption(
                    ":material/tune: Optional: describe what counts as *yes* and *no* "
                    "for sharper answers."
                )
                yes_col, no_col = st.columns(2)
                yes_col.text_input(
                    "✅ Means YES when…",
                    key=f"{qid}_true",
                    placeholder="What makes the condition true?",
                )
                no_col.text_input(
                    "❌ Means NO when…",
                    key=f"{qid}_false",
                    placeholder="What makes the condition false?",
                )

            elif qtype == "choice":
                st.caption(
                    ":material/tune: One row per option. Use **+** at the bottom of the "
                    "table to add rows (2–255 options)."
                )
                editor_key = f"{qid}_options_editor"
                if editor_key not in ss:
                    ss[f"{qid}_options_base"] = ss.get(
                        f"{qid}_options_value", ss[f"{qid}_options_init"]
                    )
                ss[f"{qid}_options_value"] = st.data_editor(
                    ss[f"{qid}_options_base"],
                    key=editor_key,
                    num_rows="dynamic",
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Option": st.column_config.TextColumn(
                            "Option key",
                            help="Short identifier returned by the API, e.g. billing",
                            width="medium",
                        ),
                        "Description": st.column_config.TextColumn(
                            "Description",
                            help="What this option means (recommended)",
                            width="large",
                        ),
                    },
                )

            elif qtype == "score":
                st.caption(
                    ":material/tune: One level per line, ordered from **lowest** to "
                    "**highest** (2–255 levels)."
                )
                st.text_area(
                    "Rubric levels",
                    key=f"{qid}_levels",
                    height=120,
                    placeholder=(
                        "Low: minor impact\n"
                        "Medium: moderate impact\n"
                        "High: severe impact"
                    ),
                )

    with st.container(horizontal=True, horizontal_alignment="center"):
        st.button(
            "Add another question",
            icon=":material/add_circle:",
            key="add_bottom",
            type="tertiary",
            on_click=add_from_template,
            args=(None,),
        )


# ---------------------------------------------------------
# RUN PANEL (sticky, right column)
# ---------------------------------------------------------

# Re-validate now that this run's widgets have been rendered.
text_value, files_value, total_size, input_errors = check_input()
questions_value, question_errors = collect_questions()
input_ready = not input_errors
questions_ready = not question_errors


def check_row(state, title, detail):
    icon = {"ok": "✓", "bad": "!", "todo": "•"}[state]
    return (
        f'<div class="dp-check {state}"><div class="dp-check-icon">{icon}</div>'
        f"<div><b>{esc(title)}</b><span>{esc(detail)}</span></div></div>"
    )


run_clicked = False

with side_col:
    with st.container(border=True, key="run_panel"):
        st.markdown("#### :material/rocket_launch: Run request")

        parts = []
        if text_value:
            parts.append(f"{len(text_value):,} chars")
        if files_value:
            parts.append(f"{len(files_value)} file(s) · {fmt_size(total_size)}")

        rows = [
            check_row(
                "ok" if input_ready else "todo",
                "Context",
                " · ".join(parts) if parts else "Nothing added yet",
            ),
            check_row(
                "ok" if questions_ready else "todo",
                "Questions",
                f"{len(ss.question_ids)} configured"
                + ("" if questions_ready else f" · {len(question_errors)} to fix"),
            ),
        ]
        st.html("".join(rows))

        has_visual = any(
            media_kind(get_mime_type(f.name)) in ("image", "video") for f in files_value
        )
        st.selectbox(
            "Image / video resolution",
            RESOLUTIONS,
            key="resolution",
            format_func=lambda r: r.replace("_", " ").title(),
            help="Ultra high is intended for images. Use low, medium or high for video.",
            disabled=not has_visual,
        )

        ready = input_ready and questions_ready
        run_clicked = st.button(
            "Run decisions",
            icon=":material/bolt:",
            type="primary",
            width="stretch",
            disabled=not ready,
            key="execute_decisions",
        )

        if not ready:
            issues = input_errors + question_errors
            with st.expander(f"{len(issues)} item(s) to complete", expanded=True, icon=":material/info:"):
                for issue in issues[:8]:
                    st.caption(f"• {issue}")
                if len(issues) > 8:
                    st.caption(f"…and {len(issues) - 8} more")
        else:
            runs_left = max(MAX_RUNS_PER_SESSION - ss.run_count, 0)
            st.caption(
                f"Sends {len(questions_value)} question(s) to `{MODEL}` in one request. "
                f"{runs_left} run(s) left in this session."
            )

        if ss.last_error:
            st.error(ss.last_error["message"], icon=":material/error:")
            if ss.last_error.get("details"):
                with st.expander("Technical details"):
                    st.code(ss.last_error["details"], language=None)

        if run_clicked:
            ss.last_error = None
            limit_message = try_reserve_run()
            if limit_message:
                ss.last_error = {"message": limit_message}
                st.rerun()
            try:
                with st.status("Evaluating your questions…", expanded=True) as status:
                    st.write(":material/inventory_2: Packaging input…")
                    api_input = build_input(text_value, files_value, ss.resolution)

                    st.write(f":material/send: Calling `{MODEL}`…")
                    client = genai.Client(
                        api_key=API_KEY,
                        http_options={"api_version": API_VERSION},
                    )
                    started = time.perf_counter()
                    result = client.decisions.create(
                        model=MODEL,
                        input=api_input,
                        questions=questions_value,
                    )
                    latency = time.perf_counter() - started

                    status.update(label="Done", state="complete", expanded=False)

                ss.last_response = result.model_dump(mode="json", exclude_none=True)
                ss.last_request = questions_value
                ss.last_meta = {
                    "latency": latency,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "files": len(files_value),
                }
                ss.pending_tab = TAB_RESULTS
                st.rerun()

            except ValueError as error:
                ss.last_error = {"message": str(error)}
                st.rerun()
            except Exception as error:  # noqa: BLE001 — surface any API failure
                ss.last_error = {
                    "message": "The API request failed. See details below.",
                    "details": f"{type(error).__name__}: {error}",
                }
                st.rerun()


# ---------------------------------------------------------
# RESULTS TAB
# ---------------------------------------------------------

def confidence_html(confidence):
    if confidence is None:
        return ""
    value = min(max(float(confidence), 0.0), 1.0)
    level = "high" if value >= 0.8 else "mid" if value >= 0.5 else "low"
    word = {"high": "High", "mid": "Moderate", "low": "Low"}[level]
    return f"""
<div class="dp-conf">
  <div class="dp-conf-top"><span>Confidence</span><b class="{level}">{word} · {value:.1%}</b></div>
  <div class="dp-conf-track"><div class="dp-conf-fill {level}" style="width:{value * 100:.1f}%"></div></div>
</div>"""


def probs_html(rows, title="Probabilities"):
    """rows: list of (label, probability, selected)."""
    if not rows:
        return ""
    out = [f'<div class="dp-probs"><div class="dp-probs-title">{esc(title)}</div>']
    for label, prob, selected in rows:
        pct = min(max(prob, 0.0), 1.0) * 100
        cls = "dp-pb sel" if selected else "dp-pb"
        out.append(
            f'<div class="{cls}"><div class="dp-pb-label" title="{esc(label)}">{esc(label)}</div>'
            f'<div class="dp-pb-track"><div class="dp-pb-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="dp-pb-val">{pct:.1f}%</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def answer_card_html(key, answer, request_question):
    qtype = answer.get("type", "unknown")
    instructions = (request_question or {}).get("instructions", "")
    criteria = (request_question or {}).get("criteria")

    head = (
        f'<div class="dp-card-head"><span class="dp-card-key">{esc(key)}</span>'
        f'<span class="dp-type {esc(qtype)}">{esc(DECISION_TYPES.get(qtype, {}).get("label", qtype))}</span></div>'
    )
    if instructions:
        head += f'<p class="dp-instr">{esc(instructions)}</p>'

    body = ""

    if qtype == "noul":
        yes = as_float(answer.get("noul"))
        is_yes = yes >= 0.5
        body += (
            f'<div class="dp-verdict {"yes" if is_yes else "no"}">{"Yes" if is_yes else "No"}</div>'
        )
        if isinstance(criteria, dict):
            hint = criteria.get("true_description" if is_yes else "false_description")
            if hint:
                body += f'<div class="dp-verdict-sub">{esc(hint)}</div>'
        body += confidence_html(answer.get("confidence"))
        body += probs_html(
            [("Yes", yes, is_yes), ("No", 1 - yes, not is_yes)],
            title="Probability",
        )

    elif qtype == "choice":
        choice = answer.get("choice", "N/A")
        body += f'<div class="dp-verdict">{esc(choice)}</div>'
        if isinstance(criteria, dict) and criteria.get(str(choice)):
            body += f'<div class="dp-verdict-sub">{esc(criteria[str(choice)])}</div>'
        body += confidence_html(answer.get("confidence"))
        probabilities = answer.get("probabilities") or {}
        if isinstance(probabilities, dict):
            rows = sorted(
                ((str(k), as_float(v), str(k) == str(choice)) for k, v in probabilities.items()),
                key=lambda r: r[1],
                reverse=True,
            )
            body += probs_html(rows)

    elif qtype == "score":
        score = answer.get("score")
        legend = answer.get("legend") or {}
        probabilities = answer.get("probabilities") or {}

        score_text = f"{as_float(score):.2f}" if score is not None else "N/A"
        body += f'<div class="dp-verdict">{score_text}</div>'

        top_label = None
        if isinstance(probabilities, dict) and probabilities:
            top_key = max(probabilities, key=lambda k: as_float(probabilities[k]))
            top_label = legend.get(str(top_key), str(top_key)) if isinstance(legend, dict) else str(top_key)
            body += f'<div class="dp-verdict-sub">Most likely: {esc(top_label)}</div>'

        # Position the weighted score on the rubric scale when keys are numeric.
        try:
            numeric_keys = sorted(float(k) for k in probabilities)
            low, high = numeric_keys[0], numeric_keys[-1]
            if score is not None and high > low:
                pos = (as_float(score) - low) / (high - low) * 100
                pos = min(max(pos, 0), 100)
                body += (
                    f'<div class="dp-scale"><div class="dp-scale-pin" style="left:{pos:.1f}%"></div></div>'
                    f'<div class="dp-scale-ends"><span>Lowest</span><span>Highest</span></div>'
                )
        except (TypeError, ValueError):
            pass

        body += confidence_html(answer.get("confidence"))
        if isinstance(probabilities, dict):
            rows = []
            for k, v in probabilities.items():
                label = legend.get(str(k), str(k)) if isinstance(legend, dict) else str(k)
                rows.append((label, as_float(v), label == top_label))
            body += probs_html(rows, title="Probability by level")

    else:
        body += f'<div class="dp-verdict">{esc(qtype)}</div>'
        body += confidence_html(answer.get("confidence"))

    return f'<div class="dp-card">{head}{body}</div>'


def render_usage(usage):
    by_modality = usage.get("input_tokens_by_modality")
    if not by_modality:
        return
    st.markdown("**Input tokens by modality**")
    if isinstance(by_modality, dict):
        df = pd.DataFrame(
            [{"Modality": k, "Tokens": v} for k, v in by_modality.items()]
        )
    elif isinstance(by_modality, list) and all(isinstance(i, dict) for i in by_modality):
        df = pd.DataFrame(by_modality)
    else:
        st.json(by_modality)
        return
    st.dataframe(df, hide_index=True, width="stretch")


with tab_results:
    st.space("small")
    response = ss.last_response

    if not response:
        st.html(
            """
<div class="dp-empty">
  <div class="dp-empty-icon">📊</div>
  <h4>No results yet</h4>
  <p>Add some context, define at least one question, then press
     <b>Run decisions</b>. Your answers will appear here.</p>
</div>"""
        )
    else:
        answers = response.get("answers") or {}
        meta = ss.last_meta or {}
        request = ss.last_request or {}

        with st.container(horizontal=True, vertical_alignment="bottom"):
            section(
                "Decision results",
                f"Completed at {meta.get('timestamp', '—')}",
            )
            st.space("stretch")
            st.download_button(
                "Download JSON",
                icon=":material/download:",
                data=json.dumps(response, indent=2, ensure_ascii=False),
                file_name="decisions_response.json",
                mime="application/json",
            )
            if st.button("Clear", icon=":material/close:", type="tertiary"):
                ss.last_response = None
                ss.last_request = None
                ss.last_meta = None
                st.rerun()

        usage = response.get("usage") or {}
        confidences = [
            as_float(a.get("confidence"))
            for a in (answers.values() if isinstance(answers, dict) else [])
            if isinstance(a, dict) and a.get("confidence") is not None
        ]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Questions answered", len(answers) if isinstance(answers, dict) else "—", border=True)
        k2.metric(
            "Avg. confidence",
            f"{sum(confidences) / len(confidences):.1%}" if confidences else "—",
            border=True,
        )
        k3.metric("Total tokens", f"{usage['total_tokens']:,}" if isinstance(usage.get("total_tokens"), int) else usage.get("total_tokens", "—"), border=True)
        k4.metric("Latency", f"{meta['latency']:.2f}s" if meta.get("latency") else "—", border=True)

        st.space("small")

        if not answers:
            st.warning("No answers were returned. Check the raw response below.", icon=":material/warning:")
        elif not isinstance(answers, dict):
            st.json(answers)
        else:
            items = list(answers.items())
            per_row = 2 if len(items) > 1 else 1
            for start in range(0, len(items), per_row):
                cols = st.columns(per_row)
                for col, (key, answer) in zip(cols, items[start:start + per_row]):
                    with col:
                        if isinstance(answer, dict):
                            st.html(answer_card_html(key, answer, request.get(key)))
                        else:
                            st.write(answer)

        st.space("small")

        with st.expander("Token usage", icon=":material/data_usage:"):
            if usage:
                u1, u2, u3 = st.columns(3)
                u1.metric("Input tokens", usage.get("total_input_tokens", "N/A"))
                u2.metric("Output tokens", usage.get("total_output_tokens", "N/A"))
                u3.metric("Total tokens", usage.get("total_tokens", "N/A"))
                render_usage(usage)
            else:
                st.caption("Token usage was not included in the response.")

        with st.expander("Raw API response", icon=":material/data_object:"):
            st.json(response, expanded=2)


# ---------------------------------------------------------
# VIDEO GUIDES TAB
# ---------------------------------------------------------

def show_video(guide):
    st.markdown(f"**{guide['title']}**")
    st.caption(guide["caption"])
    if guide["url"]:
        st.video(guide["url"])
    else:
        st.html(
            '<div class="dp-empty" style="padding:32px 20px"><div class="dp-empty-icon">🎬</div>'
            "<h4>Video coming soon</h4></div>"
        )


with tab_guides:
    st.space("small")
    section(
        "Video guides",
        "Short narrated walkthroughs recorded in this playground with real model answers.",
    )
    st.space("small")
    show_video(VIDEO_GUIDES["overview"])
    st.divider()
    section("Step-by-step demos", "Pick a decision type to see the full flow and two quick examples.")
    st.segmented_control(
        "Decision type demo",
        options=list(DECISION_TYPES),
        format_func=lambda x: DECISION_TYPES[x]["label"],
        key="guide_type",
        required=True,
        label_visibility="collapsed",
    )
    show_video(VIDEO_GUIDES[ss.guide_type])


# =========================================================
# FOOTER
# =========================================================

st.space("medium")
st.caption(f"Gemini Decisions Playground · Powered by `{MODEL}` · API `{API_VERSION}`")
