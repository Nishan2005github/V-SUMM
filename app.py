# ----------------- IMPORTS ----------------- #
import os
import gc
import uuid
import time
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import openai

import nltk
import requests
import yt_dlp
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

for _pkg in ('punkt', 'punkt_tab', 'stopwords'):
    try:
        nltk.data.find(f'tokenizers/{_pkg}' if 'punkt' in _pkg else f'corpora/{_pkg}')
    except LookupError:
        nltk.download(_pkg, quiet=True)

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(_env_path)

GROQ_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
OPENROUTER_KEY = (os.getenv("OPENROUTER_API_KEY") or GROQ_KEY or "").strip()

if not GROQ_KEY and OPENROUTER_KEY.startswith("gsk_"):
    GROQ_KEY = OPENROUTER_KEY

if OPENROUTER_KEY.startswith("gsk_"):
    AI_BASE_URL = "https://api.groq.com/openai/v1"
    AI_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    print(f"[AI] Groq API key loaded OK ({OPENROUTER_KEY[:8]}...)")
elif OPENROUTER_KEY:
    AI_BASE_URL = "https://openrouter.ai/api/v1"
    AI_MODELS = [
        "openai/gpt-oss-20b",
        "openrouter/free",
    ]
    print(f"[AI] OpenRouter API key loaded OK ({OPENROUTER_KEY[:8]}...)")
else:
    AI_BASE_URL = "https://openrouter.ai/api/v1"
    AI_MODELS = []
    print("[AI] WARNING: OPENROUTER_API_KEY / GROQ_API_KEY not found in environment")

# ----------------- FLASK & DATABASE CONFIG ----------------- #
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['FRAMES_FOLDER'] = os.path.join(BASE_DIR, 'static', 'frames')
app.config['AUDIO_FOLDER'] = os.path.join(BASE_DIR, 'static', 'audio')

for _folder in (app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER'], app.config['AUDIO_FOLDER']):
    os.makedirs(_folder, exist_ok=True)

db = SQLAlchemy(app)

# ----------------- QUALITY / SPEED PRESETS ----------------- #
WHISPER_MODELS = ["tiny", "base", "small", "medium"]
VIDEO_QUALITIES = {"480": 480, "720": 720, "1080": 1080, "best": None}
SUMMARY_LENGTHS = {"short": 2, "medium": 4, "detailed": 7}
PART_COUNTS = {"3": 3, "5": 5, "8": 8}
BEAM_SIZES = {"1": 1, "3": 3, "5": 5}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "ru": "Russian", "pt": "Portuguese", "bn": "Bengali",
    "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "ur": "Urdu",
    "english": "English", "hindi": "Hindi", "spanish": "Spanish", "french": "French",
    "german": "German", "chinese": "Chinese", "japanese": "Japanese", "korean": "Korean",
    "arabic": "Arabic", "russian": "Russian", "portuguese": "Portuguese", "bengali": "Bengali",
    "tamil": "Tamil", "telugu": "Telugu", "marathi": "Marathi", "urdu": "Urdu",
}

QUALITY_PRESETS = {
    "fast":     {"model_size": "tiny", "video_quality": "480", "summary_length": "short",    "num_parts": "3", "beam_size": "1"},
    "balanced": {"model_size": "tiny", "video_quality": "480", "summary_length": "medium",   "num_parts": "5", "beam_size": "1"},
    "high":     {"model_size": "base", "video_quality": "720", "summary_length": "detailed", "num_parts": "5", "beam_size": "3"},
}

# ----------------- AI HELPER (SAFE NONETYPE HANDLING & FALLBACK) ----------------- #
def call_ai_llm(system_prompt, user_prompt, max_tokens=800, temperature=0.3):
    if not OPENROUTER_KEY:
        return None
    client = openai.OpenAI(api_key=OPENROUTER_KEY, base_url=AI_BASE_URL)
    for model_name in AI_MODELS:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if res and res.choices and len(res.choices) > 0:
                msg = res.choices[0].message
                content = getattr(msg, "content", None) or getattr(msg, "reasoning", None) or ""
                content = content.strip()
                if content:
                    return content
        except Exception as e:
            print(f"[AI Warning - {model_name}] {e}")
            continue
    return None


def ai_summarize(text, length="medium"):
    if not text:
        return None
    detail = {
        "short":    "2-3 clear sentences",
        "medium":   "one well-structured paragraph",
        "detailed": "3-4 detailed paragraphs covering all key points",
    }.get(length, "one well-structured paragraph")

    return call_ai_llm(
        system_prompt=(
            "You are an expert video content summarizer. "
            "Write clear, informative summaries in plain English. "
            "Never mention that you are an AI."
        ),
        user_prompt=f"Summarize the following video transcript in {detail}:\n\n{text[:8000]}",
        max_tokens=1200,
        temperature=0.4,
    )


def ai_summarize_part(text):
    if not text:
        return None
    return call_ai_llm(
        system_prompt="You are a concise video segment summarizer. Reply directly with a 1-2 sentence summary.",
        user_prompt=f"Summarize this video segment in 1-2 sentences:\n\n{text[:2500]}",
        max_tokens=600,
        temperature=0.3,
    )

# ----------------- GROQ CLOUD WHISPER TRANSCRIPTION (ZERO LOCAL RAM) ----------------- #
def transcribe_audio_groq(audio_path, task="transcribe", progress_callback=None):
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add GROQ_API_KEY in Render Environment Variables.")

    if progress_callback:
        progress_callback(1, 4)

    compressed_mp3 = audio_path.rsplit(".", 1)[0] + "_groq.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
            compressed_mp3,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if progress_callback:
            progress_callback(2, 4)

        endpoint = (
            "https://api.groq.com/openai/v1/audio/translations"
            if task == "translate"
            else "https://api.groq.com/openai/v1/audio/transcriptions"
        )
        model_name = "whisper-large-v3" if task == "translate" else "whisper-large-v3-turbo"
        headers = {"Authorization": f"Bearer {GROQ_KEY}"}
        data = {
            "model": model_name,
            "response_format": "verbose_json",
        }

        with open(compressed_mp3, "rb") as f:
            files = {"file": (os.path.basename(compressed_mp3), f, "audio/mpeg")}
            resp = requests.post(endpoint, headers=headers, data=data, files=files, timeout=180)

        if resp.status_code != 200:
            raise RuntimeError(f"Groq Whisper API error ({resp.status_code}): {resp.text}")

        if progress_callback:
            progress_callback(4, 4)

        payload = resp.json()
        raw_segments = payload.get("segments") or []
        segments_data = []
        text_parts = []
        for seg in raw_segments:
            seg_text = (seg.get("text") or "").strip()
            if seg_text:
                segments_data.append({
                    "start": round(float(seg.get("start", 0.0)), 2),
                    "end": round(float(seg.get("end", 0.0)), 2),
                    "text": seg_text,
                })
                text_parts.append(seg_text)

        transcript_text = (payload.get("text") or "").strip() or " ".join(text_parts).strip()
        detected_language = (payload.get("language") or "en").lower()
        return transcript_text, segments_data, detected_language
    finally:
        if os.path.exists(compressed_mp3):
            try:
                os.remove(compressed_mp3)
            except Exception:
                pass


def format_hms(seconds):
    seconds = int(seconds or 0)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# ----------------- DATABASE MODELS ----------------- #
class VideoRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    video_path = db.Column(db.String(255), nullable=False)
    transcript = db.Column(db.Text, nullable=True)
    summary = db.Column(db.Text, nullable=True)
    detected_language = db.Column(db.String(10), nullable=True)
    original_transcript = db.Column(db.Text, nullable=True)


class VideoSegment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video_record.id'), nullable=False)
    part_title = db.Column(db.String(100))
    summary = db.Column(db.Text)
    slide_path = db.Column(db.String(255))
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)


class TimestampSegment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video_record.id'), nullable=False)
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)
    text = db.Column(db.Text)


def migrate_schema():
    with app.app_context():
        db.create_all()
        inspector = db.inspect(db.engine)
        if 'video_record' in inspector.get_table_names():
            existing_cols = {c['name'] for c in inspector.get_columns('video_record')}
            with db.engine.connect() as conn:
                if 'detected_language' not in existing_cols:
                    conn.execute(db.text("ALTER TABLE video_record ADD COLUMN detected_language VARCHAR(10)"))
                if 'original_transcript' not in existing_cols:
                    conn.execute(db.text("ALTER TABLE video_record ADD COLUMN original_transcript TEXT"))
                conn.commit()


migrate_schema()

# ----------------- IN-MEMORY JOB TRACKER ----------------- #
JOBS = {}
JOBS_LOCK = threading.Lock()


def set_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def new_job():
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "stage": "Queued",
            "progress": 0,
            "result": None,
            "error": None,
        }
    return job_id

# ----------------- PROCESSING MODULES ----------------- #
def download_video(url, job_id, height=480):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_source.mp4")
    if "youtube.com" in url or "youtu.be" in url:
        clean_url = url.split("&si=")[0].split("?si=")[0]
        height_filter = f"[height<={height}]" if height else "[height<=480]"

        def ydl_hook(d):
            if d.get('status') == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                downloaded = d.get('downloaded_bytes', 0)
                if total > 0:
                    pct = min(14, max(5, 5 + int((downloaded / total) * 9)))
                    set_job(job_id, progress=pct, stage=f"Downloading video ({int(downloaded * 100 / total)}%)")

        ydl_opts = {
            'format': f'best{height_filter}[ext=mp4]/bestvideo{height_filter}[ext=mp4]+bestaudio[ext=m4a]/best{height_filter}/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'concurrent_fragment_downloads': 4,
            'merge_output_format': 'mp4',
            'progress_hooks': [ydl_hook],
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'web'],
                }
            },
        }

        cookies_txt = os.getenv("YOUTUBE_COOKIES", "").strip()
        cookie_file = None
        if cookies_txt:
            cookie_file = os.path.join(app.config['AUDIO_FOLDER'], f"{job_id}_cookies.txt")
            with open(cookie_file, "w", encoding="utf-8") as cf:
                cf.write(cookies_txt)
            ydl_opts['cookiefile'] = cookie_file

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([clean_url])
        finally:
            if cookie_file and os.path.exists(cookie_file):
                try:
                    os.remove(cookie_file)
                except Exception:
                    pass
    else:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    return output_path


def extract_audio_ffmpeg(video_path, audio_path):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_summary(text, num_sentences=4):
    if not text or not text.strip():
        return "No speech detected in the video."
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in word_tokenize(text) if w.isalnum() and w.lower() not in stop_words]
    word_freq = Counter(words)
    scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                scores[sentence] = scores.get(sentence, 0) + word_freq[word]
    ranked = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    ranked_set = set(ranked)
    return " ".join(s for s in sentences if s in ranked_set)


def capture_frame_at_timestamp(video_path, timestamp_sec, tag):
    if not video_path or not os.path.exists(video_path):
        return ""
    filename = f"{tag}_{int(timestamp_sec)}.jpg"
    frame_path = os.path.join(app.config['FRAMES_FOLDER'], filename)
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(max(0.0, float(timestamp_sec))),
            "-i", video_path,
            "-vframes", "1", "-q:v", "4",
            frame_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(frame_path):
        return f"/static/frames/{filename}"
    return ""


def run_pipeline(job_id, video_source, options):
    app_ctx = app.app_context()
    app_ctx.push()
    try:
        num_sentences      = SUMMARY_LENGTHS[options["summary_length"]]
        num_parts          = PART_COUNTS[options["num_parts"]]
        task               = "translate" if options["translate_to_english"] else "transcribe"
        summary_length_key = options["summary_length"]

        set_job(job_id, status="running", stage="Extracting audio", progress=15)
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], f"{job_id}.wav")
        extract_audio_ffmpeg(video_source, audio_path)

        def report_progress(step, total_steps):
            frac = min(step / total_steps, 1.0) if total_steps else 0
            pct  = 25 + int(frac * 35)
            set_job(job_id, progress=pct, stage="Transcribing via Groq Cloud Whisper...")

        set_job(job_id, stage="Transcribing via Groq Cloud Whisper", progress=25)
        transcript_text, segments_data, detected_language = transcribe_audio_groq(
            audio_path, task=task, progress_callback=report_progress
        )

        original_transcript_text = None
        if options["keep_original_too"] and task == "translate":
            set_job(job_id, stage="Also generating original-language transcript", progress=55)
            original_transcript_text, _, _ = transcribe_audio_groq(audio_path, task="transcribe")

        set_job(job_id, stage="Generating AI summary", progress=68)
        summary_text = (
            ai_summarize(transcript_text, length=summary_length_key)
            or generate_summary(transcript_text, num_sentences=max(num_sentences, 3))
        )

        video_rec = VideoRecord(
            title="Processed Video",
            video_path=f"/static/uploads/{os.path.basename(video_source)}",
            transcript=transcript_text,
            summary=summary_text,
            detected_language=detected_language,
            original_transcript=original_transcript_text,
        )
        db.session.add(video_rec)
        db.session.commit()

        for seg in segments_data:
            db.session.add(TimestampSegment(
                video_id=video_rec.id,
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg["text"],
            ))
        db.session.commit()

        set_job(job_id, stage="Extracting key-scene slides", progress=80)
        total_duration = segments_data[-1]["end"] if segments_data else 0
        part_duration  = total_duration / num_parts if num_parts else 0
        parts_data     = [None] * num_parts

        def build_part(i):
            s_time, e_time = i * part_duration, (i + 1) * part_duration
            part_text = " ".join(s["text"] for s in segments_data if s_time <= s["start"] < e_time)
            if not part_text.strip():
                return None
            p_summary = (
                ai_summarize_part(part_text)
                or generate_summary(part_text, num_sentences=2)
            )
            mid_ts    = (s_time + e_time) / 2
            slide_url = capture_frame_at_timestamp(video_source, mid_ts, f"{job_id}_slide")
            p_title   = (
                f"Part {i+1} ({int(s_time//60):02d}:{int(s_time%60):02d} - "
                f"{int(e_time//60):02d}:{int(e_time%60):02d})"
            )
            return {
                "index": i, "title": p_title, "summary": p_summary,
                "slide": slide_url, "start": s_time, "end": e_time,
            }

        with ThreadPoolExecutor(max_workers=2) as pool:
            for res in pool.map(build_part, range(num_parts)):
                if res:
                    parts_data[res["index"]] = res

        parts_data = [p for p in parts_data if p]

        for p in parts_data:
            db.session.add(VideoSegment(
                video_id=video_rec.id, part_title=p["title"], summary=p["summary"],
                slide_path=p["slide"], start_time=p["start"], end_time=p["end"],
            ))
        db.session.commit()

        set_job(job_id, stage="Finalizing", progress=95)
        set_job(
            job_id, status="done", stage="Complete", progress=100,
            result={
                "video_id":               video_rec.id,
                "video_path":             video_rec.video_path,
                "summary":                summary_text,
                "parts":                  [{"title": p["title"], "summary": p["summary"], "slide": p["slide"]} for p in parts_data],
                "transcript":             transcript_text,
                "segments":               [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments_data],
                "detected_language":      detected_language,
                "detected_language_name": LANGUAGE_NAMES.get(detected_language, detected_language),
                "task":                   task,
                "original_transcript":    original_transcript_text,
            },
        )

    except Exception as exc:
        traceback.print_exc()
        set_job(job_id, status="error", stage="Failed", error=str(exc))
    finally:
        # 1. Always delete temporary audio immediately
        audio_file = os.path.join(app.config['AUDIO_FOLDER'], f"{job_id}.wav")
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception:
                pass

        # 2. Purge older videos/frames from previous runs while keeping current job_id for playback/download
        for folder in (app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER']):
            try:
                for fname in os.listdir(folder):
                    if not fname.startswith(job_id) and fname != ".gitkeep":
                        fpath = os.path.join(folder, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
            except Exception:
                pass

        gc.collect()
        app_ctx.pop()


# ----------------- ROUTING ENDPOINTS ----------------- #
@app.route('/')
def index():
    return render_template(
        'index.html',
        whisper_models=WHISPER_MODELS,
        video_qualities=list(VIDEO_QUALITIES.keys()),
        summary_lengths=list(SUMMARY_LENGTHS.keys()),
        part_counts=list(PART_COUNTS.keys()),
        beam_sizes=list(BEAM_SIZES.keys()),
        presets=QUALITY_PRESETS,
    )


@app.route('/api/process', methods=['POST'])
def process_video():
    job_id  = new_job()
    options = {
        "model_size":           request.form.get("model_size", "tiny"),
        "video_quality":        request.form.get("video_quality", "480"),
        "summary_length":       request.form.get("summary_length", "medium"),
        "num_parts":            request.form.get("num_parts", "5"),
        "beam_size":            request.form.get("beam_size", "1"),
        "translate_to_english": request.form.get("translate_to_english", "false") == "true",
        "keep_original_too":    request.form.get("keep_original_too", "false") == "true",
    }
    if options["model_size"]     not in WHISPER_MODELS:  options["model_size"]     = "tiny"
    if options["video_quality"]  not in VIDEO_QUALITIES: options["video_quality"]  = "480"
    if options["summary_length"] not in SUMMARY_LENGTHS: options["summary_length"] = "medium"
    if options["num_parts"]      not in PART_COUNTS:     options["num_parts"]      = "5"
    if options["beam_size"]      not in BEAM_SIZES:      options["beam_size"]      = "1"

    file = request.files.get('file')
    url  = (request.form.get('url') or '').strip()
    if not file and not url:
        return jsonify({"error": "No valid video provided"}), 400

    def worker():
        try:
            set_job(job_id, status="running", stage="Downloading video", progress=5)
            video_source = download_video(url, job_id, VIDEO_QUALITIES[options["video_quality"]])
            run_pipeline(job_id, video_source, options)
        except Exception as exc:
            traceback.print_exc()
            set_job(job_id, status="error", stage="Failed", error=str(exc))

    if file and file.filename:
        video_source = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_uploaded.mp4")
        file.save(video_source)
        set_job(job_id, status="running", stage="Uploaded, starting pipeline", progress=8)
        threading.Thread(target=run_pipeline, args=(job_id, video_source, options), daemon=True).start()
    else:
        threading.Thread(target=worker, daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route('/api/status/<job_id>')
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route('/api/search', methods=['POST'])
def search_keyword():
    data     = request.get_json(silent=True) or {}
    video_id = data.get("video_id")
    query    = (data.get("query") or "").strip().lower()
    if not query or not video_id:
        return jsonify({"results": []})

    video_rec = db.session.get(VideoRecord, video_id)
    if not video_rec:
        return jsonify({"results": []})

    matches = TimestampSegment.query.filter(
        TimestampSegment.video_id == video_id,
        TimestampSegment.text.ilike(f"%{query}%")
    ).all()

    abs_video_path = os.path.join(BASE_DIR, video_rec.video_path.lstrip('/'))
    results = []
    for m in matches[:5]:
        slide_url = capture_frame_at_timestamp(abs_video_path, m.start_time, f"search_{video_id}")
        time_str  = f"{int(m.start_time // 60):02d}:{int(m.start_time % 60):02d}"
        results.append({
            "timestamp":  time_str,
            "start_time": m.start_time,
            "text":       m.text,
            "slide_url":  slide_url,
        })
    return jsonify({"results": results})


# ----------------- AI Q&A ENDPOINT ----------------- #
@app.route('/api/ask', methods=['POST'])
def ask_question():
    data     = request.get_json(silent=True) or {}
    video_id = data.get("video_id")
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not OPENROUTER_KEY:
        return jsonify({"error": "API key not configured in environment"}), 500

    video_rec = db.session.get(VideoRecord, video_id) if video_id else VideoRecord.query.order_by(VideoRecord.id.desc()).first()
    if not video_rec:
        return jsonify({"error": "Video not found"}), 404

    transcript = (video_rec.transcript or "").strip()
    if not transcript:
        return jsonify({"answer": "This video has no speech transcript to answer from."})

    try:
        answer = call_ai_llm(
            system_prompt=(
                "You are an insightful, knowledgeable video assistant. Use the provided video summary and transcript as primary context to understand the video's topic. "
                "First reference any relevant details mentioned in the transcript. "
                "If a specific keyword, fact, or deeper explanation is not directly in the transcript, seamlessly expand on the topic using your own general knowledge as long as it relates to the video's subject or question—never refuse with 'the transcript does not provide this information'. "
                "Provide clear, detailed, and helpful answers."
            ),
            user_prompt=(
                f"Video Summary:\n{(video_rec.summary or '')}\n\n"
                f"Video Transcript:\n{transcript[:10000]}\n\n"
                f"User Question: {question}"
            ),
            max_tokens=1500,
            temperature=0.5,
        )
        if not answer:
            fallback_sents = [s for s in sent_tokenize(transcript) if any(w in s.lower() for w in question.lower().split() if len(w) > 3)]
            if fallback_sents:
                answer = "Based on the transcript: " + " ".join(fallback_sents[:3])
            else:
                answer = f"Summary context: {(video_rec.summary or transcript[:400]).strip()}"
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=False, use_reloader=False, port=port, threaded=True)
