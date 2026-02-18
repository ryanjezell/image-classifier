"""
app.py — Gradio Web UI for the Image Classifier
═══════════════════════════════════════════════════════════════════════════════

Tabs:
  1 · Configure    — Set Drive paths, class names, model settings
  2 · Data Setup   — Extract thumbnails from video files OR copy images
  3 · Train        — Two-phase transfer learning with live log
  4 · Sort         — Classify a mixed folder, route files to subfolders
  5 · Predict      — Upload one image and see instant prediction
  6 · Evaluate     — Full P/R/F1 report

VIDEO THUMBNAIL SUPPORT
───────────────────────
Your Drive folders contain video files (.mp4 .mov .avi .mkv etc.).
This app extracts the thumbnail from each video and uses it as the
training/classification image.

Extraction order (fastest → most accurate):
  1. Embedded thumbnail in file metadata  (what Windows Explorer shows)
  2. Frame at 10% of video duration       (fallback if no metadata thumb)

Requires (auto-installed on first run):
  opencv-python, mutagen

Launch in Colab:
  !pip install fastai gradio opencv-python mutagen --quiet
  !python app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, sys, csv, json, shutil, traceback, tempfile, subprocess
from pathlib import Path
from datetime import datetime

# ── Gradio ────────────────────────────────────────────────────────────────────
import gradio as gr

# Add project root so src/ modules are importable
sys.path.insert(0, str(Path(__file__).parent))


# ══════════════════════════════════════════════════════════════════════════════
# FILE TYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

VIDEO_EXTS = {
    '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.webm', '.ts',
}

ALL_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

STATE = {
    "config": {
        "class1_name":    "class_1",
        "class1_source":  "",
        "class2_name":    "class_2",
        "class2_source":  "",
        "class3_name":    "class_3",
        "class3_source":  "",
        "unsorted_folder":      "",
        "sorted_output":        "",
        "confidence_threshold": 0.70,
        "batch_size":           32,
        "image_size":           224,
        "architecture":         "resnet50",
        "head_epochs":          4,
        "finetune_epochs":      10,
    },
    "model_ready":   False,
    "dataset_ready": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 12px;
    color: white !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
}
.app-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 4px 0; }
.app-header p  { font-size: 0.95rem; opacity: 0.72; margin: 0; }
.section-title {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #64748b; margin-bottom: 12px;
}
.log-box textarea {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 0.78rem !important;
    background: #0f172a !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
}
.primary-btn button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important; color: white !important;
    font-weight: 600 !important; border-radius: 8px !important;
}
.warn-box {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.88rem; color: #92400e;
}
.info-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.88rem; color: #1e40af; margin-bottom: 14px;
}
"""

HEADER_HTML = """
<div class="app-header">
  <h1>🎬 Video Thumbnail Classifier</h1>
  <p>Extracts thumbnails from video files · Transfer learning · ResNet-50 · fast.ai + PyTorch</p>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# THUMBNAIL EXTRACTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_cv2():
    """Lazily imports cv2, auto-installing if missing."""
    try:
        import cv2
        return cv2
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "opencv-python", "--quiet"],
            check=True
        )
        import cv2
        return cv2


def extract_embedded_thumbnail(video_path: Path):
    """
    Extracts the thumbnail baked into a video file's metadata.

    Most phone/camera/YouTube MP4 and MOV files store a JPEG thumbnail
    in the 'covr' metadata atom — this is exactly what Windows Explorer
    and Google Drive show as the file preview icon.

    Uses mutagen to read the atom without decoding the video stream,
    making this nearly instantaneous regardless of video size.

    Returns a PIL Image (RGB) or None if no embedded thumbnail exists.
    """
    # Strategy A: mutagen MP4 covr atom
    try:
        from mutagen.mp4 import MP4
        from PIL import Image
        import io
        tags = MP4(str(video_path))
        if 'covr' in tags:
            img = Image.open(io.BytesIO(bytes(tags['covr'][0])))
            return img.convert('RGB')
    except Exception:
        pass

    # Strategy B: mutagen generic (covers MOV, M4V etc.)
    try:
        import mutagen
        from PIL import Image
        import io
        f = mutagen.File(str(video_path))
        if f and hasattr(f, 'tags') and f.tags:
            for key in f.tags.keys():
                if 'covr' in str(key).lower() or 'apic' in str(key).lower():
                    data = f.tags[key]
                    raw  = data.data if hasattr(data, 'data') else bytes(data[0])
                    img  = Image.open(io.BytesIO(raw))
                    return img.convert('RGB')
    except Exception:
        pass

    return None


def extract_frame_thumbnail(video_path: Path, pct: float = 0.10):
    """
    Reads the video stream and grabs the frame at `pct` of total duration.

    WHY 10%: Frame 0 is almost always black or a camera-initialisation
    artefact. 10% gives the first meaningful scene.

    Returns a PIL Image (RGB) or None on failure.
    """
    cv2 = _ensure_cv2()
    from PIL import Image
    import numpy as np

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * pct)))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()


def get_thumbnail(file_path: Path, save_to: Path = None):
    """
    Master dispatcher:
      - Image file  → load directly
      - Video file  → embedded metadata thumbnail → frame at 10%

    If save_to is given, writes the result as JPEG to that path.
    Returns PIL Image or None.
    """
    from PIL import Image
    suffix = file_path.suffix.lower()

    if suffix in IMAGE_EXTS:
        try:
            img = Image.open(file_path).convert('RGB')
            if save_to:
                img.save(save_to, 'JPEG', quality=95)
            return img
        except Exception:
            return None

    if suffix in VIDEO_EXTS:
        img = extract_embedded_thumbnail(file_path)
        if img is None:
            img = extract_frame_thumbnail(file_path)
        if img is not None:
            if save_to:
                img.save(save_to, 'JPEG', quality=95)
            return img

    return None


def count_media_files(folder: str) -> dict:
    """Returns counts of images, videos, and total in a folder."""
    p = Path(folder)
    if not p.exists():
        return {'images': 0, 'videos': 0, 'total': 0}
    counts = {'images': 0, 'videos': 0, 'total': 0}
    for f in p.iterdir():
        if not f.is_file():
            continue
        s = f.suffix.lower()
        if s in IMAGE_EXTS:
            counts['images'] += 1
            counts['total']  += 1
        elif s in VIDEO_EXTS:
            counts['videos'] += 1
            counts['total']  += 1
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(lines, msg):
    lines.append(f"[{_ts()}]  {msg}")
    return lines[-200:]

def _log_str(lines):
    return "\n".join(lines)

def _yaml_value(key_path, value):
    import yaml
    p = Path("config/config.yaml")
    if not p.exists():
        return
    with open(p) as f:
        cfg = yaml.safe_load(f)
    target = cfg
    for k in key_path.split(".")[:-1]:
        target = target.setdefault(k, {})
    target[key_path.split(".")[-1]] = value
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _normalize_path(raw_path: str) -> Path:
    """Normalizes Windows/Linux/macOS-friendly user-provided paths."""
    return Path(raw_path.strip().strip('"').strip("'")).expanduser()


def _mkdir_if_missing(raw_path: str, label: str, warnings):
    """Creates directory if it doesn't exist; returns Path or None for empty input."""
    if not raw_path or not raw_path.strip():
        return None
    path = _normalize_path(raw_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        warnings.append(f"Created missing folder for {label}: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONFIGURE
# ══════════════════════════════════════════════════════════════════════════════

def save_config(c1n, c1s, c2n, c2s, c3n, c3s,
                unsorted, sorted_out,
                threshold, batch_size, image_size, arch,
                head_ep, finetune_ep):

    errors, info, warnings = [], [], []

    class_rows = [
        (c1n.strip() or "class_1", c1s),
        (c2n.strip() or "class_2", c2s),
        (c3n.strip() or "class_3", c3s),
    ]

    class_paths = []
    for name, src in class_rows:
        if not src.strip():
            errors.append(f"'{name}' source path is empty.")
            continue
        path = _mkdir_if_missing(src, f"class '{name}'", warnings)
        if path is not None:
            class_paths.append((name, path))

    unsorted_path = _mkdir_if_missing(unsorted, "unsorted input", warnings)
    sorted_path = _mkdir_if_missing(sorted_out, "sorted output", warnings)

    if errors:
        return "❌  " + "\n❌  ".join(errors)

    for name, path in class_paths:
        c = count_media_files(str(path))
        if c['total'] == 0:
            warnings.append(f"'{name}': no media files yet in {path} (add files, then run Data Setup)")
        elif c['total'] < 20:
            warnings.append(f"'{name}': only {c['total']} files (20+ recommended)")
        else:
            info.append(f"'{name}': {c['images']} images + {c['videos']} videos = {c['total']} total")

    STATE["config"].update({
        "class1_name": class_rows[0][0], "class1_source": str(_normalize_path(c1s)),
        "class2_name": class_rows[1][0], "class2_source": str(_normalize_path(c2s)),
        "class3_name": class_rows[2][0], "class3_source": str(_normalize_path(c3s)),
        "unsorted_folder":      str(unsorted_path) if unsorted_path else "",
        "sorted_output":        str(sorted_path) if sorted_path else "",
        "confidence_threshold": float(threshold),
        "batch_size":           int(batch_size),
        "image_size":           int(image_size),
        "architecture":         arch,
        "head_epochs":          int(head_ep),
        "finetune_epochs":      int(finetune_ep),
    })

    try:
        _yaml_value("data.batch_size",         int(batch_size))
        _yaml_value("data.image_size",          int(image_size))
        _yaml_value("model.architecture",       arch)
        _yaml_value("training.head_epochs",     int(head_ep))
        _yaml_value("training.finetune_epochs", int(finetune_ep))
    except Exception as e:
        warnings.append(f"Could not update config.yaml: {e}")

    lines = ["✅  Configuration saved!"]
    for i in info:     lines.append(f"   📁 {i}")
    for w in warnings: lines.append(f"   ⚠️  {w}")
    return "\n".join(lines)


def build_config_tab():
    with gr.Tab("⚙️  Configure"):
        gr.HTML("""
            <div class="section-title">Local / Network Paths</div>
            <div class="info-box">
              Works with <strong>image files</strong> (.jpg .png .webp) AND
              <strong>video files</strong> (.mp4 .mov .avi .mkv etc.).<br>
              For videos, the app automatically extracts the thumbnail —
              the same preview image shown in Windows Explorer.<br>
              Missing folders are auto-created when you save this tab.
            </div>
        """)

        with gr.Group():
            with gr.Row():
                c1n = gr.Textbox(label="Class 1 Name", placeholder="e.g. ads",       scale=1)
                c1s = gr.Textbox(label="Class 1 Source Folder",
                                  placeholder="C:/datasets/videos/ads",    scale=3)
            with gr.Row():
                c2n = gr.Textbox(label="Class 2 Name", placeholder="e.g. tutorials", scale=1)
                c2s = gr.Textbox(label="Class 2 Source Folder",
                                  placeholder="C:/datasets/videos/tutorials", scale=3)
            with gr.Row():
                c3n = gr.Textbox(label="Class 3 Name", placeholder="e.g. vlogs",     scale=1)
                c3s = gr.Textbox(label="Class 3 Source Folder",
                                  placeholder="C:/datasets/videos/vlogs",  scale=3)

        gr.HTML('<div class="section-title" style="margin-top:18px">Sorting Paths</div>')
        with gr.Group():
            with gr.Row():
                unsorted   = gr.Textbox(label="Unsorted Folder (mixed input)",
                                         placeholder="C:/datasets/videos/unsorted")
                sorted_out = gr.Textbox(label="Sorted Output Folder",
                                         placeholder="C:/datasets/videos/sorted")

        gr.HTML('<div class="section-title" style="margin-top:18px">Model & Training Settings</div>')
        with gr.Group():
            with gr.Row():
                arch     = gr.Dropdown(
                    label="Architecture",
                    choices=["resnet34", "resnet50", "efficientnet_b0", "efficientnet_b3"],
                    value="resnet50",
                )
                img_size = gr.Slider(label="Image Size (px)", minimum=128, maximum=512, step=32, value=224)
                bs       = gr.Slider(label="Batch Size", minimum=8, maximum=64, step=8, value=32,
                                      info="Reduce to 8 if GPU OOM error")
            with gr.Row():
                head_ep   = gr.Slider(label="Phase 1 Epochs", minimum=1, maximum=20,  step=1, value=4)
                ftune_ep  = gr.Slider(label="Phase 2 Epochs", minimum=1, maximum=40,  step=1, value=10)
                threshold = gr.Slider(label="Confidence Threshold", minimum=0.4, maximum=0.99,
                                       step=0.01, value=0.70,
                                       info="Below this → _uncertain/ folder")

        with gr.Row(elem_classes="primary-btn"):
            save_btn = gr.Button("💾  Save Configuration", variant="primary")
        status = gr.Textbox(label="Status", lines=5, interactive=False)

        save_btn.click(
            fn=save_config,
            inputs=[c1n, c1s, c2n, c2s, c3n, c3s, unsorted, sorted_out,
                    threshold, bs, img_size, arch, head_ep, ftune_ep],
            outputs=[status],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA SETUP
# ══════════════════════════════════════════════════════════════════════════════

def run_data_setup(progress=gr.Progress(track_tqdm=True)):
    """
    For each class folder:
      - Image files  → copied directly to data/dataset/<class>/
      - Video files  → thumbnail extracted and saved as .jpg

    The resulting data/dataset/ folder contains only .jpg files.
    The training pipeline never needs to know a source was a video.
    """
    cfg   = STATE["config"]
    log   = []
    pairs = [
        (cfg["class1_name"], cfg["class1_source"]),
        (cfg["class2_name"], cfg["class2_source"]),
        (cfg["class3_name"], cfg["class3_source"]),
    ]

    if not cfg["class1_source"]:
        yield "❌  No paths configured. Go to Configure tab first.", ""
        return

    dataset_dir  = Path("data/dataset")
    total_ok     = 0
    total_failed = 0

    for i, (name, source) in enumerate(pairs):
        progress(i / len(pairs), desc=f"Processing {name}…")

        if not source:
            log = _log(log, f"SKIP  {name} — no source path")
            continue

        src  = Path(source)
        dest = dataset_dir / name
        dest.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            log = _log(log, f"ERROR {name} — folder not found: {source}")
            continue

        files = [f for f in src.iterdir()
                 if f.is_file() and f.suffix.lower() in ALL_MEDIA_EXTS]

        if not files:
            log = _log(log, f"WARN  {name} — no supported files in {source}")
            log = _log(log, f"      Supported: images {IMAGE_EXTS}")
            log = _log(log, f"                 videos {VIDEO_EXTS}")
            continue

        imgs = sum(1 for f in files if f.suffix.lower() in IMAGE_EXTS)
        vids = sum(1 for f in files if f.suffix.lower() in VIDEO_EXTS)
        log  = _log(log, f"START {name}: {imgs} images + {vids} videos = {len(files)} files")

        ok = failed = 0

        for j, file in enumerate(files):
            if j % 10 == 0:
                progress(
                    (i + j / len(files)) / len(pairs),
                    desc=f"{name}: {j}/{len(files)}"
                )

            out_path = dest / (file.stem + ".jpg")
            if out_path.exists():    # skip already-extracted
                ok += 1
                continue

            img = get_thumbnail(file, save_to=out_path)
            if img is not None:
                ok += 1
            else:
                failed += 1
                log = _log(log, f"  FAIL  {file.name}")

        total_ok     += ok
        total_failed += failed
        log = _log(log, f"DONE  {name}: {ok} thumbnails ready, {failed} failed → {dest}")
        yield f"Processing… {name} done ({ok} thumbnails)", _log_str(log)

    STATE["dataset_ready"] = total_ok > 0
    progress(1.0, desc="Done")

    summary = (
        f"✅  Dataset ready!  {total_ok} thumbnails across {len(pairs)} classes.\n"
        + (f"⚠️  {total_failed} files failed — see log for details.\n" if total_failed else "")
        + f"Location: data/dataset/"
    )
    yield summary, _log_str(log)


def build_data_tab():
    with gr.Tab("📁  Data Setup"):
        gr.HTML("""
            <div class="section-title">Extract Thumbnails & Build Dataset</div>
            <div class="info-box">
              <strong>Video files</strong> → thumbnail extracted (embedded metadata or frame at 10%)<br>
              <strong>Image files</strong> → copied directly<br>
              All saved as <code>.jpg</code> into <code>data/dataset/&lt;classname&gt;/</code><br>
              Safe to re-run — already extracted files are skipped.
            </div>
        """)

        with gr.Row(elem_classes="primary-btn"):
            btn = gr.Button("🎬  Extract Thumbnails & Build Dataset", variant="primary")

        status  = gr.Textbox(label="Status", lines=3,  interactive=False)
        log_box = gr.Textbox(label="Log",    lines=14, interactive=False,
                             elem_classes="log-box")

        btn.click(fn=run_data_setup, inputs=[], outputs=[status, log_box])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_training(use_lr_finder, progress=gr.Progress(track_tqdm=True)):
    log = []

    has_data = any(True for _ in Path("data/dataset").rglob("*.jpg")) \
               if Path("data/dataset").exists() else False

    if not has_data and not STATE["dataset_ready"]:
        yield "❌  Dataset not ready. Run Data Setup first."
        return

    log = _log(log, "Launching training pipeline…")
    progress(0.05, desc="Starting…")

    cmd = [sys.executable, "train.py"]
    if use_lr_finder:
        cmd.append("--lr-finder")
        log = _log(log, "LR Finder enabled")

    log = _log(log, f"Command: {' '.join(cmd)}")
    yield _log_str(log)

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        total_ep   = STATE["config"]["head_epochs"] + STATE["config"]["finetune_epochs"]
        epoch_seen = 0

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            log = _log(log, line)
            if "epoch" in line.lower():
                epoch_seen += 1
                progress(
                    min(0.95, 0.1 + (epoch_seen / max(total_ep, 1)) * 0.85),
                    desc=f"Epoch {epoch_seen}/{total_ep}"
                )
            yield _log_str(log)

        proc.wait()
        progress(1.0, desc="Complete")

        if proc.returncode == 0:
            STATE["model_ready"] = True
            log = _log(log, "━" * 42)
            log = _log(log, "✓  TRAINING COMPLETE")
            log = _log(log, "✓  Model → models/exported/classifier.pkl")
            log = _log(log, "━" * 42)
        else:
            log = _log(log, f"✗  Exited with code {proc.returncode}")

    except Exception:
        log = _log(log, traceback.format_exc())

    yield _log_str(log)


def build_training_tab():
    with gr.Tab("🏋️  Train"):
        gr.HTML("""
            <div class="section-title">Two-Phase Transfer Learning</div>
            <p style="color:#475569;font-size:0.9rem;margin-bottom:14px;">
              Phase 1: head only (backbone frozen).<br>
              Phase 2: all layers with discriminative learning rates.
            </p>
        """)

        lr_finder = gr.Checkbox(
            label="🔍 Run LR Finder first (recommended for new datasets)",
            value=True,
        )

        with gr.Row(elem_classes="primary-btn"):
            btn = gr.Button("🚀  Start Training", variant="primary")

        log_box = gr.Textbox(
            label="Training Log", lines=22, interactive=False,
            elem_classes="log-box",
            placeholder="Training output will stream here…",
        )
        btn.click(fn=run_training, inputs=[lr_finder], outputs=[log_box])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SORT
# ══════════════════════════════════════════════════════════════════════════════

def run_sorting(progress=gr.Progress(track_tqdm=True)):
    """
    For each file in the unsorted folder:
      1. Extract thumbnail (same logic as Data Setup)
      2. Classify the thumbnail
      3. Copy the ORIGINAL file to the matching output subfolder
      4. Files below threshold go to _uncertain/
    """
    cfg        = STATE["config"]
    log        = []
    model_path = Path("models/exported/classifier.pkl")

    if not model_path.exists():
        yield "❌  No model. Train first.", "", ""
        return

    unsorted   = cfg["unsorted_folder"]
    sorted_out = cfg["sorted_output"] or "data/sorted_output"
    threshold  = cfg["confidence_threshold"]

    if not unsorted:
        yield "❌  Unsorted folder not configured.", "", ""
        return
    if not Path(unsorted).exists():
        yield f"❌  Folder not found: {unsorted}", "", ""
        return

    log = _log(log, "Loading classifier…")
    yield "Loading model…", _log_str(log), ""

    try:
        from predict import ImageClassifier
        clf = ImageClassifier(str(model_path), cpu_only=False)
        log = _log(log, f"OK    Classes: {clf.classes}")
    except Exception as e:
        log = _log(log, f"ERROR: {e}")
        yield "❌  Model load failed.", _log_str(log), ""
        return

    out = Path(sorted_out)
    for cls in clf.classes:
        (out / cls).mkdir(parents=True, exist_ok=True)
    (out / "_uncertain").mkdir(parents=True, exist_ok=True)

    files = [f for f in Path(unsorted).iterdir()
             if f.is_file() and f.suffix.lower() in ALL_MEDIA_EXTS]

    if not files:
        yield (
            f"❌  No supported files in '{unsorted}'.\n"
            f"    Supported extensions: {sorted(ALL_MEDIA_EXTS)}"
        ), _log_str(log), ""
        return

    imgs = sum(1 for f in files if f.suffix.lower() in IMAGE_EXTS)
    vids = sum(1 for f in files if f.suffix.lower() in VIDEO_EXTS)
    log  = _log(log, f"Found {len(files)} files: {imgs} images + {vids} videos")
    log  = _log(log, f"Threshold: {threshold:.0%}  |  Output: {out}")

    counts   = {cls: 0 for cls in clf.classes}
    counts["_uncertain"] = 0
    csv_rows = []

    for i, file in enumerate(files):
        progress((i + 1) / len(files), desc=f"{i+1}/{len(files)}: {file.name}")

        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            img = get_thumbnail(file, save_to=tmp_path)
            if img is None:
                log = _log(log, f"  SKIP  {file.name} — thumbnail extraction failed")
                tmp_path.unlink(missing_ok=True)
                continue

            result     = clf.predict(tmp_path)
            label      = result["label"]
            confidence = result["confidence"]
            tmp_path.unlink(missing_ok=True)

            if confidence >= threshold:
                dest_dir   = out / label
                dest_label = label
                counts[label] += 1
            else:
                dest_dir   = out / "_uncertain"
                dest_label = f"_uncertain (best: {label} @ {confidence:.0%})"
                counts["_uncertain"] += 1

            # Copy the ORIGINAL video/image file to the output folder
            shutil.copy2(file, dest_dir / file.name)

            csv_rows.append({
                "file":       file.name,
                "type":       "video" if file.suffix.lower() in VIDEO_EXTS else "image",
                "label":      dest_label,
                "confidence": f"{confidence:.1%}",
            })
            log = _log(log, f"  {file.name:38s} → {dest_label:22s} ({confidence:.1%})")

        except Exception as e:
            log = _log(log, f"  ERROR {file.name}: {e}")

        if i % 5 == 0:
            yield f"Sorting… {i+1}/{len(files)}", _log_str(log), ""

    # Save CSV log
    log_path = out / "sort_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "type", "label", "confidence"])
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_lines = ["━" * 32, "  SORTING COMPLETE", ""]
    for lbl, cnt in counts.items():
        summary_lines.append(f"  {lbl:>22}: {cnt:>4} files")
    summary_lines += ["", f"  Output → {out}", f"  Log    → {log_path}", "━" * 32]

    for line in summary_lines:
        log = _log(log, line)

    yield "✅  Sorting complete!", _log_str(log), "\n".join(summary_lines)


def build_sort_tab():
    with gr.Tab("🗂️  Sort"):
        gr.HTML("""
            <div class="section-title">Classify & Route Mixed Folder</div>
            <div class="info-box">
              <strong>Original video/image files</strong> are copied to the output folder —
              only the thumbnail is used for classification. Your source files are never modified.
            </div>
            <p style="color:#475569;font-size:0.88rem;">
              Files below the confidence threshold → <code>_uncertain/</code> for manual review.<br>
              A <code>sort_log.csv</code> is saved with every prediction.
            </p>
        """)

        with gr.Row(elem_classes="primary-btn"):
            btn = gr.Button("▶️  Start Sorting", variant="primary")

        status  = gr.Textbox(label="Status",  lines=2,  interactive=False)
        log_box = gr.Textbox(label="Log",     lines=16, interactive=False,
                             elem_classes="log-box")
        summary = gr.Textbox(label="Summary", lines=10, interactive=False)

        btn.click(fn=run_sorting, inputs=[], outputs=[status, log_box, summary])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(uploaded_image):
    if uploaded_image is None:
        return "Upload an image first.", "", None
    if not Path("models/exported/classifier.pkl").exists():
        return "❌  No model. Train first.", "", None

    from PIL import Image as PILImg
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImg.fromarray(uploaded_image.astype(np.uint8)).save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        from predict import ImageClassifier
        clf    = ImageClassifier("models/exported/classifier.pkl", cpu_only=False)
        result = clf.predict(tmp_path)
    except Exception as e:
        return f"❌  Prediction failed: {e}", "", None
    finally:
        tmp_path.unlink(missing_ok=True)

    label = result["label"].upper()
    conf  = result["confidence"]

    lines = [f"{'Class':<18}  {'Confidence':>10}  Bar"]
    lines.append("─" * 54)
    for cls, prob in result["all_probs"].items():
        bar     = "█" * int(prob * 25) + "░" * (25 - int(prob * 25))
        pointer = "  ◄  PREDICTED" if cls == result["label"] else ""
        lines.append(f"{cls:<18}  {prob*100:>9.1f}%  {bar}{pointer}")

    return f"{label}  —  {conf*100:.1f}% confidence", "\n".join(lines), uploaded_image


def build_predict_tab():
    with gr.Tab("🔮  Predict"):
        gr.HTML("""
            <div class="section-title">Single Image Prediction</div>
            <p style="color:#475569;font-size:0.9rem;margin-bottom:14px;">
              Upload any image to classify it instantly.
            </p>
        """)
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Upload Image", type="numpy", height=300)
                with gr.Row(elem_classes="primary-btn"):
                    btn = gr.Button("🔮  Classify", variant="primary")
            with gr.Column(scale=1):
                headline  = gr.Textbox(label="Prediction",  lines=1, interactive=False)
                breakdown = gr.Textbox(label="All Classes", lines=8, interactive=False,
                                       elem_classes="log-box")

        btn.click(fn=predict_single, inputs=[img_in],
                  outputs=[headline, breakdown, img_in])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(progress=gr.Progress(track_tqdm=True)):
    log = []
    if not Path("models/exported/classifier.pkl").exists():
        yield "❌  No model. Train first.", "", ""
        return

    log = _log(log, "Running evaluation…")
    yield "Running…", _log_str(log), ""
    progress(0.1, desc="Evaluating…")

    try:
        proc = subprocess.run(
            [sys.executable, "evaluation.py",
             "--model", "models/exported/classifier.pkl",
             "--data",  "data/dataset",
             "--output", "eval_report.json"],
            capture_output=True, text=True,
        )
        for line in (proc.stdout + proc.stderr).splitlines():
            log = _log(log, line)
    except Exception as e:
        log = _log(log, f"ERROR: {e}")
        yield "❌  Evaluation failed.", _log_str(log), ""
        return

    progress(1.0, desc="Done")

    report_str = ""
    try:
        with open("eval_report.json") as f:
            report = json.load(f)
        s = report["summary"]
        lines = [
            "╔══════════════════════════════════════════╗",
            "║         EVALUATION RESULTS               ║",
            "╠══════════════════════════════════════════╣",
            f"║  Top-1 Accuracy :   {s['top1_accuracy']*100:>6.2f}%             ║",
            f"║  Top-3 Accuracy :   {s['top3_accuracy']*100:>6.2f}%             ║",
            f"║  Macro  F1      :   {s['macro_f1']*100:>6.2f}%             ║",
            f"║  Weighted F1    :   {s['weighted_f1']*100:>6.2f}%             ║",
            f"║  Total Images   :   {s['total_images']:>6}               ║",
            "╠══════════════════════════════════════════╣",
            "║  Class              Prec    Rec     F1   ║",
            "╠══════════════════════════════════════════╣",
        ]
        for cls, v in report["per_class"].items():
            lines.append(
                f"║  {cls:<18} {v['precision']:.3f}   {v['recall']:.3f}   {v['f1-score']:.3f} ║"
            )
        lines.append("╚══════════════════════════════════════════╝")
        report_str = "\n".join(lines)
    except Exception as e:
        report_str = f"Could not parse eval_report.json: {e}"

    yield "✅  Evaluation complete!", _log_str(log), report_str


def build_eval_tab():
    with gr.Tab("📊  Evaluate"):
        gr.HTML("""
            <div class="section-title">Model Evaluation Report</div>
            <p style="color:#475569;font-size:0.9rem;margin-bottom:14px;">
              Accuracy, per-class Precision/Recall/F1, confusion matrix.
            </p>
        """)
        with gr.Row(elem_classes="primary-btn"):
            btn = gr.Button("📊  Run Evaluation", variant="primary")
        status  = gr.Textbox(label="Status",  lines=1,  interactive=False)
        log_box = gr.Textbox(label="Log",     lines=10, interactive=False,
                             elem_classes="log-box")
        report  = gr.Textbox(label="Results", lines=20, interactive=False,
                             elem_classes="log-box")
        btn.click(fn=run_evaluation, inputs=[], outputs=[status, log_box, report])


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE & LAUNCH
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(
        title="Video Thumbnail Classifier",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=CUSTOM_CSS,
    ) as app:

        gr.HTML(HEADER_HTML)
        gr.HTML("""
            <div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;">
              <span style="background:#e0e7ff;color:#3730a3;padding:4px 12px;
                border-radius:20px;font-size:0.8rem;font-weight:600;">1 · Configure</span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;padding:4px 12px;
                border-radius:20px;font-size:0.8rem;font-weight:600;">2 · Data Setup</span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;padding:4px 12px;
                border-radius:20px;font-size:0.8rem;font-weight:600;">3 · Train</span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;padding:4px 12px;
                border-radius:20px;font-size:0.8rem;font-weight:600;">4 · Sort</span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;padding:4px 12px;
                border-radius:20px;font-size:0.8rem;font-weight:600;">5 · Predict</span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;padding:4px 12px;
                border-radius:20px;font-size:0.8rem;font-weight:600;">6 · Evaluate</span>
            </div>
        """)

        build_config_tab()
        build_data_tab()
        build_training_tab()
        build_sort_tab()
        build_predict_tab()
        build_eval_tab()

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        share=False,
        debug=True,
        show_error=True,
        inbrowser=False,
        server_name="127.0.0.1",
        server_port=7860,
    )
