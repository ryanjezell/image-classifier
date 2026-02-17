"""
app.py — Gradio Web UI for the Image Classifier
═══════════════════════════════════════════════════════════════════════════════

Provides a browser-based interface for the full pipeline:
  Tab 1 — Configuration   Set Drive paths and class names
  Tab 2 — Data Setup      Copy images from Drive into dataset folder
  Tab 3 — Training        Train the model with live progress feedback
  Tab 4 — Sort Images     Classify a mixed folder and route to subfolders
  Tab 5 — Predict         Upload a single image and see the prediction
  Tab 6 — Evaluation      Run the full metrics report

Launch (in Colab):
    !python app.py

Or from a notebook cell:
    %run app.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
import csv
import json
import shutil
import threading
import traceback
from pathlib import Path
from datetime import datetime

import gradio as gr

# Add project root to path so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent))

# ── Shared state ──────────────────────────────────────────────────────────────
# A simple dict that all tabs can read/write.
# In a production app you'd use a database or session store;
# for a single-user Colab tool this is perfectly sufficient.
STATE = {
    "config": {
        "class1_name":    "class_1",
        "class1_source":  "",
        "class2_name":    "class_2",
        "class2_source":  "",
        "class3_name":    "class_3",
        "class3_source":  "",
        "unsorted_folder": "",
        "sorted_output":   "",
        "confidence_threshold": 0.70,
        "batch_size":  32,
        "image_size":  224,
        "architecture": "resnet50",
        "head_epochs":  4,
        "finetune_epochs": 10,
    },
    "model_ready":   False,
    "dataset_ready": False,
    "training_log":  [],
}

# ── Colour theme ──────────────────────────────────────────────────────────────
# Gradio Soft theme + custom CSS for a clean, professional look
CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────── */
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Header banner ──────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 8px;
    color: white !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
}
.app-header h1 { 
    font-size: 1.9rem; 
    font-weight: 700; 
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.app-header p  { 
    font-size: 0.95rem; 
    opacity: 0.75; 
    margin: 0;
}

/* ── Section cards ──────────────────────────────────────────── */
.section-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.section-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 14px;
}

/* ── Status badges ──────────────────────────────────────────── */
.status-ok   { color: #16a34a; font-weight: 600; }
.status-warn { color: #d97706; font-weight: 600; }
.status-err  { color: #dc2626; font-weight: 600; }

/* ── Progress log ───────────────────────────────────────────── */
.log-box textarea {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 0.8rem !important;
    background: #0f172a !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
}

/* ── Prediction result ──────────────────────────────────────── */
.pred-label {
    font-size: 2rem;
    font-weight: 800;
    text-align: center;
    padding: 16px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Primary button ─────────────────────────────────────────── */
.primary-btn button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.primary-btn button:hover { opacity: 0.88 !important; }

/* ── Warning box ─────────────────────────────────────────────── */
.warn-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #92400e;
}
"""

HEADER_HTML = """
<div class="app-header">
  <h1>🔍 Image Classifier</h1>
  <p>Transfer learning · ResNet-50 · fast.ai + PyTorch · Google Drive integration</p>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    """Returns a short HH:MM:SS timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _log(lines: list, message: str) -> list:
    """
    Appends a timestamped message to the running log list and returns it.
    Keeping the log as a list (not a raw string) makes it easy to cap length
    and update Gradio textbox incrementally.
    """
    lines.append(f"[{_ts()}]  {message}")
    # Cap at 200 lines to avoid the textbox becoming unresponsive
    if len(lines) > 200:
        lines = lines[-200:]
    return lines


def _log_str(lines: list) -> str:
    """Joins log lines to a single string for display in the textbox."""
    return "\n".join(lines)


def _count_images(folder: str) -> int:
    """Returns the number of valid image files in a folder."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    p = Path(folder)
    if not p.exists():
        return 0
    return sum(1 for f in p.iterdir() if f.suffix.lower() in exts)


def _yaml_value(key_path: str, value) -> str:
    """
    Reads config/config.yaml, updates a nested key (dot-separated path),
    and writes it back. Used to sync UI settings to the YAML file.
    """
    import yaml
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        return "⚠️  config/config.yaml not found."
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Navigate nested dict via dot-path (e.g. "training.batch_size")
    keys = key_path.split(".")
    target = cfg
    for k in keys[:-1]:
        target = target.setdefault(k, {})
    target[keys[-1]] = value

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return "ok"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def save_config(
    c1_name, c1_src,
    c2_name, c2_src,
    c3_name, c3_src,
    unsorted, sorted_out,
    threshold, batch_size, image_size, architecture,
    head_epochs, finetune_epochs,
):
    """
    Validates all inputs and saves them to STATE and config/config.yaml.
    Returns a status message shown to the user.
    
    WHY validate here and not on train:
    Catching bad paths at config-save time gives faster, clearer feedback
    than a cryptic crash 30 seconds into training.
    """
    errors = []
    warnings = []

    # ── Validate source folders ───────────────────────────────────────────
    for name, src in [(c1_name, c1_src), (c2_name, c2_src), (c3_name, c3_src)]:
        if not src.strip():
            errors.append(f"'{name}' source path is empty.")
            continue
        if not Path(src).exists():
            errors.append(f"'{name}' folder not found: {src}")
        else:
            n = _count_images(src)
            if n < 20:
                warnings.append(f"'{name}' has only {n} images (20+ recommended).")

    if unsorted and not Path(unsorted).exists():
        warnings.append(f"Unsorted folder not found yet: {unsorted}  (OK if you'll create it later)")

    if errors:
        return "❌  " + "\n❌  ".join(errors), gr.update(visible=True)

    # ── Persist to STATE ──────────────────────────────────────────────────
    STATE["config"].update({
        "class1_name": c1_name.strip(),   "class1_source": c1_src.strip(),
        "class2_name": c2_name.strip(),   "class2_source": c2_src.strip(),
        "class3_name": c3_name.strip(),   "class3_source": c3_src.strip(),
        "unsorted_folder": unsorted.strip(),
        "sorted_output":   sorted_out.strip(),
        "confidence_threshold": float(threshold),
        "batch_size":     int(batch_size),
        "image_size":     int(image_size),
        "architecture":   architecture,
        "head_epochs":    int(head_epochs),
        "finetune_epochs": int(finetune_epochs),
    })

    # ── Sync to YAML ──────────────────────────────────────────────────────
    try:
        _yaml_value("data.batch_size",       int(batch_size))
        _yaml_value("data.image_size",        int(image_size))
        _yaml_value("model.architecture",     architecture)
        _yaml_value("training.head_epochs",   int(head_epochs))
        _yaml_value("training.finetune_epochs", int(finetune_epochs))
    except Exception as e:
        warnings.append(f"Could not update config.yaml: {e}")

    warn_str = ("\n⚠️   " + "\n⚠️   ".join(warnings)) if warnings else ""
    return f"✅  Configuration saved!{warn_str}", gr.update(visible=True)


def build_config_tab():
    with gr.Tab("⚙️  Configure"):
        gr.HTML("""
            <div class="section-title">Google Drive — Training Image Sources</div>
            <div class="warn-box">
              Mount your Drive first in Colab (<code>drive.mount('/content/drive')</code>),
              then paste the full paths below.  Example: 
              <code>/content/drive/MyDrive/images/cats</code>
            </div>
        """)

        with gr.Group():
            with gr.Row():
                c1_name = gr.Textbox(label="Class 1 Name",   placeholder="e.g. cats",  scale=1)
                c1_src  = gr.Textbox(label="Class 1 Drive Path",
                                     placeholder="/content/drive/MyDrive/...", scale=3)
            with gr.Row():
                c2_name = gr.Textbox(label="Class 2 Name",   placeholder="e.g. dogs",  scale=1)
                c2_src  = gr.Textbox(label="Class 2 Drive Path",
                                     placeholder="/content/drive/MyDrive/...", scale=3)
            with gr.Row():
                c3_name = gr.Textbox(label="Class 3 Name",   placeholder="e.g. birds", scale=1)
                c3_src  = gr.Textbox(label="Class 3 Drive Path",
                                     placeholder="/content/drive/MyDrive/...", scale=3)

        gr.HTML('<div class="section-title" style="margin-top:20px">Sorting Paths</div>')
        with gr.Group():
            with gr.Row():
                unsorted   = gr.Textbox(label="Unsorted Images Folder (mixed input)",
                                        placeholder="/content/drive/MyDrive/unsorted")
                sorted_out = gr.Textbox(label="Sorted Output Folder",
                                        placeholder="/content/drive/MyDrive/sorted_output")

        gr.HTML('<div class="section-title" style="margin-top:20px">Model & Training Settings</div>')
        with gr.Group():
            with gr.Row():
                arch         = gr.Dropdown(
                    label="Backbone Architecture",
                    choices=["resnet34", "resnet50", "efficientnet_b0", "efficientnet_b3"],
                    value="resnet50",
                )
                image_size   = gr.Slider(label="Input Image Size (px)",
                                         minimum=128, maximum=512, step=32, value=224)
                batch_size   = gr.Slider(label="Batch Size",
                                         minimum=8, maximum=64, step=8, value=32,
                                         info="Reduce to 8–16 if you get GPU out-of-memory errors")
            with gr.Row():
                head_ep      = gr.Slider(label="Phase 1 Epochs (head only)",
                                         minimum=1, maximum=20, step=1, value=4)
                finetune_ep  = gr.Slider(label="Phase 2 Epochs (full fine-tune)",
                                         minimum=1, maximum=40, step=1, value=10)
                threshold    = gr.Slider(label="Confidence Threshold for Sorting",
                                         minimum=0.4, maximum=0.99, step=0.01, value=0.70,
                                         info="Images below this confidence go to _uncertain/")

        with gr.Row(elem_classes="primary-btn"):
            save_btn = gr.Button("💾  Save Configuration", variant="primary", scale=2)

        status_box = gr.Textbox(label="Status", lines=3, interactive=False,
                                visible=False, elem_classes="status-ok")

        # Wire button → function
        save_btn.click(
            fn=save_config,
            inputs=[c1_name, c1_src, c2_name, c2_src, c3_name, c3_src,
                    unsorted, sorted_out,
                    threshold, batch_size, image_size, arch,
                    head_ep, finetune_ep],
            outputs=[status_box, status_box],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA SETUP
# ══════════════════════════════════════════════════════════════════════════════

def run_data_setup(progress=gr.Progress(track_tqdm=True)):
    """
    Copies images from the configured Drive source folders into
    data/dataset/<classname>/ ready for training.

    WHY copy instead of symlink:
    Symlinks into Drive can break when Colab reconnects. Copying guarantees
    the data is local and fast to read during training.
    """
    cfg  = STATE["config"]
    log  = []
    pairs = [
        (cfg["class1_name"], cfg["class1_source"]),
        (cfg["class2_name"], cfg["class2_source"]),
        (cfg["class3_name"], cfg["class3_source"]),
    ]

    # ── Validate config is saved ──────────────────────────────────────────
    if not cfg["class1_source"]:
        return "❌  No paths configured. Go to the Configure tab first.", ""

    dataset_dir = Path("data/dataset")
    total_copied = 0
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for i, (name, source) in enumerate(pairs):
        progress((i / len(pairs)), desc=f"Copying {name}…")

        if not source:
            log = _log(log, f"SKIP  {name} — no source path set")
            continue

        src_path  = Path(source)
        dest_path = dataset_dir / name
        dest_path.mkdir(parents=True, exist_ok=True)

        if not src_path.exists():
            log = _log(log, f"ERROR {name} — folder not found: {source}")
            continue

        # Collect all valid image files in the source
        images = [f for f in src_path.iterdir() if f.suffix.lower() in exts]

        if not images:
            log = _log(log, f"WARN  {name} — no images found in {source}")
            continue

        log = _log(log, f"Copying {len(images)} images for '{name}'…")

        for j, img in enumerate(images):
            # Simple progress update every 10 files
            if j % 10 == 0:
                progress(
                    (i + j / len(images)) / len(pairs),
                    desc=f"{name}: {j}/{len(images)}"
                )
            shutil.copy2(img, dest_path / img.name)

        total_copied += len(images)
        log = _log(log, f"OK    {name}: {len(images)} images → {dest_path}")

    STATE["dataset_ready"] = total_copied > 0
    progress(1.0, desc="Done")

    summary = (
        f"✅  Dataset ready! {total_copied} images copied across {len(pairs)} classes.\n"
        f"Location: data/dataset/"
    ) if total_copied > 0 else "❌  No images were copied. Check your paths."

    return summary, _log_str(log)


def build_data_tab():
    with gr.Tab("📁  Data Setup"):
        gr.HTML("""
            <div class="section-title">Copy Images from Google Drive</div>
            <p style="color:#475569; font-size:0.9rem; margin-bottom:16px;">
              This copies your Drive images into the local dataset folder so training
              can run at full GPU speed. Run this once, or whenever you add new images.
            </p>
        """)

        with gr.Row(elem_classes="primary-btn"):
            copy_btn = gr.Button("📋  Copy Images from Drive", variant="primary")

        status  = gr.Textbox(label="Status", lines=2,  interactive=False)
        log_box = gr.Textbox(label="Log",    lines=12, interactive=False,
                             elem_classes="log-box")

        copy_btn.click(
            fn=run_data_setup,
            inputs=[],
            outputs=[status, log_box],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def run_training(use_lr_finder: bool, progress=gr.Progress(track_tqdm=True)):
    """
    Runs the full two-phase training pipeline by calling train.py as a
    subprocess and streaming its output back to the log textbox.

    WHY subprocess instead of importing train.py directly:
    Importing and calling train() in-process means Gradio's event loop
    blocks until training finishes. As a subprocess, we can stream stdout
    in real time and keep the UI alive for progress updates.
    """
    import subprocess

    log = []

    if not STATE["dataset_ready"]:
        log = _log(log, "ERROR: Dataset not prepared. Run the Data Setup tab first.")
        return "❌  Dataset not ready.", _log_str(log)

    log = _log(log, "Starting training pipeline…")
    progress(0.05, desc="Launching trainer…")

    cmd = ["python", "train.py"]
    if use_lr_finder:
        cmd.append("--lr-finder")
        log = _log(log, "LR Finder enabled — will auto-detect optimal learning rate.")

    log = _log(log, f"Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            bufsize=1,                  # line-buffered
        )

        # Stream output line by line
        # This is what makes the log box update in real time
        epoch_count = 0
        total_epochs = (STATE["config"]["head_epochs"] +
                        STATE["config"]["finetune_epochs"])

        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            log = _log(log, line)

            # Crude progress tracking from log lines
            if "Epoch" in line or "epoch" in line:
                epoch_count += 1
                pct = min(0.9, 0.1 + (epoch_count / max(total_epochs, 1)) * 0.8)
                progress(pct, desc=f"Epoch {epoch_count}/{total_epochs}")

            yield _log_str(log)   # yield so Gradio streams updates live

        process.wait()
        progress(1.0, desc="Complete")

        if process.returncode == 0:
            STATE["model_ready"] = True
            log = _log(log, "")
            log = _log(log, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            log = _log(log, "✓  TRAINING COMPLETE")
            log = _log(log, "✓  Model saved → models/exported/classifier.pkl")
            log = _log(log, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            log = _log(log, f"✗  Training exited with code {process.returncode}")

    except Exception as e:
        log = _log(log, f"EXCEPTION: {traceback.format_exc()}")

    yield _log_str(log)


def build_training_tab():
    with gr.Tab("🏋️  Train"):
        gr.HTML("""
            <div class="section-title">Two-Phase Transfer Learning</div>
            <p style="color:#475569; font-size:0.9rem; margin-bottom:16px;">
              Phase 1 trains only the classification head (backbone frozen).<br>
              Phase 2 unfreezes all layers and fine-tunes with discriminative learning rates.
            </p>
        """)

        with gr.Row():
            lr_finder = gr.Checkbox(
                label="🔍 Run LR Finder before training (recommended for new datasets)",
                value=True,
            )

        with gr.Row(elem_classes="primary-btn"):
            train_btn = gr.Button("🚀  Start Training", variant="primary")

        log_box = gr.Textbox(
            label="Training Log",
            lines=20,
            interactive=False,
            elem_classes="log-box",
            placeholder="Training output will appear here…",
        )

        # Gradio 4+ streams automatically when the function is a generator (uses yield)
        train_btn.click(
            fn=run_training,
            inputs=[lr_finder],
            outputs=[log_box],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SORT IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def run_sorting(progress=gr.Progress(track_tqdm=True)):
    """
    Loads the trained classifier and sorts every image in the configured
    unsorted folder into subfolders named after each class.

    Images below the confidence threshold go into an '_uncertain' folder
    for manual review rather than silent misclassification.
    """
    cfg = STATE["config"]
    log = []

    if not STATE["model_ready"]:
        # Try loading from disk even if model_ready isn't set (e.g. new session)
        if not Path("models/exported/classifier.pkl").exists():
            log = _log(log, "ERROR: No trained model found. Train the model first.")
            yield "❌  No model found.", _log_str(log), ""
            return

    unsorted   = cfg["unsorted_folder"]
    sorted_out = cfg["sorted_output"]
    threshold  = cfg["confidence_threshold"]

    if not unsorted:
        log = _log(log, "ERROR: No unsorted folder configured.")
        yield "❌  Configure unsorted folder path first.", _log_str(log), ""
        return

    if not Path(unsorted).exists():
        log = _log(log, f"ERROR: Unsorted folder not found: {unsorted}")
        yield f"❌  Folder not found: {unsorted}", _log_str(log), ""
        return

    # ── Load classifier ───────────────────────────────────────────────────
    log = _log(log, "Loading classifier…")
    yield "Loading model…", _log_str(log), ""

    try:
        from predict import ImageClassifier
        clf = ImageClassifier("models/exported/classifier.pkl", cpu_only=False)
        log = _log(log, f"OK    Classes: {clf.classes}")
    except Exception as e:
        log = _log(log, f"ERROR loading model: {e}")
        yield "❌  Model load failed.", _log_str(log), ""
        return

    # ── Create output folders ─────────────────────────────────────────────
    out = Path(sorted_out) if sorted_out else Path("data/sorted_output")
    for cls in clf.classes:
        (out / cls).mkdir(parents=True, exist_ok=True)
    (out / "_uncertain").mkdir(parents=True, exist_ok=True)

    # ── Collect images ────────────────────────────────────────────────────
    exts   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in Path(unsorted).iterdir()
              if f.suffix.lower() in exts]

    if not images:
        yield "❌  No images found in unsorted folder.", _log_str(log), ""
        return

    log = _log(log, f"Found {len(images)} images to classify…")
    log = _log(log, f"Confidence threshold: {threshold:.0%}")
    log = _log(log, f"Output: {out}")

    # ── Classify & sort ───────────────────────────────────────────────────
    counts  = {cls: 0 for cls in clf.classes}
    counts["_uncertain"] = 0
    csv_rows = []

    for i, img_path in enumerate(images):
        progress((i + 1) / len(images), desc=f"{i+1}/{len(images)}")

        try:
            result = clf.predict(img_path)
            label  = result["label"]
            conf   = result["confidence"]

            if conf >= threshold:
                dest_dir = out / label
                counts[label] += 1
                dest_label = label
            else:
                dest_dir = out / "_uncertain"
                counts["_uncertain"] += 1
                dest_label = f"_uncertain (best: {label} {conf:.0%})"

            shutil.copy2(img_path, dest_dir / img_path.name)
            csv_rows.append({"file": img_path.name,
                             "label": dest_label,
                             "confidence": f"{conf:.1%}"})

            log = _log(log, f"  {img_path.name:35s} → {dest_label:25s} ({conf:.1%})")

        except Exception as e:
            log = _log(log, f"  ERROR on {img_path.name}: {e}")

        # Yield every 5 images so the UI updates smoothly
        if i % 5 == 0:
            yield f"Sorting… {i+1}/{len(images)}", _log_str(log), ""

    # ── Save CSV log ──────────────────────────────────────────────────────
    log_path = out / "sort_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "confidence"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Build summary ─────────────────────────────────────────────────────
    summary_lines = ["━━━━━━━━━━━━━━━━━━━━━━━━━━",
                     "  SORTING COMPLETE", ""]
    for lbl, cnt in counts.items():
        summary_lines.append(f"  {lbl:>18}: {cnt:>5} images")
    summary_lines += ["", f"  Output  → {out}",
                      f"  CSV log → {log_path}",
                      "━━━━━━━━━━━━━━━━━━━━━━━━━━"]

    summary_str = "\n".join(summary_lines)
    for line in summary_lines:
        log = _log(log, line)

    yield "✅  Sorting complete!", _log_str(log), summary_str


def build_sort_tab():
    with gr.Tab("🗂️  Sort Images"):
        gr.HTML("""
            <div class="section-title">Classify & Route a Mixed Image Folder</div>
            <p style="color:#475569; font-size:0.9rem; margin-bottom:16px;">
              Every image in your unsorted Drive folder will be classified and copied
              to a subfolder named after its predicted class. Low-confidence predictions
              go to <code>_uncertain/</code> for manual review.
            </p>
        """)

        with gr.Row(elem_classes="primary-btn"):
            sort_btn = gr.Button("▶️  Start Sorting", variant="primary")

        status  = gr.Textbox(label="Status",  lines=2,  interactive=False)
        log_box = gr.Textbox(label="Log",     lines=15, interactive=False,
                             elem_classes="log-box")
        summary = gr.Textbox(label="Summary", lines=10, interactive=False)

        sort_btn.click(
            fn=run_sorting,
            inputs=[],
            outputs=[status, log_box, summary],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SINGLE IMAGE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(image):
    """
    Accepts an uploaded image (numpy array from Gradio),
    saves it to a temp file, runs the classifier, and returns
    a formatted prediction breakdown.
    """
    if image is None:
        return "Upload an image first.", "", None

    if not Path("models/exported/classifier.pkl").exists():
        return "❌  No model found. Train first.", "", None

    # Save numpy array → temp JPEG so PILImage.create() can read it
    import tempfile
    import numpy as np
    from PIL import Image as PILImg

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImg.fromarray(image.astype(np.uint8)).save(tmp.name)
        tmp_path = tmp.name

    try:
        from predict import ImageClassifier
        clf    = ImageClassifier("models/exported/classifier.pkl", cpu_only=False)
        result = clf.predict(tmp_path)
    except Exception as e:
        return f"❌  Prediction failed: {e}", "", None
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    label = result["label"].upper()
    conf  = result["confidence"]

    # ── Build text breakdown ──────────────────────────────────────────────
    lines = [f"{'Class':<18}  {'Confidence':>10}  {'Bar'}"]
    lines.append("─" * 50)
    for cls, prob in result["all_probs"].items():
        bar     = "█" * int(prob * 25) + "░" * (25 - int(prob * 25))
        pointer = "  ◄  PREDICTED" if cls == result["label"] else ""
        lines.append(f"{cls:<18}  {prob*100:>9.1f}%  {bar}{pointer}")

    breakdown = "\n".join(lines)
    headline  = f"{label}  —  {conf*100:.1f}% confidence"

    return headline, breakdown, image


def build_predict_tab():
    with gr.Tab("🔮  Predict"):
        gr.HTML("""
            <div class="section-title">Single Image Prediction</div>
            <p style="color:#475569; font-size:0.9rem; margin-bottom:16px;">
              Upload any image and the trained model will classify it instantly.
            </p>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label="Upload Image", type="numpy",
                                     height=300)
                with gr.Row(elem_classes="primary-btn"):
                    pred_btn = gr.Button("🔮  Classify", variant="primary")
            with gr.Column(scale=1):
                headline  = gr.Textbox(label="Prediction",  lines=1,
                                       interactive=False, elem_classes="pred-label")
                breakdown = gr.Textbox(label="All Classes", lines=8,
                                       interactive=False, elem_classes="log-box")

        pred_btn.click(
            fn=predict_single,
            inputs=[img_input],
            outputs=[headline, breakdown, img_input],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(progress=gr.Progress(track_tqdm=True)):
    """
    Calls evaluation.py as a subprocess and streams the report back.
    Also reads eval_report.json and formats it for display.
    """
    import subprocess
    log = []

    if not Path("models/exported/classifier.pkl").exists():
        yield "❌  No model found. Train first.", "", ""
        return

    log = _log(log, "Running evaluation on full dataset…")
    yield "Running…", _log_str(log), ""

    progress(0.1, desc="Evaluating…")

    try:
        proc = subprocess.run(
            ["python", "evaluation.py",
             "--model", "models/exported/classifier.pkl",
             "--data",  "data/dataset",
             "--output", "eval_report.json"],
            capture_output=True, text=True,
        )
        output = proc.stdout + proc.stderr
        for line in output.splitlines():
            log = _log(log, line)

    except Exception as e:
        log = _log(log, f"ERROR: {e}")
        yield "❌  Evaluation failed.", _log_str(log), ""
        return

    progress(1.0, desc="Done")

    # ── Parse JSON report ─────────────────────────────────────────────────
    report_str = ""
    try:
        with open("eval_report.json") as f:
            report = json.load(f)
        s = report["summary"]
        lines = [
            "╔══════════════════════════════════════════╗",
            "║           EVALUATION RESULTS             ║",
            "╠══════════════════════════════════════════╣",
            f"║  Top-1 Accuracy : {s['top1_accuracy']*100:>6.2f}%                ║",
            f"║  Top-3 Accuracy : {s['top3_accuracy']*100:>6.2f}%                ║",
            f"║  Macro F1 Score : {s['macro_f1']*100:>6.2f}%                ║",
            f"║  Weighted F1    : {s['weighted_f1']*100:>6.2f}%                ║",
            f"║  Total Images   : {s['total_images']:>6}                   ║",
            "╠══════════════════════════════════════════╣",
            "║  Per-Class Breakdown                     ║",
            "║  Class               P       R      F1  ║",
            "╠══════════════════════════════════════════╣",
        ]
        for cls, vals in report["per_class"].items():
            lines.append(
                f"║  {cls:<18}  {vals['precision']:.3f}   "
                f"{vals['recall']:.3f}   {vals['f1-score']:.3f}  ║"
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
            <p style="color:#475569; font-size:0.9rem; margin-bottom:16px;">
              Computes accuracy, per-class Precision / Recall / F1, and a confusion
              matrix across the full dataset. Run after training to assess model quality.
            </p>
        """)

        with gr.Row(elem_classes="primary-btn"):
            eval_btn = gr.Button("📊  Run Evaluation", variant="primary")

        status  = gr.Textbox(label="Status",          lines=1,  interactive=False)
        log_box = gr.Textbox(label="Log",             lines=10, interactive=False,
                             elem_classes="log-box")
        report  = gr.Textbox(label="Results Summary", lines=18, interactive=False,
                             elem_classes="log-box")

        eval_btn.click(
            fn=run_evaluation,
            inputs=[],
            outputs=[status, log_box, report],
        )


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE & LAUNCH
# ══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    """
    Assembles all tabs into a single Gradio Blocks application.
    
    WHY Blocks over Interface:
    gr.Blocks gives full layout control — multiple tabs, custom HTML,
    column/row grids, and the ability to wire multiple inputs/outputs
    per button. gr.Interface is only suitable for single-function demos.
    """
    with gr.Blocks(
        title="Image Classifier",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=CUSTOM_CSS,
    ) as app:

        gr.HTML(HEADER_HTML)

        # ── Workflow hint ─────────────────────────────────────────────────
        gr.HTML("""
            <div style="display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap;">
              <span style="background:#e0e7ff;color:#3730a3;
                           padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;">
                1 · Configure
              </span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;
                           padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;">
                2 · Data Setup
              </span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;
                           padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;">
                3 · Train
              </span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;
                           padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;">
                4 · Sort Images
              </span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;
                           padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;">
                5 · Predict
              </span>
              <span style="color:#94a3b8;align-self:center;">→</span>
              <span style="background:#e0e7ff;color:#3730a3;
                           padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;">
                6 · Evaluate
              </span>
            </div>
        """)

        # Build each tab
        build_config_tab()
        build_data_tab()
        build_training_tab()
        build_sort_tab()
        build_predict_tab()
        build_eval_tab()

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_app()
    app.launch(
        share=True,
        debug=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
    )
