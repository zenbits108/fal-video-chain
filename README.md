# FAL LTX Video Chain Generator

A Python-based video generation pipeline using FAL.AI and LTX 2.3 that creates **continuous, chained video clips** with persistent visual style, camera motion, and audio identity.

This system solves the core limitation of LTX models:

> LTX has no memory between shots.

To compensate, this tool rebuilds **every prompt as a full cinematic shot** using a persistent scene state, continuity rules, and controlled transitions.

---

# 🚀 Features

* 🔁 Infinite or fixed-length video chaining
* 🎬 Persistent **scene state** (style, character, environment)
* 🎥 Locked **camera motion and lens grammar**
* 🔊 Consistent **audio and sound design**
* 🧠 Prompt reconstruction with **continuity memory**
* ✂️ Intelligent **transition handling**
* 📁 Local storage of clips, frames, prompts, and logs

---

# 📦 Requirements

* Linux Mint 22 (or similar Linux)
* Python 3.13
* FFmpeg
* FAL API key

---

# ⚙️ Installation

```bash
sudo apt update
sudo apt install -y ffmpeg python3.13-venv

mkdir -p ~/fal_video_chain
cd ~/fal_video_chain

python3.13 -m venv .venv
source .venv/bin/activate

pip install fal-client requests

export FAL_KEY="YOUR_API_KEY"
```

---

# 🧠 Core Concept

This is **not a prompt chain**.

This is a **shot engine**.

Each clip is generated using:

```
scene_state + transition + action + continuity rules → full prompt
```

Every prompt is rebuilt from scratch to simulate memory.

---

# 📁 Project Structure

```
fal_video_chain/
├── fal_video_chain.py
├── prompts.txt
├── scene_state.json
├── style_header.txt
├── style_footer.txt
├── negative_prompt.txt
└── outputs/
    ├── clips/
    ├── frames/
    ├── logs/
    ├── full_prompts/
    └── manifest.jsonl
```

---

# ▶️ Usage

## 1. Generate default config files

```bash
python fal_video_chain.py --write-default-files
```

## 2. Run the generator

```bash
python fal_video_chain.py \
  --first-text-to-video \
  --prompts-file prompts.txt \
  --duration 6 \
  --fps 25 \
  --iterations 10
```

---

# ✍️ How Prompts Work (IMPORTANT)

## ❗ Prompts are NOT full descriptions

Your prompts should describe **ONLY the action**, not style or camera.

### ❌ Bad Prompt

```
A cinematic shot of a glowing alien cathedral with dramatic lighting
```

### ✅ Good Prompt

```
The explorer slows down and raises the torch toward the glowing symbols on the floor
```

---

## 🎬 Prompt Responsibilities

| Element    | Controlled By      |
| ---------- | ------------------ |
| Style      | `scene_state.json` |
| Character  | `scene_state.json` |
| Lighting   | `scene_state.json` |
| Camera     | `scene_state.json` |
| Audio      | `scene_state.json` |
| Action     | `prompts.txt`      |
| Continuity | Engine logic       |

---

# 🔁 Prompt Chaining Behavior

Each iteration:

1. Takes last frame of previous video
2. Uploads it to FAL
3. Builds a **full cinematic prompt**
4. Injects:

   * Scene state
   * Camera lock
   * Audio consistency
   * Recent shot memory
   * Transition rules
5. Generates next clip

---

# 🎞️ Transitions

The system automatically detects transitions:

### Continuous shot (default)

```
Continue seamlessly from previous frame, no cut
```

### New angle

Use keywords:

```
new angle
close-up
side view
```

### New scene

Use keywords:

```
new scene
cut to
fade to
another chamber
```

---

# 🧱 Scene State (Critical)

The `scene_state.json` file defines:

* Visual style
* Environment
* Character identity
* Camera behavior
* Audio design
* Lighting rules
* Spatial anchors

This file ensures consistency across all clips.

---

# 🔊 Audio Consistency

The system maintains:

* Continuous ambient sound
* Stable music identity
* Consistent dialogue tone

Avoid describing audio in prompts unless necessary.

---

# 📌 Best Practices

## Keep prompts simple

Focus only on what changes:

```
The explorer kneels and touches the glowing floor
```

---

## Maintain slow motion

LTX performs best with:

* Slow camera movement
* Minimal action per clip
* Continuous motion

---

## Avoid conflicting instructions

Do NOT override:

* Camera movement
* Lighting style
* Character design

---

## Use anchors

Stable objects improve continuity:

* pillars
* floor cracks
* light sources

---

# ⚠️ Known Limitations

Even with all controls:

* Minor lighting drift may occur
* Camera may slightly jitter
* Character may subtly morph over time
* Audio can shift slightly

This is expected behavior with current models.

---

# 🧪 Tips for Better Results

* Use **6-second clips** for stability
* Use **25 fps** for smooth motion
* Keep **action minimal per shot**
* Avoid rapid scene changes
* Let the system control style

---

# 📊 Output

Each run generates:

* `.mp4` video clips
* Last-frame `.png` images
* Full prompts per iteration
* JSON logs
* Manifest file

---

# 🧠 Summary

To get consistent results:

* Treat prompts as **actions, not descriptions**
* Let the engine control everything else
* Think in **shots, not scenes**
* Build continuity through repetition

---

# License

Use at your own risk. The model will still occasionally hallucinate like a sleep-deprived film student.
