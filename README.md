# FAL Video Chain Generator

A Python pipeline that generates long, continuous video from a sequence of action prompts using FAL.AI video models. Each clip is seeded from the last frame of the previous one, with a persistent scene state rebuilt into every prompt to maintain visual and audio consistency across the chain.

---

## How it works

LTX and other FAL models have no memory between generations. This tool compensates by rebuilding every prompt from scratch using:

```
scene_state.json + style_header + transition + recent_history + action + style_footer → full prompt
→ FAL API → clip → extract last frame → upload → repeat
```

Your prompts describe **action only**. Everything else — style, character, camera, lighting, audio — is owned by `scene_state.json` and injected automatically into every generation.

---

## Requirements

- Python 3.13
- FFmpeg (`sudo apt install -y ffmpeg`)
- FAL API key

---

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install fal-client requests
export FAL_KEY="your-key"
```

---

## Quick start

Generate default config files on first run:

```bash
python fal_video_chain.py --write-default-files
```

Run a 10-clip chain and join them into one video:

```bash
python fal_video_chain.py \
  --first-text-to-video \
  --prompts-file prompts.txt \
  --iterations 10 \
  --concat
```

Preview what will be sent to the API without spending credits:

```bash
python fal_video_chain.py \
  --first-text-to-video \
  --prompts-file prompts.txt \
  --iterations 3 \
  --dry-run
```

---

## Model selection

Use `--model` to switch between model families. The correct payload schema is applied automatically.

```bash
python fal_video_chain.py --model kling-v3 --first-text-to-video --prompts-file prompts.txt --iterations 5 --concat
```

| Preset | Models |
|---|---|
| `ltx` | LTX 2.3 Fast (default) |
| `ltx-pro` | LTX 2.3 Pro |
| `kling-v3` | Kling v3 Pro |
| `kling-v3-std` | Kling v3 Standard |
| `seedance-2` | Seedance 2.0 |
| `seedance-2-fast` | Seedance 2.0 Fast |
| `wan-2.7` | Wan 2.7 |

For raw endpoint IDs, use `--image-model` and `--text-model` directly.

List all currently active FAL video models:

```bash
python fal_video_chain.py --list-models
python fal_video_chain.py --list-models image-to-video
python fal_video_chain.py --list-models text-to-video
```

---

## Prompts

Prompts in `prompts.txt` describe **action only** — not style, camera, or environment. One line per clip. Lines cycle if there are fewer prompts than `--iterations`.

**Wrong** — describes style:
```
A cinematic shot of a glowing alien cathedral with dramatic lighting
```

**Right** — describes only what changes:
```
The explorer kneels and touches the glowing symbols on the floor
```

### Transition keywords

The engine detects keywords in each prompt and injects the right transition instruction:

| Keyword | Effect |
|---|---|
| *(default)* | Seamless continuation from previous frame |
| `new angle`, `close-up`, `side view`, `overhead` | Match cut to new camera angle |
| `new scene`, `cut to`, `fade to`, `another chamber` | Cinematic fade to new location |

---

## Configuration files

| File | Purpose |
|---|---|
| `scene_state.json` | Visual style, character, environment, camera, audio — injected into every prompt |
| `style_header.txt` | Text prepended to every prompt |
| `style_footer.txt` | Text appended to every prompt |
| `negative_prompt.txt` | Passed as `negative_prompt` to the API |
| `prompts.txt` | Action-only lines, one per clip |

---

## All options

```
--prompts-file FILE     Text file with one action prompt per line
--prompt TEXT           Single prompt reused every clip
--iterations N          Number of clips (0 = infinite)
--duration SECS         Clip duration: 6, 8, or 10 (LTX); 3–15 (others)
--fps N                 Frame rate: 24, 25, 48, or 50 (LTX only)
--resolution RES        1080p (default), 720p
--no-audio              Disable audio generation
--model PRESET          Model preset (see table above)
--image-model ID        Raw FAL endpoint for image-to-video
--text-model ID         Raw FAL endpoint for text-to-video
--first-text-to-video   Generate clip 1 from text; subsequent clips from last frame
--initial-image PATH    Use a local image as the first frame instead
--concat                Join all clips into full_video.mp4 after the run
--dry-run               Print payloads without calling the API
--list-models [CAT]     List active FAL video models and exit
--outdir DIR            Output directory (default: outputs)
--history-depth N       Number of recent prompts injected as story memory (default: 3)
--max-prompt-chars N    Prompt character limit (default: 5000 for LTX/Wan, 8000 for others)
--seed N                Fixed seed for reproducibility
--sleep SECS            Pause between iterations (default: 2.0)
--scene-state-file      Path to scene_state.json
--header-file           Path to style_header.txt
--footer-file           Path to style_footer.txt
--negative-file         Path to negative_prompt.txt
--write-default-files   Write default config files and exit
--overwrite-default-files  Overwrite existing config files
```

---

## Output structure

```
outputs/
├── clips/          clip_0001.mp4, clip_0002.mp4, ...
├── frames/         last_frame_0001.png, last_frame_0002.png, ...
├── full_prompts/   prompt_0001.txt, prompt_0002.txt, ...
├── logs/           result_0001.json, result_0002.json, ...
├── manifest.jsonl  one JSON record per clip
└── full_video.mp4  concatenated result (with --concat)
```

---

## Known limitations

- Minor lighting drift across clips is expected with current models
- Character appearance may shift slightly over long chains
- Camera may micro-jitter despite stabilization instructions
- Audio continuity degrades over many clips

Use 6-second clips, keep action minimal per shot, and avoid rapid scene changes for best results.
