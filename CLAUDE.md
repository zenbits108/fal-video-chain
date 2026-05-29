# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install fal-client requests
export FAL_KEY="your-key"
```

FFmpeg must be installed: `sudo apt install -y ffmpeg`

## Running

Generate default config files (first time only):
```bash
python fal_video_chain.py --write-default-files
```

Run the chain:
```bash
python fal_video_chain.py \
  --first-text-to-video \
  --prompts-file prompts.txt \
  --duration 6 \
  --fps 25 \
  --iterations 10
```

Valid `--duration` values: `6`, `8`, `10`. Valid `--fps` values: `24`, `25`, `48`, `50`.

## Architecture

The entire pipeline lives in a single file: `fal_video_chain.py`.

**Core loop** (`main`):
1. Build a full cinematic prompt from `scene_state.json` + raw action line + transition classifier + recent prompt history
2. Submit to FAL.AI (`fal-ai/ltx-2.3/image-to-video/fast` or `text-to-video/fast`)
3. Download the resulting `.mp4`, extract its last frame with FFmpeg, upload that frame back to FAL
4. Use the uploaded frame URL as `image_url` for the next iteration

**Prompt architecture** — LTX has no cross-clip memory. The engine compensates by rebuilding every prompt from scratch:
- `scene_state.json`: persistent visual style, character, environment, camera grammar, audio bible
- `style_header.txt` / `style_footer.txt`: framing text prepended/appended to every prompt
- `negative_prompt.txt`: passed as the `negative_prompt` API field
- `prompts.txt`: **action-only** lines (one per clip); prompts cycle if there are fewer lines than `--iterations`
- `camera_lock.txt`: camera continuity description (currently not wired into the main prompt builder — it's a standalone reference artifact)

**Transition classifier** (`classify_transition`): reads keywords in the raw action to emit one of three transition strings — opening shot, new scene (fade), new angle (match cut), or seamless continuation.

**Prompt builder** (`build_shot_prompt`): assembles the full prompt; if it exceeds `--max-prompt-chars` (default 5000), it drops recent story memory first, then hard-truncates.

**Outputs** written per clip:
- `outputs/clips/clip_NNNN.mp4`
- `outputs/frames/last_frame_NNNN.png`
- `outputs/full_prompts/prompt_NNNN.txt`
- `outputs/logs/result_NNNN.json` (raw FAL API response)
- `outputs/manifest.jsonl` (one JSON record per clip, append-only)

## Prompts file rules

Lines in `prompts.txt` should describe **action only** — not style, camera, or environment. Those are owned by `scene_state.json`. Transition keywords (`new scene`, `cut to`, `close-up`, etc.) are parsed by `classify_transition` and change the editing instruction injected into the prompt.

## Key constraints

- `fal_client` is imported with a soft fallback (`None`) so the file can be parsed without the package installed; the `main()` guard checks and exits early if it's missing.
- `camera_lock.txt` is not currently read by the script — it exists as a manual reference / clipboard source.
- `scene_state.json` supports an optional `sound_bible` key (parsed in `format_scene_state`) that the default `DEFAULT_SCENE_STATE` dict does not include at the top level; it lives nested under `audio` in the JSON file but is accessed as a sibling key in `format_scene_state` — this is a known quirk.
