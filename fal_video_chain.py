#!/usr/bin/env python3
"""
FAL.AI LTX video chain generator with continuity prompting and shot-state control.

Flow:
1. Generate first clip from text-to-video OR from an initial image.
2. Extract the last frame from that clip.
3. Upload the last frame to FAL.
4. Generate the next clip using that frame as image_url.
5. Rebuild every prompt from persistent scene state.
6. Repeat.

Linux Mint 22 / Python 3.13
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import fal_client
except ModuleNotFoundError:
    fal_client = None  # type: ignore[assignment]

import requests


DEFAULT_IMAGE_TO_VIDEO_MODEL = "fal-ai/ltx-2.3/image-to-video/fast"
DEFAULT_TEXT_TO_VIDEO_MODEL = "fal-ai/ltx-2.3/text-to-video/fast"

VALID_DURATIONS = {6, 8, 10}
VALID_FPS = {24, 25, 48, 50}

# (text_model, image_model, payload_style)
# Payload styles: "ltx" | "kling" | "seedance" | "wan"
MODEL_PRESETS: dict[str, tuple[str, str, str]] = {
    "ltx":              ("fal-ai/ltx-2.3/text-to-video/fast",            "fal-ai/ltx-2.3/image-to-video/fast",            "ltx"),
    "ltx-pro":          ("fal-ai/ltx-2.3/text-to-video",                 "fal-ai/ltx-2.3/image-to-video",                 "ltx"),
    "kling-v3":         ("fal-ai/kling-video/v3/pro/text-to-video",      "fal-ai/kling-video/v3/pro/image-to-video",      "kling"),
    "kling-v3-std":     ("fal-ai/kling-video/v3/standard/text-to-video", "fal-ai/kling-video/v3/standard/image-to-video", "kling"),
    "seedance-2":       ("bytedance/seedance-2.0/text-to-video",         "bytedance/seedance-2.0/image-to-video",         "seedance"),
    "seedance-2-fast":  ("bytedance/seedance-2.0/fast/text-to-video",    "bytedance/seedance-2.0/fast/image-to-video",    "seedance"),
    "wan-2.7":          ("fal-ai/wan/v2.7/text-to-video",                "fal-ai/wan/v2.7/image-to-video",               "wan"),
}


DEFAULT_SCENE_STATE: dict[str, Any] = {
    "project_style": (
        "Dark art-house science fiction with a restrained cinematic tone, subtle film grain, "
        "Kodak 2383 print look, muted teal and amber color grade, realistic physical motion, "
        "no glossy CGI look, no cartoon look, no anime look."
    ),
    "environment": (
        "An impossible ancient alien cathedral interior made of black stone, bronze ribs, cracked obsidian, "
        "deep floor seams, massive shadowed arches, faint blue-green bioluminescent glyphs, drifting dust, "
        "volumetric haze, and distant architectural depth."
    ),
    "character": (
        "A lone adult human explorer, physically consistent in every shot: weathered charcoal expedition coat, "
        "dark utility pants, leather boots, small backpack, tired cautious posture, slow controlled breathing, "
        "no helmet, no armor, no costume changes, holding a brass torch in the right hand."
    ),
    "lighting": (
        "Low-key torch-lit scene with cold blue-green ambient glow from alien glyphs, deep shadows, "
        "soft practical flame flicker, stable exposure, no sudden daylight, no major lighting shift."
    ),
    "camera": {
        "shot_size": "medium-wide over-the-shoulder composition from behind the explorer's left side",
        "lens": "35mm f/2.8",
        "height": "1.6 meters, eye-level to slightly low angle",
        "movement": "tripod-stabilized slow dolly forward",
        "distance": "about half a meter over the entire clip",
        "speed": "constant slow velocity",
        "screen_direction": "the explorer moves slowly away from camera toward the deep center of the cathedral",
        "stabilization": (
            "180-degree shutter equivalent, natural motion blur, film-like cadence, no judder, "
            "no strobing, minimal micro-jitter, no handheld shake, no snap zoom, no sudden pan, "
            "no sudden tilt, no orbit, no rack focus unless explicitly requested"
        ),
    },
    "audio": {
        "ambience": (
            "continuous low-frequency cathedral hum, distant metallic echoes, faint air movement, "
            "subtle torch crackle, consistent volume, no sudden silence"
        ),
        "music": (
            "low evolving ambient drone, sparse dark orchestral texture, slow emotional tension, "
            "same sonic palette across all clips"
        ),
        "dialogue_style": (
            "If dialogue appears, the explorer speaks in a low tired voice with restrained pacing, "
            "natural lip-sync, no exaggerated performance."
        ),
    },
    "anchors": (
        "Stable foreground anchors: black stone pillar fragments near the frame edge, cracked obsidian floor lines, "
        "blue-green glyph glow along the floor center. Preserve these anchors for parallax continuity."
    ),
}


DEFAULT_NEGATIVE_PROMPT = """
subtitle, caption, text, watermark, logo, blurry, overexposed, low contrast, flickering,
distorted proportions, unnatural skin tones, extra limbs, disfigured hands, mismatched motion,
silent or muted audio, distorted voice, echo, background noise, off-sync audio, jittery movement,
awkward pauses, incorrect timing, unnatural transitions, AI artifacts, style drift, costume drift,
camera reset, subject replacement, random new characters, random new props, whip pan, snap zoom,
sudden orbit, sudden tilt, sudden pan, handheld shake, jump cut, reversed camera direction,
speed ramp, fast camera movement, aggressive parallax warp, daylight shift, new costume,
face change, modern UI, hologram text, captions
""".strip()


def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def list_models(category: str | None = None) -> None:
    categories = [category] if category else ["text-to-video", "image-to-video"]
    api_key = os.environ.get("FAL_KEY", "")
    headers = {"Authorization": f"Key {api_key}"} if api_key else {}

    for cat in categories:
        params = {"category": cat, "status": "active", "limit": 100}
        resp = requests.get("https://api.fal.ai/v1/models", params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        models = resp.json().get("models", [])

        print(f"\n{'─' * 80}")
        print(f"  {cat.upper()}  ({len(models)} models)")
        print(f"{'─' * 80}")
        print(f"  {'ENDPOINT ID':<55} {'DISPLAY NAME'}")
        print(f"  {'─' * 54} {'─' * 22}")
        for m in models:
            eid = m["endpoint_id"]
            name = m["metadata"].get("display_name", "")
            tags = m["metadata"].get("tags", [])
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            print(f"  {eid:<55} {name}{tag_str}")


def run(cmd: list[str], quiet: bool = False) -> None:
    print(" ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
        )
    except subprocess.CalledProcessError as exc:
        die(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}")


def require_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        die("ffmpeg is not available. Install it with: sudo apt install -y ffmpeg")


def validate_generation_args(duration: int, fps: int, payload_style: str = "ltx") -> None:
    if payload_style == "ltx":
        if duration not in VALID_DURATIONS:
            die(f"Invalid --duration {duration} for LTX. Expected one of: {sorted(VALID_DURATIONS)}")
        if fps not in VALID_FPS:
            die(f"Invalid --fps {fps} for LTX. Expected one of: {sorted(VALID_FPS)}")


def read_text_file(path: str | None, default: str = "") -> str:
    if not path:
        return default.strip()

    p = Path(path)
    if not p.exists():
        print(f"\nWarning: {p} not found. Using built-in default.")
        return default.strip()

    return p.read_text(encoding="utf-8").strip()


def write_default_file(path: Path, text: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        print(f"Exists, not overwritten: {path}")
        return

    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def load_json_file(path: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if not path:
        return default

    p = Path(path)
    if not p.exists():
        print(f"\nWarning: {p} not found. Using built-in scene state.")
        return default

    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"Invalid JSON in {p}: {exc}")


def download_file(url: str, out_path: Path) -> None:
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract_last_frame(video_path: Path, frame_path: Path) -> None:
    run([
        "ffmpeg",
        "-y",
        "-sseof", "-0.1",
        "-i", str(video_path),
        "-update", "1",
        "-q:v", "1",
        str(frame_path),
    ], quiet=True)


def get_video_url(result: dict[str, Any]) -> str:
    data = result.get("data", result)

    candidates = [
        data.get("video"),
        data.get("output"),
        data.get("file"),
    ]

    for item in candidates:
        if isinstance(item, dict) and item.get("url"):
            return item["url"]
        if isinstance(item, str) and item.startswith("http"):
            return item

    if isinstance(data.get("videos"), list) and data["videos"]:
        item = data["videos"][0]
        if isinstance(item, dict) and item.get("url"):
            return item["url"]

    raise ValueError(
        f"Could not find video URL in result:\n{json.dumps(result, indent=2)[:3000]}"
    )


def submit_fal(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    print(f"\nSubmitting to {model}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    try:
        result = fal_client.subscribe(
            model,
            arguments=payload,
            with_logs=True,
        )
    except Exception as exc:
        print("\nFAL submission failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise

    print("\nFAL result:")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:3000])
    return result


def upload_file(path: Path) -> str:
    print(f"\nUploading frame: {path}")
    return fal_client.upload_file(str(path))


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        path = Path(args.prompts_file)
        if not path.exists():
            die(f"Prompts file not found: {path}")

        prompts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        if not prompts:
            die("Prompts file is empty.")

        return prompts

    if args.prompt:
        return [args.prompt.strip()]

    die("Provide --prompt or --prompts-file.")


def classify_transition(raw_prompt: str, iteration: int) -> str:
    text = raw_prompt.lower()

    if iteration == 1:
        return (
            "Opening cinematic shot. Establish the explorer already inside the alien cathedral. "
            "Begin with a stable slow dolly forward."
        )

    new_scene_words = [
        "new scene",
        "new location",
        "cut to",
        "fade to",
        "transition to",
        "outside",
        "different room",
        "another chamber",
    ]

    new_angle_words = [
        "new angle",
        "side view",
        "close-up",
        "close up",
        "overhead",
        "reverse angle",
        "profile shot",
        "front view",
    ]

    if any(word in text for word in new_scene_words):
        return (
            "Slow cinematic fade transition into the new location while preserving the same character, "
            "costume, lighting style, sound palette, and art-house visual grammar. After the fade, resume "
            "a stable slow dolly."
        )

    if any(word in text for word in new_angle_words):
        return (
            "Match cut to the new camera angle while preserving the explorer's motion direction, timing, "
            "costume, lighting, lens feel, and sound continuity. The cut is editorially clean, not chaotic."
        )

    return (
        "Continue seamlessly from the supplied first frame, no cut, no camera reset, no direction reversal. "
        "This is the next few seconds of the same unbroken take."
    )


def format_scene_state(state: dict[str, Any]) -> str:
    camera = state.get("camera", {})
    audio = state.get("audio", {})
    sound_bible = state.get("sound_bible", audio.get("sound_bible", {}))
    style_fingerprint = state.get("style_fingerprint", audio.get("style_fingerprint", ""))

    return f"""
PERSISTENT SCENE STATE:
Project style: {state.get("project_style", "")}
Style fingerprint: {style_fingerprint}
Environment: {state.get("environment", "")}
Character continuity: {state.get("character", "")}
Lighting continuity: {state.get("lighting", "")}
Camera lock: {camera.get("shot_size", "")}; {camera.get("lens", "")}; {camera.get("height", "")}; {camera.get("movement", "")}; {camera.get("distance", "")}; {camera.get("speed", "")}; {camera.get("screen_direction", "")}; {camera.get("stabilization", "")}
Audio continuity: ambience: {audio.get("ambience", "")}; music: {audio.get("music", "")}; dialogue: {audio.get("dialogue_style", "")}
Sound bible: ambience loop: {sound_bible.get("ambience_loop", "")}; music identity: {sound_bible.get("music_identity", "")}; mix rules: {sound_bible.get("mix_rules", "")}
Spatial anchors: {state.get("anchors", "")}
""".strip()

def build_recent_memory(prompt_history: list[str], max_history: int) -> str:
    if max_history <= 0:
        return ""

    recent = prompt_history[-max_history:]

    if not recent:
        return ""

    return (
        "RECENT STORY MEMORY: "
        + " ".join(
            f"Previous shot {idx + 1}: {text}"
            for idx, text in enumerate(recent)
        )
    )


def build_shot_prompt(
    state: dict[str, Any],
    raw_action: str,
    transition: str,
    prompt_history: list[str],
    max_history: int,
    extra_header: str,
    extra_footer: str,
    max_chars: int,
) -> str:
    """
    Rebuild the complete shot prompt every time.

    LTX does not remember the last clip. This does.
    Not through magic. Through tedious repetition, humanity's most reliable technology.
    """
    scene_state_text = format_scene_state(state)
    recent_memory = build_recent_memory(prompt_history, max_history)

    prompt = f"""
{extra_header}

{scene_state_text}

TRANSITION / EDITING:
{transition}

{recent_memory}

CURRENT SHOT ACTION:
{raw_action}

SHOT EXECUTION:
Render this as a single flowing LTX shot. The current shot action controls only story behavior.
Do not allow it to override the persistent character, environment, camera lock, lighting continuity,
audio continuity, or spatial anchors unless the action explicitly says this is a new scene or new camera angle.
Preserve the supplied first frame as the first frame of the new clip whenever image_url is provided.
Movement must be physically plausible, slow, and continuous.

AUDIO EXECUTION:
Maintain the same ambient cathedral hum, torch crackle, distant metallic echoes, and low evolving drone.
If dialogue is included in the current shot action, keep the same voice tone, pacing, language, and emotional restraint.
No sudden music genre change. No sudden silence. No random voices.

{extra_footer}
""".strip()

    prompt = " ".join(prompt.split())

    if len(prompt) <= max_chars:
        return prompt

    # Drop recent memory and tighten execution blocks; always keep the footer.
    prompt = f"""
{extra_header}

{scene_state_text}

TRANSITION / EDITING:
{transition}

CURRENT SHOT ACTION:
{raw_action}

SHOT EXECUTION:
Render this as a single flowing LTX shot. Preserve persistent character, environment, camera lock,
lighting continuity, audio continuity, and spatial anchors. Continue from the supplied first frame when image_url is provided.

AUDIO EXECUTION:
Maintain the same ambient sound, music style, and dialogue tone.

{extra_footer}
""".strip()

    prompt = " ".join(prompt.split())

    if len(prompt) <= max_chars:
        return prompt

    # Last resort: keep header + action + footer, trim the scene state from the middle.
    fixed_parts = " ".join(f"{extra_header} CURRENT SHOT ACTION: {raw_action} {extra_footer}".split())
    if len(fixed_parts) >= max_chars:
        return fixed_parts[:max_chars]

    budget = max_chars - len(fixed_parts) - 1
    trimmed_state = " ".join(scene_state_text.split())[:budget]
    return " ".join(f"{extra_header} {trimmed_state} CURRENT SHOT ACTION: {raw_action} {extra_footer}".split())


def make_payload(
    prompt: str,
    duration: int,
    image_url: str | None,
    resolution: str,
    fps: int,
    generate_audio: bool,
    negative_prompt: str = "",
    seed: int | None = None,
    payload_style: str = "ltx",
) -> dict[str, Any]:
    if payload_style == "kling":
        return _make_kling_payload(prompt, duration, image_url, generate_audio, negative_prompt, seed)
    if payload_style == "seedance":
        return _make_seedance_payload(prompt, duration, image_url, resolution, generate_audio, seed)
    if payload_style == "wan":
        return _make_wan_payload(prompt, duration, image_url, resolution, negative_prompt, seed)
    return _make_ltx_payload(prompt, duration, image_url, resolution, fps, generate_audio, negative_prompt, seed)


def _make_ltx_payload(
    prompt: str,
    duration: int,
    image_url: str | None,
    resolution: str,
    fps: int,
    generate_audio: bool,
    negative_prompt: str,
    seed: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "fps": fps,
        "generate_audio": generate_audio,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed
    if image_url:
        payload["image_url"] = image_url
    return payload


def _make_kling_payload(
    prompt: str,
    duration: int,
    image_url: str | None,
    generate_audio: bool,
    negative_prompt: str,
    seed: int | None,
) -> dict[str, Any]:
    # Kling uses string duration, aspect_ratio, start_image_url (not image_url)
    payload: dict[str, Any] = {
        "prompt": prompt,
        "duration": str(duration),
        "aspect_ratio": "16:9",
        "generate_audio": generate_audio,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed
    if image_url:
        payload["start_image_url"] = image_url
    return payload


def _make_seedance_payload(
    prompt: str,
    duration: int,
    image_url: str | None,
    resolution: str,
    generate_audio: bool,
    seed: int | None,
) -> dict[str, Any]:
    # Seedance uses string duration, aspect_ratio; no fps, no negative_prompt
    payload: dict[str, Any] = {
        "prompt": prompt,
        "duration": str(duration),
        "aspect_ratio": "16:9",
        "resolution": resolution,
        "generate_audio": generate_audio,
    }
    if seed is not None:
        payload["seed"] = seed
    if image_url:
        payload["image_url"] = image_url
    return payload


def _make_wan_payload(
    prompt: str,
    duration: int,
    image_url: str | None,
    resolution: str,
    negative_prompt: str,
    seed: int | None,
) -> dict[str, Any]:
    # Wan uses int duration, no fps, no generate_audio; negative_prompt max 500 chars
    payload: dict[str, Any] = {
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "enable_prompt_expansion": False,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt[:500]
    if seed is not None:
        payload["seed"] = seed
    if image_url:
        payload["image_url"] = image_url
    return payload


def append_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def concat_clips(clips_dir: Path, outdir: Path) -> None:
    clips = sorted(clips_dir.glob("clip_*.mp4"))
    if not clips:
        print("\nNo clips found to concatenate.")
        return

    concat_list = outdir / "concat_list.txt"
    concat_list.write_text(
        "\n".join(f"file '{c.resolve()}'" for c in clips) + "\n",
        encoding="utf-8",
    )
    out_path = outdir / "full_video.mp4"
    print(f"\nConcatenating {len(clips)} clips into {out_path}")
    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(out_path),
    ], quiet=True)
    print(f"Concatenated video: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAL.AI LTX continuity video chain generator"
    )

    parser.add_argument("--prompt", help="Single action prompt reused for every clip.")
    parser.add_argument("--prompts-file", help="Text file with one shot action per line.")
    parser.add_argument("--outdir", default="outputs", help="Local output directory.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of clips. Use 0 for infinite.")
    parser.add_argument("--duration", type=int, default=6, help="Valid values: 6, 8, 10.")
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--fps", type=int, default=25, help="Valid values: 24, 25, 48, 50.")
    parser.add_argument("--no-audio", action="store_true")

    parser.add_argument("--initial-image", help="Optional local image path for first image-to-video clip.")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model preset shorthand. One of: {', '.join(MODEL_PRESETS)}. "
             "Overrides --image-model and --text-model and selects the correct payload schema.",
    )
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_TO_VIDEO_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_TO_VIDEO_MODEL)
    parser.add_argument("--first-text-to-video", action="store_true")

    parser.add_argument("--scene-state-file", default="scene_state.json")
    parser.add_argument("--header-file", default="style_header.txt")
    parser.add_argument("--footer-file", default="style_footer.txt")
    parser.add_argument("--negative-file", default="negative_prompt.txt")

    parser.add_argument("--history-depth", type=int, default=3)
    parser.add_argument("--max-prompt-chars", type=int, default=None,
                        help="Max prompt length in characters. Defaults to 5000 for LTX/Wan, 8000 for others.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=2.0)

    parser.add_argument("--dry-run", action="store_true", help="Print the payload for each iteration without calling the API.")
    parser.add_argument("--concat", action="store_true", help="Concatenate all clips into a single video after the run completes.")

    parser.add_argument(
        "--list-models",
        nargs="?",
        const="all",
        metavar="CATEGORY",
        help="List active FAL video models and exit. Optionally filter: text-to-video or image-to-video.",
    )

    parser.add_argument("--write-default-files", action="store_true")
    parser.add_argument("--overwrite-default-files", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        cat = None if args.list_models == "all" else args.list_models
        list_models(cat)
        return

    if args.write_default_files:
        write_default_file(
            Path(args.scene_state_file),
            json.dumps(DEFAULT_SCENE_STATE, indent=2, ensure_ascii=False),
            overwrite=args.overwrite_default_files,
        )
        write_default_file(
            Path(args.header_file),
            "LTX DIRECTOR HEADER: Treat this prompt as a precise director's shot list, not a vague mood board.",
            overwrite=args.overwrite_default_files,
        )
        write_default_file(
            Path(args.footer_file),
            "LTX CONTINUITY FOOTER: No style drift, no camera reset, no costume change, no random characters, no captions, no text overlays.",
            overwrite=args.overwrite_default_files,
        )
        write_default_file(
            Path(args.negative_file),
            DEFAULT_NEGATIVE_PROMPT,
            overwrite=args.overwrite_default_files,
        )
        print("Default files are ready.")
        return

    if not args.dry_run:
        if fal_client is None:
            die("fal-client is not installed. Run: pip install fal-client requests")

        if not os.environ.get("FAL_KEY"):
            die("FAL_KEY is not set. Run: export FAL_KEY='your-key'")

    # Resolve --model preset into endpoints + payload style.
    payload_style = "ltx"
    if args.model:
        if args.model not in MODEL_PRESETS:
            die(f"Unknown --model '{args.model}'. Valid presets: {', '.join(MODEL_PRESETS)}")
        text_model, image_model, payload_style = MODEL_PRESETS[args.model]
        args.text_model = text_model
        args.image_model = image_model
        print(f"Model preset: {args.model}  (style={payload_style})")
        print(f"  text:  {args.text_model}")
        print(f"  image: {args.image_model}")

    if args.max_prompt_chars is None:
        args.max_prompt_chars = 5000 if payload_style in ("ltx", "wan") else 8000

    require_ffmpeg()
    validate_generation_args(args.duration, args.fps, payload_style)

    outdir = Path(args.outdir)
    clips_dir = outdir / "clips"
    frames_dir = outdir / "frames"
    logs_dir = outdir / "logs"
    prompts_dir = outdir / "full_prompts"
    manifest_path = outdir / "manifest.jsonl"

    for d in (clips_dir, frames_dir, logs_dir, prompts_dir):
        d.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args)

    scene_state = load_json_file(args.scene_state_file, DEFAULT_SCENE_STATE)
    extra_header = read_text_file(
        args.header_file,
        "LTX DIRECTOR HEADER: Treat this as a precise director's shot list.",
    )
    extra_footer = read_text_file(
        args.footer_file,
        "LTX CONTINUITY FOOTER: Preserve continuity and avoid style drift.",
    )
    negative_prompt = read_text_file(args.negative_file, DEFAULT_NEGATIVE_PROMPT)

    generate_audio = not args.no_audio
    prompt_history: list[str] = []
    image_url: str | None = None

    if args.initial_image:
        initial = Path(args.initial_image)
        if not initial.exists():
            die(f"Initial image not found: {initial}")
        image_url = upload_file(initial)

    total = args.iterations
    i = 1

    while True:
        if total and i > total:
            break

        raw_action = prompts[(i - 1) % len(prompts)]
        transition = classify_transition(raw_action, i)

        full_prompt = build_shot_prompt(
            state=scene_state,
            raw_action=raw_action,
            transition=transition,
            prompt_history=prompt_history,
            max_history=args.history_depth,
            extra_header=extra_header,
            extra_footer=extra_footer,
            max_chars=args.max_prompt_chars,
        )

        full_prompt_path = prompts_dir / f"prompt_{i:04d}.txt"
        full_prompt_path.write_text(full_prompt + "\n", encoding="utf-8")

        if i == 1 and args.first_text_to_video and not image_url:
            model = args.text_model
            payload = make_payload(
                prompt=full_prompt,
                duration=args.duration,
                image_url=None,
                resolution=args.resolution,
                fps=args.fps,
                generate_audio=generate_audio,
                negative_prompt=negative_prompt,
                seed=args.seed,
                payload_style=payload_style,
            )
        else:
            if not image_url and not args.dry_run:
                die("No image_url available. Use --initial-image or --first-text-to-video.")

            model = args.image_model
            payload = make_payload(
                prompt=full_prompt,
                duration=args.duration,
                image_url=image_url,
                resolution=args.resolution,
                fps=args.fps,
                generate_audio=generate_audio,
                negative_prompt=negative_prompt,
                seed=args.seed,
                payload_style=payload_style,
            )

        label = f"{i}/{total}" if total else str(i)
        print(f"\n=== ITERATION {label} ===")
        print(f"Raw action: {raw_action}")
        print(f"Transition: {transition}")
        print(f"Prompt chars: {len(full_prompt)}")
        print(f"Full prompt saved to: {full_prompt_path}")

        if args.dry_run:
            print("\n[DRY RUN] Payload:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            i += 1
            continue

        result = submit_fal(model, payload)

        log_path = logs_dir / f"result_{i:04d}.json"
        log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        video_url = get_video_url(result)
        clip_path = clips_dir / f"clip_{i:04d}.mp4"
        frame_path = frames_dir / f"last_frame_{i:04d}.png"

        print(f"\nDownloading video:\n{video_url}")
        download_file(video_url, clip_path)

        print(f"\nExtracting last frame from {clip_path}")
        extract_last_frame(clip_path, frame_path)

        image_url = upload_file(frame_path)
        prompt_history.append(raw_action)

        record = {
            "iteration": i,
            "model": model,
            "raw_action": raw_action,
            "transition": transition,
            "full_prompt_path": str(full_prompt_path),
            "clip_path": str(clip_path),
            "last_frame_path": str(frame_path),
            "result_log_path": str(log_path),
            "duration": args.duration,
            "fps": args.fps,
            "resolution": args.resolution,
            "seed": args.seed,
            "generated_audio": generate_audio,
            "scene_state_file": args.scene_state_file,
        }

        append_manifest(manifest_path, record)

        print("\nSaved:")
        print(f"  clip:        {clip_path}")
        print(f"  frame:       {frame_path}")
        print(f"  log:         {log_path}")
        print(f"  full prompt: {full_prompt_path}")
        print(f"  manifest:    {manifest_path}")

        i += 1
        time.sleep(args.sleep)

    if args.concat and not args.dry_run:
        concat_clips(clips_dir, outdir)

    print("\nDone. The machine has eaten enough frames for now.")


if __name__ == "__main__":
    main()