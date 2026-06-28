import os
import glob
import json
import cv2
import pybboxes as pbx
import yaml
import argparse
from ultralytics import YOLO
import shutil
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn, MofNCompleteColumn
from natsort import natsorted
from os.path import join as osj
import gc
import torch
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()
console = Console()


console.print(f"Reading the Configuration file from {args.config}", style="bold green")
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


console.print("Loading YOLO Model...", style="bold green")
model = YOLO(config["model_path"])

# ── Discover videos ───────────────────────────────────────────────────────────
video_extensions = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV',
                    '*.mkv', '*.MKV', '*.wmv', '*.WMV', '*.flv', '*.FLV']
videos = []
for ext in video_extensions:
    videos.extend(glob.glob(os.path.join(config['videos_path'], ext)))
videos = natsorted(videos)

# ── Create output folder ──────────────────────────────────────────────────────
anonymized_videos_path = config["output_folder"]
os.makedirs(anonymized_videos_path, exist_ok=True)

# =============================================================================
# Helper functions — all defined before any code calls them
# =============================================================================

def blur_regions(image, regions):
    """Blur bounding-box regions in-place using repeated Gaussian blur passes."""
    blur_k = config["blur_radius"] if config["blur_radius"] % 2 != 0 else config["blur_radius"] + 1
    passes = config.get("blur_passes", 1)
    for region in regions:
        x1, y1, x2, y2 = (round(v) for v in region)
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        if x1 < x2 and y1 < y2:
            roi = image[y1:y2, x1:x2]
            for _ in range(passes):
                roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
            image[y1:y2, x1:x2] = roi
    return image


def load_frame_detections(frame_num, detections_dir, width, height):
    """Load detections for a single frame on-demand from the YOLO label files."""
    pattern = os.path.join(detections_dir, f"*_{frame_num}.txt")
    files = glob.glob(pattern)
    if not files:
        return []
    detections = []
    with open(files[0], 'r') as fin:
        for line in fin:
            coords = [float(v) for v in line.split()[1:]]
            coords = pbx.convert_bbox(coords, from_type="yolo", to_type="voc",
                                      image_size=(int(width), int(height)))
            detections.append(coords)
    return detections


def stream_to_ffmpeg(input_path, output_path, fps, frame_width, frame_height, detections_dir, total_frames):
    """
    Read frames from input_path, optionally blur detected regions, and stream
    raw bytes to an ffmpeg subprocess for H.264 encoding.
    Detections are loaded one frame at a time (on-demand) to keep memory flat.
    """
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{frame_width}x{frame_height}',
        '-framerate', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',  # required for broad player compatibility
        output_path
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    cap = cv2.VideoCapture(input_path)
    frame_idx = 1
    try:
        with Progress(
            TextColumn("  Encoding "),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("", total=total_frames)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if detections_dir and os.path.exists(detections_dir):
                    dets = load_frame_detections(frame_idx, detections_dir, frame_width, frame_height)
                    if dets:
                        frame = blur_regions(frame, dets)

                proc.stdin.write(frame.tobytes())
                del frame
                progress.advance(task)

                if frame_idx % 100 == 0:
                    gc.collect()

                frame_idx += 1

        proc.stdin.close()
        proc.wait(timeout=600)

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {proc.stderr.read().decode()}")

    except Exception:
        proc.terminate()
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        cap.release()
        gc.collect()


def process_video(video_path, model, anonymized_videos_path):
    """
    Full pipeline for a single video:
      1. YOLO detection  (written to a temp dir)
      2. Frame-by-frame blur  (detections read on-demand from temp dir)
      3. H.264 encoding via ffmpeg
    The temp dir is deleted automatically after each video, keeping memory flat.
    """
    vid_name, _ = os.path.splitext(os.path.basename(video_path))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            # ── Get video info up front ───────────────────────────────────
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # ── Phase 1: detect ───────────────────────────────────────────
            console.print(f"  [1/2] Detecting  {vid_name}  ({total_frames} frames)", style="cyan")
            device = 'cuda:0' if config["gpu_avail"] else 'cpu'
            # stream=True yields one result per frame — keeps YOLO memory flat.
            # We must iterate the generator so frames are actually processed and
            # label files are written to disk.
            with Progress(
                TextColumn("  Detecting "),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("", total=total_frames)
                for _ in model(
                    source=video_path,
                    save=False,
                    save_txt=True,
                    conf=config['detection_conf_thresh'],
                    imgsz=config.get('imgsz', 1280),
                    device=device,
                    project=tmp,
                    name="det",
                    verbose=False,
                    stream=True,
                ):
                    progress.advance(task)

            if config["gpu_avail"]:
                torch.cuda.empty_cache()
            gc.collect()

            # ── Phase 2: blur + encode ────────────────────────────────────
            console.print(f"  [2/2] Encoding   {vid_name}", style="cyan")
            detections_dir = os.path.join(tmp, "det", "labels")

            out_path = osj(anonymized_videos_path, vid_name + '.mp4')
            stream_to_ffmpeg(video_path, out_path, fps, w, h, detections_dir, total_frames)

        console.print(f"  ✓ Done  {vid_name}", style="bold green")

    except Exception as e:
        console.print(f"  ✗ Failed {vid_name}: {e}", style="bold red")


# =============================================================================
# Main execution
# =============================================================================

if not videos:
    console.print("No videos found. Check 'videos_path' in your config.", style="bold red")
else:
    console.print(f"Found {len(videos)} video(s). Processing...", style="bold green")
    for i, video in enumerate(videos, 1):
        console.print(f"\nVideo {i}/{len(videos)}: {os.path.basename(video)}", style="bold white")
        process_video(video, model, anonymized_videos_path)

    del model
    gc.collect()
    if config["gpu_avail"]:
        torch.cuda.empty_cache()

console.print(f"Blurred videos are stored in {anonymized_videos_path}", style="bold yellow")
