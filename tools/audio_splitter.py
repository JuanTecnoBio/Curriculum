#!/usr/bin/env python3
"""Command line utility for splitting audio and video files into equal parts.

Examples
--------
Split an audio file into three equal parts::

    python tools/audio_splitter.py -i song.m4a --parts 3

Split an MP3 file into clips of 30 seconds each::

    python tools/audio_splitter.py -i interview.mp3 --part-duration 30

Customize the output file name pattern::

    python tools/audio_splitter.py -i lecture.mp4 --parts 4 --pattern "{stem}_seg{idx}.{ext}"

Enable audio fades at the boundaries of each segment::

    python tools/audio_splitter.py -i ambient.m4a --parts 2 --crossfade-ms 500
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


class SplitterError(RuntimeError):
    """Custom error for predictable failures."""


@dataclass
class Segment:
    index: int
    start: float
    duration: float

    @property
    def pretty_index(self) -> int:
        return self.index + 1


@dataclass
class MediaInfo:
    duration: float
    has_video: bool


SUPPORTED_EXTENSIONS = {"m4a", "mp3", "mp4"}


def ensure_dependencies(verbose: bool) -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        joined = ", ".join(missing)
        raise SplitterError(
            f"Required tool(s) not found in PATH: {joined}. Please install ffmpeg."  # noqa: E501
        )
    if verbose:
        print("Dependencies found: ffmpeg, ffprobe")


def run_command(cmd: List[str], *, verbose: bool, dry_run: bool) -> subprocess.CompletedProcess:
    if verbose or dry_run:
        print("$", " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, returncode=0)
    return subprocess.run(cmd, check=True)


def probe_media(path: Path, *, verbose: bool) -> MediaInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    if verbose:
        print("$", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:  # pragma: no cover - handled elsewhere
        raise SplitterError("ffprobe executable not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise SplitterError(f"ffprobe failed: {exc.stderr.strip() or exc.stdout.strip()}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SplitterError("Unable to parse ffprobe JSON output.") from exc

    try:
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitterError("Could not read media duration from ffprobe output.") from exc

    has_video = any(stream.get("codec_type") == "video" for stream in payload.get("streams", []))
    if verbose:
        print(f"Duration detected: {duration:.3f} seconds")
        print(f"Video stream detected: {'yes' if has_video else 'no'}")
    return MediaInfo(duration=duration, has_video=has_video)


def generate_segments(total_duration: float, *, parts: Optional[int], part_duration: Optional[float]) -> List[Segment]:
    if parts is None and part_duration is None:
        raise SplitterError("Either --parts or --part-duration must be provided.")
    if parts is not None and part_duration is not None:
        raise SplitterError("--parts and --part-duration are mutually exclusive.")

    segments: List[Segment] = []
    if parts is not None:
        if parts <= 0:
            raise SplitterError("--parts must be a positive integer.")
        base_duration = total_duration / parts
        for idx in range(parts):
            start = idx * base_duration
            end = total_duration if idx == parts - 1 else (idx + 1) * base_duration
            segments.append(Segment(index=idx, start=start, duration=end - start))
    else:
        assert part_duration is not None
        if part_duration <= 0:
            raise SplitterError("--part-duration must be greater than zero.")
        parts = max(1, int(math.ceil(total_duration / part_duration)))
        for idx in range(parts):
            start = idx * part_duration
            end = min(total_duration, (idx + 1) * part_duration)
            segments.append(Segment(index=idx, start=start, duration=end - start))

    return segments


def format_output_name(pattern: str, input_path: Path, segment: Segment) -> Path:
    stem = input_path.stem
    ext = input_path.suffix.lstrip(".")
    try:
        filename = pattern.format(stem=stem, idx=segment.pretty_index, ext=ext)
    except KeyError as exc:
        raise SplitterError(
            f"Invalid output pattern: missing placeholder {{{exc.args[0]}}}."
        )
    except IndexError as exc:
        raise SplitterError(f"Invalid output pattern formatting: {exc}.")
    return input_path.with_name(filename)


def determine_codecs(ext: str, *, has_video: bool, reencode: bool) -> List[str]:
    if not reencode:
        return ["-c", "copy"]

    ext = ext.lower()
    args: List[str] = []
    if has_video:
        args.extend(["-c:v", "copy"])

    if ext in {"m4a", "mp4"}:
        args.extend(["-c:a", "aac", "-b:a", "192k"])
    elif ext == "mp3":
        args.extend(["-c:a", "libmp3lame", "-b:a", "192k"])
    else:
        args.extend(["-c:a", "aac", "-b:a", "192k"])
    return args


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    segment: Segment,
    *,
    reencode: bool,
    has_video: bool,
    ext: str,
    crossfade_s: float,
    verbose: bool,
) -> List[str]:
    cmd: List[str] = ["ffmpeg", "-y"]
    if verbose:
        cmd.append("-hide_banner")
    else:
        cmd.extend(["-hide_banner", "-loglevel", "error"])

    cmd.extend(["-ss", f"{segment.start:.6f}", "-i", str(input_path), "-t", f"{segment.duration:.6f}"])
    cmd.extend(["-map", "0"])

    filters: List[str] = []
    if crossfade_s > 0:
        if segment.duration <= crossfade_s:
            raise SplitterError(
                "Segment duration is too short for the requested crossfade."
            )
        fade_out_start = max(segment.duration - crossfade_s, 0)
        filters.append(f"afade=t=in:st=0:d={crossfade_s}")
        filters.append(f"afade=t=out:st={fade_out_start}:d={crossfade_s}")

    if filters:
        cmd.extend(["-af", ",".join(filters)])

    cmd.extend(determine_codecs(ext, has_video=has_video, reencode=reencode or crossfade_s > 0))
    cmd.append(str(output_path))
    return cmd


def process_segments(
    input_path: Path,
    segments: Iterable[Segment],
    *,
    media_info: MediaInfo,
    pattern: str,
    crossfade_ms: int,
    verbose: bool,
    dry_run: bool,
) -> None:
    crossfade_s = crossfade_ms / 1000.0
    ext = input_path.suffix.lstrip(".")
    reencode_required = crossfade_s > 0

    for segment in segments:
        output_path = format_output_name(pattern, input_path, segment)
        if output_path.exists() and not dry_run:
            raise SplitterError(f"Output file already exists: {output_path}")

        cmd = build_ffmpeg_command(
            input_path,
            output_path,
            segment,
            reencode=reencode_required,
            has_video=media_info.has_video,
            ext=ext,
            crossfade_s=crossfade_s,
            verbose=verbose,
        )

        try:
            run_command(cmd, verbose=verbose, dry_run=dry_run)
        except subprocess.CalledProcessError:
            if reencode_required:
                raise
            # Retry with re-encoding
            if verbose:
                print("Copy mode failed; retrying with re-encoding due to potential keyframe issues.")
            retry_cmd = build_ffmpeg_command(
                input_path,
                output_path,
                segment,
                reencode=True,
                has_video=media_info.has_video,
                ext=ext,
                crossfade_s=crossfade_s,
                verbose=verbose,
            )
            run_command(retry_cmd, verbose=verbose, dry_run=dry_run)



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split audio/video files into evenly sized parts or by duration.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input media file (.m4a, .mp3, .mp4)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parts", type=int, help="Number of equal parts to create")
    group.add_argument(
        "--part-duration",
        type=float,
        help="Target duration (in seconds) for each part",
    )

    parser.add_argument(
        "--pattern",
        default="{stem}_part{idx:02d}.{ext}",
        help="Output filename pattern (default: {stem}_part{idx:02d}.{ext})",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=int,
        default=0,
        help="Apply fade in/out of the given duration in milliseconds",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing them")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")

    args = parser.parse_args(argv)

    if args.crossfade_ms < 0:
        raise SplitterError("--crossfade-ms must be zero or a positive integer.")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        input_path = Path(args.input)
        if not input_path.exists():
            raise SplitterError(f"Input file not found: {input_path}")
        if input_path.suffix.lstrip(".").lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise SplitterError(f"Unsupported input extension. Supported: {allowed}.")

        ensure_dependencies(verbose=args.verbose)
        media_info = probe_media(input_path, verbose=args.verbose)
        segments = generate_segments(
            media_info.duration,
            parts=args.parts,
            part_duration=args.part_duration,
        )
        if args.verbose:
            for segment in segments:
                print(
                    f"Segment {segment.pretty_index}: start={segment.start:.3f}s "
                    f"duration={segment.duration:.3f}s"
                )

        process_segments(
            input_path,
            segments,
            media_info=media_info,
            pattern=args.pattern,
            crossfade_ms=args.crossfade_ms,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print("Dry run complete. No files were created.")
        return 0
    except SplitterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Operation cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
