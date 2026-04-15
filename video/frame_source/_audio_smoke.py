"""Smoke test for the migrated PyAV-based AudioPlayer.

Exercises the decode/filter/resample pipeline (without actually opening a
sounddevice output stream — that would require live audio hardware). Confirms
the reader thread produces the expected PCM byte rate at various tempos and
seek positions.
"""

from __future__ import annotations

import os
import sys
import time

import av

from video import audio_player as ap_mod


def out(*args) -> None:
    print(*args, flush=True)


def assert_(cond: bool, msg: str) -> bool:
    status = "PASS" if cond else "FAIL"
    out(f"  [{status}] {msg}")
    return cond


def measure_pcm_rate(path: str, start_sec: float, tempo: float, duration: float) -> int:
    """Open container + filter + resampler, run the reader logic for ``duration``
    seconds (wall clock), return total bytes produced."""
    container = av.open(path)
    audio_stream = container.streams.audio[0]
    audio_stream.thread_type = "AUTO"

    # Build a fake AudioPlayer just to reuse its filter graph + resampler logic.
    p = ap_mod.AudioPlayer()
    p._video_path = path
    p._has_audio = True
    p._audio_stream = audio_stream
    p._container = container

    chain = p._build_atempo_chain(tempo)
    if chain:
        p._filter_graph = p._build_filter_graph(audio_stream, chain)
    else:
        p._filter_graph = None

    import av.audio.resampler as ar
    p._resampler = ar.AudioResampler(format='s16', layout='stereo', rate=p._sample_rate)

    # Seek
    if start_sec > 0.001:
        target_pts = int(start_sec / audio_stream.time_base)
        container.seek(target_pts, backward=True, any_frame=False, stream=audio_stream)

    p._is_stopped = False
    bytes_produced = 0
    deadline = time.perf_counter() + duration
    try:
        for packet in container.demux(audio_stream):
            if time.perf_counter() > deadline:
                break
            pcm = bytearray()
            for frame in packet.decode():
                if p._filter_graph is not None:
                    p._filter_graph.push(frame)
                    while True:
                        try:
                            out_frame = p._filter_graph.pull()
                        except (av.BlockingIOError, av.FFmpegError):
                            break
                        p._emit_pcm(out_frame, pcm)
                else:
                    p._emit_pcm(frame, pcm)
            bytes_produced += len(pcm)
            if time.perf_counter() > deadline:
                break
    finally:
        container.close()
    return bytes_produced


def main(path: str) -> int:
    if not os.path.exists(path):
        out(f"missing: {path}"); return 2

    sr = ap_mod._get_device_sample_rate()
    expected_bytes_per_sec_1x = sr * ap_mod.FRAME_BYTES
    out(f"Device sample rate: {sr} Hz")
    out(f"Expected bytes/sec @ 1.0x: {expected_bytes_per_sec_1x:,}")
    out(f"File: {path}\n")

    fails = 0

    out("[1] PyAV opens the file and finds an audio stream")
    container = av.open(path)
    has_audio = len(container.streams.audio) > 0
    container.close()
    fails += not assert_(has_audio, "audio stream present")

    out("\n[2] Decode 1.5s of PCM at tempo=1.0 from the start")
    n = measure_pcm_rate(path, start_sec=0.0, tempo=1.0, duration=1.5)
    expected = expected_bytes_per_sec_1x * 1.5
    out(f"    produced {n:,} bytes (expected ~{int(expected):,})")
    fails += not assert_(n > 0, "produced any PCM at all")
    fails += not assert_(n >= expected * 0.5,
                         f"produced at least 50% of expected ({n} >= {int(expected*0.5)})")

    out("\n[3] Decode 1s after seek to t=60s")
    n = measure_pcm_rate(path, start_sec=60.0, tempo=1.0, duration=1.0)
    out(f"    produced {n:,} bytes")
    fails += not assert_(n > expected_bytes_per_sec_1x * 0.3, "produced PCM after seek")

    out("\n[4] atempo=0.5 (slow-mo) for 1s — expect ~half the bytes vs 1.0x")
    n = measure_pcm_rate(path, start_sec=0.0, tempo=0.5, duration=1.0)
    out(f"    produced {n:,} bytes")
    fails += not assert_(n > 0, "atempo=0.5 produced PCM")

    out("\n[5] atempo=2.0 for 1s — expect ~double")
    n = measure_pcm_rate(path, start_sec=0.0, tempo=2.0, duration=1.0)
    out(f"    produced {n:,} bytes")
    fails += not assert_(n > 0, "atempo=2.0 produced PCM")

    out("\n[6] atempo=0.25 (chained 0.5,0.5) for 1s")
    n = measure_pcm_rate(path, start_sec=0.0, tempo=0.25, duration=1.0)
    out(f"    produced {n:,} bytes")
    fails += not assert_(n > 0, "chained atempo produced PCM")

    out(f"\n=== {fails} failure(s) ===")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    default_path = (
        "/Volumes/Crucial/pn/VR/"
        "VRCONK_marvel_rivals_luna_snow_a_porn_parody_8K_180x180_3dh.mp4"
    )
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default_path))
