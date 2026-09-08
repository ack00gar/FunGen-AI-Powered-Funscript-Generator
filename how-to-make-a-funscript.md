# How to make a funscript

A funscript is a small JSON file that describes motion over time: a list of
timestamps, each with a position from 0 to 100. A player reads it alongside a
video and drives a device to those positions, so the motion matches what is on
screen.

There are four practical ways to make one, and they mix freely. This guide
covers all four with **[FunGen 2](https://fungen.app)**, which is free to
download, runs on Windows, macOS and Linux, and does its AI generation on your
own machine.

> [!NOTE]
> This guide describes FunGen 2, the current release. It lives in this
> repository because this is where most people find us. The application itself,
> its documentation and its downloads are at **[fungen.app](https://fungen.app)**.

---

## Before you start

1. Download FunGen 2 from **[fungen.app](https://fungen.app)** or the
   [releases page](https://github.com/ack00gar/FunGen/releases).
2. Run it. It is a single native binary per platform. There is no Python to
   install, no virtual environment and no dependency chain to resolve.
3. Open a video, or drop one onto the Welcome screen.

No account and no card are needed to download or to script by hand.

---

## Route 1: generate it with AI

The fastest route. FunGen tracks the motion in the video and writes a full
script you can then refine.

1. Open your video.
2. Start a generation run. From the Welcome screen you can drop a video in and
   go; from the timeline you can run the full pipeline over the whole file.
3. Wait for the pass to finish. It runs locally on your GPU. Nothing is
   uploaded.
4. Review the result on the timeline and fix what you disagree with.

Treat the output as a strong draft, not a finished script. The AI is good at
finding the beat and the range; you are better at deciding what matters.

**Try before you commit.** *FunGenerate* proposes the next 20 seconds of motion
from the playhead and shows it as a live ghost over your script. You accept or
reject it before anything is written. This is the honest way to judge quality on
your own content rather than on a demo.

Base algorithmic generation is free for everyone. The AI generation itself is
also free behind a short metered wait, so you can genuinely try it before
deciding whether the [Pro Pass](https://fungen.app/pricing) is worth it to you.

More detail: [Generate a funscript from any
video](https://fungen.app/generate-funscripts-from-video).

---

## Route 2: script it by hand

Full control, and the route every scripter eventually uses for the parts that
matter.

1. Open the video in the Studio, FunGen's frame-accurate editor.
2. Move the playhead to where a stroke should begin and add a point.
3. Add the matching point at the other end of the stroke. Two points make one
   stroke: a down and an up.
4. Repeat through the scene, then refine.

Working faster:

- **Snap to the reversal.** Endpoint snap puts a stroke end on the computed
  reversal instead of where your cursor happened to land.
- **Select in bulk.** Select peaks and valleys, tops, middles or bottoms, by
  chapter, or everything left or right of the playhead.
- **Transform rather than redraw.** Equalize, isolate, ripple-delete,
  repeat-stroke and parametric stroke generation all work on a selection, with a
  live preview before you commit.
- **Undo without fear.** Undo and redo are unlimited, with a visual history
  browser that survives closing and reopening the project.

More detail: [The funscript editor](https://fungen.app/funscript-editor).

---

## Route 3: record it live

Sometimes the fastest way to express motion is to perform it.

1. Play the video.
2. Drive the motion yourself with the mouse or a controller.
3. FunGen records what you do as a script.

This gets the feel right in one pass. Clean up the timing afterwards on the
timeline.

---

## Route 4: from the audio

For music-led material, FunGen can generate motion from the soundtrack. It
follows the tempo and the musical-section changes across the track, with live
controls to shape the result and a preview before you apply it.

This is the quickest way to script a PMV or any cut where the motion should
follow the music rather than the picture.

---

## Clean it up

A script that is technically valid can still be unpleasant to feel. Two tools
help.

**Funscript Doctor** scores the script's quality and flags what is wrong: speed
limits your device cannot physically follow, gaps where nothing happens, strokes
that are missing entirely. Most findings have a one-click fix, and you see the
before and after on the real timeline with matching counts.

**Chapters** divide the scene into named sections, so you can work on one part
at a time, apply a transform to just that part, or export a single section as a
clip with its own funscript. A colored heatmap shows intensity across the whole
script at a glance.

---

## Play it on a device

FunGen plays scripts straight to hardware, so you can feel an edit immediately
rather than exporting and switching apps.

Supported directly: The Handy (USB, Bluetooth and network), Kiiroo Keon and
Onyx, Fleshlight Launch, Lovense Solace Pro, Autoblow, Vacuglide, Vorze Piston,
OSSM, Hismith machines, and OSR2 / SR6 / SSR1 over T-Code. Anything else
connects through [Buttplug.io](https://buttplug.io) / Intiface.

More detail: [Supported devices](https://fungen.app/supported-devices), and the
per-device guides for [The Handy](https://fungen.app/funscripts-for-the-handy)
and [OSR2 / SR6](https://fungen.app/funscripts-for-osr2-sr6).

---

## Multi-axis and VR

A single-axis script moves up and down. A multi-axis script also moves in
surge, sway, twist, roll and pitch, which is what multi-axis hardware such as
the OSR2 / SR6 can actually use.

FunGen edits all six axes on one timeline. You can open several scripts or axes
as lanes side by side, and ghost one over another to reference while you edit.

VR video is handled natively: fisheye, equirectangular, side-by-side and
top/bottom, up to 8K, with the projection detected automatically.

More detail: [VR funscripts](https://fungen.app/vr-funscripts).

---

## Saving and sharing

Funscripts save next to the video as `<video name>.funscript`. Multi-axis scripts
save one file per axis, using the standard axis suffixes, so other players and
tools read them correctly.

---

## Where next

- [What is a funscript?](https://fungen.app/what-is-a-funscript)
- [Generate a funscript from any video](https://fungen.app/generate-funscripts-from-video)
- [The funscript editor](https://fungen.app/funscript-editor), and the [editor guide in this repository](funscript-editor.md)
- [VR funscripts](https://fungen.app/vr-funscripts)
- [Download FunGen 2](https://fungen.app)
- [Discord community](https://discord.gg/WYkjMbtCZA)
