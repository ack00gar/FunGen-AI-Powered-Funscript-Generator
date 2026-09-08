# Funscript editor

Generating a script gets you a draft. Editing is where it becomes good. This
page describes the editor in **[FunGen 2](https://fungen.app)**, called the
Studio: a frame-accurate, multi-axis funscript editor with the video, the
timeline and the device all in one window.

> [!NOTE]
> This page describes FunGen 2, the current release. It lives in this repository
> because this is where most people find us. The application, its documentation
> and its downloads are at **[fungen.app](https://fungen.app)**.

FunGen is free to download and runs on Windows, macOS and Linux as a single
native binary. Editing by hand is free and always has been; nothing in this
page is behind a paywall.

---

## The timeline

Points are edited at frame accuracy, against the decoded video rather than an
approximate clock. A stroke is two points, a reversal and its opposite, and the
timeline is built around manipulating strokes rather than isolated dots.

- **Endpoint snap** places a stroke end on the computed reversal instead of
  wherever the cursor landed. On by default.
- **A speed limit that adds time, not depth.** When a stroke is faster than the
  device can follow, the sensible correction is to give it more time rather than
  flatten it. Depth is what the viewer feels; duration is what the hardware
  negotiates.
- **Trim** from the left or the right without disturbing what is between.

## Selecting

Most editing time is spent selecting the right points, so selection is the part
worth learning first.

Select peaks and valleys, tops, middles or bottoms, everything within a chapter,
or everything left or right of the playhead. Marquee-select a region directly on
the timeline. Selections then feed every transform below.

## Transforming

Every transform previews live before it commits.

- **Equalize** evens out inconsistent stroke heights.
- **Isolate** keeps what you selected and clears the rest.
- **Ripple-delete** removes a span and closes the gap behind it.
- **Repeat-stroke** takes a stroke you like and repeats it across a range.
- **Fade** ramps a selection, finishing at a depth you choose and leaning
  earlier or later. Vibrate, stroke generation and Amplify take the same shaping
  control.
- **Parametric stroke generation** builds a run of strokes from parameters
  rather than by hand.
- **Snap-to-playhead magnet** pulls nearby points onto the playhead.

## Multi-axis

Full multi-axis scripting: stroke, surge, sway, twist, roll and pitch, all on
one timeline.

Open several scripts or axes as lanes side by side and ghost one over another,
so you can reference the stroke axis while shaping roll without switching
context. Copying between timelines replaces the whole destination axis rather
than leaving an old tail behind, and a deleted axis stays deleted.

## Chapters, bookmarks and the heatmap

A chapter bar divides the scene into named sections with custom types. Chapters
scope your selections and transforms, and several selected chapters can be
exported together as one clip with a matching funscript, leaving out the
material between them.

Bookmarks mark places to come back to. A colored heatmap shows intensity across
the whole script, so a flat section or an over-driven one is visible without
scrubbing for it.

## Funscript Doctor

The Doctor scores a script and flags what is wrong with it: speed limits the
device cannot physically follow, gaps where nothing happens, missing strokes.
Most findings carry a one-click fix, and the before and after are shown on the
real timeline with matching counts, so you can see exactly what a fix did rather
than trusting a summary.

This is the fastest way to make an AI-generated draft feel deliberate.

## Undo, and a history you can browse

Undo and redo are unlimited. Not a rolling buffer of the last few hundred
steps: the whole history, with a visual browser, and it survives closing and
reopening the project. A generation run that would overwrite points you authored
by hand asks first.

## Plugins and macros

A plugin and macro system records repeatable edits, so a cleanup you perform on
every script becomes one action instead of twenty.

## Playing while you edit

The built-in player decodes on the GPU, so scrubbing an 8K VR file stays
responsive. Scripts play straight to hardware while you edit, single- or
multi-axis, with an on-screen position gauge.

Devices supported directly: The Handy (USB, Bluetooth and network), Kiiroo Keon
and Onyx, Fleshlight Launch, Lovense Solace Pro, Autoblow, Vacuglide, Vorze
Piston, OSSM, Hismith machines, and OSR2 / SR6 / SSR1 over T-Code. Anything else
through [Buttplug.io](https://buttplug.io) / Intiface.

Feeling an edit immediately, on the device you actually own, changes what you
choose to fix.

---

## A workflow that works

1. Generate a draft, or open an existing script.
2. Run the Doctor and take the fixes you agree with.
3. Set chapters, so the rest of the work has structure.
4. Fix the sections that matter most, by hand, with the device connected.
5. Equalize and fade the parts that are merely fine.
6. Save. Multi-axis scripts write one file per axis, with the standard suffixes.

---

## Where next

- [The funscript editor](https://fungen.app/funscript-editor)
- [How to make a funscript](how-to-make-a-funscript.md)
- [What is a funscript?](https://fungen.app/what-is-a-funscript)
- [VR funscripts](https://fungen.app/vr-funscripts)
- [Supported devices](https://fungen.app/supported-devices)
- [Download FunGen 2](https://fungen.app)
- [Discord community](https://discord.gg/WYkjMbtCZA)
