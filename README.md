# FunGen: AI funscript generator for 2D and VR

> [!IMPORTANT]
> **This is the archived FunGen 1 source repository.** It is unmaintained. For
> the current FunGen 2 application, documentation and official downloads, use
> **[fungen.app](https://fungen.app)**.

FunGen generates funscripts from 2D and VR video with AI, entirely on your own
machine, and nothing is uploaded. It is also a full frame-accurate, multi-axis
funscript editor and player, so you can build a script by hand, clean up what
the AI generated, and drive your device straight from the timeline.

## → **[fungen.app](https://fungen.app)**

[![Current FunGen 2 Studio interface](https://fungen.app/shots/studio-current.webp)](https://fungen.app/studio/)

The current release is **FunGen 2**: a single native binary for Windows, macOS
(Apple Silicon native) and Linux, with no Python, no venv and no dependencies to chase.

- **AI generation for 2D and VR**: fisheye, equirectangular, side-by-side and
  top/bottom, up to 8K, tracked on your GPU
- **Frame-accurate multi-axis editor**: stroke, surge, sway, twist, roll and
  pitch on one timeline
- **Plays straight to your device**: The Handy, OSR2 / SR6, Autoblow, and
  anything on Buttplug.io / Intiface
- **Runs entirely on your own machine**: AI generation is local, nothing is
  uploaded
- **Free to download**: no account, no card

**[Get FunGen 2](https://fungen.app)** ·
[Download](https://github.com/ack00gar/FunGen/releases) ·
[How it generates funscripts](https://fungen.app/generate-funscripts-from-video/) ·
[Supported devices](https://fungen.app/supported-devices/) ·
[Pricing](https://fungen.app/pricing/) ·
[Discord](https://discord.gg/WYkjMbtCZA)

## Guides

Step-by-step, for FunGen 2:

- **[How to make a funscript](how-to-make-a-funscript.md)**: the four ways to
  build one (AI generation, by hand, recorded live, or from the audio), then
  cleaning it up and playing it on a device.
- **[Funscript editor](funscript-editor.md)**: the timeline, selection,
  transforms, multi-axis, chapters, the Funscript Doctor and device playback.

---

## About this repository

This repository holds **FunGen 1**, the original Python implementation. It has
been completely rewritten as FunGen 2 (above), which is where all development
happens. FunGen 1 is kept here for reference and is no longer developed.

The source, its documentation and its release history stay available:

- **[Installation & usage](DOCS-v1.md)**: installers, model downloads, GUI,
  CLI, trackers, plugins, performance tuning, troubleshooting
- **[Version history](CHANGELOG-v1.md)**: v0.6.0 through v1.0.0 release notes

For anything new, use **[FunGen 2](https://fungen.app)**.

---

### Licence

FunGen is free for personal, noncommercial use only, under the
[PolyForm Strict License 1.0.0](LICENSE) together with the
[FunGen Supplemental Terms](LICENSE-SUPPLEMENTAL). Nobody may sell, resell,
sublicense or charge for FunGen in any form. FunGen 1 is additionally provided
as-is and unmaintained. Canonical, always-current text:
<https://fungen.app/license>.
