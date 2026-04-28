"""
Standalone splash-theme viewer. Runs the real StandaloneSplashWindow without
loading the rest of FunGen, so we can iterate on themes in isolation.

Usage:
    # Random theme, one run (auto-close after 60s, Esc to quit)
    python3 application/gui_components/splash_play.py

    # Force a specific theme
    python3 application/gui_components/splash_play.py --theme starwars
    FUNGEN_SPLASH_THEME=breaking python3 application/gui_components/splash_play.py

    # Cycle through all 17 themes, 12 seconds each (tune with --duration)
    python3 application/gui_components/splash_play.py --all
    python3 application/gui_components/splash_play.py --all --duration 20

Valid theme names: matrix, terminator, tron, starwars, breaking, invaders, mars,
clippy, tetris, pacman, blade, bsod, sonic, xfiles, tmnt, et, mario.
Press Esc (or close the window) to skip to the next theme / quit.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

# Make sure the repo root is importable (so "application.*" resolves
# regardless of where this is invoked from).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


THEME_ORDER = ['matrix', 'terminator', 'tron', 'starwars',
               'breaking', 'invaders', 'mars',
               'clippy', 'tetris', 'pacman', 'blade', 'bsod',
               'sonic', 'xfiles', 'tmnt',
               'et', 'mario']


def _run_one(duration_s):
    """Run a single splash session in this process, for at most duration_s
    seconds. Called as the child process when --all is dispatching, or directly
    for the single-theme path."""
    import glfw  # local import so top-level argparse stays cheap
    from application.gui_components.splash_screen import StandaloneSplashWindow

    splash = StandaloneSplashWindow()

    def esc_watchdog():
        while splash.running:
            time.sleep(0.05)
            if splash.window is None:
                continue
            try:
                if glfw.get_key(splash.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    glfw.set_window_should_close(splash.window, True)
                    break
            except Exception:
                break

    def timeout_close():
        time.sleep(duration_s)
        if splash.running:
            splash.stop()
            # Nudge the render loop out of glfw.poll_events on macOS
            try:
                if splash.window is not None:
                    glfw.set_window_should_close(splash.window, True)
            except Exception:
                pass

    threading.Thread(target=esc_watchdog, daemon=True).start()
    threading.Thread(target=timeout_close, daemon=True).start()

    splash.start()
    try:
        splash.cleanup()
    except Exception:
        pass


def _cycle_all(duration_s):
    """Spawn a fresh subprocess per theme so GLFW can fully re-initialize
    between runs (macOS is picky about re-opening windows in the same
    process). Each child is this same script with --theme NAME --single."""
    for theme in THEME_ORDER:
        print(f"\n=== {theme} (Esc to skip, {duration_s}s max) ===", flush=True)
        env = os.environ.copy()
        env['FUNGEN_SPLASH_THEME'] = theme
        cmd = [sys.executable, __file__,
               '--single', '--duration', str(duration_s)]
        try:
            rc = subprocess.call(cmd, env=env)
        except KeyboardInterrupt:
            print("\n[interrupted]", flush=True)
            break
        if rc not in (0, None):
            print(f"  (theme exited with code {rc})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--theme', '-t',
                        help="Force a specific theme (same as setting "
                             "FUNGEN_SPLASH_THEME).")
    parser.add_argument('--all', '-a', action='store_true',
                        help="Cycle through all themes, one subprocess each.")
    parser.add_argument('--duration', '-d', type=float,
                        help="Max seconds per theme. Default 60 for single, "
                             "12 for --all.")
    parser.add_argument('--single', action='store_true',
                        help=argparse.SUPPRESS)  # internal: child-process flag
    args = parser.parse_args()

    if args.theme:
        os.environ['FUNGEN_SPLASH_THEME'] = args.theme

    if args.all and not args.single:
        _cycle_all(args.duration if args.duration is not None else 12.0)
        return

    _run_one(args.duration if args.duration is not None else 60.0)


if __name__ == '__main__':
    main()
