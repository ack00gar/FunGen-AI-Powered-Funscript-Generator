"""Theme orchestration.

One public surface:

- ``apply_imgui_theme(theme_cls)`` — applies imgui's native color palette
  (window/child/frame/menubar backgrounds, text, accent slots) and restores
  our custom rounding overrides, which imgui's ``style_colors_*`` wipes back
  to pack defaults. Contains the LightTheme imgui-side tuning.
- ``apply_theme_live(theme_name)`` — full end-to-end theme swap: updates the
  ``CurrentTheme`` proxy, refreshes the class-level cached tuples in
  ``element_group_colors``, and re-applies ``apply_imgui_theme``. Call this
  from the main thread during a render frame to change themes without an app
  restart.

The two layers below (``config.constants_colors`` and
``config.element_group_colors``) are intentionally kept separate: the former
is the palette source of truth with proxy-based swapping; the latter is a
caching layer that groups palette entries by semantic component (timeline,
video display, etc). Keeping those as two concerns makes grep-ability of
"which widget reads which color" much cleaner than a single mega-module.
"""

from config.constants_colors import DarkTheme, LightTheme, set_active_theme


def apply_imgui_theme(theme_cls):
    """Apply the imgui color style + restore our custom rounding overrides.

    imgui.style_colors_light/dark resets every color AND every style var back
    to its pack defaults, so the rounding values set once during bootstrap
    (app_gui.py ~L508) get wiped. Re-apply them here so the toggle is a true
    in-place theme swap that leaves window_rounding / frame_rounding intact.

    On top of that, for Light theme we tone the pack-default light palette
    several stops darker: imgui's stock light is near-pure-white which the
    user flagged as "blinding / lacking contrast". We darken window / child /
    frame / menubar backgrounds, deepen the text color, and recolor the
    accent slots (buttons, headers, tabs, slider grabs) into a bluish-gray
    palette that reads against the darker canvas without losing the "light"
    feel.
    """
    try:
        import imgui
    except ImportError:
        return
    try:
        if theme_cls is LightTheme and hasattr(imgui, 'style_colors_light'):
            imgui.style_colors_light()
        elif hasattr(imgui, 'style_colors_dark'):
            imgui.style_colors_dark()
    except Exception:
        return
    try:
        style = imgui.get_style()
        style.window_rounding = 6.0
        style.frame_rounding = 4.0
        style.child_rounding = 6.0
        style.popup_rounding = 6.0
        style.tab_rounding = 4.0
        style.scrollbar_rounding = 6.0
        style.grab_rounding = 4.0
    except Exception:
        pass

    if theme_cls is LightTheme:
        try:
            style = imgui.get_style()
            c = style.colors
            c[imgui.COLOR_TEXT]                     = (0.90, 0.89, 0.86, 1.0)
            c[imgui.COLOR_TEXT_DISABLED]            = (0.55, 0.54, 0.51, 1.0)
            c[imgui.COLOR_WINDOW_BACKGROUND]        = (0.28, 0.27, 0.25, 1.0)
            c[imgui.COLOR_CHILD_BACKGROUND]         = (0.24, 0.23, 0.21, 1.0)
            c[imgui.COLOR_POPUP_BACKGROUND]         = (0.30, 0.29, 0.27, 1.0)
            c[imgui.COLOR_BORDER]                   = (0.18, 0.17, 0.15, 0.80)
            c[imgui.COLOR_FRAME_BACKGROUND]         = (0.22, 0.21, 0.19, 1.0)
            c[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.28, 0.28, 0.34, 1.0)
            c[imgui.COLOR_FRAME_BACKGROUND_ACTIVE]  = (0.32, 0.34, 0.42, 1.0)
            c[imgui.COLOR_TITLE_BACKGROUND]         = (0.20, 0.19, 0.17, 1.0)
            c[imgui.COLOR_TITLE_BACKGROUND_ACTIVE]  = (0.24, 0.26, 0.32, 1.0)
            c[imgui.COLOR_MENUBAR_BACKGROUND]       = (0.22, 0.21, 0.19, 1.0)
            c[imgui.COLOR_SCROLLBAR_BACKGROUND]     = (0.20, 0.19, 0.17, 1.0)
            c[imgui.COLOR_SCROLLBAR_GRAB]           = (0.38, 0.37, 0.34, 1.0)
            c[imgui.COLOR_SCROLLBAR_GRAB_HOVERED]   = (0.46, 0.45, 0.42, 1.0)
            c[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE]    = (0.52, 0.52, 0.56, 1.0)
            c[imgui.COLOR_CHECK_MARK]               = (0.45, 0.60, 0.90, 1.0)
            c[imgui.COLOR_SLIDER_GRAB]              = (0.40, 0.55, 0.85, 1.0)
            c[imgui.COLOR_SLIDER_GRAB_ACTIVE]       = (0.32, 0.46, 0.75, 1.0)
            c[imgui.COLOR_BUTTON]                   = (0.32, 0.34, 0.40, 1.0)
            c[imgui.COLOR_BUTTON_HOVERED]           = (0.38, 0.42, 0.52, 1.0)
            c[imgui.COLOR_BUTTON_ACTIVE]            = (0.30, 0.38, 0.55, 1.0)
            c[imgui.COLOR_HEADER]                   = (0.30, 0.34, 0.46, 0.85)
            c[imgui.COLOR_HEADER_HOVERED]           = (0.36, 0.42, 0.54, 0.95)
            c[imgui.COLOR_HEADER_ACTIVE]            = (0.32, 0.40, 0.56, 1.0)
            c[imgui.COLOR_SEPARATOR]                = (0.22, 0.22, 0.24, 0.8)
            c[imgui.COLOR_SEPARATOR_HOVERED]        = (0.32, 0.44, 0.65, 0.8)
            c[imgui.COLOR_SEPARATOR_ACTIVE]         = (0.40, 0.55, 0.82, 1.0)
            c[imgui.COLOR_TAB]                      = (0.26, 0.28, 0.32, 0.88)
            c[imgui.COLOR_TAB_HOVERED]              = (0.34, 0.40, 0.50, 1.0)
            c[imgui.COLOR_TAB_ACTIVE]               = (0.32, 0.40, 0.55, 1.0)
            c[imgui.COLOR_TAB_UNFOCUSED]            = (0.22, 0.22, 0.24, 0.85)
            c[imgui.COLOR_TAB_UNFOCUSED_ACTIVE]     = (0.28, 0.32, 0.40, 0.95)
            c[imgui.COLOR_RESIZE_GRIP]              = (0.32, 0.40, 0.55, 0.35)
            c[imgui.COLOR_RESIZE_GRIP_HOVERED]      = (0.38, 0.50, 0.72, 0.7)
            c[imgui.COLOR_RESIZE_GRIP_ACTIVE]       = (0.42, 0.58, 0.85, 1.0)
        except Exception:
            pass


def apply_theme_live(theme_name: str) -> bool:
    """Swap the active theme at runtime. Returns True on success.

    1. Swaps CurrentTheme's inner reference via set_active_theme — every
       CurrentTheme.X lookup picks up the new values on the next render frame.
    2. Refreshes element_group_colors' class-level cached tuples. Those
       `X = CurrentTheme.Y` assignments run once at import time and would
       otherwise hold the original theme's tuple forever.
    3. Re-applies imgui's color style + custom rounding so native widgets
       (menu bars, scrollbars, window backgrounds) repaint in the new theme.
    Safe to call from the main thread during a render frame.
    """
    t = str(theme_name).strip().lower()
    target = LightTheme if t == "light" else DarkTheme
    set_active_theme(target)
    try:
        from config import element_group_colors
        element_group_colors.refresh_from_theme()
    except Exception:
        pass
    apply_imgui_theme(target)
    return True