"""Live Capture UI — STUB.

As of v1.1.3 all GUI rendering has moved to the main app
(application/gui_components/cp_batch_ui.py).

This stub keeps backward compatibility with older main-app versions
that still call ``render_capture_panel(app)``.
"""


def render_capture_panel(app):
    """No-op — GUI now lives in the main app's BatchMixin."""
    pass
