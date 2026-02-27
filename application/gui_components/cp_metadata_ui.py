"""Metadata Editor tab UI mixin for ControlPanelUI.

Provides input fields for funscript metadata: creator, title, description,
tags, performers, URLs, license, and notes. Persisted in project file and
included in funscript exports.
"""
import imgui
from application.utils.section_card import section_card


_METADATA_FIELDS = [
    ("creator", "Creator", "Author or studio name", False),
    ("title", "Title", "Script title", False),
    ("description", "Description", "Script description", True),
    ("tags", "Tags", "Comma-separated tags (e.g., blowjob, cowgirl, POV)", False),
    ("performers", "Performers", "Comma-separated performer names", False),
    ("script_url", "Script URL", "URL where this script can be downloaded", False),
    ("video_url", "Video URL", "URL of the source video", False),
    ("license", "License", "License type (e.g., Free, CC-BY, etc.)", False),
    ("notes", "Notes", "Additional notes for personal use", True),
]


class MetadataEditorMixin:
    """Mixin providing Metadata tab rendering methods for ControlPanelUI."""

    def _render_metadata_tab(self):
        """Render the metadata editor panel."""
        metadata = self._get_project_metadata()

        with section_card("Script Metadata##MetadataEditor", tier="primary",
                          open_by_default=True) as _open:
            if not _open:
                return

            imgui.text_wrapped(
                "Metadata is saved in your project file and included in funscript exports."
            )
            imgui.spacing()

            changed = False
            for key, label, tooltip, is_multiline in _METADATA_FIELDS:
                current_value = metadata.get(key, "")
                if current_value is None:
                    current_value = ""

                imgui.text(label)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(tooltip)

                imgui.push_item_width(-1)
                if is_multiline:
                    c, new_value = imgui.input_text_multiline(
                        f"##{key}_meta",
                        current_value,
                        2048,
                        width=-1,
                        height=60,
                    )
                else:
                    c, new_value = imgui.input_text(
                        f"##{key}_meta",
                        current_value,
                        512,
                    )
                imgui.pop_item_width()

                if c:
                    metadata[key] = new_value
                    changed = True

                imgui.spacing()

            if changed:
                self._set_project_metadata(metadata)

    def _get_project_metadata(self):
        """Read metadata from project manager."""
        pm = getattr(self.app, 'project_manager', None)
        if pm and hasattr(pm, 'get_metadata'):
            return pm.get_metadata()
        # Fallback: store on app instance
        if not hasattr(self.app, '_project_metadata'):
            self.app._project_metadata = {}
        return self.app._project_metadata

    def _set_project_metadata(self, metadata):
        """Write metadata to project manager and mark dirty."""
        pm = getattr(self.app, 'project_manager', None)
        if pm and hasattr(pm, 'set_metadata'):
            pm.set_metadata(metadata)
        else:
            self.app._project_metadata = metadata
        # Mark project as dirty
        if pm:
            pm.project_dirty = True
