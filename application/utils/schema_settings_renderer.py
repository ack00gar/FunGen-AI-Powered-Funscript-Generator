"""Auto-generate imgui widgets from a JSON-Schema property map.

Used by trackers that choose *Path B* (schema-driven settings) instead of
implementing ``render_settings_ui()`` directly.

Type mapping
------------
* ``number``  → ``slider_float``
* ``integer`` → ``slider_int``
* ``boolean`` → ``checkbox``
* ``string`` with ``enum`` → ``combo``
* ``string`` (plain) → ``input_text``

On change the value is written to *app_settings* **and**
``tracker.on_setting_changed(key, value)`` is called.
"""

import imgui
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered


def render_schema_settings(schema, app_settings, tracker_instance) -> bool:
    """Render imgui widgets for every property in *schema*.

    Parameters
    ----------
    schema : dict
        JSON-Schema with a ``properties`` dict.
    app_settings :
        Settings store (must support ``.get(key, default)`` / ``.set(key, val)``).
    tracker_instance :
        The tracker whose ``on_setting_changed`` will be called.

    Returns
    -------
    bool
        ``True`` if at least one widget was rendered.
    """
    props = schema.get("properties")
    if not props:
        return False

    rendered = False
    for key, prop in props.items():
        ptype = prop.get("type", "string")
        title = prop.get("title", key)
        desc = prop.get("description", "")
        default = prop.get("default")

        if ptype == "number":
            cur = app_settings.get(key, default if default is not None else 0.0)
            lo = prop.get("minimum", 0.0)
            hi = prop.get("maximum", 100.0)
            ch, nv = imgui.slider_float("##schema_%s" % key, cur, lo, hi, "%.2f")
            imgui.same_line()
            imgui.text(title)
            if desc:
                _tooltip_if_hovered(desc)
            if ch and nv != cur:
                app_settings.set(key, nv)
                tracker_instance.on_setting_changed(key, nv)
            rendered = True

        elif ptype == "integer":
            cur = app_settings.get(key, default if default is not None else 0)
            lo = prop.get("minimum", 0)
            hi = prop.get("maximum", 100)
            ch, nv = imgui.slider_int("##schema_%s" % key, cur, lo, hi)
            imgui.same_line()
            imgui.text(title)
            if desc:
                _tooltip_if_hovered(desc)
            if ch and nv != cur:
                app_settings.set(key, nv)
                tracker_instance.on_setting_changed(key, nv)
            rendered = True

        elif ptype == "boolean":
            cur = app_settings.get(key, default if default is not None else False)
            ch, nv = imgui.checkbox("%s##schema_%s" % (title, key), cur)
            if desc:
                _tooltip_if_hovered(desc)
            if ch:
                app_settings.set(key, nv)
                tracker_instance.on_setting_changed(key, nv)
            rendered = True

        elif ptype == "string":
            enum_values = prop.get("enum")
            if enum_values:
                cur = app_settings.get(key, default if default is not None else (enum_values[0] if enum_values else ""))
                try:
                    idx = enum_values.index(cur)
                except ValueError:
                    idx = 0
                ch, nidx = imgui.combo("%s##schema_%s" % (title, key), idx, enum_values)
                if desc:
                    _tooltip_if_hovered(desc)
                if ch and nidx != idx:
                    nv = enum_values[nidx]
                    app_settings.set(key, nv)
                    tracker_instance.on_setting_changed(key, nv)
            else:
                cur = app_settings.get(key, default if default is not None else "")
                ch, nv = imgui.input_text("%s##schema_%s" % (title, key), cur, 256)
                if desc:
                    _tooltip_if_hovered(desc)
                if ch and nv != cur:
                    app_settings.set(key, nv)
                    tracker_instance.on_setting_changed(key, nv)
            rendered = True

    return rendered
