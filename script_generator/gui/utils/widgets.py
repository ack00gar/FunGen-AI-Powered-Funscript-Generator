import os
import tkinter as tk
from tkinter import ttk, filedialog, Button

from script_generator.constants import LOGO
from script_generator.gui.utils.tooltip import Tooltip
from script_generator.gui.utils.utils import disable_widgets

PADDING_X = 5
PADDING_Y = 5
LABEL_WIDTH = 150


class Widgets:
    @staticmethod
    def frame(parent, title=None, main_section=False, **grid_kwargs):
        if title:
            frame_cls = ttk.LabelFrame
            if main_section:
                style = ttk.Style(parent)
                style.configure("Bold.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
                frame = frame_cls(parent, text=title, style="Bold.TLabelframe")
            else:
                frame = frame_cls(parent, text=title)
        else:
            frame = ttk.Frame(parent)

        grid_kwargs.setdefault("sticky", "nsew")  # Default to stretch in all directions
        grid_kwargs.setdefault("padx", 5)
        grid_kwargs.setdefault("pady", (5, (10 if main_section else 5)))

        frame.grid(**grid_kwargs)

        # Ensure the frame dynamically adjusts grid weights for nested widgets
        def configure_grid(event):
            for i in range(frame.grid_size()[0]):  # Columns
                if frame.grid_columnconfigure(i)["weight"] == 0:  # Avoid overwriting custom weights
                    frame.grid_columnconfigure(i, weight=1)
            for j in range(frame.grid_size()[1]):  # Rows
                if frame.grid_rowconfigure(j)["weight"] == 0:  # Avoid overwriting custom weights
                    frame.grid_rowconfigure(j, weight=1)

        frame.bind("<Configure>", configure_grid)

        return frame

    @staticmethod
    def label(frame, text, tooltip_text=None, **grid_kwargs):
        label = ttk.Label(frame, text=text)
        label.grid(**grid_kwargs)

        if tooltip_text:
            Tooltip(label, tooltip_text)

        return label

    @staticmethod
    def entry(frame, default_value="", tooltip_text=None, **grid_kwargs):
        entry = ttk.Entry(frame, textvariable=tk.StringVar(value=default_value))
        entry.grid(**grid_kwargs)

        if tooltip_text:
            Tooltip(entry, tooltip_text)

        return entry

    @staticmethod
    def input(parent, label_text, state, attr, row=0, column=0, label_width_px=LABEL_WIDTH, entry_width_px=200, callback=None, tooltip_text=None):
        container = ttk.Frame(parent)
        container.grid(row=row, column=column, sticky="ew", padx=5, pady=5)
        container.columnconfigure(1, weight=1)

        initial_value = getattr(state, attr)
        if initial_value is None:
            initial_value = ""

        value = tk.StringVar(value=initial_value)

        def on_value_change(*args):
            setattr(state, attr, value.get())
            if callback:
                callback(value.get())

        value.trace_add("write", on_value_change)

        label = tk.Label(container, text=label_text, anchor="w", width=label_width_px // 7)
        label.grid(row=0, column=0, sticky="w", padx=(5, 2))

        entry_container = ttk.Frame(container, width=entry_width_px)
        entry_container.grid(row=0, column=1, sticky="ew", padx=(2, 5))
        entry_container.grid_propagate(False)  # Prevent resizing
        entry = ttk.Entry(entry_container, textvariable=value)
        entry.pack(fill="both", expand=True)

        if tooltip_text:
            Tooltip(entry, tooltip_text)

        return container, entry, value

    @staticmethod
    def button(parent, button_text, on_click, row=0, column=0, tooltip_text=None, style_name="Custom.TButton"):
        style = ttk.Style()
        style.configure(style_name, padding=(10, 3))

        button = ttk.Button(parent, text=button_text, command=on_click, style=style_name)
        button.grid(row=row, column=column, sticky="w", padx=PADDING_X, pady=PADDING_Y)

        if tooltip_text:
            Tooltip(button, tooltip_text)

        return button

    @staticmethod
    def file_selection(parent, label_text, button_text, file_selector_title, file_types, state, attr, command=None, row=0, label_width_px=150, button_width_px=100, tooltip_text=None):
        container = tk.Frame(parent)
        container.grid(row=row, column=0, sticky="nsew", padx=5, pady=5)

        def on_change(val):
            setattr(state, attr, val)
            state_val = val if os.path.exists(val) else None
            if command:
                command(state_val)

        # Ensure the container scales properly
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Configure the container's internal layout
        container.columnconfigure(1, weight=1)

        file_path = tk.StringVar()

        # Label for file selection with fixed pixel width
        label = tk.Label(container, text=label_text, anchor="w", width=label_width_px // 7)
        label.grid(row=0, column=0, sticky="w", padx=(5, 2))

        # Entry widget that expands to fill available space
        entry = ttk.Entry(container, textvariable=file_path)
        entry.grid(row=0, column=1, sticky="ew", padx=(2, 5))

        # Button for browsing files with fixed pixel width
        button_container = tk.Frame(container, width=button_width_px, bg="lightgreen")  # Set button container background
        button_container.grid(row=0, column=2, sticky="e", padx=(2, 5))
        button_container.grid_propagate(False)  # Prevent resizing
        button = ttk.Button(button_container, text=button_text, command=lambda: Widgets._browse_file(file_path, file_selector_title, file_types, lambda val: setattr(state, attr, val)))
        button.pack(fill="both", expand=True)

        # Update state whenever the file path changes
        file_path.trace("w", lambda *args: on_change(file_path.get()))

        if tooltip_text:
            Tooltip(entry, tooltip_text)
            Tooltip(button, tooltip_text)

        return container, entry, button, file_path

    @staticmethod
    def labeled_progress(parent, label_text, row=0, column=0, progress_length=300, label_width_px=LABEL_WIDTH, label_percentage_width_px=LABEL_WIDTH, tooltip_text=None):
        container = ttk.Frame(parent)
        container.grid(row=row, column=column, sticky="ew", padx=5, pady=5)
        container.columnconfigure(1, weight=1)  # Allow the progress bar to expand

        # Label for progress description with fixed pixel width
        progress_label = tk.Label(container, text=label_text, anchor="w", width=label_width_px // 7)
        progress_label.grid(row=0, column=0, sticky="w", padx=(5, 2))

        # Progress bar widget
        progress_bar = ttk.Progressbar(container, orient="horizontal", mode="determinate", length=progress_length)
        progress_bar.grid(row=0, column=1, sticky="ew", padx=(2, 5))

        # Percentage label with fixed pixel width
        percentage_label = tk.Label(container, text="0%", anchor="e", width=label_percentage_width_px // 7)
        percentage_label.grid(row=0, column=2, sticky="e", padx=(2, 5))

        if tooltip_text:
            Tooltip(progress_bar, tooltip_text)

        return container, progress_bar, progress_label, percentage_label

    @staticmethod
    def dropdown(parent, label_text, options, default_value, state, attr, row=0, column=0, label_width_px=LABEL_WIDTH, tooltip_text=None, **grid_kwargs):
        selected_value = tk.StringVar(value=default_value)

        # Create a container for the dropdown
        container = ttk.Frame(parent)
        grid_kwargs.setdefault("sticky", "ew")
        container.grid(row=row, column=column, padx=5, pady=5, **grid_kwargs)
        container.columnconfigure(1, weight=1)  # Allow the dropdown to expand

        # Label with fixed pixel width
        label = tk.Label(container, text=label_text, anchor="w", width=label_width_px // 7)
        label.grid(row=0, column=0, sticky="w", padx=(5, 2))

        # Dropdown (Combobox) widget
        dropdown = ttk.Combobox(container, textvariable=selected_value, values=options, state="readonly")
        dropdown.grid(row=0, column=1, sticky="ew", padx=(2, 5))

        # Bind selection changes to update the state
        dropdown.bind("<<ComboboxSelected>>", lambda _: setattr(state, attr, selected_value.get()))

        if tooltip_text:
            Tooltip(dropdown, tooltip_text)

        return container, label, dropdown, selected_value

    @staticmethod
    def range_selector(parent, label_text, row, state, attr, values, column=0, tooltip_text=None):
        Widgets.label(parent, label_text, row=row, column=column, sticky="w", padx=PADDING_X, pady=PADDING_Y)

        selected_value = tk.StringVar(value=str(getattr(state, attr)))

        dropdown = ttk.Combobox(parent, textvariable=selected_value, values=values, width=5, state="readonly")
        dropdown.grid(row=row, column=column + 1, sticky="w", padx=PADDING_X, pady=PADDING_Y)

        dropdown.bind("<<ComboboxSelected>>", lambda _: setattr(state, attr, int(selected_value.get())))

        if tooltip_text:
            Tooltip(dropdown, tooltip_text)

        return dropdown

    @staticmethod
    def checkbox(parent, label_text, state, attr, label_left=True, row=0, column=0, label_width_px=150, tooltip_text=None, **grid_kwargs):
        # Ensure the parent has a _checkbox_vars attribute to track BooleanVars
        if not hasattr(parent, "_checkbox_vars"):
            parent._checkbox_vars = {}

        # Check if the BooleanVar for this attr already exists; create it if not
        if attr not in parent._checkbox_vars:
            parent._checkbox_vars[attr] = tk.BooleanVar(value=getattr(state, attr))

        is_checked = parent._checkbox_vars[attr]

        # Create a container for the checkbox and label
        container = ttk.Frame(parent)
        container.grid(row=row, column=column, sticky="ew", padx=5, pady=5)
        container.columnconfigure(1, weight=1)  # Allow checkbox to adjust dynamically

        if label_left:
            # Label with fixed pixel width and sticky west alignment
            label = tk.Label(container, text=label_text, anchor="w", width=label_width_px // 7)
            label.grid(row=0, column=0, sticky="w", padx=(5, 2))

            # Checkbox widget
            checkbox = ttk.Checkbutton(
                container,
                text="",  # No text since the label is on the left
                variable=is_checked,
                command=lambda: setattr(state, attr, is_checked.get())
            )
            checkbox.grid(row=0, column=1, sticky="w")  # Explicit sticky west for checkbox alignment
        else:
            # Checkbox widget with text (label on the right)
            checkbox = ttk.Checkbutton(
                container,
                text=label_text,
                variable=is_checked,
                command=lambda: setattr(state, attr, is_checked.get())
            )
            checkbox.grid(row=0, column=0, sticky="w", **grid_kwargs)

        if tooltip_text:
            Tooltip(checkbox, tooltip_text)

        return container, checkbox, is_checked

    @staticmethod
    def create_popup(title, master, content_builder, width=400, height=300):
        # Variable to store user choice
        user_action = tk.StringVar(value=None)  # Stores user action ('cancel', 'yes', 'no', etc.)

        # Create a Toplevel window
        window = tk.Toplevel(master)
        window.title(title)
        window.geometry(f"{width}x{height}")
        window.iconphoto(False, tk.PhotoImage(file=LOGO))

        # Center the popup
        if master:
            master.update_idletasks()  # Ensure master's dimensions are up-to-date
            master_width = master.winfo_width()
            master_height = master.winfo_height()
            master_x = master.winfo_x()
            master_y = master.winfo_y()

            position_right = master_x + (master_width // 2) - (width // 2)
            position_down = master_y + (master_height // 2) - (height // 2)
        else:
            # Center on the screen if no master provided
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()

            position_right = (screen_width // 2) - (width // 2)
            position_down = (screen_height // 2) - (height // 2)

        window.geometry(f"{width}x{height}+{position_right}+{position_down}")

        # Handle the close button (X) action
        def on_close():
            user_action.set("cancel")  # Set action to 'cancel'
            window.destroy()

        # Bind the on_close callback to the default close button
        window.protocol("WM_DELETE_WINDOW", on_close)

        # Call the content builder if provided
        if content_builder:
            content_builder(window, user_action)

        # Ensure the popup is modal
        window.grab_set()
        window.wait_window()

        return user_action.get()

    @staticmethod
    def messagebox(title, message, yes_text="Yes", no_text="No", master=None, width=400, height=200):
        # Call `create_popup` and get the action result
        def build_content(window, user_action):
            # Add message label
            tk.Label(window, text=message, wraplength=300, justify="left").pack(pady=20)

            # Add buttons
            button_frame = tk.Frame(window)
            button_frame.pack(pady=10)

            def on_yes():
                user_action.set("yes")
                window.destroy()

            def on_no():
                user_action.set("no")
                window.destroy()

            Widgets.button(button_frame, yes_text, on_yes, row=0, column=0)
            Widgets.button(button_frame, no_text, on_no, row=0, column=1)

        action_result = Widgets.create_popup(title, master, content_builder=build_content, width=width, height=height)
        return action_result  # Returns 'yes', 'no', or 'cancel'

    @staticmethod
    def custom_popup(title, content_builder, master=None, width=400, height=300):
        Widgets.create_popup(title, master, width, height, content_builder)

    @staticmethod
    def disclaimer(parent, tooltip_text=None):
        footer_label = ttk.Label(
            parent,
            text="Individual and personal use only.\nNot for commercial use.\nhttps://github.com/ack00gar",
            font=("Arial", 10, "italic", "bold"), justify="center"
        )
        footer_label.grid(row=8, column=0, columnspan=100, padx=5, pady=5, sticky="s")

        if tooltip_text:
            Tooltip(footer_label, tooltip_text)

    @staticmethod
    def _browse_file(file_path, title, file_types, callback):
        file = filedialog.askopenfilename(filetypes=file_types, title=title)
        if file:
            file_path.set(file)
            if callback:
                callback(file)

    @staticmethod
    def frames_input(
            parent,
            label_text,
            state,
            attr,
            row=0,
            column=0,
            label_width_px=LABEL_WIDTH,
            entry_width_px=200,
            callback=None,
            tooltip_text=None
    ):
        """Creates a row of controls for editing time in H:M:S and frames format."""

        def hms_to_frames(hours, minutes, seconds, fps):
            if fps <= 0:
                return 0
            return int(hours * 3600 * fps + minutes * 60 * fps + seconds * fps)

        def frames_to_hms(frames, fps):
            if fps <= 0:
                return (0, 0, 0)
            total_seconds = frames / fps
            h = int(total_seconds // 3600)
            total_seconds = total_seconds % 3600
            m = int(total_seconds // 60)
            total_seconds = total_seconds % 60
            s = int(total_seconds)
            return (h, m, s)

        # Main container
        container = ttk.Frame(parent)
        container.grid(row=row, column=column, sticky="ew", padx=5, pady=5)
        container.columnconfigure(1, weight=1)

        initial_frames = getattr(state, attr, 0) or 0
        value = tk.StringVar(value=str(initial_frames))

        hours_var = tk.StringVar()
        minutes_var = tk.StringVar()
        seconds_var = tk.StringVar()
        frames_var = tk.StringVar()

        # A small mutable flag to prevent "ping-pong" re-entrancy
        in_update = [False]

        def on_fps_change(*_):
            """ Recalculate HH:MM:SS whenever FPS or 'value' changes (externally). """
            fps = getattr(state.video_info, "fps", 0) or 0
            # fps_label.config(text=" {state.video_info.fps} fps" if state.video_info else ' ? fps')

            try:
                f_ = int(value.get())
            except ValueError:
                f_ = 0

            h_, m_, s_ = frames_to_hms(f_, fps)
            hours_var.set(str(h_))
            minutes_var.set(str(m_))
            seconds_var.set(str(s_))

        fps_var = tk.StringVar()
        fps_var.trace_add("write", on_fps_change)

        def update_from_hms(*_):
            if in_update[0]:
                return
            in_update[0] = True
            try:
                fps = getattr(state.video_info, "fps", 0) or 0
                if fps <= 0:
                    return

                try:
                    h = int(hours_var.get())
                    m = int(minutes_var.get())
                    s = int(seconds_var.get())
                except ValueError:
                    return

                new_frames = hms_to_frames(h, m, s, fps)
                frames_var.set(str(new_frames))
                value.set(str(new_frames))

                if callback:
                    callback(new_frames)
            finally:
                in_update[0] = False

        def update_from_frames(*_):
            if in_update[0]:
                return
            in_update[0] = True
            try:
                fps = getattr(state.video_info, "fps", 0) or 0
                if fps <= 0:
                    return

                try:
                    f_ = int(frames_var.get())
                except ValueError:
                    return

                h_, m_, s_ = frames_to_hms(f_, fps)
                hours_var.set(str(h_))
                minutes_var.set(str(m_))
                seconds_var.set(str(s_))
                value.set(str(f_))

                if callback:
                    callback(f_)
            finally:
                in_update[0] = False

        hours_var.trace_add("write", update_from_hms)
        minutes_var.trace_add("write", update_from_hms)
        seconds_var.trace_add("write", update_from_hms)
        frames_var.trace_add("write", update_from_frames)
        value.trace_add("write", on_fps_change)  # in case value is set externally

        # Label on the left
        label = tk.Label(container, text=label_text, anchor="w", width=label_width_px // 7)
        label.grid(row=0, column=0, sticky="w", padx=(5, 2))

        # Sub-container for the time fields
        entry_container = ttk.Frame(container, width=entry_width_px)
        entry_container.grid(row=0, column=1, sticky="ew", padx=(2, 5))
        entry_container.grid_propagate(False)

        def make_entry(parent_, textvar, width=4):
            e = ttk.Entry(parent_, textvariable=textvar, width=width)
            e.pack(side="left", padx=(0, 2))
            return e

        hours_entry = make_entry(entry_container, hours_var, width=6)
        tk.Label(entry_container, text=":").pack(side="left")
        minutes_entry = make_entry(entry_container, minutes_var, width=6)
        tk.Label(entry_container, text=":").pack(side="left")
        seconds_entry = make_entry(entry_container, seconds_var, width=6)
        tk.Label(entry_container, text=" frame ").pack(side="left")
        frames_entry = make_entry(entry_container, frames_var, width=23)
        # fps_label = tk.Label(entry_container, text=" ? fps")
        # fps_label.pack(side="left")

        # TODO remove disabled and fix logic
        disable_widgets([hours_entry, minutes_entry, seconds_entry])

        # Optional tooltip
        if tooltip_text:
            Tooltip(hours_entry, tooltip_text)
            Tooltip(minutes_entry, tooltip_text)
            Tooltip(seconds_entry, tooltip_text)
            Tooltip(frames_entry, tooltip_text)

        # Initialize the fields so they match initial_frames (and current FPS if set)
        on_fps_change()
        # TODO readd hours_entry, minutes_entry, seconds_entry,
        return container, (frames_entry,), value

