import tkinter as tk
from tkinter import ttk, filedialog, Button

from script_generator.gui.utils.tooltip import Tooltip

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

        value = tk.StringVar(value=getattr(state, attr))

        if callback:
            value.trace_add("write", lambda *args: setattr(state, attr, value.get()))

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
            if command:
                command(val)

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
            Tooltip(button, tooltip_text)

        return container, entry, file_path

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
    def messagebox(title, message, yes_text="Yes", no_text="No"):
        # Create a Toplevel window
        window = tk.Toplevel()
        window.title(title)
        window.geometry("350x140")

        # Add message label
        tk.Label(window, text=message, wraplength=350, justify="left").pack(pady=20)

        # Container for buttons
        button_frame = tk.Frame(window)
        button_frame.pack(pady=10)

        # Variable to store user choice
        user_choice = tk.BooleanVar(value=None)

        # Yes button
        def on_yes():
            user_choice.set(True)
            window.destroy()

        Widgets.button(button_frame, yes_text, on_yes, row=0, column=0)

        # No button
        def on_no():
            user_choice.set(False)
            window.destroy()

        Widgets.button(button_frame, no_text, on_no, row=0, column=1)

        # Wait for the user to close the dialog
        window.grab_set()
        window.wait_window()

        return user_choice.get()

    @staticmethod
    def disclaimer(parent, tooltip_text=None):
        footer_label = ttk.Label(
            parent,
            text="Individual and personal use only.\nNot for commercial use.\nk00gar 2025 - https://github.com/ack00gar",
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

