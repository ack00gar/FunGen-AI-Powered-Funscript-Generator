import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import subprocess
import json
import os


class VideoSegmentExtractor:
    def __init__(self, root):
        self.root = root
        self.video_path = None
        self.funscript_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.start_time = 0  # Start time in seconds
        self.duration = 30  # Default duration in seconds
        self.category = "HJ-BJ"  # Default category

        # GUI Elements
        self.canvas = tk.Canvas(root, width=640, height=360)
        self.canvas.pack(pady=10)

        # Video and Funscript name labels
        self.video_name_label = tk.Label(root, text="Selected Video: None", anchor="w")
        self.video_name_label.pack(fill="x", padx=10, pady=5)

        self.funscript_name_label = tk.Label(root, text="Selected Funscript: None", anchor="w")
        self.funscript_name_label.pack(fill="x", padx=10, pady=5)

        # Start time input (hours, minutes, seconds)
        self.time_frame = tk.Frame(root)
        self.time_frame.pack(pady=10)

        self.hour_label = tk.Label(self.time_frame, text="Hours:")
        self.hour_label.grid(row=0, column=0, padx=5)
        self.hour_var = tk.IntVar(value=0)
        self.hour_entry = tk.Entry(self.time_frame, textvariable=self.hour_var, width=5)
        self.hour_entry.grid(row=0, column=1, padx=5)

        self.minute_label = tk.Label(self.time_frame, text="Minutes:")
        self.minute_label.grid(row=0, column=2, padx=5)
        self.minute_var = tk.IntVar(value=0)
        self.minute_entry = tk.Entry(self.time_frame, textvariable=self.minute_var, width=5)
        self.minute_entry.grid(row=0, column=3, padx=5)

        self.second_label = tk.Label(self.time_frame, text="Seconds:")
        self.second_label.grid(row=0, column=4, padx=5)
        self.second_var = tk.IntVar(value=0)
        self.second_entry = tk.Entry(self.time_frame, textvariable=self.second_var, width=5)
        self.second_entry.grid(row=0, column=5, padx=5)

        self.update_time_button = tk.Button(self.time_frame, text="Update Start Time", command=self.update_start_time)
        self.update_time_button.grid(row=0, column=6, padx=10)

        # Duration dropdown with label
        self.duration_frame = tk.Frame(root)
        self.duration_frame.pack(pady=10)

        self.duration_label = tk.Label(self.duration_frame, text="Select Duration (seconds):")
        self.duration_label.grid(row=0, column=0, padx=5)

        self.duration_var = tk.IntVar(value=self.duration)
        self.duration_dropdown = ttk.Combobox(self.duration_frame, textvariable=self.duration_var, values=[10, 30, 60, 120, 240, 300], width=10)
        self.duration_dropdown.grid(row=0, column=1, padx=5)
        self.duration_dropdown.bind("<<ComboboxSelected>>", self.update_duration)

        # Category dropdown with label
        self.category_frame = tk.Frame(root)
        self.category_frame.pack(pady=10)

        self.category_label = tk.Label(self.category_frame, text="Select Category:")
        self.category_label.grid(row=0, column=0, padx=5)

        self.category_var = tk.StringVar(value=self.category)
        self.category_dropdown = ttk.Combobox(self.category_frame, textvariable=self.category_var, 
                                              values=["HJ-BJ", "Doggy", "Cowgirl", "RevCowgirl", "Pronebone"], width=15)
        self.category_dropdown.grid(row=0, column=1, padx=5)
        self.category_dropdown.bind("<<ComboboxSelected>>", self.update_category)

        # Progress label
        self.progress_label = tk.Label(root, text="Progress: 0%")
        self.progress_label.pack(pady=10)

        # Buttons frame (Select Video, Select Funscript, Extract Segment)
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack(pady=10)

        self.select_video_button = ttk.Button(self.buttons_frame, text="Select Video", command=self.select_video)
        self.select_video_button.grid(row=0, column=0, padx=5)

        self.select_funscript_button = ttk.Button(self.buttons_frame, text="Select Funscript", command=self.select_funscript)
        self.select_funscript_button.grid(row=0, column=1, padx=5)

        self.extract_button = ttk.Button(self.buttons_frame, text="Extract Segment", command=self.extract_segment, state=tk.DISABLED)
        self.extract_button.grid(row=0, column=2, padx=5)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv"), ("All Files", "*.*")])
        if self.video_path:
            self.video_name_label.config(text=f"Selected Video: {os.path.basename(self.video_path)}")
            self.load_video()

    def select_funscript(self):
        self.funscript_path = filedialog.askopenfilename(
            filetypes=[("Funscript Files", "*.funscript"), ("All Files", "*.*")])
        if self.funscript_path:
            self.funscript_name_label.config(text=f"Selected Funscript: {os.path.basename(self.funscript_path)}")
            if self.video_path:
                self.extract_button.config(state=tk.NORMAL)

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            messagebox.showerror("Error", "Invalid video FPS.")
            return

        self.update_frame(0)

    def update_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 360))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def update_start_time(self):
        try:
            hours = self.hour_var.get()
            minutes = self.minute_var.get()
            seconds = self.second_var.get()
            self.start_time = hours * 3600 + minutes * 60 + seconds
            self.update_frame(int(self.start_time * self.fps))
            self.progress_label.config(text=f"Selected Start Time: {self.start_time:.2f}s")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid time input: {str(e)}")

    def update_duration(self, event):
        self.duration = self.duration_var.get()

    def update_category(self, event):
        self.category = self.category_var.get()

    def extract_segment(self):
        start_time = self.start_time
        duration = self.duration
        category = self.category

        # Create Extracts folder and category subfolder
        extracts_folder = os.path.join(os.path.expanduser("~"), "Bench_extracts")
        category_folder = os.path.join(extracts_folder, category)
        os.makedirs(category_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_video = f"{base_name}_segment_{int(start_time)}.mp4"
        output_funscript = f"{base_name}_segment_{int(start_time)}-ref.funscript"

        output_video = os.path.join(category_folder, output_video)
        output_funscript = os.path.join(category_folder, output_funscript)

        # Extract video segment
        self.extract_video_segment(self.video_path, output_video, start_time, duration)

        # Extract funscript segment if funscript is selected
        if self.funscript_path:
            self.extract_funscript_segment(self.funscript_path, output_funscript, start_time, duration)

    def extract_video_segment(self, input_video, output_video, start_time, duration):
        try:
            self.progress_label.config(text="Extracting video...")
            command = ['ffmpeg', '-i', input_video, '-ss', str(start_time), '-t', str(duration),
                       '-c', 'copy', '-reset_timestamps', '1', output_video]
            print(f"Running command: {' '.join(command)}")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(stdout.decode('utf-8'))
            print(stderr.decode('utf-8'))
            if process.returncode != 0:
                raise Exception(stderr.decode('utf-8'))

            if os.path.exists(output_video):
                messagebox.showinfo("Success", f"Video segment saved to {output_video}")
            else:
                messagebox.showerror("Error", "Video extraction failed.")
        except Exception as e:
            messagebox.showerror("Error", f"Video extraction failed: {str(e)}")
        finally:
            self.progress_label.config(text="Progress: 100%")

    def extract_funscript_segment(self, input_funscript, output_funscript, start_time, duration):
        try:
            with open(input_funscript, 'r') as f:
                funscript_data = json.load(f)

            # Convert times to milliseconds
            start_time_ms = int(start_time * 1000)
            end_time_ms = start_time_ms + int(duration * 1000)

            # Filter actions within the time range
            segment_actions = [action for action in funscript_data['actions']
                               if start_time_ms <= action['at'] <= end_time_ms]

            # Adjust timestamps relative to new start time
            for action in segment_actions:
                action['at'] -= start_time_ms

            funscript_data['actions'] = segment_actions
            with open(output_funscript, 'w') as f:
                json.dump(funscript_data, f, indent=4)

            messagebox.showinfo("Success", f"Funscript segment saved to {output_funscript}")
        except Exception as e:
            messagebox.showerror("Error", f"Funscript extraction failed: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Video Segment Extractor")
    app = VideoSegmentExtractor(root)
    root.mainloop()