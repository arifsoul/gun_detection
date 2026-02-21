import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import os
import time
import yaml
import torch
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from ultralytics import YOLO
import TKinterModernThemes as TKMT

# Constants
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "docs"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
INFERENCE_OUTPUT_DIR = PROJECT_ROOT / "runs" / "inference"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Ensure output directory exists
INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ScrollableSidebar(tk.Frame):
    """A frame with a vertical scrollbar that makes content scrollable."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.canvas.yview
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class GunDetectionApp(TKMT.ThemedTKinterFrame):
    def __init__(self):
        super().__init__("Gun Detection System", "sun-valley", "dark")
        self.root.geometry("1400x900")

        # Variables
        self.video_source_path = tk.StringVar(value="")
        self.conf_threshold = tk.DoubleVar(value=0.40)
        self.brightness = tk.DoubleVar(value=1.0)
        self.day_night_thresh = tk.IntVar(value=100)
        self.save_output = tk.BooleanVar(value=False)
        self.output_format = tk.StringVar(value="mp4")
        self.is_previewing = tk.BooleanVar(value=False)
        self.working_resolution = tk.StringVar(value="720p")
        self.env_status_var = tk.StringVar(value="DAY")

        # Playback Controls
        self.is_paused = tk.BooleanVar(value=False)
        self.video_pos = tk.IntVar(value=0)
        self.total_frames = 0
        self.video_fps = 30
        self.user_seeking = False

        # UI Responsiveness
        self.available_canvas_width = 1100
        self.available_canvas_height = 700

        # Internal state
        self.is_running = False
        self.thread = None
        self.cap = None
        self.writer = None
        self.model_map = {}

        self.setup_ui()

    # ──────────────────────────────────────────────
    #  UI SETUP
    # ──────────────────────────────────────────────

    def setup_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── Scrollable Sidebar ──
        sidebar_container = ttk.Frame(main_paned, width=300)
        main_paned.add(sidebar_container, weight=1)

        scrollable = ScrollableSidebar(sidebar_container)
        scrollable.pack(fill=tk.BOTH, expand=True)
        inner = scrollable.inner  # All widgets go here

        ttk.Label(inner, text="Configuration", font=("Segoe UI", 18, "bold")).pack(
            pady=(0, 15)
        )

        # 1. Model
        model_frame = ttk.LabelFrame(inner, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5)

        self.load_model_options()

        model_list_frame = ttk.Frame(model_frame)
        model_list_frame.pack(fill=tk.BOTH, expand=True)

        self.model_listbox = tk.Listbox(
            model_list_frame, selectmode=tk.MULTIPLE, height=5, exportselection=False
        )
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(
            model_list_frame, orient=tk.VERTICAL, command=self.model_listbox.yview
        )
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.config(yscrollcommand=sb.set)

        for name in self.model_map.keys():
            self.model_listbox.insert(tk.END, name)
        if self.model_map:
            self.model_listbox.selection_set(0)

        self.model_listbox.bind(
            "<<ListboxSelect>>", lambda e: self._update_target_name()
        )

        # 2. Input Video
        input_frame = ttk.LabelFrame(inner, text="Input Video", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Entry(input_frame, textvariable=self.video_source_path).pack(
            fill=tk.X, pady=5
        )
        ttk.Button(input_frame, text="Browse...", command=self.browse_file).pack(
            fill=tk.X
        )

        # 3. Resolution
        res_frame = ttk.LabelFrame(inner, text="Working Resolution", padding=10)
        res_frame.pack(fill=tk.X, pady=5)

        resolutions = ["Original", "1080p", "720p", "480p"]
        self.res_combobox = ttk.Combobox(
            res_frame,
            textvariable=self.working_resolution,
            values=resolutions,
            state="readonly",
        )
        self.res_combobox.pack(fill=tk.X)
        ttk.Label(
            res_frame,
            text="Lower resolution = Faster speed",
            font=("Segoe UI", 8, "italic"),
        ).pack(pady=(5, 0))

        # 4. Confidence
        conf_frame = ttk.LabelFrame(inner, text="Confidence Threshold", padding=10)
        conf_frame.pack(fill=tk.X, pady=5)

        self.conf_label = ttk.Label(conf_frame, text=f"{self.conf_threshold.get():.2f}")
        self.conf_label.pack()
        ttk.Scale(
            conf_frame,
            variable=self.conf_threshold,
            from_=0.0,
            to=1.0,
            command=lambda v: self.conf_label.config(text=f"{float(v):.2f}"),
        ).pack(fill=tk.X)

        # 4. Environment Simulation
        sim_frame = ttk.LabelFrame(inner, text="Environment Simulation", padding=10)
        sim_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sim_frame, text="Brightness").pack(anchor=tk.W)
        self.env_status_label = ttk.Label(
            sim_frame,
            textvariable=self.env_status_var,
            font=("Segoe UI", 10, "bold"),
            foreground="#f1c40f",
        )
        self.env_status_label.pack()
        self.brit_label = ttk.Label(sim_frame, text=f"{self.brightness.get():.2f}")
        self.brit_label.pack()
        ttk.Scale(
            sim_frame,
            variable=self.brightness,
            from_=0.1,
            to=2.0,
            command=self._on_sim_slider_change,
        ).pack(fill=tk.X)

        # 5. Output
        out_frame = ttk.LabelFrame(inner, text="Output", padding=10)
        out_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            out_frame, text="Save Output Video", variable=self.save_output
        ).pack(anchor=tk.W)

        fmt_row = ttk.Frame(out_frame)
        fmt_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(fmt_row, text="Format:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            fmt_row,
            text="MP4",
            variable=self.output_format,
            value="mp4",
            command=self._update_target_name,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            fmt_row,
            text="GIF",
            variable=self.output_format,
            value="gif",
            command=self._update_target_name,
        ).pack(side=tk.LEFT, padx=5)

        self.output_name_var = tk.StringVar(value="[video]_output_[model].ext")
        ttk.Label(
            out_frame,
            textvariable=self.output_name_var,
            font=("Consolas", 8),
            foreground="#666666",
        ).pack(anchor=tk.W, pady=(5, 0))

        # 6. Results
        results_frame = ttk.LabelFrame(inner, text="Results", padding=10)
        results_frame.pack(fill=tk.X, pady=5)

        res_list_frame = ttk.Frame(results_frame)
        res_list_frame.pack(fill=tk.X)

        self.results_listbox = tk.Listbox(res_list_frame, height=6)
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        res_scroll = ttk.Scrollbar(
            res_list_frame, orient=tk.VERTICAL, command=self.results_listbox.yview
        )
        res_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_listbox.config(yscrollcommand=res_scroll.set)

        res_btns = ttk.Frame(results_frame)
        res_btns.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            res_btns, text="Play Selected", command=self.play_selected_result
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        ttk.Button(res_btns, text="Delete", command=self.delete_selected_result).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0)
        )

        self.refresh_results_list()

        # ── Main Content ──
        content_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(content_frame, weight=4)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            content_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 16, "bold"),
            background="#ffffff",
            anchor=tk.CENTER,
            padding=10,
        )
        self.status_label.pack(fill=tk.X, pady=(5, 0))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            content_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        metrics_frame = ttk.Frame(content_frame)
        metrics_frame.pack(fill=tk.X, pady=5)

        self.metric_condition = tk.StringVar(value="Condition: -")
        self.metric_device = tk.StringVar(value="Device: -")
        self.metric_detections = tk.StringVar(value="Detections: 0")
        self.metric_fps = tk.StringVar(value="FPS: -")
        self.metric_latency = tk.StringVar(value="Inf: -")
        self.metric_res = tk.StringVar(value="Res: -")

        # Row 1
        row1 = ttk.Frame(metrics_frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, textvariable=self.metric_condition, font=("Consolas", 10)).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Label(row1, textvariable=self.metric_device, font=("Consolas", 10)).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Label(row1, textvariable=self.metric_fps, font=("Consolas", 10)).pack(
            side=tk.LEFT, padx=10
        )

        # Row 2
        row2 = ttk.Frame(metrics_frame)
        row2.pack(fill=tk.X)
        ttk.Label(
            row2, textvariable=self.metric_detections, font=("Consolas", 10)
        ).pack(side=tk.LEFT, padx=10)
        ttk.Label(row2, textvariable=self.metric_latency, font=("Consolas", 10)).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Label(row2, textvariable=self.metric_res, font=("Consolas", 10)).pack(
            side=tk.LEFT, padx=10
        )

        self.video_label = ttk.Label(content_frame, background="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ── Video Controls ──
        self.controls_frame = ttk.Frame(content_frame, padding=5)
        self.controls_frame.pack(fill=tk.X)

        self.start_pre_btn = ttk.Button(
            self.controls_frame, text="🔍 Preview", command=self.toggle_preview
        )
        self.start_pre_btn.pack(side=tk.LEFT, padx=5)

        self.start_inf_btn = ttk.Button(
            self.controls_frame, text="🚀 Inference", command=self.start_inference
        )
        self.start_inf_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            self.controls_frame,
            text="⏹ Stop",
            command=self.stop_playback,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(
            self.controls_frame, text="⏸ Pause", command=self.toggle_pause, width=10
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.time_label = ttk.Label(
            self.controls_frame, text="00:00 / 00:00", font=("Consolas", 10)
        )
        self.time_label.pack(side=tk.LEFT, padx=10)

        self.seeker = ttk.Scale(
            self.controls_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.video_pos,
            command=self._on_seeker_change,
        )
        self.seeker.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.seeker.bind("<ButtonPress-1>", self._on_seeker_press)
        self.seeker.bind("<ButtonRelease-1>", self._on_seeker_release)

        self.set_control_mode("idle")

        content_frame.bind("<Configure>", self._on_content_resize)

    def _on_content_resize(self, event):
        """Update available space for video when window/frame is resized."""
        # Use a small margin to avoid scrollbar jitters or overflow
        self.available_canvas_width = max(event.width - 20, 100)
        self.available_canvas_height = max(event.height - 100, 100)

    # ──────────────────────────────────────────────
    #  MODEL LOADING
    # ──────────────────────────────────────────────

    def load_model_options(self):
        self.model_map = {}

        if MLRUNS_DIR.exists():
            for args_path in MLRUNS_DIR.rglob("artifacts/training_results/args.yaml"):
                try:
                    artifacts_dir = args_path.parent.parent
                    weights_path = artifacts_dir / "weights" / "best.pt"

                    if weights_path.exists():
                        with open(args_path, "r") as f:
                            args = yaml.safe_load(f)
                            model_name = args.get("name", "Unknown")

                        run_id = artifacts_dir.parent.name
                        display_name = f"{model_name} ({run_id[:8]})"
                        self.model_map[display_name] = str(weights_path)
                except Exception:
                    continue

        if not self.model_map:
            self.model_map["Default (yolov8n)"] = "yolov8n.pt"

    # ──────────────────────────────────────────────
    #  BROWSING
    # ──────────────────────────────────────────────

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if filename:
            self.video_source_path.set(filename)
            self.refresh_results_list()
            self._update_target_name()
            self.show_thumbnail(filename)
            self.set_control_mode("idle")

    def show_thumbnail(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.seeker.config(to=self.total_frames - 1)

                # Update time label
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_sec = self.total_frames / fps
                self.time_label.config(text=f"00:00 / {self._format_time(total_sec)}")

                # Display frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # We need available_canvas_width/height but they might be 0 if window not shown yet
                # Use a default if needed
                w_w = (
                    self.available_canvas_width
                    if self.available_canvas_width > 100
                    else 640
                )
                w_h = (
                    self.available_canvas_height
                    if self.available_canvas_height > 100
                    else 480
                )

                h, w = frame.shape[:2]
                ratio = min(w_w / w, w_h / h)
                disp_w, disp_h = int(w * ratio), int(h * ratio)
                resized = cv2.resize(rgb, (disp_w, disp_h))
                pil = Image.fromarray(resized)
                self._show_image(pil)
            cap.release()

    def set_control_mode(self, mode):
        """idle, preview, inference"""
        # Pack everything off first to reset positions
        for widget in self.controls_frame.winfo_children():
            widget.pack_forget()

        # Re-pack based on mode
        if mode == "idle":
            self.start_pre_btn.pack(side=tk.LEFT, padx=5)
            self.start_inf_btn.pack(side=tk.LEFT, padx=5)
            self.start_pre_btn.config(state=tk.NORMAL)
            self.start_inf_btn.config(state=tk.NORMAL)
        elif mode == "preview":
            self.start_pre_btn.pack(side=tk.LEFT, padx=5)
            self.start_inf_btn.pack(side=tk.LEFT, padx=5)
            self.start_pre_btn.config(state=tk.DISABLED)
            self.start_inf_btn.config(state=tk.DISABLED)

            self.stop_btn.pack(side=tk.RIGHT, padx=5)
            self.play_btn.pack(side=tk.RIGHT, padx=5)
            self.seeker.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
            self.time_label.pack(side=tk.RIGHT, padx=10)

            self.stop_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL)
        elif mode == "inference":
            self.start_pre_btn.pack(side=tk.LEFT, padx=5)
            self.start_inf_btn.pack(side=tk.LEFT, padx=5)
            self.start_pre_btn.config(state=tk.DISABLED)
            self.start_inf_btn.config(state=tk.DISABLED)

            self.stop_btn.pack(side=tk.RIGHT, padx=5)
            self.seeker.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
            self.time_label.pack(side=tk.RIGHT, padx=10)

            self.stop_btn.config(state=tk.NORMAL)

    # ──────────────────────────────────────────────
    #  RESULTS
    # ──────────────────────────────────────────────

    def refresh_results_list(self):
        self.results_listbox.delete(0, tk.END)
        if INFERENCE_OUTPUT_DIR.exists():
            files = sorted(
                list(INFERENCE_OUTPUT_DIR.glob("*.mp4"))
                + list(INFERENCE_OUTPUT_DIR.glob("*.gif")),
                key=os.path.getmtime,
                reverse=True,
            )
            for file_path in files:
                self.results_listbox.insert(tk.END, file_path.name)

    def play_selected_result(self):
        sel = self.results_listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No result video selected.")
            return

        filename = self.results_listbox.get(sel[0])
        file_path = INFERENCE_OUTPUT_DIR / filename

        if not file_path.exists():
            messagebox.showerror("Error", f"File not found: {file_path}")
            return

        self.stop_playback()
        self.is_running = True
        self.stop_btn.config(state=tk.NORMAL)

        self.thread = threading.Thread(
            target=self.playback_loop, args=(str(file_path),), daemon=True
        )
        self.thread.start()

    def delete_selected_result(self):
        sel = self.results_listbox.curselection()
        if not sel:
            messagebox.showwarning("Delete", "No result video selected.")
            return

        filename = self.results_listbox.get(sel[0])
        file_path = INFERENCE_OUTPUT_DIR / filename

        if messagebox.askyesno(
            "Delete", f"Are you sure you want to delete {filename}?"
        ):
            try:
                if file_path.exists():
                    file_path.unlink()
                self.refresh_results_list()
                messagebox.showinfo("Delete", "File deleted successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete file: {e}")

    # ──────────────────────────────────────────────
    #  PREVIEW
    # ──────────────────────────────────────────────

    # ──────────────────────────────────────────────
    #  CONTROLS
    # ──────────────────────────────────────────────

    def toggle_pause(self):
        self.is_paused.set(not self.is_paused.get())
        if self.is_paused.get():
            self.play_btn.config(text="▶ Play")
        else:
            self.play_btn.config(text="⏸ Pause")

    def stop_playback(self):
        self.is_previewing.set(False)
        self.is_running = False
        self.is_paused.set(False)
        self.play_btn.config(text="⏸ Pause")
        self.video_pos.set(0)
        self.status_var.set("Ready")
        self.status_label.configure(background="#ffffff", foreground="#000000")

        # Reset to thumbnail
        video_path = self.video_source_path.get()
        if video_path and Path(video_path).exists():
            self.show_thumbnail(video_path)

        self.set_control_mode("idle")

    def _on_seeker_change(self, value):
        # Update time label instantly
        if hasattr(self, "total_frames") and self.total_frames > 0:
            fps = 30  # fallback
            video_path = self.video_source_path.get()
            if video_path and Path(video_path).exists():
                # Try to get real fps
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    cap.release()

            curr_sec = int(float(value)) / fps
            total_sec = self.total_frames / fps
            self.time_label.config(
                text=f"{self._format_time(curr_sec)} / {self._format_time(total_sec)}"
            )

        # If stopped, show frame
        if not self.is_previewing.get() and not self.is_running:
            video_path = self.video_source_path.get()
            if video_path and Path(video_path).exists():
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(value)))
                    ret, frame = cap.read()
                    if ret:
                        self._display_frame_manual(frame)
                    cap.release()

    def _display_frame_manual(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Apply brightness simulation even for manual seeker
        alpha = self.brightness.get()
        if alpha != 1.0:
            rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=0)

        w_w = self.available_canvas_width if self.available_canvas_width > 100 else 640
        w_h = (
            self.available_canvas_height if self.available_canvas_height > 100 else 480
        )

        h, w = frame.shape[:2]
        ratio = min(w_w / w, w_h / h)
        disp_w, disp_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(rgb, (disp_w, disp_h))
        pil = Image.fromarray(resized)

        imgtk = ImageTk.PhotoImage(image=pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _on_seeker_press(self, event):
        self.user_seeking = True

    def _on_seeker_release(self, event):
        self.user_seeking = False

    def _format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _update_target_name(self, current_conditions=None):
        if not hasattr(self, "video_source_path"):
            return
        video_path = self.video_source_path.get()
        if not video_path:
            return

        base_name = Path(video_path).stem
        ext = self.output_format.get()

        if current_conditions:
            # If it's a list (from inference loop), join with hyphen
            if isinstance(current_conditions, (list, tuple)):
                cond_str = "-".join([c.upper() for c in current_conditions])
            else:
                cond_str = "_".join(sorted(list(current_conditions))).lower()
        else:
            # Estimate based on current slider
            v = self.brightness.get()
            cond_str = "DAY" if v > 0.8 else "NIGHT"

        # Get current model names (simplified for label)
        sel = self.model_listbox.curselection()
        model_str = (
            "multiple"
            if len(sel) > 1
            else (self.model_listbox.get(sel[0]) if sel else "model")
        )

        target = f"{base_name}_{cond_str}_{model_str}.{ext}"
        self.output_name_var.set(target)

    def _on_sim_slider_change(self, value=None):
        """Called when brightness or day/night sliders change."""
        v = self.brightness.get()
        self.brit_label.config(text=f"{v:.2f}")

        # Update Status Label
        new_status = "DAY" if v > 0.8 else "NIGHT"
        self.env_status_var.set(new_status)
        self.env_status_label.config(
            foreground="#f1c40f" if new_status == "DAY" else "#3498db"
        )

        self._update_target_name()

    def toggle_preview(self):
        # Stop existing stuff
        self.stop_playback()

        video_path = self.video_source_path.get()
        if not video_path or not Path(video_path).exists():
            messagebox.showwarning("Preview", "Please select a video file first.")
            return

        self.is_previewing.set(True)
        self.set_control_mode("preview")
        thread = threading.Thread(target=self.preview_loop, daemon=True)
        thread.start()
        # Else loop will exit by itself on next iteration

    def preview_loop(self):
        """Play video through simulation (brightness/day-night) without inference."""
        video_path = self.video_source_path.get()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        self.status_var.set("🔍 Preview Mode")

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.seeker.config(to=self.total_frames - 1)

        # Get original resolution
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_time = time.time()
        frame_count = 0

        try:
            while self.is_previewing.get() and cap.isOpened():
                loop_start = time.time()

                if self.is_paused.get() and not self.user_seeking:
                    time.sleep(0.1)
                    continue

                if self.user_seeking:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_pos.get())

                ret, frame = cap.read()
                if not ret:
                    if not self.user_seeking:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.video_pos.set(0)
                    continue

                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if not self.user_seeking:
                    self.video_pos.set(current_frame)

                # Update metrics periodically (every 10 frames) to save GUI overhead
                if current_frame % 10 == 0:
                    curr_sec = current_frame / self.video_fps
                    total_sec = self.total_frames / self.video_fps
                    self.time_label.config(
                        text=f"{self._format_time(curr_sec)} / {self._format_time(total_sec)}"
                    )

                # DYNAMIC RESOLUTION
                res_sel = self.working_resolution.get()
                if res_sel == "Original":
                    working_w, working_h = orig_w, orig_h
                else:
                    target_w = 1280
                    if res_sel == "1080p":
                        target_w = 1920
                    elif res_sel == "720p":
                        target_w = 1280
                    elif res_sel == "480p":
                        target_w = 854
                    ratio = target_w / orig_w
                    working_w, working_h = target_w, int(orig_h * ratio)

                if current_frame % 30 == 0:
                    self.metric_res.set(
                        f"Res: {orig_w}x{orig_h} → {working_w}x{working_h}"
                    )

                # Display resize - Combine with early downscale if possible
                max_w, max_h = self.available_canvas_width, self.available_canvas_height
                ratio_display = min(max_w / working_w, max_h / working_h)
                disp_w, disp_h = (
                    int(working_w * ratio_display),
                    int(working_h * ratio_display),
                )

                # OPTIMIZATION: One resize is enough for display
                resized_frame = cv2.resize(
                    frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
                )

                # Brightness
                brightness_val = self.brightness.get()
                if brightness_val != 1.0:
                    resized_frame = cv2.convertScaleAbs(
                        resized_frame, alpha=brightness_val, beta=0
                    )

                # Day/Night ONLY every 30 frames
                if current_frame % 30 == 0:
                    small_frame_hsv = cv2.resize(
                        resized_frame, (100, 100)
                    )  # Even smaller
                    hsv = cv2.cvtColor(small_frame_hsv, cv2.COLOR_BGR2HSV)
                    avg_v = np.mean(hsv[:, :, 2])
                    condition = (
                        "Day" if avg_v > self.day_night_thresh.get() else "Night"
                    )
                    self.metric_condition.set(f"Condition: {condition} (V={avg_v:.1f})")
                    # For preview, we don't track run_conditions or update target name based on them

                # Progress Bar
                prog = (current_frame / self.total_frames) * 100
                self.progress_var.set(prog)

                # Show
                rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                self.root.after(0, lambda p=pil: self._show_image(p))

                # FPS tracking
                frame_count += 1
                curr_t = time.time()
                if curr_t - last_time >= 1.0:
                    self.metric_fps.set(
                        f"FPS: {frame_count / (curr_t - last_time):.1f}"
                    )
                    last_time, frame_count = curr_t, 0

                # Frame-rate control: sleep only if we are faster than the video's FPS
                process_time = time.time() - loop_start
                target_delay = 1.0 / self.video_fps
                sleep_time = max(0.001, target_delay - process_time)
                time.sleep(sleep_time)
        finally:
            cap.release()
            self.status_var.set("Ready")
            self.progress_var.set(0)  # Reset progress bar

    def _show_image(self, pil_image):
        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # ──────────────────────────────────────────────
    #  PLAYBACK
    # ──────────────────────────────────────────────

    def playback_loop(self, video_path):
        try:
            self.status_var.set(f"Playing: {Path(video_path).name}")
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video.")
                self.stop_playback()
                return

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                max_w, max_h = 1100, 750
                h, w = frame.shape[:2]
                ratio = min(max_w / w, max_h / h)
                resized_frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb_frame)

                self.root.after(0, lambda p=pil: self._show_image(p))

                time.sleep(0.03)

            cap.release()
            self.stop_playback()
        except Exception as e:
            print(f"Error in playback: {e}")
            self.stop_playback()

    # ──────────────────────────────────────────────
    #  INFERENCE CONTROL
    # ──────────────────────────────────────────────

    def start_inference(self):
        # Stop preview if running
        self.stop_playback()

        video_path = self.video_source_path.get()
        if not video_path:
            default_path = DATA_DIR / "real_test.mp4"
            if default_path.exists():
                self.video_source_path.set(str(default_path))
                video_path = str(default_path)
            else:
                messagebox.showerror("Error", "Please select a video file.")
                return

        if not os.path.exists(video_path):
            messagebox.showerror("Error", f"File not found: {video_path}")
            return

        selected_indices = self.model_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one model.")
            return

        models_to_run = []
        for idx in selected_indices:
            name = self.model_listbox.get(idx)
            path = self.model_map.get(name)
            if path:
                models_to_run.append((name, path))

        self.is_running = True
        self.set_control_mode("inference")

        self.thread = threading.Thread(
            target=self.inference_loop,
            args=(video_path, models_to_run),
            daemon=True,
        )
        self.thread.start()

    def stop_inference(self):
        self.stop_playback()
        self.status_var.set("Stopped")

    # ──────────────────────────────────────────────
    #  INFERENCE LOOP
    # ──────────────────────────────────────────────

    def inference_loop(self, video_path, models_list):
        total_models = len(models_list)
        output_format = self.output_format.get()
        video_stem = Path(video_path).stem  # Original video name without extension

        run_conditions = []  # Track env sequence for dynamic naming
        gun_found = False  # Track if any gun was detected during the whole run

        for idx, (model_name, model_path) in enumerate(models_list):
            if not self.is_running:
                break

            gif_frames = []

            try:
                # Build sanitized model name
                sanitized_name = (
                    model_name.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(".pt", "")
                )

                # ── Temporary Naming (will rename at end based on runs) ──
                temp_filename = (
                    f"{video_stem}_processing_{sanitized_name}.{output_format}"
                )
                output_path = INFERENCE_OUTPUT_DIR / temp_filename

                self.status_var.set(
                    f"Processing ({idx + 1}/{total_models}): {model_name} → {output_format.upper()}"
                )

                # Device
                if torch.cuda.is_available():
                    device = 0
                    device_name = "GPU"
                else:
                    device = "cpu"
                    device_name = "CPU"

                self.metric_device.set(f"Device: {device_name}")

                model = YOLO(model_path)

                self.cap = cv2.VideoCapture(video_path)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open video source.")
                    self.stop_inference()
                    return

                # Get original resolution
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.seeker.config(to=self.total_frames - 1)
                orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Writer (MP4 only) - Always save at working resolution for consistency and size
                # This needs to be set *before* the loop, and resolution cannot change for MP4.
                # So, for MP4, we'll use the initial working resolution.
                initial_res_sel = self.working_resolution.get()
                initial_working_w, initial_working_h = orig_w, orig_h
                if initial_res_sel != "Original":
                    initial_target_w = 1280
                    if initial_res_sel == "1080p":
                        initial_target_w = 1920
                    elif initial_res_sel == "720p":
                        initial_target_w = 1280
                    elif initial_res_sel == "480p":
                        initial_target_w = 854
                    initial_ratio = initial_target_w / orig_w
                    initial_working_w, initial_working_h = (
                        initial_target_w,
                        int(orig_h * initial_ratio),
                    )

                if self.save_output.get() and output_format == "mp4":
                    fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                    self.writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (initial_working_w, initial_working_h),
                    )
                else:
                    self.writer = None

                last_time = time.time()
                frame_count = 0

                while self.is_running and self.cap.isOpened():
                    if self.is_paused.get():
                        time.sleep(0.1)
                        continue

                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    current_f = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.video_pos.set(current_f)

                    # Time Label
                    curr_p = current_f / (self.cap.get(cv2.CAP_PROP_FPS) or 30)
                    total_p = self.total_frames / (self.cap.get(cv2.CAP_PROP_FPS) or 30)
                    self.time_label.config(
                        text=f"{self._format_time(curr_p)} / {self._format_time(total_p)}"
                    )

                    # Progress Bar
                    prog = (current_f / self.total_frames) * 100
                    self.progress_var.set(prog)

                    # DYNAMIC RESOLUTION
                    res_sel = self.working_resolution.get()
                    if res_sel == "Original":
                        w_w, w_h = orig_w, orig_h
                    else:
                        t_w = 1280
                        if res_sel == "1080p":
                            t_w = 1920
                        elif res_sel == "720p":
                            t_w = 1280
                        elif res_sel == "480p":
                            t_w = 854

                        ratio_w = t_w / orig_w
                        w_w, w_h = t_w, int(orig_h * ratio_w)

                    s_needed = w_w != orig_w
                    if current_f % 30 == 0:
                        self.metric_res.set(f"Res: {orig_w}x{orig_h} → {w_w}x{w_h}")

                    # EARLY DOWNSCALE (only if needed for model or save consistency)
                    if s_needed:
                        frame = cv2.resize(
                            frame, (w_w, w_h), interpolation=cv2.INTER_AREA
                        )

                    # Simulation (on working frame)
                    brightness_val = self.brightness.get()
                    if brightness_val != 1.0:
                        frame = cv2.convertScaleAbs(frame, alpha=brightness_val, beta=0)

                    # Day/Night ONLY every 30 frames
                    if current_f % 30 == 0:
                        small_frame_hsv = cv2.resize(frame, (100, 100))
                        hsv = cv2.cvtColor(small_frame_hsv, cv2.COLOR_BGR2HSV)
                        avg_v = np.mean(hsv[:, :, 2])
                        condition = (
                            "Day" if avg_v > self.day_night_thresh.get() else "Night"
                        )
                        condition_label = condition.upper()
                        if not run_conditions or run_conditions[-1] != condition_label:
                            run_conditions.append(condition_label)
                            self._update_target_name(run_conditions)

                        self.metric_condition.set(
                            f"Condition: {condition} (V={avg_v:.1f})"
                        )

                    # Inference (on working frame)
                    t0 = time.time()
                    results = model.track(
                        frame,
                        conf=self.conf_threshold.get(),
                        persist=True,
                        verbose=False,
                        device=device,
                        imgsz=640,  # Ensure YOLO uses consistent input size
                    )
                    t_inf = (time.time() - t0) * 1000
                    res = results[0]
                    annotated_frame = res.plot()

                    gun_detected = len(res.boxes) > 0
                    if gun_detected:
                        gun_found = True  # Flag for final naming

                    if current_f % 5 == 0:
                        self.metric_detections.set(f"Detections: {len(res.boxes)}")
                        self.metric_latency.set(f"Inf: {t_inf:.1f}ms")

                    # Save (MP4 Writer caveat: Resolution must stay consistent for video file)
                    if self.save_output.get():
                        if output_format == "mp4" and self.writer:
                            # Note: Dynamic resolution during save will corrupt MP4 if w_w/w_h changes
                            # We keep simple for now, using the initial resolution for the writer.
                            # If the dynamic resolution (w_w, w_h) changes, the frame will be resized
                            # to the initial_working_w, initial_working_h before writing.
                            if w_w != initial_working_w or w_h != initial_working_h:
                                annotated_frame_for_save = cv2.resize(
                                    annotated_frame,
                                    (initial_working_w, initial_working_h),
                                )
                            else:
                                annotated_frame_for_save = annotated_frame
                            self.writer.write(annotated_frame_for_save)
                        elif output_format == "gif":
                            rgb_for_gif = cv2.cvtColor(
                                annotated_frame, cv2.COLOR_BGR2RGB
                            )
                            gif_frames.append(Image.fromarray(rgb_for_gif))

                    # Responsive resize for display
                    max_w = self.available_canvas_width
                    max_h = self.available_canvas_height

                    ratio_disp = min(max_w / w_w, max_h / w_h)
                    disp_w, disp_h = int(w_w * ratio_disp), int(w_h * ratio_disp)
                    resized_frame = cv2.resize(annotated_frame, (disp_w, disp_h))
                    rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)

                    self.root.after(
                        0, lambda p=pil, d=gun_detected: self.update_video_label(p, d)
                    )

                    # FPS Calculation
                    frame_count += 1
                    curr_time = time.time()
                    elapsed = curr_time - last_time
                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        self.metric_fps.set(f"FPS: {fps:.1f}")
                        last_time = curr_time
                        frame_count = 0

                    time.sleep(0.001)

                # Per-model cleanup
                self.cap.release()
                if self.writer:
                    self.writer.release()

                # Save GIF
                if self.save_output.get() and output_format == "gif" and gif_frames:
                    self.status_var.set(f"Saving GIF: {model_name}...")
                    try:
                        fps_source = 30
                        duration = int(1000 / fps_source)
                        # We can't easily show progress of gif_frames[0].save,
                        # so we just set to 100% or indeterminate.
                        # But we can at least show the status.
                        self.progress_var.set(95)
                        gif_frames[0].save(
                            output_path,
                            save_all=True,
                            append_images=gif_frames[1:],
                            optimize=False,
                            duration=duration,
                            loop=0,
                        )
                        self.progress_var.set(100)
                    except Exception as e:
                        print(f"Error saving GIF: {e}")

                # ── FINAL RENAMING ──
                if self.save_output.get() and output_path.exists():
                    # Environment sequence joined by hyphen
                    env_str = "-".join(run_conditions) if run_conditions else "UNKNOWN"

                    # Add gun tag if found
                    cond_str = f"{env_str}_GUN_DETECTED" if gun_found else env_str

                    final_filename = (
                        f"{video_stem}_{cond_str}_{sanitized_name}.{output_format}"
                    )
                    final_path = INFERENCE_OUTPUT_DIR / final_filename

                    # If target exists, add index
                    counter = 1
                    while final_path.exists():
                        final_path = (
                            INFERENCE_OUTPUT_DIR
                            / f"{video_stem}_{cond_str}_{sanitized_name}_{counter}.{output_format}"
                        )
                        counter += 1

                    output_path.rename(final_path)
                    print(f"Exported: {final_path.name}")

            except Exception as e:
                print(f"Error in inference loop for {model_name}: {e}")

            # Reset progress for next model
            self.progress_var.set(0)

        self.stop_inference()
        self.root.after(0, self.refresh_results_list)

    # ──────────────────────────────────────────────
    #  UI HELPERS
    # ──────────────────────────────────────────────

    def update_video_label(self, pil_image, gun_detected):
        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if gun_detected:
            self.status_label.configure(background="#ff3333", foreground="#ffffff")
            self.status_var.set("⚠️ GUN DETECTED!")
        else:
            self.status_label.configure(background="#ccffcc", foreground="#006600")
            self.status_var.set("✅ SAFE")


if __name__ == "__main__":
    app = GunDetectionApp()
    app.root.mainloop()
