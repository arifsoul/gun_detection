import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import os
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

        # 2. Input Video
        input_frame = ttk.LabelFrame(inner, text="Input Video", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Entry(input_frame, textvariable=self.video_source_path).pack(
            fill=tk.X, pady=5
        )
        ttk.Button(input_frame, text="Browse...", command=self.browse_file).pack(
            fill=tk.X
        )

        # 3. Confidence
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
        self.brit_label = ttk.Label(sim_frame, text=f"{self.brightness.get():.2f}")
        self.brit_label.pack()
        ttk.Scale(
            sim_frame,
            variable=self.brightness,
            from_=0.1,
            to=2.0,
            command=self._on_sim_slider_change,
        ).pack(fill=tk.X)

        # Preview Toggle
        preview_row = ttk.Frame(sim_frame)
        preview_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            preview_row,
            text="Live Preview (before inference)",
            variable=self.is_previewing,
            command=self.toggle_preview,
        ).pack(anchor=tk.W)

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
            fmt_row, text="MP4", variable=self.output_format, value="mp4"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            fmt_row, text="GIF", variable=self.output_format, value="gif"
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(
            out_frame, text="Naming: [video]_output_[model].ext", font=("Consolas", 9)
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

        ttk.Button(
            results_frame, text="Play Selected", command=self.play_selected_result
        ).pack(fill=tk.X, pady=(5, 0))
        self.refresh_results_list()

        # 7. Buttons
        btn_frame = ttk.Frame(inner, padding=(0, 10))
        btn_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(
            btn_frame, text="START INFERENCE", command=self.start_inference
        )
        self.start_btn.pack(fill=tk.X, pady=5)

        self.stop_btn = ttk.Button(
            btn_frame, text="STOP", command=self.stop_inference, state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=5)

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
        self.status_label.pack(fill=tk.X, pady=5)

        metrics_frame = ttk.Frame(content_frame)
        metrics_frame.pack(fill=tk.X, pady=5)

        self.metric_condition = tk.StringVar(value="Condition: -")
        self.metric_device = tk.StringVar(value="Device: -")
        self.metric_detections = tk.StringVar(value="Detections: 0")

        ttk.Label(
            metrics_frame, textvariable=self.metric_condition, font=("Consolas", 11)
        ).pack(side=tk.LEFT, padx=15)
        ttk.Label(
            metrics_frame, textvariable=self.metric_device, font=("Consolas", 11)
        ).pack(side=tk.LEFT, padx=15)
        ttk.Label(
            metrics_frame, textvariable=self.metric_detections, font=("Consolas", 11)
        ).pack(side=tk.LEFT, padx=15)

        self.video_label = ttk.Label(content_frame, background="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

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
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if filename:
            self.video_source_path.set(filename)

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

        if self.is_running:
            messagebox.showwarning("Warning", "Inference is running. Stop it first.")
            return

        self.is_running = True
        self.stop_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.DISABLED)

        self.thread = threading.Thread(
            target=self.playback_loop, args=(str(file_path),), daemon=True
        )
        self.thread.start()

    # ──────────────────────────────────────────────
    #  PREVIEW
    # ──────────────────────────────────────────────

    def _on_sim_slider_change(self, value=None):
        """Called when brightness or day/night sliders change."""
        # Update brightness label only (day/night slider was removed from UI)
        self.brit_label.config(text=f"{self.brightness.get():.2f}")

    def toggle_preview(self):
        if self.is_previewing.get():
            video_path = self.video_source_path.get()
            if not video_path or not Path(video_path).exists():
                messagebox.showwarning(
                    "Preview", "Please select a valid video file first."
                )
                self.is_previewing.set(False)
                return
            if self.is_running:
                messagebox.showwarning("Preview", "Stop inference first.")
                self.is_previewing.set(False)
                return
            thread = threading.Thread(target=self.preview_loop, daemon=True)
            thread.start()
        # Else loop will exit by itself on next iteration

    def preview_loop(self):
        """Play video through simulation (brightness/day-night) without inference."""
        video_path = self.video_source_path.get()
        cap = cv2.VideoCapture(video_path)
        self.status_var.set("🔍 Preview Mode")

        try:
            while self.is_previewing.get() and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
                    continue

                # Apply brightness
                brightness_val = self.brightness.get()
                if brightness_val != 1.0:
                    frame = cv2.convertScaleAbs(frame, alpha=brightness_val, beta=0)

                # Compute day/night
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                avg_v = np.mean(hsv[:, :, 2])
                condition = "Day" if avg_v > self.day_night_thresh.get() else "Night"
                self.metric_condition.set(f"Condition: {condition} (V={avg_v:.1f})")

                # Display
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                max_w, max_h = 1100, 750
                w, h = pil.size
                ratio = min(max_w / w, max_h / h)
                pil = pil.resize(
                    (int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS
                )

                imgtk = ImageTk.PhotoImage(image=pil)
                self.root.after(0, lambda i=imgtk: self._show_image(i))

                cv2.waitKey(33)
        finally:
            cap.release()
            if self.is_previewing.get():
                self.is_previewing.set(False)
            self.status_var.set("Ready")

    def _show_image(self, imgtk):
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
                self.stop_inference()
                return

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb_frame)

                max_w, max_h = 1100, 750
                w, h = pil.size
                ratio = min(max_w / w, max_h / h)
                pil = pil.resize(
                    (int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS
                )

                imgtk = ImageTk.PhotoImage(image=pil)
                self.root.after(0, lambda i=imgtk: self._show_image(i))
                self.video_label.imgtk = imgtk

                cv2.waitKey(33)

            cap.release()
            self.stop_inference()
        except Exception as e:
            print(f"Error in playback: {e}")
            self.stop_inference()

    # ──────────────────────────────────────────────
    #  INFERENCE CONTROL
    # ──────────────────────────────────────────────

    def start_inference(self):
        # Stop preview if running
        self.is_previewing.set(False)

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
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.thread = threading.Thread(
            target=self.inference_loop,
            args=(video_path, models_to_run),
            daemon=True,
        )
        self.thread.start()

    def stop_inference(self):
        self.is_running = False
        self.is_previewing.set(False)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self.status_label.config(background="#ffffff", foreground="#000000")

    # ──────────────────────────────────────────────
    #  INFERENCE LOOP
    # ──────────────────────────────────────────────

    def inference_loop(self, video_path, models_list):
        total_models = len(models_list)
        output_format = self.output_format.get()
        video_stem = Path(video_path).stem  # Original video name without extension

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

                # ── NEW Naming: [video_stem]_output_[model].ext ──
                final_filename = f"{video_stem}_output_{sanitized_name}.{output_format}"
                output_path = INFERENCE_OUTPUT_DIR / final_filename

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

                # Writer (MP4 only)
                if self.save_output.get() and output_format == "mp4":
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                    self.writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                else:
                    self.writer = None

                while self.is_running and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # Simulation
                    brightness_val = self.brightness.get()
                    if brightness_val != 1.0:
                        frame = cv2.convertScaleAbs(frame, alpha=brightness_val, beta=0)

                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    avg_v = np.mean(hsv[:, :, 2])
                    condition = (
                        "Day" if avg_v > self.day_night_thresh.get() else "Night"
                    )
                    self.metric_condition.set(f"Condition: {condition} (V={avg_v:.1f})")

                    # Inference
                    results = model.track(
                        frame,
                        conf=self.conf_threshold.get(),
                        persist=True,
                        verbose=False,
                        device=device,
                    )
                    res = results[0]
                    annotated_frame = res.plot()

                    gun_detected = len(res.boxes) > 0
                    self.metric_detections.set(f"Detections: {len(res.boxes)}")

                    # Save
                    if self.save_output.get():
                        if output_format == "mp4" and self.writer:
                            self.writer.write(annotated_frame)
                        elif output_format == "gif":
                            rgb_for_gif = cv2.cvtColor(
                                annotated_frame, cv2.COLOR_BGR2RGB
                            )
                            gif_frames.append(Image.fromarray(rgb_for_gif))

                    # Display
                    rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)

                    max_w, max_h = 1100, 750
                    w, h = pil.size
                    ratio = min(max_w / w, max_h / h)
                    pil = pil.resize(
                        (int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS
                    )

                    imgtk = ImageTk.PhotoImage(image=pil)
                    self.root.after(
                        0, lambda i=imgtk, d=gun_detected: self.update_video_label(i, d)
                    )

                # Per-model cleanup
                self.cap.release()
                if self.writer:
                    self.writer.release()

                # Save GIF
                if self.save_output.get() and output_format == "gif" and gif_frames:
                    self.status_var.set(f"Saving GIF: {output_path.name}…")
                    try:
                        fps_source = 30
                        duration = int(1000 / fps_source)
                        gif_frames[0].save(
                            output_path,
                            save_all=True,
                            append_images=gif_frames[1:],
                            optimize=False,
                            duration=duration,
                            loop=0,
                        )
                    except Exception as e:
                        print(f"Error saving GIF: {e}")

            except Exception as e:
                print(f"Error in inference loop for {model_name}: {e}")

        self.stop_inference()
        self.root.after(0, self.refresh_results_list)

    # ──────────────────────────────────────────────
    #  UI HELPERS
    # ──────────────────────────────────────────────

    def update_video_label(self, imgtk, gun_detected):
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
