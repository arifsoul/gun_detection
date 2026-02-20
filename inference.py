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
DATA_DIR = PROJECT_ROOT / "docs"  # Updated based on user's successful run path
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
INFERENCE_OUTPUT_DIR = PROJECT_ROOT / "runs" / "inference"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Ensure output directory exists
INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GunDetectionApp(TKMT.ThemedTKinterFrame):
    def __init__(self):
        super().__init__(
            "Gun Detection System", "sun-valley", "dark"
        )  # specific title, theme, mode
        self.root.geometry("1400x900")

        # Variables
        self.video_source_path = tk.StringVar(value="")
        self.model_path = tk.StringVar(value="")
        self.conf_threshold = tk.DoubleVar(value=0.40)
        self.brightness = tk.DoubleVar(value=1.0)
        self.day_night_thresh = tk.IntVar(value=100)
        self.save_output = tk.BooleanVar(value=False)
        self.output_filename = tk.StringVar(value="output.mp4")

        # Internal state
        self.is_running = False
        self.thread = None
        self.cap = None
        self.writer = None
        self.current_image = None
        self.model_map = {}  # Map display name -> path

        self.setup_ui()

    def setup_ui(self):
        # Main Layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Sidebar ---
        sidebar = ttk.Frame(main_paned, width=300, padding=10)
        main_paned.add(sidebar, weight=1)

        # Title
        ttk.Label(sidebar, text="Configuration", font=("Segoe UI", 20, "bold")).pack(
            pady=(0, 20)
        )

        # 1. Model Selection
        model_frame = ttk.LabelFrame(sidebar, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5)

        self.load_model_options()
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_path,
            values=list(self.model_map.keys()),
        )
        if self.model_map:
            self.model_combo.current(0)
        self.model_combo.pack(fill=tk.X)

        # 2. Input Source
        input_frame = ttk.LabelFrame(sidebar, text="Input Video", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Entry(input_frame, textvariable=self.video_source_path).pack(
            fill=tk.X, pady=5
        )
        ttk.Button(input_frame, text="Browse...", command=self.browse_file).pack(
            fill=tk.X
        )

        # 3. Confidence
        conf_frame = ttk.LabelFrame(sidebar, text="Confidence Threshold", padding=10)
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

        # 4. Simulation
        sim_frame = ttk.LabelFrame(sidebar, text="Environment Simulation", padding=10)
        sim_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sim_frame, text="Brightness").pack(anchor=tk.W)
        self.brit_label = ttk.Label(sim_frame, text=f"{self.brightness.get():.2f}")
        self.brit_label.pack()
        ttk.Scale(
            sim_frame,
            variable=self.brightness,
            from_=0.1,
            to=2.0,
            command=lambda v: self.brit_label.config(text=f"{float(v):.2f}"),
        ).pack(fill=tk.X)

        ttk.Label(sim_frame, text="Day/Night Threshold").pack(anchor=tk.W, pady=(10, 0))
        self.dn_label = ttk.Label(sim_frame, text=f"{self.day_night_thresh.get()}")
        self.dn_label.pack()
        ttk.Scale(
            sim_frame,
            variable=self.day_night_thresh,
            from_=0,
            to=255,
            command=lambda v: self.dn_label.config(text=f"{int(float(v))}"),
        ).pack(fill=tk.X)

        # 5. Output
        out_frame = ttk.LabelFrame(sidebar, text="Output", padding=10)
        out_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            out_frame, text="Save Output Video", variable=self.save_output
        ).pack(anchor=tk.W)
        ttk.Entry(out_frame, textvariable=self.output_filename).pack(fill=tk.X, pady=5)

        # Buttons
        btn_frame = ttk.Frame(sidebar, padding=20)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(
            btn_frame, text="START INFERENCE", command=self.start_inference
        )
        # Note: TKMT buttons might need specific styling via style argument if standard ttk style isn't enough,
        # but usually ThemedTK handles it.
        self.start_btn.pack(fill=tk.X, pady=5)

        self.stop_btn = ttk.Button(
            btn_frame, text="STOP", command=self.stop_inference, state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=5)

        # --- Main Content ---
        content_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(content_frame, weight=4)

        # Status Bar
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

        # Metrics Bar
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

        # Video Canvas
        self.video_container = ttk.Frame(content_frame)  # Container to center
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(self.video_container, background="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def load_model_options(self):
        self.model_map = {"Default (yolo26n)": "yolo26n.pt"}

        if MLRUNS_DIR.exists():
            for args_path in MLRUNS_DIR.rglob("artifacts/training_results/args.yaml"):
                try:
                    artifacts_dir = args_path.parent.parent
                    weights_path = artifacts_dir / "weights" / "best.pt"

                    if weights_path.exists():
                        # Parse yaml for name
                        with open(args_path, "r") as f:
                            args = yaml.safe_load(f)
                            model_name = args.get("name", "Unknown")

                        # Use parent folder name (run_id) for uniqueness
                        run_id = artifacts_dir.parent.name
                        display_name = f"{model_name} ({run_id[:8]})"
                        self.model_map[display_name] = str(weights_path)
                except Exception:
                    continue

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if filename:
            self.video_source_path.set(filename)

    def start_inference(self):
        # Validate Inputs
        video_path = self.video_source_path.get()
        if not video_path:
            # Try default check
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

        # Get actual model path from map
        selected_name = self.model_path.get()
        actual_model_path = self.model_map.get(selected_name, "yolo26n.pt")

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Run in thread
        self.thread = threading.Thread(
            target=self.inference_loop,
            args=(video_path, actual_model_path),
            daemon=True,
        )
        self.thread.start()

    def stop_inference(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self.status_label.config(background="#ffffff", foreground="#000000")

    def inference_loop(self, video_path, model_path):
        try:
            # Check for CUDA
            if torch.cuda.is_available():
                device = 0
                device_name = "GPU"
            else:
                device = "cpu"
                device_name = "CPU"

            self.metric_device.set(f"Device: {device_name}")

            # Load Model
            model = YOLO(model_path)

            # Open Video
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video source.")
                self.stop_inference()
                return

            # Writer Setup
            if self.save_output.get():
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                output_path = INFERENCE_OUTPUT_DIR / self.output_filename.get()
                self.writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # --- Simulation ---
                brightness_val = self.brightness.get()
                if brightness_val != 1.0:
                    frame = cv2.convertScaleAbs(frame, alpha=brightness_val, beta=0)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                avg_v = np.mean(hsv[:, :, 2])
                condition = "Day" if avg_v > self.day_night_thresh.get() else "Night"
                self.metric_condition.set(f"Condition: {condition} (V={avg_v:.1f})")

                # --- Inference ---
                results = model.track(
                    frame,
                    conf=self.conf_threshold.get(),
                    persist=True,
                    verbose=False,
                    device=device,
                )
                res = results[0]
                annotated_frame = res.plot()

                # Metrics Update
                gun_detected = len(res.boxes) > 0
                self.metric_detections.set(f"Detections: {len(res.boxes)}")

                if gun_detected:
                    self.status_var.set("⚠️ GUN DETECTED!")
                    # Configure label colors safely on main thread via update function or polling
                    # For simplicity, we just set var here. Color change needs safer handling.
                    # We'll pass status to update_video_label
                else:
                    self.status_var.set("✅ SAFE")

                # --- Video Writer ---
                if self.writer:
                    self.writer.write(annotated_frame)

                # --- Display ---
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Simple resize logic
                max_w = 1100
                max_h = 750
                orig_w, orig_h = pil_image.size
                ratio = min(max_w / orig_w, max_h / orig_h)
                new_size = (int(orig_w * ratio), int(orig_h * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                imgtk = ImageTk.PhotoImage(image=pil_image)

                # Schedule UI update
                # Pass gun_detected so we can update color on main thread
                self.root.after(
                    0, lambda i=imgtk, d=gun_detected: self.update_video_label(i, d)
                )

                # Small sleep to yield CPU if needed, though waitKey logic isn't here since it's a loop
                # With tk we rely on loop speed.
                # To limit FPS we could calculate delay, but full speed is usually fine on GPU

            self.cap.release()
            if self.writer:
                self.writer.release()
            self.stop_inference()

        except Exception as e:
            print(f"Error in inference loop: {e}")
            self.stop_inference()

    def update_video_label(self, imgtk, gun_detected):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if gun_detected:
            self.status_label.configure(background="#ff3333", foreground="#ffffff")
        else:
            self.status_label.configure(background="#ccffcc", foreground="#006600")


if __name__ == "__main__":
    app = GunDetectionApp()
    app.root.mainloop()
