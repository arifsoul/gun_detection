import os
from pathlib import Path
from PIL import Image, ImageSequence


def compress_gif_adaptive(input_path, output_path, target_size_mb=10.0):
    """
    Mengompres GIF secara adaptif hingga mencapai target ukuran tertentu.
    """
    # Skip if already small enough
    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    if orig_size <= target_size_mb:
        return False

    scale = 0.5  # Penurunan dimensi lebih signifikan untuk kompresi cepat
    skip_frames = 2  # Ambil setiap frame genap
    colors = 96  # Limit warna lebih agresif

    print(f"Opening {input_path.name}...")
    with Image.open(input_path) as img:
        duration = img.info.get("duration", 100)
        all_frames = list(ImageSequence.Iterator(img))
        total_frames = len(all_frames)

        frames = []
        print(f"Processing {total_frames} frames...")
        for i, frame in enumerate(all_frames):
            if i % skip_frames != 0:
                continue

            # Update progress
            if (i // skip_frames) % 10 == 0:
                print(f"  > Frame {i}/{total_frames} processed...", end="\r")

            # Pemrosesan Frame - Convert to "RGB" instead of "RGBA" to save space/speed
            new_frame = frame.convert("RGB")
            new_size = (int(new_frame.width * scale), int(new_frame.height * scale))
            new_frame = new_frame.resize(new_size, Image.NEAREST)  # Fast resize
            new_frame = new_frame.convert("P", palette=Image.ADAPTIVE, colors=colors)
            frames.append(new_frame)

        print(f"\nSaving compressed GIF to {output_path.name}...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration * skip_frames,
            loop=0,
        )
    return True


def process_folder(input_folder, output_folder, target_mb=10.0):
    """
    Membaca semua file .gif dalam folder dan memprosesnya.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Buat folder output jika belum ada
    output_path.mkdir(parents=True, exist_ok=True)

    # Ambil semua file .gif (case-insensitive)
    gif_files = list(input_path.glob("*.gif")) + list(input_path.glob("*.GIF"))

    if not gif_files:
        print(f"Tidak ada file GIF yang ditemukan di {input_folder}")
        return

    print(f"Ditemukan {len(gif_files)} file. Memulai kompresi...\n")

    for gif in gif_files:
        # Jangan proses file yang sudah di-kompres (punya suffix _compressed)
        if "_compressed.gif" in gif.name.lower():
            continue

        print(f"Memproses: {gif.name}...")
        save_to = output_path / f"{gif.stem}_compressed.gif"

        try:
            success = compress_gif_adaptive(gif, save_to, target_mb)
            if success:
                final_size = os.path.getsize(save_to) / (1024 * 1024)
                print(f"✅ Berhasil: {gif.name} -> {final_size:.2f} MB\n")
            else:
                print(f"ℹ️ Dialihkan: {gif.name} tidak memerlukan kompresi tambahan.\n")
        except Exception as e:
            print(f"❌ Gagal memproses {gif.name}: {e}\n")


if __name__ == "__main__":
    # Konfigurasi Path Default ke folder inference
    PROJECT_ROOT = Path(__file__).parent
    FOLDER_INFERENCE = PROJECT_ROOT / "runs" / "inference"

    if not FOLDER_INFERENCE.exists():
        print(f"Folder tidak ditemukan: {FOLDER_INFERENCE}")
    else:
        # Gunakan target_mb 10MB sebagai default
        process_folder(FOLDER_INFERENCE, FOLDER_INFERENCE, target_mb=10.0)
