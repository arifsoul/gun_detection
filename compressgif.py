import os
from pathlib import Path
from PIL import Image, ImageSequence


def compress_gif_adaptive(input_path, output_path, target_size_mb=10.0):
    """
    Mengompres GIF secara adaptif hingga mencapai target ukuran tertentu.
    """
    scale = 0.6  # Mulai dengan 60% dimensi asli
    skip_frames = 2  # Ambil 1 dari setiap 2 frame (FPS dipotong setengah)
    colors = 128  # Limit warna

    with Image.open(input_path) as img:
        duration = img.info.get("duration", 100)
        frames = []

        for i, frame in enumerate(ImageSequence.Iterator(img)):
            if i % skip_frames != 0:
                continue

            # Pemrosesan Frame
            new_frame = frame.convert("RGBA")
            new_size = (int(new_frame.width * scale), int(new_frame.height * scale))
            new_frame = new_frame.resize(new_size, Image.LANCZOS)
            new_frame = new_frame.convert("P", palette=Image.ADAPTIVE, colors=colors)
            frames.append(new_frame)

        # Simpan sementara untuk cek ukuran
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration * skip_frames,
            loop=0,
        )


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
        print(f"Memproses: {gif.name}...")
        save_to = output_path / f"{gif.name}"

        try:
            compress_gif_adaptive(gif, save_to, target_mb)
            final_size = os.path.getsize(save_to) / (1024 * 1024)
            print(f"✅ Berhasil: {gif.name} -> {final_size:.2f} MB\n")
        except Exception as e:
            print(f"❌ Gagal memproses {gif.name}: {e}\n")


if __name__ == "__main__":
    # Konfigurasi Path
    BASE_DIR = Path(__file__).parent
    FOLDER_INPUT = BASE_DIR / "gif_ori"
    FOLDER_OUTPUT = BASE_DIR / "gif_compressed"

    process_folder(FOLDER_INPUT, FOLDER_OUTPUT, target_mb=9.5)  # Target 9.5MB agar aman
