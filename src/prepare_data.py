import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm

# Configuration
SEED = 42
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # Train, Val, Test

# Paths
# This script is in src/, so project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Input Datasets
REAL_DATASET_ROOT = (
    PROJECT_ROOT
    / "BS Interview CV Test_ Weapon Object Detection"
    / "real_dataset_KAC_PDW_Blackgun"
    / "dataset"
)
SYN_V2_ROOT = (
    PROJECT_ROOT
    / "BS Interview CV Test_ Weapon Object Detection"
    / "synthetic_dataset_KAC_PDW_Blackgun_v2"
    / "KACPDW"
)
SYN_V3_ROOT = (
    PROJECT_ROOT
    / "BS Interview CV Test_ Weapon Object Detection"
    / "synthetic_dataset_KAC_PDW_Blackgun_v3"
)


def setup_directories():
    """Creates the YOLO directory structure."""
    if DATA_ROOT.exists():
        # We don't necessarily want to wipe everything if we are just updating manifests,
        # but for consistency and to avoid stale files from previous runs with different logic, clean start is safer.
        # However, copying images takes time. If images are already there, we could skip copying?
        # For now, let's keep it clean to ensure exact state.
        shutil.rmtree(DATA_ROOT)

    for split in ["train", "val", "test"]:
        (DATA_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def get_real_data_pairs():
    """Gathers (image, label) pairs from the Real Dataset."""
    img_dir = REAL_DATASET_ROOT / "images"
    lbl_dir = REAL_DATASET_ROOT / "labels"

    pairs = []
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Warning: Real dataset directories not found at {REAL_DATASET_ROOT}")
        return pairs

    # Map stem to files
    img_files = {
        f.stem: f
        for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }
    lbl_files = {
        f.stem: f
        for f in lbl_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    }

    # Intersection
    common = set(img_files.keys()) & set(lbl_files.keys())

    for stem in common:
        pairs.append({"img": img_files[stem], "lbl": lbl_files[stem], "source": "real"})

    print(f"Found {len(pairs)} pairs in Real Dataset")
    return pairs


def get_syn_v2_pairs():
    """Gathers pairs from Synthetic V2."""
    # Structure: KACPDW/Screenshots and KACPDW/Annotations
    img_dir = SYN_V2_ROOT / "Screenshots"
    lbl_dir = SYN_V2_ROOT / "Annotations"

    pairs = []
    if not img_dir.exists():
        print(f"Warning: Syn V2 screenshots not found at {img_dir}")
        return pairs

    if not lbl_dir.exists():
        print(f"Warning: Syn V2 annotations not found at {lbl_dir}. Skipping V2.")
        return pairs

    img_files = {
        f.stem: f
        for f in img_dir.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    }
    lbl_files = {f.stem: f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"}

    common = set(img_files.keys()) & set(lbl_files.keys())

    for stem in common:
        pairs.append(
            {"img": img_files[stem], "lbl": lbl_files[stem], "source": "syn_v2"}
        )

    print(f"Found {len(pairs)} pairs in Syn V2 Dataset")
    return pairs


def get_syn_v3_pairs():
    """Gathers pairs from Synthetic V3 (Dataset_0 and Dataset_1)."""
    pairs = []
    if not SYN_V3_ROOT.exists():
        print(f"Warning: Syn V3 root not found at {SYN_V3_ROOT}")
        return pairs

    for subdir_name in ["Dataset_0", "Dataset_1"]:
        subdir = SYN_V3_ROOT / subdir_name
        img_dir = subdir / "Screenshots"
        lbl_dir = subdir / "Annotations"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_files = {
            f.stem: f
            for f in img_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        }
        lbl_files = {f.stem: f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"}

        common = set(img_files.keys()) & set(lbl_files.keys())

        for stem in common:
            pairs.append(
                {
                    "img": img_files[stem],
                    "lbl": lbl_files[stem],
                    "source": f"syn_v3_{subdir_name}",
                }
            )

    print(f"Found {len(pairs)} pairs in Syn V3 Dataset")
    return pairs


def copy_files(pairs, split):
    """Copies image and label files to the destination split folder."""
    dest_img_dir = DATA_ROOT / "images" / split
    dest_lbl_dir = DATA_ROOT / "labels" / split

    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(pairs, desc=f"Processing {split}"):
        src_img = p["img"]
        src_lbl = p["lbl"]

        # Unique prefix to prevent name collisions between datasets
        prefix = p["source"] + "_"
        dst_name = prefix + src_img.name
        dst_lbl_name = prefix + src_lbl.name

        shutil.copy(src_img, dest_img_dir / dst_name)
        shutil.copy(src_lbl, dest_lbl_dir / dst_lbl_name)


def write_manifest(name, pairs_list, split_name):
    """Writes a text file list of absolute image paths."""
    paths = []
    for p in pairs_list:
        prefix = p["source"] + "_"
        dst_name = prefix + p["img"].name
        abs_path = (DATA_ROOT / "images" / split_name / dst_name).absolute()
        paths.append(str(abs_path))

    with open(DATA_ROOT / f"{name}.txt", "w") as f:
        f.write("\n".join(paths))


def create_yaml(filename, train_txt, val_txt, test_txt):
    """Generates a data.yaml file."""
    base_names = {0: "Black Gun"}

    yaml_content = {
        "path": str(DATA_ROOT.absolute()),
        "train": str(train_txt),
        "val": str(val_txt),
        "test": str(test_txt),
        "names": base_names,
    }
    with open(DATA_ROOT / filename, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Created {filename}")


def main():
    print(f"Project Root: {PROJECT_ROOT}")

    # 1. Gather all data
    real_pairs = get_real_data_pairs()
    syn_v2_pairs = get_syn_v2_pairs()
    syn_v3_pairs = get_syn_v3_pairs()

    all_pairs = real_pairs + syn_v2_pairs + syn_v3_pairs

    if len(all_pairs) == 0:
        print("No data found. Aborting.")
        return

    # 2. Setup Dirs
    setup_directories()

    # 3. Stratified Split (per Source)
    # We split each source independently so any combination maintains structure
    random.seed(SEED)

    def split_list(input_list):
        l = list(input_list)
        random.shuffle(l)
        n = len(l)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        return (
            l[:n_train],
            l[n_train : n_train + n_val],
            l[n_train + n_val :],
        )

    # Split each source
    real_train, real_val, real_test = split_list(real_pairs)
    syn_v2_train, syn_v2_val, syn_v2_test = split_list(syn_v2_pairs)
    syn_v3_train, syn_v3_val, syn_v3_test = split_list(syn_v3_pairs)

    # 4. Copy ALL files to physical directories
    # We combine lists just for the copy loop to avoid code rep
    all_train = real_train + syn_v2_train + syn_v3_train
    all_val = real_val + syn_v2_val + syn_v3_val
    all_test = real_test + syn_v2_test + syn_v3_test

    copy_files(all_train, "train")
    copy_files(all_val, "val")
    copy_files(all_test, "test")

    # 5. Generate Manifests & Configs (The Core Logic)

    # Helper to generate manifest set
    def generate_set(name_suffix, train_list, val_list, test_list):
        t_name = f"{name_suffix}_train.txt"
        v_name = f"{name_suffix}_val.txt"
        tt_name = f"{name_suffix}_test.txt"

        write_manifest(f"{name_suffix}_train", train_list, "train")
        write_manifest(f"{name_suffix}_val", val_list, "val")
        write_manifest(f"{name_suffix}_test", test_list, "test")

        create_yaml(
            f"data_{name_suffix}.yaml",
            (DATA_ROOT / t_name).absolute(),
            (DATA_ROOT / v_name).absolute(),
            (DATA_ROOT / tt_name).absolute(),
        )

    # --- Individual Datasets ---
    generate_set("real", real_train, real_val, real_test)
    generate_set("syn_v2", syn_v2_train, syn_v2_val, syn_v2_test)
    generate_set("syn_v3", syn_v3_train, syn_v3_val, syn_v3_test)

    # --- Combinations ---
    # Real + Syn V2
    generate_set(
        "real_syn_v2",
        real_train + syn_v2_train,
        real_val + syn_v2_val,
        real_test + syn_v2_test,
    )

    # Real + Syn V3
    generate_set(
        "real_syn_v3",
        real_train + syn_v3_train,
        real_val + syn_v3_val,
        real_test + syn_v3_test,
    )

    # All Combined
    generate_set("combined", all_train, all_val, all_test)

    # Standard data.yaml (pointing to combined files implicitly or explicitly)
    # Let's make the default data.yaml point to the combined manifest for "maximal" default behavior
    create_yaml(
        "data.yaml",
        (DATA_ROOT / "combined_train.txt").absolute(),
        (DATA_ROOT / "combined_val.txt").absolute(),
        (DATA_ROOT / "combined_test.txt").absolute(),
    )

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
