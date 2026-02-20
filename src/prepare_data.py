import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

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
        shutil.rmtree(DATA_ROOT)

    for split in ["train", "val", "test"]:
        (DATA_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def analyze_dataset(
    dataset_name, label_type, img_dir, lbl_dir, extensions=[".jpg", ".jpeg", ".png"]
):
    """
    Scans image and label directories and returns (pairs, stats_dict).
    Does NOT print to terminal.
    """
    stats = {
        "Dataset": dataset_name,
        "Label Type": label_type,
        "Images": 0,
        "Labels": 0,
        "Pairs": 0,
        "Missing Labels": 0,
        "Extra Labels": 0,
    }

    if not img_dir.exists():
        # If image dir doesn't exist, we can't do anything
        return [], stats

    # Gather files
    img_files = {
        f.stem: f
        for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    }

    lbl_files = {}
    if lbl_dir.exists():
        lbl_files = {
            f.stem: f
            for f in lbl_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        }

    n_img = len(img_files)
    n_lbl = len(lbl_files)

    # Intersection
    common = set(img_files.keys()) & set(lbl_files.keys())
    n_pairs = len(common)

    # Missing
    missing_labels = set(img_files.keys()) - set(lbl_files.keys())
    extra_labels = set(lbl_files.keys()) - set(img_files.keys())

    # Populate Stats
    stats["Images"] = n_img
    stats["Labels"] = n_lbl
    stats["Pairs"] = n_pairs
    stats["Missing Labels"] = len(missing_labels)
    stats["Extra Labels"] = len(extra_labels)

    pairs = []
    for stem in common:
        pairs.append({"img": img_files[stem], "lbl": lbl_files[stem]})

    return pairs, stats


def get_real_data_pairs():
    """Gathers (image, label) pairs from the Real Dataset."""
    stats_list = []

    # 1. Analyze with OLD labels
    _, stats_old = analyze_dataset(
        "Real Dataset",
        "Original",
        REAL_DATASET_ROOT / "images",
        REAL_DATASET_ROOT / "labels",
    )
    stats_list.append(stats_old)

    # 2. Analyze with FIXED labels (for actual use)
    fixed_pairs, stats_fix = analyze_dataset(
        "Real Dataset",
        "Fixed",
        REAL_DATASET_ROOT / "images",
        REAL_DATASET_ROOT / "labels_fix",
    )
    stats_list.append(stats_fix)

    return [{**p, "source": "real"} for p in fixed_pairs], stats_list


def get_syn_v2_pairs():
    """Gathers pairs from Synthetic V2."""
    stats_list = []

    # 1. Analyze with OLD labels
    _, stats_old = analyze_dataset(
        "Syn V2", "Original", SYN_V2_ROOT / "Screenshots", SYN_V2_ROOT / "Annotations"
    )
    stats_list.append(stats_old)

    # 2. Analyze with FIXED labels
    fixed_pairs, stats_fix = analyze_dataset(
        "Syn V2", "Fixed", SYN_V2_ROOT / "Screenshots", SYN_V2_ROOT / "labels_fix"
    )
    stats_list.append(stats_fix)

    return [{**p, "source": "syn_v2"} for p in fixed_pairs], stats_list


def get_syn_v3_pairs():
    """Gathers pairs from Synthetic V3 (Dataset_0 and Dataset_1)."""
    pairs = []
    stats_list = []

    if not SYN_V3_ROOT.exists():
        return pairs, stats_list

    for subdir_name in ["Dataset_0", "Dataset_1"]:
        subdir = SYN_V3_ROOT / subdir_name

        # 1. Old Labels (Annotations)
        sub_pairs_old, stats_old = analyze_dataset(
            f"Syn V3 {subdir_name}",
            "Original",
            subdir / "Screenshots",
            subdir / "Annotations",
        )
        stats_list.append(stats_old)

        # 2. Fixed Labels (labels_fix) - For Syn V3, 'labels' ARE the fixed/correct labels.
        # User confirmed Syn V3 has no 'labels_fix' folder, only 'labels'.
        # So we treat 'labels' as the 'Fixed' version for consistency in data prep.
        sub_pairs_fix, stats_fix = analyze_dataset(
            f"Syn V3 {subdir_name}",
            "Fixed",  # We call it Fixed so it gets picked up by the logic below
            subdir / "Screenshots",
            subdir / "Annotations",  # Use 'labels' instead of 'labels_fix'
        )
        stats_list.append(stats_fix)

        # Decide which to use for training.
        # For Syn V3, we trust 'labels' (mapped to 'Fixed' here).
        if stats_fix["Pairs"] > 0:
            for p in sub_pairs_fix:
                pairs.append({**p, "source": f"syn_v3_{subdir_name}"})
        else:
            # If even 'labels' are missing/unmatched, we have a problem with V3 data
            print(
                f"WARNING: No valid pairs found for Syn V3 {subdir_name} in 'Annotations' folder."
            )

    return pairs, stats_list


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


def plot_dataset_inventory(stats_list):
    """
    Plots a comparison of Old vs Fixed labels for all datasets.
    Includes a table with detailed counts.
    """
    if not stats_list:
        return

    df = pd.DataFrame(stats_list)

    # Setup plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1]}
    )
    sns.set_theme(style="whitegrid")

    # 1. Bar Chart (Pairs)
    sns.barplot(
        data=df, x="Dataset", y="Pairs", hue="Label Type", palette="viridis", ax=ax1
    )
    ax1.set_title(
        "Dataset Inventory: Matched Image-Label Pairs (Original vs Fixed)", fontsize=16
    )
    ax1.set_ylabel("Count (Pairs)", fontsize=12)
    ax1.legend(title="Label Type")

    # Add values on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%d")

    # 2. Data Table
    # Filter columns for table
    table_cols = [
        "Dataset",
        "Label Type",
        "Images",
        "Labels",
        "Pairs",
        "Missing Labels",
        "Extra Labels",
    ]
    table_data = df[table_cols].values.tolist()
    column_headers = table_cols

    ax2.axis("tight")
    ax2.axis("off")
    table = ax2.table(
        cellText=table_data, colLabels=column_headers, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax2.set_title("Detailed Statistics", fontsize=14, pad=20)

    plt.tight_layout()
    output_path = DATA_ROOT / "dataset_inventory_comparison.png"
    plt.savefig(output_path)
    print(f"[Visualization] Inventory plot saved to: {output_path}")


def plot_splits(stats_list):
    """Generates a Seaborn bar plot to visualize dataset splits."""
    if not stats_list:
        return

    df = pd.DataFrame(stats_list)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Create barplot
    chart = sns.barplot(data=df, x="Dataset", y="Count", hue="Split", palette="viridis")

    plt.title("Dataset Distribution by Split (Train/Val/Test)", fontsize=16)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Split")
    plt.tight_layout()

    # Add values on top of bars
    for container in chart.containers:
        chart.bar_label(container, fmt="%d")

    output_path = DATA_ROOT / "data_split_distribution.png"
    plt.savefig(output_path)
    print(f"[Visualization] Split distribution plot saved to: {output_path}")


def main():
    print(f"Project Root: {PROJECT_ROOT}")

    # 1. Gather all data
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS (Graph Generation)")
    print("-" * 70)

    # Collect stats instead of printing
    all_stats = []

    real_pairs, real_stats = get_real_data_pairs()
    all_stats.extend(real_stats)

    syn_v2_pairs, syn_v2_stats = get_syn_v2_pairs()
    all_stats.extend(syn_v2_stats)

    syn_v3_pairs, syn_v3_stats = get_syn_v3_pairs()
    all_stats.extend(syn_v3_stats)

    # Plot Inventory (Old vs Fix)
    setup_directories()  # Ensure DATA_ROOT exists for saving plots
    plot_dataset_inventory(all_stats)

    print("=" * 70 + "\n")

    all_pairs = real_pairs + syn_v2_pairs + syn_v3_pairs

    if len(all_pairs) == 0:
        print(
            "No valid pairs found for training (labels_fix might be empty). Aborting."
        )
        return

    # 3. Stratified Split (per Source)
    random.seed(SEED)

    def split_list(input_list):
        input_list_copy = list(input_list)
        random.shuffle(input_list_copy)
        n = len(input_list_copy)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        return (
            input_list_copy[:n_train],
            input_list_copy[n_train : n_train + n_val],
            input_list_copy[n_train + n_val :],
        )

    # Split each source
    real_train, real_val, real_test = split_list(real_pairs)
    syn_v2_train, syn_v2_val, syn_v2_test = split_list(syn_v2_pairs)
    syn_v3_train, syn_v3_val, syn_v3_test = split_list(syn_v3_pairs)

    # Calculate Combined Split
    all_train = real_train + syn_v2_train + syn_v3_train
    all_val = real_val + syn_v2_val + syn_v3_val
    all_test = real_test + syn_v2_test + syn_v3_test

    # Prepare Stats for Visualization (Split)
    split_stats = []

    def add_stats(name, t, v, tt):
        split_stats.append({"Dataset": name, "Split": "Train", "Count": len(t)})
        split_stats.append({"Dataset": name, "Split": "Val", "Count": len(v)})
        split_stats.append({"Dataset": name, "Split": "Test", "Count": len(tt)})

    add_stats("Real Data", real_train, real_val, real_test)
    add_stats("Synthetic V2", syn_v2_train, syn_v2_val, syn_v2_test)
    add_stats("Synthetic V3", syn_v3_train, syn_v3_val, syn_v3_test)
    add_stats(
        "Real + Syn V2",
        real_train + syn_v2_train,
        real_val + syn_v2_val,
        real_test + syn_v2_test,
    )
    add_stats(
        "Real + Syn V3",
        real_train + syn_v3_train,
        real_val + syn_v3_val,
        real_test + syn_v3_test,
    )
    add_stats("Combined", all_train, all_val, all_test)

    # Generate Visualization (After split and summary)
    plot_splits(split_stats)

    # 4. Copy ALL files
    copy_files(all_train, "train")
    copy_files(all_val, "val")
    copy_files(all_test, "test")

    # 5. Generate Manifests & Configs (The Core Logic)
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
    generate_set(
        "real_syn_v2",
        real_train + syn_v2_train,
        real_val + syn_v2_val,
        real_test + syn_v2_test,
    )
    generate_set(
        "real_syn_v3",
        real_train + syn_v3_train,
        real_val + syn_v3_val,
        real_test + syn_v3_test,
    )

    # All Combined
    generate_set("combined", all_train, all_val, all_test)

    create_yaml(
        "data.yaml",
        (DATA_ROOT / "combined_train.txt").absolute(),
        (DATA_ROOT / "combined_val.txt").absolute(),
        (DATA_ROOT / "combined_test.txt").absolute(),
    )

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
