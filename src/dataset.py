import os
import torch
import torch.utils.data
from PIL import Image


class GunDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # Assume directory structure:
        # root/
        #   images/
        #   labels/  <-- YOLO format txt files

        self.img_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")

        # Robust matching of images and labels
        if not os.path.exists(self.img_dir) or not os.path.exists(self.label_dir):
            self.imgs = []
            self.lbls = []
        else:
            all_imgs = sorted(os.listdir(self.img_dir))
            all_lbls = sorted(os.listdir(self.label_dir))

            # Map basename to filename
            img_map = {
                os.path.splitext(f)[0]: f
                for f in all_imgs
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            }
            lbl_map = {
                os.path.splitext(f)[0]: f
                for f in all_lbls
                if f.lower().endswith(".txt")
            }

            # Intersection
            common_names = sorted(list(set(img_map.keys()) & set(lbl_map.keys())))

            self.imgs = [img_map[n] for n in common_names]
            self.lbls = [lbl_map[n] for n in common_names]

            if len(self.imgs) != len(self.lbls):
                print(
                    f"Warning: Root {root} has mismatch: Imgs={len(all_imgs)}, Labels={len(all_lbls)}, Matched={len(self.imgs)}"
                )

        # Label mapping - tailored for Gun Detection (Single class usually)
        # YOLO class 0 is typically the target.
        self.label_map = {0: "Black Gun"}

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        lbl_path = os.path.join(self.label_dir, self.lbls[idx])

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        # Parse YOLO TXT
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx = float(parts[1])
                        cy = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])

                        # Convert YOLO format (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
                        # cx, cy, bw, bh are normalized

                        xmin = (cx - bw / 2) * w
                        ymin = (cy - bh / 2) * h
                        xmax = (cx + bw / 2) * w
                        ymax = (cy + bh / 2) * h

                        boxes.append([xmin, ymin, xmax, ymax])
                        # In Faster R-CNN, labels usually start at 1 for object, 0 for background.
                        # But here we are just loading data. Let's keep raw class ID + 1 if we were using R-CNN,
                        # but for general YOLO purpose, keeping 0 is fine.
                        # However, since this class inherits torch.utils.data.Dataset and follows
                        # torchvision conventions often used with R-CNN in the old code:
                        # old code: "with_mask": 1.
                        # We will map 0 (YOLO) -> 1 (TorchVision convention) to simulate old behavior refactored.
                        labels.append(cls_id + 1)

        # Handle cases with no valid boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # Fix: torchvision transforms only take image
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
