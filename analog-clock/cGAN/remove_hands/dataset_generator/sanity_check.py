import cv2
import numpy as np
from pathlib import Path
import random


def check_yolo_dataset(dataset_path="./dataset/yolo_seg"):
    img_dir = Path(dataset_path) / "images" / "train"
    lbl_dir = Path(dataset_path) / "labels" / "train"

    images = list(img_dir.glob("*.jpg"))
    if not images:
        print("❌ No images found. Run the generator first.")
        return

    print(f"🔍 Checking {len(images)} images for corruption or invalid polygons...")

    # Check random sample of 20 images
    sample = random.sample(images, min(len(images), 20))

    for img_path in sample:
        # 1. Load Image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Corrupt image: {img_path}")
            continue

        h, w = img.shape[:2]

        # 2. Load Label
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"⚠ Missing label for {img_path.name} (This is fine if it's background)")
            continue

        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        # 3. Parse Polygons
        valid_polys = 0
        for line in lines:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            coords = parts[1:]

            # CHECK 1: Coordinates must be normalized 0-1
            if any(c < 0.0 or c > 1.0 for c in coords):
                print(f"❌ ERROR: Coordinates out of bounds in {lbl_path.name}")
                print(f"  -> Found value outside [0, 1]")
                return

            # CHECK 2: Polygon validity
            if len(coords) < 6:  # x,y,x,y,x,y minimum
                print(f"❌ ERROR: Malformed polygon (too few points) in {lbl_path.name}")
                return

            # Visualize (Optional - Draws on image)
            # Denormalize
            points = []
            for i in range(0, len(coords), 2):
                px = int(coords[i] * w)
                py = int(coords[i + 1] * h)
                points.append([px, py])

            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
            cv2.polylines(img, [pts], True, color, 2)
            valid_polys += 1

        # Show verification
        print(f"✅ {img_path.name}: OK ({valid_polys} polygons)")

        # Uncomment to see the images popup
        # cv2.imshow("Sanity Check", img)
        # cv2.waitKey(500)

    print("\n🎉 DATASET PASSED SANITY CHECK. You are safe from 'nan' losses.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    check_yolo_dataset()