# Robust Clock Dataset Generator - Setup Instructions

Complete guide for setting up the dataset generator and integrating with the reference repository.

---

## 📋 Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Asset Preparation](#asset-preparation)
5. [Reference Repository Integration](#reference-repository-integration)
6. [Configuration](#configuration)
7. [Running the Generator](#running-the-generator)
8. [Troubleshooting](#troubleshooting)

---

## 📁 Project Structure

After setup, your project should look like this:

```
clock-dataset-generator/
├── analog_sketch_creator.py     # YOUR sketch creator file
├── config.py                    # Configuration settings
├── augmentations.py             # Augmentation pipelines
├── clock_renderer.py            # Hand rendering logic
├── sketch_generator.py          # Sketch generation wrapper
├── asset_loader.py              # Asset loading utilities
├── dataset_generator.py         # Main generator class
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
│
├── data/                        # Input assets
│   ├── textures-dataset/        # Kaggle texture backgrounds
│   ├── clean_clocks/            # Inpainted clock faces
│   └── clock_hands/             # Hand PNG assets (hour + minute only)
│       ├── style_1/
│       │   ├── hour.png
│       │   └── minute.png
│       ├── style_2/
│       │   └── ...
│       └── ...
│
└── dataset/                     # Generated output
    ├── train/
    │   ├── source/
    │   ├── target/
    │   └── sketch/
    ├── val/
    │   ├── source/
    │   ├── target/
    │   └── sketch/
    └── metadata.csv
```

---

## 🔧 Prerequisites

- **Python 3.8+** (recommended: 3.10)
- **pip** package manager
- **Git** (for cloning reference repository)
- **~10GB disk space** (for datasets and generated images)

---

## 📦 Installation Steps

### Step 1: Create Project Directory

```bash
mkdir clock-dataset-generator
cd clock-dataset-generator
```

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Create requirements.txt

Create a file named `requirements.txt` with the following content:

```txt
# Core dependencies
numpy>=1.21.0
opencv-python>=4.6.0
opencv-contrib-python>=4.6.0
Pillow>=9.0.0
pandas>=1.3.0

# Augmentations
albumentations>=1.3.0

# Image processing extras
scikit-image>=0.19.0
imageio>=2.9.0

# Optional but recommended
tqdm>=4.65.0
matplotlib>=3.5.0
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import cv2; import albumentations; import numpy; print('✓ All dependencies installed successfully!')"
```

---

## 🎨 Asset Preparation

### 1. Download Kaggle Texture Dataset

**Option A: Using Kaggle CLI (Recommended)**

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (get from https://www.kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/kaggle.json

# Download dataset
mkdir -p data
cd data
kaggle datasets download -d roustoumabdelmoula/textures-dataset
unzip textures-dataset.zip -d textures-dataset
cd ..
```

**Option B: Manual Download**

1. Visit: https://www.kaggle.com/datasets/roustoumabdelmoula/textures-dataset
2. Click "Download"
3. Extract to `data/textures-dataset/`

### 2. Prepare Clean Clock Faces

Clean clock faces are clock images with hands removed (inpainted).

**Creating Clean Clocks:**

```python
# Example script to inpaint clock hands (basic approach)
import cv2
import numpy as np

def create_clean_clock(input_path, output_path):
    img = cv2.imread(input_path)
    
    # Create mask for clock center (where hands are)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.4)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Inpaint
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, result)

# Use on your clock images
create_clean_clock('clock_with_hands.jpg', 'data/clean_clocks/clock_001.png')
```

**Alternative:** Use any clock face images or create synthetic backgrounds.

### 3. Prepare Hand Assets

Hand assets should be transparent PNGs of clock hands. **Only hour and minute hands needed - no second hand.**

**Directory Structure (Option 1 - Recommended):**

```
data/clock_hands/
├── classic/
│   ├── hour.png      ← Required
│   └── minute.png    ← Required
├── modern/
│   ├── hour.png
│   └── minute.png
└── ornate/
    ├── hour.png
    └── minute.png
```

**Directory Structure (Option 2 - Flat):**

```
data/clock_hands/
├── classic_hour.png
├── classic_minute.png
├── modern_hour.png
├── modern_minute.png
└── ...
```

**Creating Hand Assets:**

You can:
- Extract from existing clock images using image editing software
- Create synthetic hands programmatically
- Use simple shapes (rectangles, triangles) as hands

**Example: Create Simple Hand Assets**

```python
import numpy as np
import cv2
from pathlib import Path

def create_simple_hand(output_path, length_ratio=0.7, width=8, is_hour=False):
    """Create a simple rectangular clock hand"""
    size = 256
    img = np.zeros((size, size, 4), dtype=np.uint8)  # RGBA
    
    center = size // 2
    hand_length = int(size * length_ratio / 2)
    
    if is_hour:
        width = 10  # Hour hand is thicker
        hand_length = int(size * 0.45 / 2)  # Shorter
    
    # Draw vertical hand (pointing up)
    cv2.rectangle(
        img,
        (center - width//2, center - hand_length),
        (center + width//2, center + 5),  # Slight extension below center
        (0, 0, 0, 255),  # Black with full opacity
        -1
    )
    
    # Optional: Add pointed tip
    pts = np.array([
        [center, center - hand_length - 10],
        [center - width, center - hand_length],
        [center + width, center - hand_length]
    ], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0, 255))
    
    cv2.imwrite(str(output_path), img)
    print(f"Created: {output_path}")

# Create a simple hand style
Path("data/clock_hands/simple").mkdir(parents=True, exist_ok=True)
create_simple_hand("data/clock_hands/simple/hour.png", is_hour=True)
create_simple_hand("data/clock_hands/simple/minute.png", is_hour=False)
```

---

## 🔗 Reference Repository Integration

### Step 1: Clone Reference Repository

```bash
git clone https://github.com/VictorSuarezVara/Reading-analog-clocks-with-neural-networks.git
```

### Step 2: Verify Required Files

Check that these files exist:

```bash
ls Reading-analog-clocks-with-neural-networks/Dataset\ of\ Clocks\ Generators/
```

You should see:
- `analog_sketch_creator.py`
- `auxiliaryFunctions.py`

### Step 3: Test Integration

Create a test script `test_integration.py`:

```python
import sys
from pathlib import Path

# Add reference repo to path
sys.path.insert(0, str(Path("Reading-analog-clocks-with-neural-networks/Dataset of Clocks Generators")))

try:
    from analog_sketch_creator import draw_analog_clock
    print("✓ Successfully imported draw_analog_clock")
    
    # Test it
    sketch = draw_analog_clock(hh=3, mm=15, return_array=True)
    print(f"✓ Generated sketch with shape: {sketch.shape}")
    print("✓ Integration successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("The generator will use fallback sketch creation.")
```

Run it:

```bash
python test_integration.py
```

### Step 4: Update config.py

Edit `config.py` to point to your reference repository:

```python
# In config.py
REFERENCE_REPO_PATH = "./Reading-analog-clocks-with-neural-networks"
```

---

## ⚙️ Configuration

### Basic Configuration

Edit `config.py` to customize your dataset generation:

```python
class Config:
    # === PATHS ===
    BACKGROUNDS_DIR = "./data/textures-dataset"
    CLEAN_CLOCKS_DIR = "./data/clean_clocks"
    HANDS_DIR = "./data/clock_hands"
    OUTPUT_DIR = "./dataset"
    REFERENCE_REPO_PATH = "./Reading-analog-clocks-with-neural-networks"
    
    # === DATASET PARAMETERS ===
    N_SAMPLES = 20000          # Total samples to generate
    TRAIN_SPLIT = 0.9          # 90% train, 10% validation
    IMAGE_SIZE = 256           # Output image size (256x256)
    USE_TEXTURES_AS_BG = True  # Use textures vs clean clocks
    
    # === AUGMENTATION PARAMETERS ===
    # Adjust these to control augmentation intensity
    PERSPECTIVE_SCALE = (0.05, 0.1)
    ROTATION_RANGE = (-15, 15)
    GAUSSIAN_NOISE_VAR = (10.0, 50.0)
    # ... etc
```

### Advanced Configuration

**For lighter augmentations:**

```python
GAUSSIAN_NOISE_VAR = (5.0, 20.0)  # Less noise
ROTATION_RANGE = (-5, 5)          # Less rotation
GLARE_PROBABILITY = 0.3           # Less glare
```

**For heavier augmentations:**

```python
GAUSSIAN_NOISE_VAR = (20.0, 80.0) # More noise
ROTATION_RANGE = (-25, 25)        # More rotation
GLARE_PROBABILITY = 0.7           # More glare
```

---

## 🚀 Running the Generator

### Quick Start

```bash
python main.py
```

This will:
1. Load all assets
2. Validate configuration
3. Generate 20,000 samples (18,000 train, 2,000 val)
4. Save to `./dataset/`

### Custom Generation

Create a custom script `generate_custom.py`:

```python
from dataset_generator import ClockDatasetGenerator

# Create generator with custom settings
generator = ClockDatasetGenerator(
    backgrounds_dir="./data/textures-dataset",
    clean_clocks_dir="./data/clean_clocks",
    hands_dir="./data/clock_hands",
    output_dir="./my_dataset",
    image_size=512,  # Higher resolution
    use_textures_as_bg=True
)

# Generate smaller test dataset
generator.generate_dataset(
    n_samples=1000,
    train_split=0.8
)
```

Run it:

```bash
python generate_custom.py
```

### Generate Single Sample (Testing)

```python
from dataset_generator import ClockDatasetGenerator

generator = ClockDatasetGenerator(
    backgrounds_dir="./data/textures-dataset",
    clean_clocks_dir="./data/clean_clocks",
    hands_dir="./data/clock_hands",
    output_dir="./test_output"
)

# Generate just one sample
metadata = generator.generate_sample(idx=0, split='train')
print(f"Generated sample: {metadata}")
```

### Monitor Progress

The generator provides progress updates:

```
============================================================
DATASET GENERATION
============================================================
Total samples: 20000
Training: 18000
Validation: 2000
============================================================

Generating 18000 training samples...
  Progress: 100/18000
  Progress: 200/18000
  ...
```

---

## 🔍 Verifying Output

### Check Generated Files

```bash
# Count generated files
ls dataset/train/source/ | wc -l    # Should match n_train
ls dataset/train/target/ | wc -l
ls dataset/train/sketch/ | wc -l

ls dataset/val/source/ | wc -l      # Should match n_val
```

### Visualize Samples

Create `visualize.py`:

```python
import cv2
import numpy as np
from pathlib import Path

def visualize_triplet(idx=0, split='train'):
    dataset_dir = Path('./dataset')
    
    # Load triplet
    source = cv2.imread(str(dataset_dir / split / 'source' / f'{idx:06d}.png'))
    target = cv2.imread(str(dataset_dir / split / 'target' / f'{idx:06d}.png'))
    sketch = cv2.imread(str(dataset_dir / split / 'sketch' / f'{idx:06d}.png'))
    
    # Concatenate horizontally
    combined = np.hstack([source, target, sketch])
    
    # Display
    cv2.imshow(f'Sample {idx} - Source | Target | Sketch', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# View first 5 samples
for i in range(5):
    visualize_triplet(i)
```

### Inspect Metadata

```python
import pandas as pd

df = pd.read_csv('./dataset/metadata.csv')
print(df.head(10))
print(f"\nDataset statistics:")
print(df.describe())
```

---

## 🐛 Troubleshooting

### Issue: "No backgrounds or clock faces found"

**Solution:**
- Verify paths in `config.py`
- Check that directories exist and contain images
- Ensure image extensions are `.jpg`, `.jpeg`, or `.png`

```bash
# Check directories
ls -la data/textures-dataset/
ls -la data/clean_clocks/
```

### Issue: "No hand assets found"

**Solution:**
- Check `data/clock_hands/` structure
- Ensure files are named correctly: `hour.png`, `minute.png`
- Try both directory structures (subdirs or flat)

```bash
# Verify hand assets
find data/clock_hands/ -name "*.png"
```

### Issue: "Could not import analog_sketch_creator.py"

**Solution:**
- Verify `analog_sketch_creator.py` is in your project root
- Check the file has the `draw_analog_clock` function
- Update `SKETCH_CREATOR_PATH` in `config.py` if file is elsewhere

```bash
# Verify file exists
ls analog_sketch_creator.py

# Check it can be imported
python -c "from analog_sketch_creator import draw_analog_clock; print('OK')"
```

### Issue: "Module 'cv2' has no attribute..."

**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

### Issue: Out of memory during generation

**Solution:**
- Reduce `IMAGE_SIZE` in config.py (e.g., 128 instead of 256)
- Generate in smaller batches
- Close other applications

### Issue: Augmentations too extreme/mild

**Solution:**
- Adjust parameters in `config.py`
- Modify augmentation pipelines in `augmentations.py`
- Test with single samples first

---

## 📊 Performance Tips

### Speed Optimization

1. **Use SSD storage** for faster I/O
2. **Reduce image size** during testing (128x128)
3. **Disable second hands** if not needed
4. **Limit augmentation probability** for faster generation

### Quality Optimization

1. **Use high-quality backgrounds** (>512x512 source images)
2. **Prepare multiple hand styles** for diversity
3. **Balance train/val split** (90/10 is good)
4. **Verify samples visually** before full generation

---

## 🎯 Next Steps

After generating your dataset:

1. **Train Pix2Pix Model**
   - Use frameworks like PyTorch pix2pix
   - Train on Source→Target mapping
   - Use Sketch as conditional input

2. **Evaluate Results**
   - Visual inspection of generated clocks
   - Quantitative metrics (SSIM, PSNR)
   - Time reading accuracy

3. **Iterate**
   - Adjust augmentation parameters
   - Add more hand styles
   - Experiment with different backgrounds

---

## 📝 Example Commands Reference

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Place analog_sketch_creator.py in project root
# Verify: ls analog_sketch_creator.py

# Download Kaggle dataset
kaggle datasets download -d roustoumabdelmoula/textures-dataset

# Generate dataset
python main.py

# Test single sample
python -c "from dataset_generator import ClockDatasetGenerator; g = ClockDatasetGenerator('./data/textures-dataset', './data/clean_clocks', './data/clock_hands'); g.generate_sample(0)"

# Verify output
ls -R dataset/
```

---

## 📚 Additional Resources

- **Reference Repository**: https://github.com/VictorSuarezVara/Reading-analog-clocks-with-neural-networks
- **Kaggle Dataset**: https://www.kaggle.com/datasets/roustoumabdelmoula/textures-dataset
- **Albumentations Docs**: https://albumentations.ai/docs/
- **Pix2Pix Paper**: https://arxiv.org/abs/1611.07004

---

## ✅ Quick Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `analog_sketch_creator.py` in project root
- [ ] Texture dataset downloaded
- [ ] Hand assets prepared (hour + minute PNG files)
- [ ] Paths configured in `config.py`
- [ ] Test sketch creation works (`python test_sketch.py`)

Ready to generate!

```bash
python main.py
```

---

**Need help?** Check the troubleshooting section or review the code comments in each file.