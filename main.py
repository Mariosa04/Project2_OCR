import os
import random
import zipfile

# --- SETTINGS ---
IMG_DIR = r"C:\Users\LENOVO\Desktop\OCR\Scratch\Dataset\Total Images"
GT_DIR = r"C:\Users\LENOVO\Desktop\OCR\Scratch\Dataset\Total GT"
OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\OCR_Splits_words"

# Split ratios (80% Train, 10% Val, 10% Test)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# --- SCRIPT START ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image extensions to look for
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# 1. Find all images and match them with their GT text files
valid_pairs = []
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(IMG_EXTS)]

for img_file in image_files:
    base_name, _ = os.path.splitext(img_file)
    txt_file = base_name + ".txt"

    gt_path = os.path.join(GT_DIR, txt_file)
    if os.path.exists(gt_path):
        valid_pairs.append((img_file, txt_file))
    else:
        print(f"Warning: Missing text file for {img_file}")

print(f"Found {len(valid_pairs)} valid image/label pairs.")

# 2. Shuffle the data randomly
random.shuffle(valid_pairs)

# 3. Calculate split sizes
total_items = len(valid_pairs)
train_end = int(total_items * TRAIN_RATIO)
val_end = train_end + int(total_items * VAL_RATIO)

train_pairs = valid_pairs[:train_end]
val_pairs = valid_pairs[train_end:val_end]
test_pairs = valid_pairs[val_end:]

print(f"Splitting into: {len(train_pairs)} Train, {len(val_pairs)} Val, {len(test_pairs)} Test")


# 4. Function to create the zip files WITH FOLDERS INSIDE
def create_zip(pairs, zip_name):
    zip_path = os.path.join(OUTPUT_DIR, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img_file, txt_file in pairs:
            # Add image to "images/" folder inside the zip
            img_path = os.path.join(IMG_DIR, img_file)
            zf.write(img_path, os.path.join("images", img_file))

            # Add text to "labels/" folder inside the zip
            txt_path = os.path.join(GT_DIR, txt_file)
            zf.write(txt_path, os.path.join("labels", txt_file))

    print(f"Created {zip_name} successfully!")


# 5. Create the 3 zip files
create_zip(train_pairs, "train.zip")
create_zip(val_pairs, "val.zip")
create_zip(test_pairs, "test.zip")

print("\nAll done! Check your Desktop for the 'OCR_Splits' folder.")