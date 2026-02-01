import cv2
import numpy as np
import os

# --- STEP 1: DYNAMIC PATH SEARCH ---
# This looks for any folder containing .jpg files
target_folder = ""
base_search = 'celeba-dataset'

print("Searching for images...")
for root, dirs, files in os.walk(base_search):
    # If we find a folder with a lot of jpgs, that's our target!
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    if len(jpg_files) > 10:
        target_folder = root
        print(f"✅ Target found: {target_folder}")
        break

if not target_folder:
    print("❌ ERROR: Could not find images inside the folder.")
    print("Check if the folder is still unzipping!")
else:
    # --- STEP 2: LOAD AND PROCESS ---
    all_imgs = [f for f in os.listdir(target_folder) if f.lower().endswith('.jpg')]
    sample_path = os.path.join(target_folder, all_imgs[0])
    img = cv2.imread(sample_path)

    # 1. Automatic Eye Detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

    # 2. Apply Goggles (Occlusion)
    occluded = img.copy()
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(occluded, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), -1)

    # 3. Calculate Cosine Similarity (The DRDO Metric)
    v1 = cv2.resize(img, (16, 16)).flatten().astype(float)
    v2 = cv2.resize(occluded, (16, 16)).flatten().astype(float)
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # --- STEP 4: SAVE EVIDENCE ---
    # Create a nice side-by-side comparison for Ma'am
    final_view = np.hstack((img, occluded))
    cv2.imwrite('DRDO_Success_Proof.jpg', final_view)

    print("-" * 40)
    print("DRDO TASK: COMPLETED")
    print(f"RELIABILITY SCORE: {similarity:.4f}")
    print("Check 'DRDO_Success_Proof.jpg' for your result!")
    print("-" * 40)