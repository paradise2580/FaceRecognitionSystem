import cv2
import numpy as np
import os
import pandas as pd

# 1. Setup paths
img_folder = 'celeba-dataset/img_align_celeba/' # Update if your path is different
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 2. Preparing to store results
results = []
files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')][:10] # Just the first 10

print("Processing batch for FRS Report...")

for filename in files:
    img = cv2.imread(os.path.join(img_folder, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

    occluded = img.copy()
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(occluded, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), -1)

    # Cosine Similarity Calculation
    v1 = cv2.resize(img, (16, 16)).flatten().astype(float)
    v2 = cv2.resize(occluded, (16, 16)).flatten().astype(float)
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    results.append({"Image": filename, "Similarity": round(similarity, 4)})

# 3. Save a professional CSV Report
df_report = pd.DataFrame(results)
df_report.to_csv('FRS_Reliability_Analysis.csv', index=False)

print("\n--- FINAL BATCH REPORT ---")
print(df_report)
print("\nSuccess! 'FRS_Reliability_Analysis.csv' created.")

import cv2
import numpy as np
import os
import pandas as pd

# 1. SETUP - Finding where those 1.3GB of images are hiding
base_path = 'celeba-dataset'
img_folder = ""

# This loop searches every subfolder for the images
for root, dirs, files in os.walk(base_path):
    jpgs = [f for f in files if f.lower().endswith('.jpg')]
    if len(jpgs) > 10: # If we find more than 10 images, this is the right place!
        img_folder = root
        break

if not img_folder:
    print("❌ ERROR: Still can't find the images. Did you delete the dataset?")
else:
    print(f"✅ Found images in: {img_folder}")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    results = []
    # Grabing the first 10 images found
    files = [f for f in os.listdir(img_folder) if f.lower().endswith('.jpg')][:10]

    for filename in files:
        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        
        # Detecting Eyes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

        # Drawing Goggles
        occluded = img.copy()
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(occluded, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), -1)

        # Math: Cosine Similarity
        v1 = cv2.resize(img, (16, 16)).flatten().astype(float)
        v2 = cv2.resize(occluded, (16, 16)).flatten().astype(float)
        
        # Formula: (A . B) / (||A|| * ||B||)
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        results.append({"Image": filename, "Similarity_Score": round(float(similarity), 4)})
        print(f"Processed {filename}: Score {similarity:.4f}")

    # 2. SAVING THE DATA
    df_report = pd.DataFrame(results)
    df_report.to_csv('FRS_Reliability_Analysis.csv', index=False)
    print("\n--- DONE! ---")
    print("The file 'FRS_Reliability_Analysis.csv' should now have 10 rows of data.")