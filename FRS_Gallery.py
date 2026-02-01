import cv2
import os
import numpy as np

# 1. FIND THE FOLDER
base_path = 'celeba-dataset'
img_folder = ""
for root, dirs, files in os.walk(base_path):
    jpgs = [f for f in files if f.lower().endswith('.jpg')]
    if len(jpgs) > 10:
        img_folder = root
        break

if not img_folder:
    print("❌ Could not find images. Make sure the folder is in your sidebar!")
else:
    # 2. SETUP DETECTOR
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # 3. PROCESS 3 DIFFERENT IMAGES
    all_files = [f for f in os.listdir(img_folder) if f.lower().endswith('.jpg')]
    # We take images at different intervals to get a variety of faces
    sample_files = [all_files[0], all_files[10], all_files[20]] 

    for i, filename in enumerate(sample_files):
        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

        occluded = img.copy()
        for (ex, ey, ew, eh) in eyes:
            # Drawing the goggles
            cv2.rectangle(occluded, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), -1)

        # Save the result
        output_name = f'DRDO_Sample_Face_{i+1}.jpg'
        cv2.imwrite(output_name, occluded)
        print(f"✅ Generated: {output_name}")

    print("\n--- GALLERY COMPLETE ---")
    print("You now have 3 distinct images to show in your report!")