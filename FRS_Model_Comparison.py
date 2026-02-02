import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace

#  STEP 1: THE CORRECT PATH 
img_folder = r'C:\Users\HP\FRS\celeba-dataset\img_align_celeba\img_align_celeba' 

#  STEP 2: VALIDATION 
if not os.path.exists(img_folder):
    print(f"❌ Folder still not found. Please double check the path.")
else:
    all_files = [f for f in os.listdir(img_folder) if f.lower().endswith('.jpg')]
    print(f"✅ Found {len(all_files)} images. Starting Comparative Analysis...")

    models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace']
    sample_files = all_files[:3] 
    all_data = []
    threshold = 0.7 

    for model_name in models:
        print(f"\n🔄 Testing Model: {model_name}...")
        similarities = []
        
        for filename in sample_files:
            img_path = os.path.join(img_folder, filename)
            img = cv2.imread(img_path)
            
            # Create Occlusion (Goggles)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
            
            occluded = img.copy()
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(occluded, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), -1)

            try:
                # DeepFace verification (Similarity = 1 - distance)
                res = DeepFace.verify(img, occluded, model_name=model_name, enforce_detection=False, distance_metric='cosine')
                sim = 1 - res['distance']
                similarities.append(sim)
            except Exception as e:
                print(f"⚠️ Skipping {filename} due to model error.")

        if similarities:
            # FNMR: Percentage where similarity is below threshold (failed match)
            fnm_count = sum(1 for s in similarities if s < threshold)
            fnmr = (fnm_count / len(similarities)) * 100
            
            all_data.append({
                'Model': model_name, 
                'Avg_Similarity': round(np.mean(similarities), 4), 
                'FNMR_Percent': round(fnmr, 2)
            })

    # STEP 3: OUTPUT 
    if all_data:
        df = pd.DataFrame(all_data)
        print("\n--- FINAL COMPARISON REPORT ---")
        print(df)
        
        # Saving Graph
        df.plot(x='Model', y='Avg_Similarity', kind='bar', color='teal', title='Model Robustness Comparison')
        plt.ylabel('Cosine Similarity')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig('FRS_Comparison_Graph.png')
        print("\n✅ Results saved to 'FRS_Comparison_Graph.png'")