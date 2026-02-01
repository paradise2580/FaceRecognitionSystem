# Biometric Reliability Assessment: Ocular Occlusion Analysis

This project evaluates the robustness of **Face Recognition Systems (FRS)** when subjected to synthetic ocular occlusion (simulating spectacles or masks). The assessment utilizes the CelebA dataset to quantify identity retention using advanced similarity metrics.

##  Technical Approach
1. **Detection:** Automated eye localization using **Haar-Cascade Classifiers**.
2. **Occlusion:** Programmatic application of synthetic "goggles" to obscure the ocular manifold.
3. **Metric:** Quantitative analysis via **Cosine Similarity** to measure the distance between the baseline face vector ($A$) and the occluded face vector ($B$).



## Reliability Metric
The system's stability is calculated using the Cosine Similarity formula:

$$\text{Similarity} = \frac{A \cdot B}{\|A\| \|B\|}



A score closer to **1.0** indicates high robustness, proving the FRS can still identify the subject despite the presence of spectacles.

##  Repository Structure
* `FRS_Final_Batch.py`: The core processing engine.
* `FRS_Reliability_Analysis.csv`: Generated report containing similarity scores for the test batch.
* `FRS_Sample_Face_n.jpg`: Visual evidence of successful occlusion and detection.

##  How to Run
1. Ensure `opencv-python`, `numpy`, and `pandas` are installed.
2. Place the dataset in the `celeba-dataset/` directory.
3. Run `python FRS_Final_Batch.py`.
