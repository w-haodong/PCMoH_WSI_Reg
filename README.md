All code and examples will be released after the paper is officially published.

Title: "PCMoH: Radiation-Insensitive Descriptor for Automatic Registration of Histopathology Whole Slide Images"

Precise registration of high-resolution Whole Slide Images (WSIs) is crucial for digital pathology, especially in diagnosing complex cases like breast cancer. This paper introduces a robust, fully automatic hierarchical registration framework. It begins with global alignment using a novel Phase Congruency Moment-weighted Orientation Histogram (PCMoH) descriptor, engineered for resilience against common intensity, rotation, and scaling variations in multi-stain histopathology, ensuring reliable keypoint correspondences. These inliers then guide a two-stage adaptive non-rigid registration: an initial Radial Basis Function (RBF) warping followed by a multi-scale iterative ADAM optimization with relative diffusion regularization for fine-grained deformation capture. Evaluations on ANHIR 2019 and ACROBAT 2023 benchmarks show our method significantly surpasses classical techniques and achieves state-of-the-art performance, evidenced by superior MMrTRE and MP90TRE metrics. Notably, the framework requires no prior alignment, fine-tuning, or retraining, highlighting its strong generalizability and applicability in diverse histopathological contexts. 

![image](https://github.com/user-attachments/assets/a0198246-d0c8-4e52-a650-b82d78b442ea)

