
Title: "PCMoH: Radiation-Insensitive Descriptor for Automatic Registration of Histopathology Whole Slide Images"

Precise registration of high-resolution Whole Slide Images (WSIs) is crucial for digital pathology, especially in diagnosing complex cases like breast cancer. This paper introduces a robust, fully automatic hierarchical registration framework. It begins with global alignment using a novel Phase Congruency Moment-weighted Orientation Histogram (PCMoH) descriptor, engineered for resilience against common intensity, rotation, and scaling variations in multi-stain histopathology, ensuring reliable keypoint correspondences. These inliers then guide a two-stage adaptive non-rigid registration: an initial Radial Basis Function (RBF) warping followed by a multi-scale iterative ADAM optimization with relative diffusion regularization for fine-grained deformation capture. Evaluations on ANHIR 2019 and ACROBAT 2023 benchmarks show our method significantly surpasses classical techniques and achieves state-of-the-art performance, evidenced by superior MMrTRE and MP90TRE metrics. Notably, the framework requires no prior alignment, fine-tuning, or retraining, highlighting its strong generalizability and applicability in diverse histopathological contexts. 

![image](https://github.com/user-attachments/assets/a0198246-d0c8-4e52-a650-b82d78b442ea)

Detail the steps:
1. Run reg_with_PCMoH_gui.py, which has an interface to manually pass in an image for registration, as follows:

![image](https://github.com/user-attachments/assets/3521fc0c-c821-4c42-895c-b9ea10cf18ee)

1. To view the resulting image, double-click on the resulting image, as follows:

![image](https://github.com/user-attachments/assets/9adba50c-f239-4c03-92a2-f452b9533a95)

![image](https://github.com/user-attachments/assets/68403f08-6c37-4dc4-8b94-3f59143b4ba5)




