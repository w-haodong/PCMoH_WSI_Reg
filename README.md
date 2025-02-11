All code and examples will be released after the paper is officially published.

Title: Automatic Histopathology-Registration of Whole Slide Images based on Radiation-Variation Insensitive

Abstract: In digital pathology, high-resolution image registration is crucial for accurate analysis of multi-stained tissues, particularly in breast cancer diagnosis. This paper introduces a novel affine registration method tailored for whole slide images (WSIs), specifically addressing the challenges posed by intensity, rotation, and scaling variations in multi-stained pathology images. The proposed approach combines a FAST keypoint extractor with a modified RIFT descriptor to ensure robustness against both geometric and radiometric distortions. Additionally, an iterative multi-scale ADAM framework with relative diffusion regularization enhances non-rigid alignment accuracy. Experimental results indicate that our method outperforms classical techniques, such as SIFT and SuperPoint, in alignment accuracy across resolutions, especially in high-resolution cases where traditional methods degrade significantly. Tests on the ANHIR 2019 and ACROBAT 2023 datasets confirm the stability and precision of the method, achieving state-of-the-art performance with superior median-rTRE and MP90TRE metrics. Notably, the approach requires no pre-alignment, fine-tuning, or retraining across datasets, demonstrating its adaptability and effectiveness in complex histopathological contexts.

![image](https://github.com/user-attachments/assets/a0198246-d0c8-4e52-a650-b82d78b442ea)

