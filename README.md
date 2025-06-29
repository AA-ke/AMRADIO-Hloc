# AMRADIO-Hloc
This project utilizes the VFM: AM-RADIO released by NVIDIA in October 2024 to address the issue of autonomous position identification and localization of drones without GNSS. This visual large model distills three mainstream SOTA models (SAM, CLIP, DINOv2) into one large model, demonstrating superior performance in tasks such as object recognition, multimodal processing, and scene segmentation. The project employs this large model with a Hierarchical Localization architecture, transforming the original CNN-based HF-Net model into one based on a hybrid Transformer architecture, and hierarchically performing global descriptor retrieval and local descriptor matching, demonstrating feasibility on internal drone top-view datasets. The project uses AM-RADIO in conjunction with local descriptor matching methods such as NNM/SuperGlue/LightGlue to achieve local alignment, tested on the HPatches dataset, showing approximately a 15% improvement (on average) over traditional SIFT in matching and retrieval effectiveness, and about 10% over SuperPoint.
![image](https://github.com/user-attachments/assets/7892ae21-0a56-4a84-bc2a-6fe5465fd8ac)
the structure of RADIO
![image](https://github.com/user-attachments/assets/ec336d66-5a53-404b-940a-f5721ddeffe8)
the principle of Hloc
![image](https://github.com/user-attachments/assets/2c08b902-de19-4caf-a6dd-a31e61ddb5d9)
Using the RADIO model for spatial feature extraction to improve local matching effects of feature points
![image](https://github.com/user-attachments/assets/fcabc263-ead0-44e7-9383-c58948e100a8)
Local matching alignment of drone downward view scenes
![image](https://github.com/user-attachments/assets/c365506e-25e1-428a-bb71-07b79837e86c)

![image](https://github.com/user-attachments/assets/d6b1c644-7c1f-478c-be8c-f0c82c85008b)
Use RADIO model Summary tokens for global retrieval
![image](https://github.com/user-attachments/assets/2a9c4b71-0def-4867-bcfd-c1b1103db226)
HPatches benchmark
