# Diagnosing-Osteoarthritis-Using-AI
An AI-assisted diagnostic framework for osteoarthritis screening that combines questionnaire-based clinical features with image-based deep learning analysis. This project investigates a multimodal learning pipeline for the early detection of joint diseases, focusing on osteoarthritis (OA), rheumatoid arthritis (RA), and combined RA+OA cases.

This project proposes an AI-enabled screening system that integrates:

- **Questionnaire-based clinical analysis**
- **Convolutional Neural Network (CNN)-based image classification**
- **Probability-level fusion using a Joint Probability Function**
- **Multi-class diagnosis for `CONTROL`, `OA`, `RA`, and `RA + OA`**

## Research Scope

This repository documents the **questionnaire analysis phase** of the project, which forms one branch of a larger multimodal diagnostic pipeline.  
In this phase, patient-reported symptoms and demographic information are analyzed through an AI-based questionnaire model to estimate disease probability and screening risk.

## Methodology

The proposed framework includes the following components:

### 1. Dataset Preprocessing
Medical images and questionnaire data are prepared through standard preprocessing steps to improve model consistency and training quality.

### 2. CNN-Based Image Analysis
A convolutional neural network is used to extract discriminative features from hand/joint images.

Typical CNN stages include:

- Input image
- Convolution layers
- Feature map extraction
- Pooling
- Flattening
- Dense layers
- Softmax classification

### 3. Questionnaire-Based Neural Network
A second branch processes clinical questionnaire features such as:

- Pain
- Stiffness
- Swelling
- Age
- Other relevant symptom-based inputs

### 4. Multimodal Fusion
The outputs of the image branch and questionnaire branch are fused using probability-based strategies, including:

- Early fusion
- Late fusion
- Joint Probability Function

### 5. Final Classification
The fused model predicts one of the following classes:

- `CONTROL`
- `OA`
- `RA`
- `RA + OA`

- ## Research Architecture

The system employs a sophisticated multimodal fusion architecture. It processes data through two parallel branches:

1.  **Image Branch:** A Convolutional Neural Network (CNN) extracts features from hand/wrist images to identify visual markers associated with OA and RA.
2.  **Questionnaire Branch:** A dense neural network processes patient demographics and symptom-related clinical inputs (e.g., pain, stiffness, swelling).

The final classification is achieved through a **Joint Probability Function**, which fuses the outputs from both branches to determine the probability of four distinct classes: `CONTROL`, `OA`, `RA`, and `RA + OA`.

<img width="975" height="524" alt="image" src="https://github.com/user-attachments/assets/3c52541d-5284-4e14-ae7c-8400ab580eaf" />
*Figure 1. Multimodal joint-probability fusion architecture combining CNN-based image classification with questionnaire-based clinical features.*



This repository serves as the practical implementation of the **questionnaire-based screening phase**. It provides a user-friendly GUI (built with `customtkinter`) that allows clinicians to enter patient data, run the AI-based inference engine, and generate standardized reports.

### Key Features
*   **Clinical Interface:** Streamlined patient record management and symptom evaluation.
*   **AI Inference:** Real-time prediction of joint disease risk level based on questionnaire data.
*   **Explainable Results:** Visualization of class probability distributions and confidence scoring.
*   **Automated Reporting:** Generate patient screening reports in `PDF`, `HTML`, `TXT`, and `Excel` formats.
*   **Clinic Branding:** Supports custom clinic logo integration for formal reporting.

### Screenshots

**Main Questionnaire Interface**
<img width="1919" height="1024" alt="API" src="https://github.com/user-attachments/assets/e9a07a91-f57a-4867-b9ec-b26a3535e800" />

*Clinical workflow interface for patient data entry and automated symptom screening.*

**Results Dashboard**
<img width="756" height="790" alt="API2" src="https://github.com/user-attachments/assets/c709e34a-8b8b-465b-9df0-0ced9ee7a601" />
*Prediction summary dashboard illustrating confidence levels, risk assessment, and probability distributions.*

---

## Technical Stack

- **GUI Development:** `customtkinter`
- **Data Analysis & Processing:** `pandas`, `numpy`, `scikit-learn`
- **Inference Pipeline:** `joblib` (for model deployment)
- **Visualization:** `matplotlib`
- **Report Generation:** `reportlab`, `openpyxl`
- **Architecture:** Multimodal Neural Networks (CNN + MLP fusion)

## Access & Privacy Notice

**Please Note:** The source code, trained models, and underlying datasets associated with this project are **proprietary**. They are currently maintained as confidential research artifacts and are **not available for public release, download, or distribution**. 

This repository is maintained solely as a portfolio showcase to demonstrate the research methodology, multimodal architecture, and clinical application design.

---

## Contact

For academic inquiries, collaboration opportunities, or technical discussions regarding this research, please feel free to reach out through the following channels:

- **Email:** [soroush.salari74@gmail.com]
- **LinkedIn:** [[Your LinkedIn Profile URL](https://www.linkedin.com/in/soroush-salari-381518221/)]
- **Scholar:** [[Your Academic Profile URL](https://scholar.google.com/citations?user=qyNaVM4AAAAJ&hl=en&oi=ao)]



