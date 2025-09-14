# Face Paralysis Classification

This repository contains the implementation of the **Face Paralysis Classification System**, a trial-and-error experimental work for a final project developed by Informatics Engineering students at **Sriwijaya University**.  
The goal of this project is to assist in the early detection of facial paralysis by classifying facial images into two categories: **Paralysis** and **Normal**, using deep learning with transfer learning.

---

## 🎯 Project Objective
The system aims to provide a **fast, consistent, and accessible screening tool** to help support preliminary detection of facial paralysis. In the future, the project is planned to be extended into a **multi-class classification system using the House–Brackmann scale** for standardized clinical evaluation.

---

## 📊 Dataset Description
- **Dataset faces**: Private dataset from *Massachusetts Eye and Ear Infirmary (MEEI)*.   
- **Total samples**: 60 images  
  - Complete: 10 samples
  - Mild: 10 samples
  - Moderate: 10 samples
  - Near Normal: 10 samples
  - Normal: 10 samples
  - Severe: 10 samples

---

## 🔄 Project Workflow
1. **Dataset Preparation**  
   - Data sourced from MEEI dataset
   - Augmented data
   - Images cleaned, resized, and preprocessed for model input  

2. **Model Training**  
   - Transfer learning with **MobileNetV2** using TensorFlow  
   - Achieved accuracy: **98%**  

3. **Deployment**  
   - Model integrated into a Streamlit web app  
   - Accessible online for interactive predictions  

---

## 🛠️ Tech Stack
- **Deep Learning Framework**: TensorFlow  
- **Model**: MobileNetV2 (Transfer Learning)  
- **Deployment**: Streamlit  

---

## 👨‍🎓 About
This project was developed by **Informatics Engineering students at Sriwijaya University** as a **Final Project** in the completion of their studies.  

---
