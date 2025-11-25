<h1 align="center">🚆⚡ VoltGuard — Railway Electrical Fault Detection System</h1>

<p align="center">
  <b>AI-powered monitoring • Visual inspection • Predictive analytics</b><br>
  <b>Built by <a href="https://github.com/Ashish-890">Ashish Tripathi</a></b>
</p>

<div align="center">

<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
<img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge">
<img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge">

</div>

---

## 📝 Overview

**VoltGuard** is an advanced **AI-powered Railway Electrical Fault Detection System** that analyzes electrical readings, photos, and real-time telemetry to detect railway equipment faults.

It provides an **industrial-grade dashboard UI** for railway technicians, inspectors, and electrical engineers.

Built for:

- Overhead Equipment (OHE)
- Railway traction substations
- Electrical panels & transformers
- Condition monitoring & maintenance

---

## ⭐ Features

### 🔍 1. Quick Scan — Instant Fault Detection
Enter parameters:
- Voltage
- Current
- Temperature
- Load

VoltGuard outputs:
- **OK / Warning / Fault**
- Confidence score
- Suggested action

---

### 🖼️ 2. Photo Analysis — AI Visual Inspection
Upload a photo to detect:

- Burn marks  
- Loose clamps  
- Damaged insulators  
- Overheating  
- Panel faults  
- Cable degradation  

AI returns:
- Health score  
- Fault classification  
- Suggested repair steps  

---

### 📂 3. Bulk Upload (Photos + CSV)
Upload:
- Multiple equipment photos  
- CSV reading logs  

VoltGuard performs:
- Batch fault detection  
- Summary table  
- Fault distribution  
- Analysis PDF  

---

### 📡 4. Live Telemetry (IoT Mode)
Simulated real-time stream:
- Voltage
- Current
- Temperature
- Vibration
- Spark probability

Updates every second with dynamic **color-coded alerts**.

---

### 📊 5. History & Analytics
Track:
- Fault trend charts  
- Warning vs OK ratio  
- MTBF / MTTR insights  
- Equipment health curve  

---

## 🎨 UI/UX Highlights
- Modern industrial dark theme  
- Sidebar navigation with icons  
- Smooth animations  
- Operator profile section  
- Responsive layout  
- Clean analytics visuals  

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| UI | Streamlit |
| ML | Random Forest |
| Data | NumPy, Pandas |
| Visualization | Matplotlib |
| Image Processing | OpenCV (future-ready) |
| Backend | Python 3.10+ |

---

## 📁 Project Structure
VoltGuard-Railway-Electrical-Fault-Detection/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── models/
│   └── fault_detector.pkl
│
├── src/
│   ├── streamlit_app.py
│   ├── train_model.py
│   ├── simulate_sensors.py
│   └── __init__.py

---

## Create a Virtual Environment
python -m venv .venv

---

## Install Dependencies
pip install -r requirements.txt

---

## ▶️ Run the Application
streamlit run src/streamlit_app.py

---

## 🧪 Retrain the Machine Learning Model
python src/train_model.py

---

## 👤 Author
Ashish Tripathi
🔗 GitHub: https://github.com/Ashish-890

❤️ Passion: AI • Machine Learning • Robotics • Electrical Engineering

---


