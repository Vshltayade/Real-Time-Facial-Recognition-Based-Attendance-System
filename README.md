
# 🎯 Real-Time Facial Recognition Based Attendance System

A **secure, contactless, and real-time attendance management system** built using advanced **computer vision and deep learning** techniques.  
The system integrates **face detection, anti-spoofing, liveness verification, and face recognition** in a multi-stage verification pipeline to prevent proxy attendance and spoofing attacks.

---

## 📌 Project Overview

Traditional attendance systems (manual roll calls, RFID, fingerprint) are inefficient, error-prone, and vulnerable to manipulation.  
This project addresses these issues by implementing a **real-time facial recognition-based attendance system** with **strong security mechanisms** such as anti-spoofing and behavioral liveness detection.

The system ensures that **only a real, live person** can mark attendance, making it suitable for **classrooms, offices, conferences, and smart campus environments**.

---

## 🚀 Key Features

- 🎥 **Real-Time Face Detection** using YOLOv8-Face  
- 🛡️ **Anti-Spoofing Detection** using MiniFASNet (40-frame analysis)  
- 👁️ **Liveness Verification** via dual blink detection (MediaPipe Face Mesh)  
- 🧠 **Face Recognition** using ArcFace (ResNet-50) with cosine similarity  
- 🗂️ **Secure Attendance Storage** in MySQL database  
- 🌐 **Web-Based Dashboard** for attendance monitoring (Flask)  
- 🚫 **Duplicate Attendance Prevention** per session  
- ⚡ **Low-Latency & Real-Time Performance** on CPU  

---

## 🧠 System Pipeline (Fail-Fast Security Architecture)

1. **Video Capture & Preprocessing**
2. **Face Detection (YOLOv8-Face)**
3. **Anti-Spoofing Detection (MiniFASNet)**
4. **Liveness Verification (MediaPipe Face Mesh)**
5. **Face Recognition (ArcFace ResNet-50)**
6. **Attendance Recording (MySQL)**

---

## 🛠️ Technology Stack

- **Language:** Python 3.x  
- **Face Detection:** YOLOv8-Face (Ultralytics)  
- **Anti-Spoofing:** MiniFASNet (ONNX)  
- **Liveness Detection:** MediaPipe Face Mesh  
- **Face Recognition:** ArcFace ResNet-50 (ONNX)  
- **Backend:** Flask  
- **Database:** MySQL  
- **Libraries:** OpenCV, NumPy, SciPy, ONNX Runtime  

---

## 📊 Performance Highlights

- ~95% mAP face detection  
- ~97% spoof detection accuracy  
- ~99.8% recognition accuracy (good lighting conditions)  
- < 2 seconds end-to-end latency  

---

## 🎯 Applications

- Educational Institutions  
- Corporate Offices  
- Conferences & Events  
- Smart Campus Systems  

---

## 🔮 Future Enhancements

- Cloud deployment  
- Mobile application  
- Advanced analytics  
- Multi-modal biometrics  
- Blockchain-based attendance  

---

## 👨‍💻 Project Team

- Nikhil Kolhe  
- Rutuja Jadhav  
- Abhijeet Singh  
- Vishal Tayade  
- Sukeshanee Zende  

**Guided by:** Malkeet Singh (Module Lead – CDAC Mumbai)

---

⭐ If you find this project useful, consider starring the repository!
