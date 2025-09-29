# Smart Traffic Management using RL, SUMO, OpenCV & IoT

An intelligent traffic control solution for **SIH 2025 Problem Statement 50**, focused on mitigating urban traffic congestion through **Reinforcement Learning (RL)**, **SUMO simulation**, **OpenCV-based computer vision**, and **IoT integration**.

---

## 📖 Problem Statement

Traffic congestion arises due to unbalanced signal timing, fluctuating traffic density, and the limited adaptability of traditional controllers. Our solution leverages **adaptive, data-driven methods** that integrate **computer vision** and **IoT** to make real-time, intelligent traffic management decisions.

---

## ⚙️ Implementation

* **Data Collection**: Roadside IoT sensors (cameras, inductive loops, GPS-enabled vehicles) capture real-time traffic data.
* **Computer Vision**: OpenCV processes video feeds to detect and track vehicles, estimate queue lengths, and classify vehicle types.
* **IoT Data Fusion**: Streams such as air quality, weather, and vehicle-to-infrastructure (V2I) signals are integrated to form a rich state vector.
* **RL Controller**: Trained in **SUMO** via TraCI, the RL agent dynamically optimizes green splits, phase durations, and offsets.
* **Reward Function**: Minimizes waiting time, reduces emissions, and maximizes throughput. Domain randomization ensures resilience across varying conditions.

---

## ✨ Key Features

* **Adaptive Signal Control**: RL-driven timing that continuously learns from live data.
* **IoT-Enabled Data Fusion**: Combines inputs from GPS, sensors, and connected vehicles.
* **Computer Vision Insights**: Vehicle detection, tracking, and congestion estimation using OpenCV.
* **Scalable Simulation**: SUMO-based training across diverse and dynamic scenarios.
* **Deployment Ready**: Lightweight, edge-compatible design for IoT networks.

---

## 🌍 Impact

By uniting **RL, SUMO, OpenCV, and IoT**, our system evolves beyond static traffic control to deliver a **sustainable, intelligent, and responsive** solution for modern smart cities.

---

## 👥 Team

Developed by Team **CROSSFAIR** for **SIH 2025**.

---
