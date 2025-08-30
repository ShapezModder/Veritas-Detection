Veritas AI 🚨  
A real-time surveillance system** built with Python, OpenCV, and YOLO.  
It can detect motion, humans, and faces from webcam input.  
When enabled, the system can trigger an alarm and record video evidence into folders automatically.  

---

✨ Features
- 🎥 Motion Detection** – detects and records movements in real-time  
- 🧑 Human Detection (YOLOv8)** – tracks humans in frame  
- 🙂 Face Detection** – detects and records faces  
- 🚨 Alarm System** – siren plays when a human is detected (if enabled)  
- 💾 Recording – saves events into `MOTION/`, `HUMAN/`, `FACE/` folders  
- 🖥️ Modern UI – built with CustomTkinter  

---

 🛠️ Tech Stack
- Python 3.10 (Recommended)
- [OpenCV](https://opencv.org/)  
- [face_recognition](https://github.com/ageitgey/face_recognition)  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)  
- [pygame](https://www.pygame.org/)  

---

📦 Installation

Clone this repository:
```bash
git clone https://github.com/ShapezModder/Veritas-Detection

📦 Create a Virtual Environment
python -m venv .venv
.venv\Scripts\activate     # On Windows
source .venv/bin/activate  # On Mac/Linux

📦 Install dependencies:
pip install -r requirements.txt

▶️ Usage
Run the app:
python veritas_final_ui.py

Controls inside the UI:
- Toggle Motion Detection
- Toggle Human Detection
- Toggle Face Detection
- Toggle Alarm (siren on human)

Recordings will be saved into:
- MOTION/
- HUMAN/
- FACE/

📂 Project Structure
veritas-ai/
│── veritas_final_ui.py     # Main script
│── requirements.txt        # Dependencies
│── .gitignore
│── README.md
│── LICENSE
│── siren.wav               # Alarm sound
│── background.png (optional)
│── splash.png (optional)
│── MOTION/                 # Motion recordings
│── HUMAN/                  # Human recordings
│── FACE/                   # Face recordings

⚠️ Notes
- YOLO weights (yolov8n.pt) are not included in the repo (too large).
- By default, Ultralytics will auto-download them on first run.
- On Windows, installing face_recognition may require Visual C++ Build Tools and CMake.

📜 License
This project is licensed under the MIT License.

🚀 Future Improvements
- Multi-camera support
- Email / SMS alerts
- Cloud storage for recordings
- Web-based dashboard (Flask/Streamlit)


