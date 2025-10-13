# RoboCup RM25 Victim Detection Model

Welcome to the RoboCup RM25 custom victim detection repository. This project is centered around a lightweight TinyML model designed for the **OpenMV H7** camera, capable of real-time **letter ('H', 'S', 'U')** and **color (red, yellow, green)** victim detection, as required by the RoboCup Rescue Maze 2025 competition.

---

## 🔧 Project Structure

```bash
RM25-Model/
├── Main/               # Main program for OpenMV H7
│   └── main.py         # ✔️ Current working detection script (upload to OpenMV)
├── Scripts/            # Examples and experiments
├── Training/           # Model training pipeline and instructions
├── requirements.txt    # For dataset prep and model training
└── README.md           # You're reading it
```

---

## 🎯 Main Program

The primary code can be found in:

```
Main/main.py
```

This is the working **OpenMV MicroPython script** uploaded to the **OpenMV H7** camera. It performs:

- Real-time object detection using a custom-trained model.
- Victim classification based on color or letter.
- GPIO output using P4 and P5.
- An interrupt signal via P6 (e.g., for STM32 communication).

---

## 🤖 Model Info

This is a **custom-trained FOMO (Fast Object Detection)** model tailored for:

- RoboCup Rescue Maze rules
- Detecting letters
- Recognizing colored squares
  
It’s built for **OpenMV H7**, but works with any OpenMV camera that supports TFLite models.

🟢 Achieves ~90% detection accuracy in good conditions  
⚡ Runs at ~60 FPS on H7 — highly optimized and lightweight!

---

## 🛠️ Installation & Setup

1. 🔽 **Download OpenMV IDE**  
   [https://openmv.io/pages/download](https://openmv.io/pages/download)

2. ⚙️ **Install custom firmware** for model support  
   → Follow instructions and get the firmware from:  
   [Firmware-via-wmiuns](https://github.com/s3r5-robotics/RM25-Model/tree/main/Main/Firmware-via-wmiuns)

3. 📂 **Upload** `main.py` to your OpenMV cam using the IDE.

4. ✅ Done! The camera will now detect victims and send outputs via pins.

---

## 📡 Output Protocol

- `P4` and `P5`: encode victim type (e.g., color or letter)
- `P6`: sends a HIGH pulse (20 ms) as an interrupt signal to STM32
- LED indicators also signal detections visually

---

## 🧪 Example Scripts

Look inside the `Scripts/` folder for experimental code and small demos. Useful for:
- Tuning thresholds
- Trying out display options
- Debugging color vs. letter inference

---

## 🧠 Training Your Own Model

If you want to retrain:

1. Go into the `Training/` folder
2. Follow the tensorflow workflow
3. Use `requirements.txt` to set up your environment:

```bash
pip install -r requirements.txt
```

4. Upload the `.tflite` model back to the OpenMV cam using the IDE.

---

## 🚀 Performance

| Metric          | Value       |
|-----------------|-------------|
| Accuracy        | ~90%        |
| FPS on H7       | ~60         |
| Flash Usage     | ~<500KB     |
| RAM Usage       | <1MB        |


---

## 💬 Contact & Credits

Created by the **Maj Korent | RM25 RoboCup Team**  
For RoboCup Rescue Maze 2025 technical challenge  
Firmware tools and deployment inspired by OpenMV docs.

> If you like this, star the repo and share your implementation with the community!

