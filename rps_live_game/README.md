# GesturePlay — Real-Time Hand Gesture Recognition with Rule-Constrained Game Logic

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A real-time Rock-Paper-Scissors game engine that integrates a fine-tuned **MobileNetV2 CNN** for live webcam gesture perception with a **deterministic rule-constrained decision layer** for game logic enforcement — demonstrating the integration of neural pattern recognition with explicit symbolic constraint satisfaction.

---

## 🎯 Key Features

- **Real-time gesture classification** via fine-tuned MobileNetV2 (Transfer Learning)
- **Live webcam inference** with ROI-based hand detection using OpenCV
- **Rule-constrained decision engine** — deterministic game logic enforcing RPS constraints
- **Score tracking** with 5-second round intervals
- **Modular architecture** — perception and decision layers cleanly separated

---

## 🏗️ System Architecture

```
Webcam Feed
     │
     ▼
┌─────────────────────────────┐
│   ROI Extraction (OpenCV)   │  ← Crops 300×300 hand region
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  MobileNetV2 CNN Classifier │  ← Fine-tuned on RPS dataset
│  (Neural Perception Module) │     224×224 input, ImageNet weights
└─────────────────────────────┘
     │
     ▼  Predicted Gesture: {rock | paper | scissors}
     │
     ▼
┌─────────────────────────────┐
│  Rule-Constrained Decision  │  ← Deterministic constraint engine
│        Engine               │     Enforces RPS game rules strictly
└─────────────────────────────┘
     │
     ▼
  Game Outcome + Score Update
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Model | MobileNetV2 (Fine-Tuned) |
| Input Size | 224 × 224 × 3 |
| Classes | Rock, Paper, Scissors |
| Training Strategy | Transfer Learning + Fine-Tuning (last 30 layers) |
| Optimizer | Adam (lr=1e-5 for fine-tuning) |
| Augmentation | Rotation, Zoom, Shift, Flip |

---

## 📁 Project Structure

```
GesturePlay/
│
├── rps_live_game/
│   ├── rps_live_predict.py        # Main inference + game loop
│   ├── utils.py                   # Modular helper functions
│   ├── rps.ipynb                  # Model training notebook (Colab)
│   ├── rps_mobilenetv2_final.keras # Trained model weights
│   ├── labels.txt                 # Class label definitions
│   └── requirements.txt           # Dependencies
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AmudhanManimaran/GesturePlay.git
cd GesturePlay/rps_live_game
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Game
```bash
python rps_live_predict.py
```

---

## 🎮 How to Play

1. Run `rps_live_predict.py`
2. Position your hand inside the **blue ROI box** on the webcam feed
3. Show **Rock**, **Paper**, or **Scissors** gesture
4. Every **5 seconds**, the system captures and classifies your gesture
5. The rule engine determines the winner and updates the score
6. Press **`q`** to quit

---

## 🧠 Technical Details

### Neural Perception Module
- **Base Model:** MobileNetV2 pre-trained on ImageNet
- **Fine-tuning:** Last 30 layers unfrozen for domain adaptation
- **Head:** GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(3, Softmax)
- **Input Pipeline:** ImageDataGenerator with augmentation (rotation, zoom, shift, horizontal flip)

### Rule-Constrained Decision Engine
The `decide_winner()` function in `utils.py` enforces strict RPS constraints:
```python
# Constraint table
rock     > scissors
paper    > rock
scissors > paper
# All other combinations → Draw
```
This deterministic constraint layer ensures **zero tolerance** for invalid game outcomes regardless of neural prediction confidence.

---

## 🔧 Model Training

The full training pipeline is available in `rps.ipynb` (Google Colab):

1. **Phase 1 — Feature Extraction** (5 epochs, lr=0.0001): Base frozen, head trained
2. **Phase 2 — Fine-Tuning** (10 epochs, lr=1e-5): Last 30 layers unfrozen
3. **Evaluation:** Classification report + Confusion matrix on validation set

---

## 📦 Requirements

```
tensorflow>=2.10.0
opencv-python>=4.5.0
numpy>=1.21.0
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Amudhan Manimaran**
- 🌐 Portfolio: [amudhanmanimaran.github.io/Portfolio](https://amudhanmanimaran.github.io/Portfolio/)
- 💼 LinkedIn: [linkedin.com/in/amudhan-manimaran-3621bb32a](https://www.linkedin.com/in/amudhan-manimaran-3621bb32a)
- 🐙 GitHub: [github.com/AmudhanManimaran](https://github.com/AmudhanManimaran)
