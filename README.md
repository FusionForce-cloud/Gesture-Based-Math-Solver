# 🤖 Gesture-Based Math Solver

This project uses **MediaPipe**, **OpenCV**, and **Streamlit** to build a real-time, gesture-controlled math expression solver via webcam.

## 💡 Features
- Detects hand gestures using MediaPipe
- Maps finger counts and two-hand gestures to digits/operators
- Supports +, -, *, /, =, clear, delete, exit
- Live webcam-based interaction

## 🛠️ Requirements
- Python 3.7–3.11
- Streamlit
- OpenCV
- MediaPipe
- NumPy

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/gesture-math-solver.git
cd gesture-math-solver
pip install -r requirements.txt
streamlit run app.py
