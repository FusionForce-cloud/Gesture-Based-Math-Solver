import streamlit as st
from gesture_solver import run_gesture_solver

st.set_page_config(page_title="Gesture-Based Math Solver", layout="centered")

st.title("🖐️ Gesture-Based Math Solver")
st.markdown("""
Use hand gestures via webcam to perform basic math operations!  
✋ Fingers represent digits and two-hand gestures control operators.
""")

if st.button("Start Gesture Solver"):
    st.warning("Press 'q' or 'Esc' to exit the webcam window.")
    run_gesture_solver()
