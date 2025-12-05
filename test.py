"""Quick smoke test to verify critical imports and runtime.

Run:
    python test.py
"""

def smoke():
    ok = True
    try:
        import cv2
        import numpy as np
        print("OK: cv2 and numpy imported")
    except Exception as e:
        print("ERROR: cv2/numpy import failed:", e)
        ok = False

    try:
        import mediapipe as mp
        print("OK: mediapipe imported")
    except Exception as e:
        print("WARN: mediapipe import failed:", e)

    try:
        import pytesseract
        print("OK: pytesseract imported")
    except Exception as e:
        print("WARN: pytesseract import failed:", e)

    print("Smoke test finished.\nExit code =", 0 if ok else 2)


if __name__ == '__main__':
    smoke()
