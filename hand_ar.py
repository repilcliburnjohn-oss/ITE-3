import collections
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
) 
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

CAM_SOURCE = 0 
FRAME_W, FRAME_H = 1280, 720

TRAIL_LENGTH = 18
FINGERTIP_IDS = [4, 8, 12, 16, 20]

SKELETON_COLOR = (255, 0, 180)
SKELETON_GLOW_COLOR = (255, 120, 255)    
JOINT_COLOR = (0, 255, 255)

TRAIL_COLORS = [
    (255, 80, 80),
    (80, 255, 255),
    (80, 255, 80),
    (255, 80, 255),
    (80, 160, 255),
]
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def glow_line(img, p1, p2, color,glow_color, thickness=2):
    cv2.line(img, p1, p2, glow_color, thickness + 6, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def glow_circle(img, center, radius, color, glow_color):
    cv2.circle(img, center, radius + 8, glow_color, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand landmark model (one-time, ~8 mb)...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model Downloaded.")
    
 
def main():
    ensure_model()  
    
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )
    
    
    
    if isinstance(CAM_SOURCE, int):
        cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW)  
    else:
        cap = cv2.VideoCapture(CAM_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAM_SOURCE / camera permissions.")
    
    trails = [ 
              [collections.deque(maxlen=TRAIL_LENGTH) for _ in FINGERTIP_IDS]
              for _ in range(2)
              ]
    
    print("Hand & Finger AR Tracker running. Press 'q' to quit.") 
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms =int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms) 
        
        overlay_layer = np.zeros_like(frame)
        
        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[:2]):
                pts =[(int(p.x * w), int(p.y * h )) for p in hand_landmarks]
                
                for a, b in HAND_CONNECTIONS:
                    glow_line(overlay_layer, pts[a], pts[b], SKELETON_COLOR, SKELETON_GLOW_COLOR,2)
                for p in pts:
                    glow_circle(overlay_layer, p, 4, JOINT_COLOR, (0, 255, 255))   
        
                for f_idx, tip_id in enumerate(FINGERTIP_IDS):
                    trails[hand_idx][f_idx].append(pts[tip_id])
                    trail= trails[hand_idx][f_idx]
                    n = len(trail)
                    for i in range(1, n):
                        fade = i / n 
                        radius = max(1, int(6 * fade))
                        color = tuple(int(c * fade) for c in TRAIL_COLORS[f_idx])
                        cv2.circle(overlay_layer, trail[i], radius, color, -1, cv2.LINE_AA)
                    glow_circle(overlay_layer, pts[tip_id], 8, TRAIL_COLORS[f_idx], TRAIL_COLORS[f_idx] )
                        
                
        frame = cv2.addWeighted(frame, 0.65, overlay_layer, 1.0, 0) 
        cv2.imshow("Hand & Finger AR Tracker",frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break 
         
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    
if __name__=="__main__":
        main()
    