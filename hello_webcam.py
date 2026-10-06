import cv2

CAM_SOURCE = 0

if isinstance(CAM_SOURCE, int):
    cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(CAM_SOURCE)
    
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check CAM_SOURCE / camera permissions.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    
    frame = cv2.flip(frame, 1)
    cv2.imshow("Hello, Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): # pyright: ignore[reportUndefinedVariable]
        break
    
cap.release()
cv2.destroyAllWindows()