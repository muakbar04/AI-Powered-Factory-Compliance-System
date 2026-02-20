import cv2
import pickle
import numpy as np

frame = cv2.imread("worker.png")  # Extract a sample frame from your video
clone = frame.copy()
rois = []
roi_names = []

current_roi = []

def click_event(event, x, y, flags, param):
    global current_roi, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        current_roi.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_roi) >= 3:
            rois.append(current_roi.copy())
            roi_name = input(f"Enter name for ROI #{len(rois)}: ")
            roi_names.append(roi_name)
            cv2.polylines(clone, [np.array(current_roi)], True, (255, 0, 0), 2)
        current_roi = []

cv2.namedWindow("Draw ROI - Left Click: Points, Right Click: Save ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Draw ROI - Left Click: Points, Right Click: Save ROI", 1920, 1080)
cv2.setMouseCallback("Draw ROI - Left Click: Points, Right Click: Save ROI", click_event)

while True:
    cv2.imshow("Draw ROI - Left Click: Points, Right Click: Save ROI", clone)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# Save ROIs to file
with open("rois.pkl", "wb") as f:
    pickle.dump({"rois": rois, "names": roi_names}, f)