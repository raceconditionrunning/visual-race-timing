import cv2
import sys


# Mouse callback function to get the coordinates
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: ({x}, {y})")


# Load the video
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', get_coordinates)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
