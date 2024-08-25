import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera could not be opened.")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

camera.release()
cv2.destroyAllWindows()
