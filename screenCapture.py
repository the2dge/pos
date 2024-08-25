import cv2
import os

def save_frame(img, save_path, prefix="img", img_format=".jpg"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate the filename based on the current number of images in the folder
    file_index = len(os.listdir(save_path))
    filename = f"{prefix}_{file_index:04d}{img_format}"
    file_path = os.path.join(save_path, filename)

    # Save the image
    cv2.imwrite(file_path, img)
    print(f"Image saved as {file_path}")

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera could not be opened.")
    exit()

save_path = "/home/ming/images/"  # Define the path to save the images

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):  # Press 'q' to exit
        break
    elif k == ord('c'):  # Press 'c' to capture the image
        save_frame(frame, save_path)

camera.release()
cv2.destroyAllWindows()
