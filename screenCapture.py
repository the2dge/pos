import os
import imutils
import cv2

preChar = "n"
videoFile = "0"
webCamSize = (1920, 1080)
framesSavePath = "/home/ming/images/"
resizeWidth = 0
rotate = 0

if not os.path.exists(framesSavePath):
    os.makedirs(framesSavePath)

# Initialize the camera
if videoFile.isdigit():
    camera = cv2.VideoCapture(int(videoFile))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, webCamSize[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, webCamSize[1])
else:
    camera = cv2.VideoCapture(videoFile)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Camera or video file could not be opened.")
    sys.exit()

i = 0
while True:
    grabbed, img = camera.read()
    if not grabbed:
        print("Frame not grabbed. Exiting...")
        break

    if rotate > 0:
        img = imutils.rotate_bound(img, rotate)

    cv2.imshow("Frame", imutils.resize(img, width=600))
    k = cv2.waitKey(1)
    if k == 99:  # 'c' key to capture
        filename = preChar + "_" + str(i).zfill(8) + ".jpg"
        if resizeWidth > 0:
            img = imutils.resize(img, width=resizeWidth)

        cv2.imwrite(os.path.join(framesSavePath, filename), img)
        print(f"{filename} saved.")
        i += 1
    elif k == 27:  # Escape key to exit
        break

camera.release()
cv2.destroyAllWindows()
