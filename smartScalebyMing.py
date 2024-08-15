import cv2
import numpy as np
import RPi.GPIO as GPIO
from hx711 import HX711
import configparser
import time

# Load configuration from pos.ini
config = configparser.ConfigParser()
config.read('pos.ini')

# Load YOLO model
model_type = config['yoloModel']['modeltype']
weights_path = config['yoloModel']['weights']
cfg_path = config['yoloModel']['cfg']
objnames_path = config['yoloModel']['objnames']

net = cv2.dnn.readNet(weights_path, cfg_path)
with open(objnames_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Load product labels from pos.ini
labels_tw = eval(config['products']['labels_tw'])

# Initialize camera
cam_id = int(config['camera']['cam_id'])
flip_frame = eval(config['camera']['flipFrame'])
record_video = config['camera'].getboolean('record_video')
video_out = config['camera']['video_out']
frame_rate = int(config['camera']['frame_rate'])

cap = cv2.VideoCapture(cam_id)
if record_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

# Initialize GPIO and HX711
hx = HX711(5, 6)  # Initialize with GPIO pin numbers
hx.set_reading_format("MSB", "MSB")
hx.set_reference_unit(2280)  # You may need to calibrate this
hx.reset()
hx.tare()  # Tare the scale

def get_weight():
    weight = hx.get_weight(5)  # Take the average of 5 readings
    hx.power_down()
    hx.power_up()
    return max(0, weight)  # Ensure no negative values

# Load background image and resize to fit the screen
bg_img = cv2.imread('images/bg.jpg')
bg_img = cv2.resize(bg_img, (800, 480))  # Adjust to the screen resolution

# Hide the window frame
cv2.namedWindow(config['system']['name_win'], cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(config['system']['name_win'], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Define the region where the webcam feed will be placed (coordinates need to match your layout)
webcam_width = 500
webcam_height = 380
webcam_target_x = 0
webcam_target_y = 100

# Define the region for the weight display
weight_x_start = 550
weight_x_end = 750
weight_y_start = 200  # Adjust as needed for vertical positioning

# Process each frame
while True:
    ret, frame = cap.read()
    if flip_frame[0]:
        frame = cv2.flip(frame, 1)  # Horizontal flip
    if flip_frame[1]:
        frame = cv2.flip(frame, 0)  # Vertical flip

    # Resize the webcam feed to fit the target region
    resized_frame = cv2.resize(frame, (webcam_width, webcam_height))

    # Prepare the frame for YOLO
    height, width = resized_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(resized_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Place the webcam feed onto the background
    combined_frame = bg_img.copy()
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            weight = get_weight()
            cv2.putText(resized_frame, f"Weight: {weight:.2f}g", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Lookup price and calculate cost
            product_info = eval(config['products']['labels_tw']).get(label)
            if product_info:
                product_name, price, unit = product_info
                if unit == "twkg":
                    cost = weight * price / 600  # Convert to '台斤' unit
                elif unit == "kg":
                    cost = weight * price / 1000  # Convert to kilograms
                elif unit == "gram":
                    cost = weight * price  # Grams
                else:
                    cost = price  # Single item

                cv2.putText(resized_frame, f"Cost: NT$ {cost:.2f}", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Overlay the detection frame onto the background image
    combined_frame[webcam_target_y:webcam_target_y + resized_frame.shape[0],
                   webcam_target_x:webcam_target_x + resized_frame.shape[1]] = resized_frame

    # Display the result frame
    cv2.imshow(config['system']['name_win'], combined_frame)
    
    if record_video:
        out.write(combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if record_video:
    out.release()
cv2.destroyAllWindows()
