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
hx = HX711(dout_pin=5, pd_sck_pin=6)
hx.set_scale_ratio(2280)  # You may need to calibrate this scale ratio

def get_weight():
    weight = hx.get_weight_mean(20)  # Take the average of 20 readings
    return max(0, weight)  # Return the weight, ensuring no negative values

# Process each frame
while True:
    ret, frame = cap.read()
    if flip_frame[0]:
        frame = cv2.flip(frame, 1)  # Horizontal flip
    if flip_frame[1]:
        frame = cv2.flip(frame, 0)  # Vertical flip

    height, width = frame.shape[:2]

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
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
    
    # Display detection and weight
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            weight = get_weight()
            cv2.putText(frame, f"Weight: {weight:.2f}g", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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

                cv2.putText(frame, f"Cost: NT$ {cost:.2f}", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display the result frame
    cv2.imshow(config['system']['name_win'], frame)
    
    if record_video:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if record_video:
    out.release()
cv2.destroyAllWindows()
