import cv2
import numpy as np
from hx711 import HX711
import configparser
from opencvYOLO import opencvYOLO

# Load configuration from pos.ini
config = configparser.ConfigParser()
config.read('pos.ini')

# Initialize the YOLO model using the opencvYOLO class
yolo = opencvYOLO(modeltype=config['yoloModel']['modeltype'],
                  objnames=config['yoloModel']['objnames'],
                  weights=config['yoloModel']['weights'],
                  cfg=config['yoloModel']['cfg'])

# Set detection parameters
yolo.setScore(0.5)  # Set the confidence threshold
yolo.setNMS(0.4)    # Set the non-maximum suppression threshold

# Load product labels from pos.ini
labels_tw = eval(config['products']['labels_tw'])

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

# Initialize the camera
cam_id = int(config['camera']['cam_id'])
cap = cv2.VideoCapture(cam_id)

# Set flip frame configuration
flip_frame = eval(config['camera']['flipFrame'])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break
    
    if flip_frame[0]:
        frame = cv2.flip(frame, 1)  # Horizontal flip
    if flip_frame[1]:
        frame = cv2.flip(frame, 0)  # Vertical flip

    # Perform object detection
    yolo.getObject(frame, labelWant=("",), drawBox=True, bold=2, textsize=0.6, bcolor=(0,255,0), tcolor=(255,255,255))

    # Display the detected objects with labels
    for i in range(len(yolo.bbox)):
        left, top, width, height, label, score = yolo.list_Label(i)
        if label in labels_tw:
            product_name, price, unit = labels_tw[label]
            weight = get_weight()
            cost = 0
            if unit == "twkg":
                cost = weight * price / 600  # Convert to '台斤' unit
            elif unit == "kg":
                cost = weight * price / 1000  # Convert to kilograms
            elif unit == "gram":
                cost = weight * price  # Grams
            else:
                cost = price  # Single item

            text = f"{product_name}: NT$ {cost:.2f}, Weight: {weight:.2f}g"
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
