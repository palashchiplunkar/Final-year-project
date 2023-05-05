from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time
from picamera2 import Picamera2, Preview
import RPi.GPIO as GPIO
from time import sleep
from adafruit_servokit import ServoKit
import time


kit = ServoKit(channels=8)
def get_distance():
    GPIO.setmode(GPIO.BCM)

    TRIG = 23

    ECHO = 24

    print("Distance Measurement In Progress")

    GPIO.setup(TRIG,GPIO.OUT)

    GPIO.setup(ECHO,GPIO.IN)

    GPIO.output(TRIG, False)

    print("Waiting For Sensor To Settle")

    time.sleep(2)

    GPIO.output(TRIG, True)

    time.sleep(0.00001)

    GPIO.output(TRIG, False)
    
    while GPIO.input(ECHO)==0:

        pulse_start = time.time()
  
    while GPIO.input(ECHO)==1:

        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    print("Distance:",distance,"cm")
    GPIO.cleanup()
    return distance

    

def get_waste_image():
    picam = Picamera2()
    config = picam.create_preview_configuration()
    picam.configure(config)

    picam.start_preview(Preview.QTGL)

    picam.start()
    time.sleep(2)
    picam.capture_file("picture.jpg")

    picam.close()
    
def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    print("Index of the input tensor: ", tensor_index, end="\n\n")
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image):

    set_input_tensor(interpreter, image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    print("\nDetails about the input tensors:\n   ", output_details, end="\n\n")

    scores = interpreter.get_tensor(output_details['index'])[0]
    print("Predicted class label score      =", np.max(np.unique(scores)))
    scale, zero_point = output_details['quantization']
    scores_dequantized = scale * (scores - zero_point)

    dequantized_max_score = np.max(np.unique(scores_dequantized))
    print("Predicted class label probability=", dequantized_max_score, end="\n\n")

    max_score_index = np.where(scores_dequantized == np.max(np.unique(scores_dequantized)))[0][0]
    print("Predicted class label ID=", max_score_index)

    return max_score_index, dequantized_max_score

def open_servo1():
    kit.servo[0].angle = 0
    for i in range(0, 121, 1):
        kit.servo[0].angle = i
        sleep(0.02)
def close_servo1():
    for i in range(120, -1, -1):
        kit.servo[0].angle = i
        sleep(0.02)
    
def open_other_servo(index):
    kit.servo[index].angle = 120
    for i in range(120, -1, -1):
        kit.servo[index].angle = i
        sleep(0.02)
def close_other_servo(index):
    for i in range(0, 121):
        kit.servo[index].angle = i
        sleep(0.02)

def run_model():
    model_path = "./model.tflite"
    label_path = "./labels.txt"

    interpreter = Interpreter(model_path)
    print("Model Loaded Successfully.", end="\n\n")

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Input tensor size: (", width, ",", height, ")")

    get_waste_image()
    image = Image.open("./picture.jpg").convert('RGB')
    print("Original image size:", image.size)

    image = image.resize((width, height))
    print("New image size:", image.size, end="\n\n")

    label_id, prob = classify_image(interpreter, image)

    with open(label_path, 'r') as f:
        labels = [line.strip() for i, line in enumerate(f.readlines())]

    classification_label = labels[label_id]
    print("Image Label:", classification_label)
    if(classification_label=="plastic"):
        open_other_servo(1)
        open_servo1()
        sleep(1)
        close_servo1()
        close_other_servo(1)  
    elif(classification_label=="wetwaste"):
        open_other_servo(2)
        open_servo1()
        sleep(1)
        close_servo1()
        close_other_servo(2)
    elif(classification_label=="metal"):
        open_other_servo(3)
        open_servo1()
        sleep(1)
        close_servo1()
        close_other_servo(3)
    else:
        open_servo1()
        sleep(1)
        close_servo1()
        
dist = get_distance()
#if(dist>=0 and dist<=34):
run_model()
print("DONE")
