# ---------------- Imports ----------------
import sensor
import time
import ml
import math
import image
import gc
from machine import Pin, LED
from ml.utils import NMS

# ---------------- Camera Setup ----------------
sensor.reset()
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time=2000)

# ---------------- Pins Setup ----------------
p1_out = Pin('P1', Pin.OUT)  # OUTPUT P1
p2_out = Pin('P2', Pin.OUT)  # OUTPUT P2
p3_in  = Pin('P3', Pin.IN, Pin.PULL_DOWN)  # INPUT P3 from STM32

blue_led = LED("LED_BLUE")
red_led = LED("LED_RED")
green_led = LED("LED_GREEN")

# ---------------- Model Load ----------------
model = ml.Model("trained")
print(model)

# ---------------- Detection Settings ----------------
min_confidence = 0.4
threshold_list = [(math.ceil(min_confidence * 255), 255)]

red_thresh    = (5, 100, 16, 127, -2, 31)
yellow_thresh = (0, 100, -19, 116, 39, 127)
green_thresh  = (0, 75, -128, -13, -79, 51)

color_labels = {
    "red": red_thresh,
    "yellow": yellow_thresh,
    "green": green_thresh
}

# ---------------- Victim Categorization ----------------
green_victims = ["green", "U"]
yellow_victims = ["yellow", "S", "s"]
red_victims = ["red", "H"]

# ---------------- Parameters ----------------
min_blob_pixels = 300
group_blobs = True
FRAME_WINDOW = 15  # How many frames to confirm detection

# ---------------- States ----------------
frame_counter = 0
victim_history = []

# ---------------- Functions ----------------

# Post-process outputs from FOMO model
def fomo_post_process(model, inputs, outputs):
    n, oh, ow, oc = model.output_shape[0]
    nms = NMS(ow, oh, inputs[0].roi)
    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(threshold_list, x_stride=1, area_threshold=1, pixels_threshold=1)
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            nms.add_bounding_box(x, y, x + w, y + h, score, i)
    return nms.get_bounding_boxes()

# Detect color blobs
def detect_color_blobs(img):
    detections = []
    for color_name, threshold in color_labels.items():
        blobs = img.find_blobs([threshold], area_threshold=min_blob_pixels, pixels_threshold=min_blob_pixels, merge=group_blobs)
        for b in blobs:
            detections.append(color_name)
    return detections

# Find most frequent victim
def analyze_victims(history):
    if not history:
        return None
    counts = {}
    for label in history:
        counts[label] = counts.get(label, 0) + 1
    most_common = max(counts, key=counts.get)
    return most_common, counts[most_common]

# Set P1 and P2 based on detected victim
def set_pins_for_victim(victim_label):
    if victim_label in green_victims:
        p1_out.low()
        p2_out.high()
    elif victim_label in yellow_victims:
        p1_out.high()
        p2_out.low()
    elif victim_label in red_victims:
        p1_out.high()
        p2_out.high()
    else:
        p1_out.low()
        p2_out.low()

# Reset both output pins
def reset_pins():
    p1_out.low()
    p2_out.low()

# Reset detection history
def reset_history():
    global frame_counter, victim_history
    frame_counter = 0
    victim_history.clear()

# ---------------- Main Loop ----------------
clock = time.clock()
reset_pins()  # Make sure P1 and P2 are low at start
blue_led.on()
red_led.on()
green_led.on()

while True:
    clock.tick()

    try:
        img = sensor.snapshot()
    except Exception as e:
        print("[WARNING] Frame capture failed:", e)
        continue

    frame_victims = []

    # --- Detect color blobs ---
    color_detections = detect_color_blobs(img)
    frame_victims.extend([str(c) for c in color_detections])

    # --- Detect letter victims using model ---
    for i, detection_list in enumerate(model.predict([img], callback=fomo_post_process)):
        if i == 0:
            continue  # skip background
        if len(detection_list) == 0:
            continue
        label = model.labels[i] if i < len(model.labels) else f"class_{i}"
        for (x, y, w, h), score in detection_list:
            frame_victims.append(str(label))

    # --- Handle STM32 Reset Command ---
    if p3_in.value() == 1:
        print("[P3 HIGH] STM acknowledged. Resetting pins and history.")
        reset_pins()
        reset_history()
        time.sleep_ms(200)  # Delay to allow movement into new tile
        continue

    # --- Update history ---
    victim_history.extend(frame_victims)
    frame_counter += 1

    # --- Confirm detection if enough frames ---
    if frame_counter >= FRAME_WINDOW:
        most_common = analyze_victims(victim_history)
        if most_common:
            label, count = most_common
            if count >= FRAME_WINDOW:
                print(f"[DETECTED] Victim: {label} ({count}/{FRAME_WINDOW})")
                set_pins_for_victim(label)
            else:
                print("[DETECTED] No consistent victim detected.")
        reset_history()

    gc.collect()
