# Import required modules for hardware control, ML, image processing, and memory management
import sensor, time, ml, math, image, gc
from machine import Pin, LED
from ml.utils import NMS
import neopixel

# ========== CONFIG ========== #
LED_BRIGHTNESS = 120                # Brightness for NeoPixel LEDs
EXPOSURE_US = 8000                  # Camera exposure setting in microseconds
FRAME_WINDOW = 20                   # Number of frames to aggregate victim detection votes
SEND_WINDOW = 2                     # Rounds required before sending detection result
MIN_LETTER_BLOB_AREA = 100          # Minimum area for letter detection
REQUIRED_COLOR_BLOB_AREA = 600      # Minimum area for color victim detection
REQUIRED_COLOR_CONFIRMATIONS = 2    # Number of consecutive frames to confirm color detection
DEBUG = False                       # Debug mode flag
# ============================ #

# ------- Camera Setup -------
sensor.reset()                      # Initialize camera sensor
sensor.set_vflip(True)              # Flip image vertically
sensor.set_hmirror(True)            # Mirror image horizontally
sensor.set_pixformat(sensor.RGB565) # Set pixel format
sensor.set_framesize(sensor.QQVGA)  # Set frame size
sensor.set_auto_gain(False)         # Disable auto gain
sensor.set_auto_whitebal(True)      # Enable auto white balance
sensor.set_auto_exposure(False, exposure_us=EXPOSURE_US) # Manual exposure
sensor.skip_frames(time=1000)       # Wait for sensor to stabilize

# ------- NeoPixel Setup (LEDs) -------
p = Pin('P7', Pin.OUT)
n = neopixel.NeoPixel(p, 5, bpp=4)  # Initialize 5 NeoPixels with brightness control
for i in range(5):
    n[i] = (0, 0, 0, LED_BRIGHTNESS) # Set all LEDs to off with specified brightness
n.write()

# ------- GPIO Setup -------
p4_out = Pin('P4', Pin.OUT)         # Output pin for victim signaling
p5_out = Pin('P5', Pin.OUT)         # Output pin for victim signaling
p2_interrupt = Pin('P2', Pin.OUT)   # Interrupt pin for STM32 communication
p3_in = Pin('P3', Pin.IN)           # Input pin for STM32 handshake
p2_interrupt.low()                  # Set interrupt pin low
blue_led = LED("LED_BLUE")          # Onboard blue LED for status indication

# ------- Load ML Model -------
model = ml.Model("trained")         # Load custom-trained FOMO model
model_labels = model.labels if model.labels else ["background", "H", "S", "U"] # Default labels if none provided

# ------- Detection Thresholds -------
min_confidence = 0.7
threshold_list = [(math.ceil(min_confidence * 255), 255)] # Convert confidence threshold to pixel value

# ------- LAB Color Thresholds (tight) -------
red_thresh    = (30, 85, 25, 127, 0, 60)
yellow_thresh = (65, 100, -15, 15, 15, 127)
green_thresh  = (42, 88, -128, -20, -10, 37)

color_labels = {
    "red": red_thresh,
    "yellow": yellow_thresh,
    "green": green_thresh
}

# ------- Buffer Variables for Voting and State -------
color_confirm_buffer = []  # Buffer for consecutive color blob detections
victim_history = []        # Stores detected victim labels to vote on
average_results = []       # (Unused) for possible averaging logic
frame_counter = 0          # Counts processed frames

# ------- Victim Encoding for Output Logic -------
green_victims = ["green", "U"]   # Green color or 'U' letter
yellow_victims = ["yellow", "S"] # Yellow color or 'S' letter
red_victims   = ["red", "H"]     # Red color or 'H' letter

# ================= Function Definitions =================

def fomo_post_process(model, inputs, outputs):
    """
    Post-processes FOMO model output with Non-Maximum Suppression (NMS).
    Converts raw output to bounding boxes for victim detection.
    """
    n, oh, ow, oc = model.output_shape[0]
    nms = NMS(ow, oh, (0, 0, sensor.width(), sensor.height()))
    raw = outputs[0][0]
    for i in range(oc):
        img = image.Image(raw[:, :, i] * 255)
        blobs = img.find_blobs(threshold_list, area_threshold=1, pixels_threshold=1)
        for b in blobs:
            x, y, w, h = b.rect()
            nms.add_bounding_box(x, y, x + w, y + h, 1.0, i)
    return nms.get_bounding_boxes()

def detect_color_blobs(img):
    """
    Detects largest color blob in the image that matches victim color thresholds.
    Returns the color label with the largest valid area.
    """
    max_blob = None
    max_area = 0
    max_label = None
    for color_name, threshold in color_labels.items():
        blobs = img.find_blobs([threshold], area_threshold=REQUIRED_COLOR_BLOB_AREA,
                               pixels_threshold=REQUIRED_COLOR_BLOB_AREA, merge=True)
        for b in blobs:
            area = b.w() * b.h()
            if area > max_area:
                max_area = area
                max_blob = b
                max_label = color_name
    return max_label

def output_victim_pins(victim_label):
    """
    Sets GPIO pins according to the detected victim type and signals STM32 via interrupt.
    Also blinks onboard blue LED for status and waits for STM32 handshake.
    """
    # Set output pins based on victim label
    if victim_label in green_victims:
        p4_out.low()
        p5_out.high()
    elif victim_label in yellow_victims:
        p4_out.high()
        p5_out.low()
    elif victim_label in red_victims:
        p4_out.high()
        p5_out.high()
    else:
        p4_out.low()
        p5_out.low()

    # Trigger interrupt to STM32
    p2_interrupt.high()
    time.sleep_ms(10)
    p2_interrupt.low()

    # Blink blue LED for send indication
    blue_led.on()
    time.sleep_ms(30)
    blue_led.off()

    # Wait for STM32 handshake
    timeout = 0
    while not p3_in.value() and timeout < 100:
        time.sleep_ms(10)
        timeout += 1

    # Reset output pins
    p4_out.low()
    p5_out.low()
    if DEBUG:
        print("[SEND] Victim sent:", victim_label)

def analyze_victims(history):
    """
    Votes on detected victim labels within history window.
    Returns the most frequent label or None if no detections.
    """
    if not history:
        return None
    label = max(set(history), key=history.count)
    return label

def reset_all():
    """
    Resets frame counter, detection history, confirmation buffer, and output pins.
    Prepares for next voting window.
    """
    global frame_counter, victim_history, average_results, color_confirm_buffer
    frame_counter = 0
    victim_history.clear()
    average_results.clear()
    color_confirm_buffer.clear()
    p4_out.low()
    p5_out.low()
    p2_interrupt.low()

# ================= Main Loop =================

clock = time.clock()  # For FPS measurement
reset_all()           # Initialize all buffers and pins

while True:
    clock.tick()      # Start timing for frame rate calculation
    try:
        img = sensor.snapshot() # Capture camera frame
    except:
        continue      # Skip frame if capture fails

    frame_victims = []
    max_letter_area = 0
    max_letter_label = None

    # Run victim detection model and process results
    detections = model.predict([img], callback=fomo_post_process)
    for i, detection_list in enumerate(detections):
        if i == 0: continue   # Skip background class
        label = model_labels[i] if i < len(model_labels) else f"class_{i}"
        for (x, y, w, h), score in detection_list:
            area = w * h
            if area > MIN_LETTER_BLOB_AREA and area > max_letter_area:
                max_letter_area = area
                max_letter_label = label

    # If a letter victim is found, add to frame's detections
    if max_letter_label:
        frame_victims.append(max_letter_label)
        if DEBUG:
            print("[DETECT] Letter:", max_letter_label)
    else:
        # Otherwise, attempt to detect color victim blobs
        color_label = detect_color_blobs(img)
        if color_label:
            color_confirm_buffer.append(color_label)
            if len(color_confirm_buffer) > REQUIRED_COLOR_CONFIRMATIONS:
                color_confirm_buffer.pop(0)
            # Confirm color detection only if it appears in consecutive frames
            if color_confirm_buffer.count(color_label) == REQUIRED_COLOR_CONFIRMATIONS:
                frame_victims.append(color_label)
                if DEBUG:
                    print("[DETECT] Color Confirmed:", color_label)
                color_confirm_buffer.clear()

    # Record frame victim(s) for voting
    if frame_victims:
        victim_history.extend(frame_victims)

    frame_counter += 1

    # If voting window complete, analyze and output result
    if frame_counter >= FRAME_WINDOW:
        final_label = analyze_victims(victim_history)
        if final_label:
            if DEBUG:
                print("[VOTE] Final Label:", final_label)
            output_victim_pins(final_label)
        reset_all()

    # Collect garbage for memory management
    gc.collect()
    if DEBUG:
        print("FPS:", clock.fps(), "| Free RAM:", gc.mem_free())