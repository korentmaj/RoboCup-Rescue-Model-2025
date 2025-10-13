import sensor, time, ml, math, image, gc, ustruct
from machine import UART
from ml.utils import NMS

# ---------------- Camera Setup ----------------
sensor.reset()
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)  # 160x120 - smallest available
sensor.skip_frames(time=2000)

# ---------------- UART Setup ----------------
uart = UART(3, 115200)  # UART 3 is standard on OpenMV H7

# ---------------- Model Load ----------------
model = ml.Model("trained")
# print(model)

min_confidence = 0.4
threshold_list = [(math.ceil(min_confidence * 255), 255)]

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]

# LAB thresholds
red_thresh    = (30, 100, 15, 127, 15, 127)
yellow_thresh = (70, 100, -10, 10, 60, 127)
green_thresh  = (30, 100, -128, -10, 20, 60)

color_labels = {
    "red": red_thresh,
    "yellow": yellow_thresh,
    "green": green_thresh
}

victim_id_map = {
    "red": 3,
    "yellow": 4,
    "green": 5,
    "H": 6,
    "s": 7,
    "U": 8
}

# Blob detection settings
min_blob_pixels = 50
group_blobs = True

# Frame and averaging settings
FRAME_WINDOW = 10   # How many frames to average per detection
SEND_WINDOW = 3     # How many averages to gather before sending UART

# Internal counters
frame_counter = 0
victim_history = []
average_results = []

# ---------------- Post-Processing Functions ----------------
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

def detect_color_blobs(img):
    detections = []
    for color_name, threshold in color_labels.items():
        blobs = img.find_blobs([threshold], area_threshold=min_blob_pixels, pixels_threshold=min_blob_pixels, merge=group_blobs)
        for b in blobs:
            detections.append(color_name)
    return detections

def analyze_victims(history):
    if not history:
        return None
    counts = {}
    for label in history:
        counts[label] = counts.get(label, 0) + 1
    most_common = max(counts, key=counts.get)
    percent = (counts[most_common] / len(history)) * 100
    return most_common, percent

# ---------------- UART Functions ----------------
def send_victim_uart(victim_label):
    victim_id = victim_id_map.get(victim_label, None)
    if victim_id is not None:
        packet = ustruct.pack("<B", victim_id)  # Send 1 byte
        uart.write(packet)
        # print("[UART] Sent victim ID:", victim_id)
    # else:
        # print("[UART] Unknown label, not sending:", victim_label)

def check_for_new_tile():
    if uart.any():
        data = uart.read(1)
        if data and data[0] == 1:
            # print("[UART] New tile detected, resetting history!")
            reset_all()

def reset_all():
    global frame_counter, victim_history, average_results
    frame_counter = 0
    victim_history.clear()
    average_results.clear()

# ---------------- Main Loop ----------------
clock = time.clock()

while True:
    clock.tick()

    check_for_new_tile()

    try:
        img = sensor.snapshot()
    except Exception as e:
        # print("[WARNING] Frame capture failed:", e)
        continue  # Skip frame

    frame_victims = []

    # --- Color detection ---
    color_detections = detect_color_blobs(img)
    frame_victims.extend(color_detections)

    # --- FOMO detection ---
    for i, detection_list in enumerate(model.predict([img], callback=fomo_post_process)):
        if i == 0:
            continue
        if len(detection_list) == 0:
            continue
        label = model.labels[i] if i < len(model.labels) else f"class_{i}"
        for (x, y, w, h), score in detection_list:
            frame_victims.append(label)

    # --- Update victim history ---
    victim_history.extend(frame_victims)
    frame_counter += 1

    if frame_counter >= FRAME_WINDOW:
        result = analyze_victims(victim_history)
        if result:
            most_common_label, percent = result
            # print(f"[VICTIM DETECTED] {most_common_label} ({percent:.1f}%)")
            average_results.append(most_common_label)
        # else:
            # print("[VICTIM DETECTED] None")

        frame_counter = 0
        victim_history.clear()

        # --- Send if enough averages collected ---
        if len(average_results) >= SEND_WINDOW:
            # Find most common among averages
            final_label = max(set(average_results), key=average_results.count)
            send_victim_uart(final_label)
            average_results.clear()

    # print(clock.fps(), "fps")

    gc.collect()  # Always clean memory
