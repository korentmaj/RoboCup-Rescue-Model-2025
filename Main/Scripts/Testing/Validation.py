import sensor, time, ml, math, image
from ml.utils import NMS

# ---------------- Camera Setup ----------------
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

# ---------------- Model Load ----------------
model = ml.Model("trained")
print(model)

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

# LAB thresholds for color blobs
red_thresh    = (30, 100, 15, 127, 15, 127)
yellow_thresh = (70, 100, -10, 10, 60, 127)
green_thresh  = (30, 100, -128, -10, 20, 60)

color_labels = {
    "red": red_thresh,
    "yellow": yellow_thresh,
    "green": green_thresh
}

# Blob detection settings
min_blob_pixels = 100
group_blobs = True

# Frame averaging settings
FRAME_WINDOW = 10  # <-- You can change this anytime!

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
            cx, cy = b.cx(), b.cy()
            size = b.pixels()
            detections.append(color_name)
    return detections

# ---------------- Victim Analysis ----------------
FRAME_COUNTER = 0
victim_history = []

def analyze_victims(history):
    if not history:
        return None

    counts = {}
    for label in history:
        counts[label] = counts.get(label, 0) + 1
    most_common = max(counts, key=counts.get)
    percent = (counts[most_common] / len(history)) * 100
    return most_common, percent

# ---------------- Main Loop ----------------
clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()
    frame_victims = []

    # --- FOMO Detection (Letters) ---
    for i, detection_list in enumerate(model.predict([img], callback=fomo_post_process)):
        if i == 0:
            continue  # Skip background
        if len(detection_list) == 0:
            continue
        label = model.labels[i] if i < len(model.labels) else f"class_{i}"
        for (x, y, w, h), score in detection_list:
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            frame_victims.append(label)

    # --- Color Detection ---
    color_detections = detect_color_blobs(img)
    frame_victims.extend(color_detections)

    # --- Update history ---
    victim_history.extend(frame_victims)
    FRAME_COUNTER += 1

    if FRAME_COUNTER >= FRAME_WINDOW:
        result = analyze_victims(victim_history)
        if result:
            most_common_label, percent = result
            print(f"[VICTIM DETECTED] {most_common_label} ({percent:.1f}%)")
        else:
            print("[VICTIM DETECTED] None")
        
        # Clear memory and reset counter
        FRAME_COUNTER = 0
        victim_history.clear()

    # print(clock.fps(), "fps")
