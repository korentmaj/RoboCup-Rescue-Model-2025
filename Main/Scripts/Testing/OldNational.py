import sensor, time, ml, math, image, gc
from machine import Pin, LED
from ml.utils import NMS

# Sensor setup
sensor.reset()
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_whitebal(False)
sensor.set_auto_gain(False)

# GPIO Pins
p2 = Pin('P2', Pin.OUT); p2.low()  # Optional LED indicator pin
p4 = Pin('P4', Pin.OUT); p4.low()
p5 = Pin('P5', Pin.OUT); p5.low()

# LEDs
r = LED("LED_RED")
g = LED("LED_GREEN")
b = LED("LED_BLUE")

# Load ML Model
model = ml.Model("trained")
print(model)

# Parameters
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

victim_id_map = {
    "red": 3,
    "yellow": 4,
    "green": 5,
    "H": 6,
    "s": 7,
    "S": 7,
    "U": 8
}

min_blob_pixels = 50
group_blobs = True
VOTING_WINDOW = 20
CONFIDENCE_THRESHOLD = 40

# Blink function
def blink_white_led_once():
    r.on()
    g.on()
    b.on()
    time.sleep_ms(10)
    r.off()
    g.off()
    b.off()

# FOMO post-processing
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

# Color blob detection
def detect_color_blobs(img):
    detections = []
    for color_name, threshold in color_labels.items():
        blobs = img.find_blobs([threshold], area_threshold=min_blob_pixels, pixels_threshold=min_blob_pixels, merge=group_blobs)
        for b in blobs:
            detections.append(color_name)
    return detections

# Voting function
def most_common_label(labels):
    if not labels:
        return None, 0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    most_common = max(counts, key=counts.get)
    percent = (counts[most_common] / len(labels)) * 100
    return most_common, percent

# Main Loop
clock = time.clock()
victim_votes = []

while True:
    clock.tick()
    try:
        img = sensor.snapshot()
    except Exception as e:
        print("[WARNING] Frame capture failed:", e)
        continue

    frame_victims = []

    # Color blob detection
    frame_victims.extend(detect_color_blobs(img))

    # FOMO letter detection
    for i, detection_list in enumerate(model.predict([img], callback=fomo_post_process)):
        if i == 0:
            continue
        if len(detection_list) == 0:
            continue
        label = model.labels[i] if i < len(model.labels) else f"class_{i}"
        frame_victims.append(label)

    if frame_victims:
        victim_votes.extend(frame_victims)

    if len(victim_votes) > VOTING_WINDOW:
        victim_votes = victim_votes[-VOTING_WINDOW:]

    if len(victim_votes) >= VOTING_WINDOW:
        label, confidence = most_common_label(victim_votes)
        print(f"[VOTE] Most common: {label} ({confidence:.1f}%)")
        if confidence >= CONFIDENCE_THRESHOLD:
            victim_id = victim_id_map.get(label, None)
            if victim_id is not None:
                print("[VICTIM CONFIRMED]", label)

                # GPIO logic for victim signaling
                if label in ["U", "green"]:
                    p4.low()
                    p5.high()
                elif label in ["S", "s", "yellow"]:
                    p4.high()
                    p5.low()
                elif label in ["H", "red"]:
                    p4.high()
                    p5.high()
                else:
                    print("[GPIO ERROR] Unmapped label:", label)
                    p4.low()
                    p5.low()

                print(f"[GPIO] P4={'HIGH' if p4.value() else 'LOW'}, P5={'HIGH' if p5.value() else 'LOW'}")

                blink_white_led_once()
                time.sleep_ms(200)
                p4.low()
                p5.low()
                print("[GPIO] Pins reset to LOW")

                victim_votes.clear()
            else:
                print("[VICTIM ERROR] Unknown victim label:", label)

    gc.collect()
