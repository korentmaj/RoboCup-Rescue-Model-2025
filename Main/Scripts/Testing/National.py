import sensor, time, ml, math, image, gc
from machine import Pin, LED
from ml.utils import NMS

# ---------------- Camera Setup ----------------
sensor.reset()
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)  # 160x120
sensor.skip_frames(time=2000)

# ---------------- GPIO Setup ----------------
p4_out = Pin('P4', Pin.OUT)
p5_out = Pin('P5', Pin.OUT)

blue_led = LED("LED_BLUE")
red_led = LED("LED_RED")
green_led = LED("LED_GREEN")

def led_blink():
    blue_led.on()
    red_led.on()
    green_led.on()
    time.sleep_ms(100)
    blue_led.off()
    red_led.off()
    green_led.off()

def log_output_pins():
    print(f"[PIN STATE] P4: {'HIGH' if p4_out.value() else 'LOW'}, P5: {'HIGH' if p5_out.value() else 'LOW'}")

# ---------------- Model Load ----------------
model = ml.Model("trained")
print("Model loaded:", model)

# ---------------- Thresholds ----------------
min_confidence = 0.5
threshold_list = [(math.ceil(min_confidence * 255), 255)]

# LAB color thresholds
red_thresh    = (30, 100, 15, 127, 15, 127)
yellow_thresh = (70, 100, -10, 10, 60, 127)
green_thresh  = (30, 100, -128, -10, 20, 60)

color_labels = {
    "red": red_thresh,
    "yellow": yellow_thresh,
    "green": green_thresh
}

green_victims = ["green", "U"]
yellow_victims = ["yellow", "s"]
red_victims = ["red", "H"]

# ---------------- Settings ----------------
min_blob_pixels = 100
group_blobs = True
FRAME_WINDOW = 10        
SEND_WINDOW = 2          
NO_VICTIM_RESET_FRAMES = 20  


# ---------------- State ----------------
frame_counter = 0
victim_history = []
average_results = []
no_victim_frame_count = 0

# ---------------- Functions ----------------
def fomo_post_process(model, inputs, outputs):
    n, oh, ow, oc = model.output_shape[0]
    nms = NMS(ow, oh, (0, 0, sensor.width(), sensor.height()))
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
    max_blob = None
    max_area = 0
    max_label = None
    for color_name, threshold in color_labels.items():
        blobs = img.find_blobs([threshold], area_threshold=min_blob_pixels,
                               pixels_threshold=min_blob_pixels, merge=group_blobs)
        for b in blobs:
            area = b.w() * b.h()
            if area > max_area:
                max_area = area
                max_blob = b
                max_label = color_name
    if max_blob:
        img.draw_rectangle(max_blob.rect(), color=255)
        img.draw_string(max_blob.cx(), max_blob.cy(), max_label, mono_space=False)
        return max_label
    return None

def analyze_victims(history):
    if not history:
        return None
    counts = {}
    for label in history:
        counts[label] = counts.get(label, 0) + 1
    most_common = max(counts, key=counts.get)
    percent = (counts[most_common] / len(history)) * 100
    return most_common, percent

def output_victim_pins(victim_label):
    print(f"[OUTPUT] Triggering output for victim: {victim_label}")
    prev_p4 = p4_out.value()
    prev_p5 = p5_out.value()
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

    if (p4_out.value() and not prev_p4) or (p5_out.value() and not prev_p5):
        blue_led.on()
        time.sleep_ms(100)
        blue_led.off()

def reset_all():
    global frame_counter, victim_history, average_results
    frame_counter = 0
    victim_history.clear()
    average_results.clear()
    p4_out.low()
    p5_out.low()
    print("[RESET] Output pins and state cleared")

# ---------------- Main Loop ----------------
clock = time.clock()
reset_all()

while True:
    clock.tick()

    try:
        img = sensor.snapshot()
    except Exception as e:
        print("[ERROR] Snapshot failed:", e)
        continue

    frame_victims = []

    # --- Color Detection (max area only) ---
    max_color_label = detect_color_blobs(img)
    if max_color_label:
        frame_victims.append(max_color_label)

    # --- Letter Detection (max area only) ---
    max_letter_area = 0
    max_letter_label = None

    for i, detection_list in enumerate(model.predict([img], callback=fomo_post_process)):
        if i == 0: continue
        label = model.labels[i] if i < len(model.labels) else f"class_{i}"
        for (x, y, w, h), score in detection_list:
            area = w * h
            if area > max_letter_area:
                max_letter_area = area
                max_letter_label = label
            img.draw_rectangle((x, y, w, h), color=255)
            img.draw_string(x, y - 8, label, mono_space=False)

    if max_letter_label:
        frame_victims.append(max_letter_label)

    print(f"[FRAME] Detected (max only): {frame_victims}")

    # --- No detection counter ---
    if len(frame_victims) == 0:
        no_victim_frame_count += 1
        print(f"[NO DETECTION] No victims for {no_victim_frame_count} frames")
    else:
        no_victim_frame_count = 0

    if no_victim_frame_count >= NO_VICTIM_RESET_FRAMES:
        print("[FAILSAFE] No victims detected for long duration. Resetting outputs.")
        reset_all()
        no_victim_frame_count = 0

    # --- Update history and process result ---
    victim_history.extend(frame_victims)
    frame_counter += 1

    if frame_counter >= FRAME_WINDOW:
        result = analyze_victims(victim_history)
        if result:
            most_common_label, percent = result
            print(f"[ANALYSIS] Most common: {most_common_label} ({percent:.1f}%)")
            average_results.append(most_common_label)
        else:
            print("[ANALYSIS] No consistent victim")

        frame_counter = 0
        victim_history.clear()

        if len(average_results) >= SEND_WINDOW:
            final_label = max(set(average_results), key=average_results.count)
            print(f"[CONFIRM] Final label to output: {final_label}")
            output_victim_pins(final_label)
            average_results.clear()

    print(f"[INFO] FPS: {clock.fps():.1f}")
    log_output_pins()
    gc.collect()
