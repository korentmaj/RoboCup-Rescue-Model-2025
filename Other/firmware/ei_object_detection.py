# RoboCup Rescue Maze: Dual Victim Detection (Letters + Colors)

import sensor, time, ml, math, image
from ml.utils import NMS

# ---------------- Setup ----------------
sensor.reset()
sensor.set_pixformat(sensor.RGB565)      # Use RGB for color detection
sensor.set_framesize(sensor.QQVGA)       # 160x120
sensor.skip_frames(time=2000)

# ---------------- Load FOMO Model for Letters ----------------
model = ml.Model("trained")
print("Model loaded:", model)

NUM_CLASSES = model.output_shape[0][-1]
model.labels = ["background", "H", "S", "U"]  # Override for RoboCup

min_confidence = 0.4
threshold_list = [(math.ceil(min_confidence * 255), 255)]
colors = [255, 192, 128, 64]

# ---------------- FOMO Post Processing ----------------
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

# ---------------- Color Thresholds ----------------
# Tune these for your exact field lighting
red_thresh    = (30, 100, 15, 127, 15, 127)     # L, A, B for red
yellow_thresh = (70, 100, -10, 10, 60, 127)     # L, A, B for yellow
green_thresh  = (30, 100, -128, -10, 20, 60)    # L, A, B for green

color_labels = {
    "red": red_thresh,
    "yellow": yellow_thresh,
    "green": green_thresh
}

# ---------------- Main Loop ----------------
clock = time.clock()
while True:
    clock.tick()
    img = sensor.snapshot()

    # --- FOMO LETTER DETECTION ---
    for i, detections in enumerate(model.predict([img], callback=fomo_post_process)):
        if i == 0: continue
        if len(detections) == 0: continue

        label = model.labels[i] if i < len(model.labels) else f"class_{i}"
        for (x, y, w, h), score in detections:
            cx, cy = math.floor(x + w / 2), math.floor(y + h / 2)
            print(f"[LETTER] {label} at ({cx},{cy}) | score={score:.2f}")
            img.draw_rectangle((x, y, w, h), color=colors[i % len(colors)])

    # --- COLOR VICTIM DETECTION ---
    img_lab = img.to_lab()
    for color_name, threshold in color_labels.items():
        blobs = img_lab.find_blobs([threshold], area_threshold=100, pixels_threshold=80)
        for b in blobs:
            cx, cy = b.cx(), b.cy()
            print(f"[COLOR] {color_name} at ({cx},{cy}) | size={b.pixels()}")
            img.draw_circle((cx, cy, 8), color=128)

    print("FPS:", clock.fps())
