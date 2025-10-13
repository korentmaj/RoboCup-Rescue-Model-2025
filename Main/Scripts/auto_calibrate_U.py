import sensor, image, time, math
import ml
from machine import Pin, LED
import neopixel
import sys
import select
from ml.utils import NMS

# ---------------- SETUP ----------------
# Pins and hardware
led_pin = Pin('P7', Pin.OUT)
n = neopixel.NeoPixel(led_pin, 5, bpp=4)
blue_led = LED("LED_BLUE")

# Initial settings
white_brightness = 120
exposure_us = 8000
brightness_step = 10
exposure_step = 1000
target_label = "H"

# Set LED
def set_white_led(level):
    for i in range(5):
        n[i] = (0, 0, 0, level)
    n.write()

set_white_led(white_brightness)

# Camera init
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(True)
sensor.set_auto_exposure(False, exposure_us=exposure_us)
sensor.skip_frames(time=1000)

# Model
model = ml.Model("trained")
threshold_list = [(math.ceil(0.5 * 255), 255)]

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

# Poller to read serial input
poller = select.poll()
poller.register(sys.stdin, select.POLLIN)

print("\n--- Manual U Calibration ---")
print("w/s = LED ++/--, e/d = Exposure ++/--, q = Quit\n")

# ---------------- MAIN LOOP ----------------
while True:
    img = sensor.snapshot()

    # Stats
    stats = img.get_statistics()
    mean_luma = stats.l_mean()
    stdev = stats.l_stdev()

    max_score = 0
    detected = False

    for i, detection_list in enumerate(model.predict([img], callback=fomo_post_process)):
        if i == 0:
            continue
        label = model.labels[i] if i < len(model.labels) else f"class_{i}"
        for (x, y, w, h), score in detection_list:
            if label == target_label:
                img.draw_rectangle((x, y, w, h), color=255)
                img.draw_string(x, y - 10, f"{label} {score:.2f}", mono_space=False)
                detected = True
                if score > max_score:
                    max_score = score

    print(f"[INFO] LED={white_brightness}, EXP={exposure_us} µs, Luma={mean_luma}, Stdev={stdev}, Score={max_score:.2f}")

    # Handle key input
    if poller.poll(0):
        key = sys.stdin.read(1)

        if key == 'w':
            white_brightness = min(255, white_brightness + brightness_step)
            set_white_led(white_brightness)
            print(f"[SET] LED ++ -> {white_brightness}")

        elif key == 's':
            white_brightness = max(0, white_brightness - brightness_step)
            set_white_led(white_brightness)
            print(f"[SET] LED -- -> {white_brightness}")

        elif key == 'e':
            exposure_us += exposure_step
            sensor.set_auto_exposure(False, exposure_us=exposure_us)
            print(f"[SET] Exposure ++ -> {exposure_us} µs")

        elif key == 'd':
            exposure_us = max(1000, exposure_us - exposure_step)
            sensor.set_auto_exposure(False, exposure_us=exposure_us)
            print(f"[SET] Exposure -- -> {exposure_us} µs")

        elif key == 'q':
            print("[QUIT] Exiting calibration.")
            break
