import sensor
import time
import image

# -------- CONFIG --------
threshold = (0, 100)  # <<< Set your threshold (L min, L max)
invert = False        # <<< Set to True if you want to invert black/white
# ------------------------

sensor.reset()  # Reset and initialize the sensor
sensor.set_pixformat(sensor.GRAYSCALE)  # Grayscale is needed for binary
sensor.set_framesize(sensor.QVGA)  # Set frame size to QVGA (320x240)
sensor.skip_frames(time=2000)  # Let the sensor adjust
clock = time.clock()  # Create a clock object to track FPS

def apply_binary(img):
    img.binary([threshold], invert=invert)  # Apply binary thresholding with optional invert

while True:
    clock.tick()
    img = sensor.snapshot()
    apply_binary(img)  # Apply binary threshold + optional invert
    print(clock.fps())
