import time
from machine import Pin
from pyb import LED

# --- Pin Setup ---
p2 = Pin('P2', Pin.OUT)
p3 = Pin('P3', Pin.OUT)
p4 = Pin('P4', Pin.OUT)
p5 = Pin('P5', Pin.OUT)

# --- LED Setup ---
red_led = LED(1)    # RED
green_led = LED(2)  # GREEN
blue_led = LED(3)   # BLUE

# --- Helper to turn all off ---
def reset_all():
    for pin in [p2, p3, p4, p5]:
        pin.low()
    red_led.off()
    green_led.off()
    blue_led.off()

# --- Main Loop ---
while True:
    # Step 1: P2 + WHITE
    reset_all()
    print("[CYCLE] P2 HIGH - WHITE ON (all LEDs)")
    p2.high()
    red_led.on()
    green_led.on()
    blue_led.on()
    time.sleep(5)

    # Step 2: P3 + RED
    reset_all()
    print("[CYCLE] P3 HIGH - RED ON")
    p3.high()
    red_led.on()
    time.sleep(5)

    # Step 3: P4 + BLUE
    reset_all()
    print("[CYCLE] P4 HIGH - BLUE ON")
    p4.high()
    blue_led.on()
    time.sleep(5)

    # Step 4: P5 + GREEN
    reset_all()
    print("[CYCLE] P5 HIGH - GREEN ON")
    p5.high()
    green_led.on()
    time.sleep(5)
