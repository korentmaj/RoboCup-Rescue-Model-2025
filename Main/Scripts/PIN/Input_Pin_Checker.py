import  time
from machine import Pin, LED




green_led = LED("LED_GREEN")

# ---------------- Pin P3 as Input ----------------
p3_in = Pin('P3', Pin.IN, Pin.PULL_DOWN)
time.sleep_ms(100)
# Check input and control LED at startup


while True:
    if p3_in.value():
        green_led.on()
        time.sleep_ms(100)
    else:
        green_led.off()

