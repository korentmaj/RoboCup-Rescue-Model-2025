import sensor, time, ml, math, image, gc, ustruct
from machine import UART, Pin, LED
from ml.utils import NMS
sensor.reset()
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_whitebal(False)
sensor.set_auto_gain(False)
uart = UART(3, 115200)
p2 = Pin('P2', Pin.OUT)
p2.low()
r = LED("LED_RED")
g = LED("LED_GREEN")
b = LED("LED_BLUE")
model = ml.Model("trained")
print(model)
min_confidence = 0.4
threshold_list = [(math.ceil(min_confidence * 255), 255)]
red_thresh	= (5, 100, 16, 127, -2, 31)
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
def blink_white_led_once():
	r.on()
	g.on()
	b.on()
	time.sleep_ms(300)
	r.off()
	g.off()
	b.off()
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
def most_common_label(labels):
	if not labels:
		return None, 0
	counts = {}
	for label in labels:
		counts[label] = counts.get(label, 0) + 1
	most_common = max(counts, key=counts.get)
	percent = (counts[most_common] / len(labels)) * 100
	return most_common, percent
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
	frame_victims.extend(detect_color_blobs(img))
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
				p2.high()
				print("[P2] HIGH - Sending victim")
				try:
					packet = ustruct.pack("<B", victim_id)
					uart.write(packet)
					print("[UART] Sent victim ID:", victim_id)
				except Exception as e:
					print("[UART ERROR]", e)
				blink_white_led_once()
				p2.low()
				print("[P2] LOW - Done sending")
				victim_votes.clear()
			else:
				print("[VICTIM ERROR] Unknown victim label:", label)
	gc.collect()