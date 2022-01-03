import cv2

class Tracker():
	def __init__(self, image, selected_area):
		self.tracker = cv2.TrackerCSRT_create()
		self.tracker.init(image, selected_area)

	def update(self, image):
		success, selected_area = self.tracker.update(image)
		if success:
			p0 = (int(selected_area[0]), int(selected_area[1]))
			p1 = (int(p0[0] + selected_area[2]), int(p0[1] + selected_area[3]))
			cv2.rectangle(image, p0, p1, (255,0,0), 1)
		else:
			print("[INFO] CV2 tracking method, tracking failure")

		return success, image

	