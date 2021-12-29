class Rectangle():
	def __init__(self):
		self.p0, self.p1, self.ptemp = None, None, None

	def get_points(self):
		p0, p1 = self.p0, self.p1
		if p1 is None: p1 = self.ptemp
		return p0, p1

	def get_rec(self):
		p0, p1 = self.get_points()
		x_min = min(p0[0], p1[0])
		x_max = max(p0[0], p1[0])
		y_min = min(p0[1], p1[1])
		y_max = max(p0[1], p1[1])
		x_len = x_max-x_min
		y_len = y_max-y_min

		return (x_min, y_min, x_len, y_len)

	def is_active(self):
		active = bool(self.p0 is not None)
		if active: active = bool(self.p0!=self.p1)
		return active

	def is_finished(self):
		return self.p1 is not None

	def clear(self):
		self.p0, self.p1, self.ptemp = None, None, None

	def mouse_test(self, event, x, y, flags, param):
		if event == 1: #mouse1 click
			self.p0 = (x,y)

		elif event == 4: #mouse1 release
			self.p1 = (x,y)

		if self.p0 is not None: self.ptemp = (x,y)

def crop_image(base_image, selected_area, factor=0):
	if factor<0: factor=0 # ändra så att man kan minska?

	(x_min, y_min, x_len, y_len) = selected_area
	x_max, y_max = x_min+x_len, y_min+y_len

	if factor > 0:
		x = [x_min-int(x_len//factor), x_max+int(x_len//factor)]
		y = [y_min-int(y_len//factor), y_max+int(y_len//factor)]
	else: x, y = [x_min, x_max], [y_min, y_max]
	if x[0] < 0: x[0] = 0
	if y[0] < 0: y[0] = 0
	if x[1] > base_image.shape[1]: x[1] = base_image.shape[1]
	if y[1] > base_image.shape[0]: y[1] = base_image.shape[0]
	
	cropped_image = base_image[y[0]:y[1], x[0]:x[1]]
	new_selected_area = (x_min-x[0], y_min-y[0], x_len, y_len)
	return cropped_image, new_selected_area