class Rectangle():
	def __init__(self):
		self.p0, self.p1, self.ptemp = None, None, None

	def get_points(self):
		p0, p1 = self.p0, self.p1
		if p1 is None: p1 = self.ptemp
		return p0, p1

	def is_active(self):
		return self.p0 is not None

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