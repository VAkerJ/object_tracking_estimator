import cv2
import time
import numpy as np

from .image_tools import crop_image

def kmeans(base_image, selected_area, iterC, num_clusters, verbose=1):
	# pre prossessing
	cropped_image, _, _ = crop_image(base_image, selected_area, factor=0)

	image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB)
	image = np.float32(image.reshape((-1,3)))

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # vet ej vad dehär gör

	# applicera kmeans
	start = time.time()
	_, label, center = cv2.kmeans(image, num_clusters, None, criteria, iterC, cv2.KMEANS_PP_CENTERS) # inte helt koll på denhär
	end = time.time()
	print("[INFO] applying Kmeans took {:.2f} seconds".format(end - start))

	res = np.uint8(center[label.flatten()])
	output = res.reshape((cropped_image.shape))

	if verbose > 0:
		cv2.imshow("Cropped image", cropped_image)
		cv2.imshow("Kmeans output", output)
		cv2.waitKey(1)

	return _, output, selected_area