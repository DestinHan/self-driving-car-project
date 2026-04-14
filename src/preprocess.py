import cv2
import random
import numpy as np

def preprocess_image(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    return img


def balance_data(data, bin_size=1000):
	bins = 35

	_, bin_edges = np.histogram(data["steering"], bins=bins)

	extra_vals = []
	for i in range(bins):
		left = bin_edges[i]
		right = bin_edges[i+1]

		bin_vals = []
		for j, row in data.iterrows():

			# if inside the bin
			if row["steering"] >= left and row["steering"] < right:
				bin_vals.append(j)

		if len(bin_vals) > bin_size:
			leftover = len(bin_vals) - bin_size
			drop = random.sample(bin_vals, leftover)
			
			extra_vals.extend(drop)

	# delete and fixes index
	data = data.drop(extra_vals).reset_index(drop=True)
	return data

    