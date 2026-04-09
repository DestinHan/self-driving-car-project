import cv2
import numpy as np
import random


def augment_image(img, steering_angle, training_model=True):
    if not training_model:
        return img, steering_angle

    # flipping
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        steering_angle = -steering_angle

    # brightness
    if random.random() < 0.5:
        # hsv is better for brightness
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        np_img = np.array(hsv_img, dtype=np.float64)
        factor = 0.5 + np.random.uniform()  # range from 0.5 to 1.5
        np_img[:, :, 2] = np_img[:, :, 2] * factor
        np_img = np.clip(np_img, 0, 255)
        img = np.array(np_img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # zooming
    if random.random() < 0.5:
        h, w = img.shape[:2]
        factor = 1.0 + (np.random.rand() * 0.5)

        if factor > 1.0:
            new_h, new_w = int(h / factor), int(w / factor)
            if new_w < w and new_h < h:
                left_right_point_x = np.random.randint(0, w - new_w + 1)
                top_bottom_point_y = np.random.randint(0, h - new_h + 1)
                cropped_img = img[
                    top_bottom_point_y : top_bottom_point_y + new_h,
                    left_right_point_x : left_right_point_x + new_w,
                ]
                img = cv2.resize(cropped_img, (w, h))

    # panning
    if random.random() < 0.5:
        h, w = img.shape[:2]

        # left right shift
        x_shift = int((np.random.rand() - 0.5) * 0.2 * w)
        # top bottom shift
        y_shift = int((np.random.rand() - 0.5) * 0.2 * h)

        # shift matrix
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        img = cv2.warpAffine(img, M, (w, h))

    # rotation
    if random.random() < 0.5:
        h, w = img.shape[:2]

        angle = (np.random.rand() - 0.5) * 20

        # rotation matrix
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

    return img, steering_angle
