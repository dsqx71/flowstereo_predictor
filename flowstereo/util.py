import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_velocity_vector(flow):

    img = np.ones(flow.shape[:2] + (3,))
    for i in range(0, img.shape[0] - 20, 30):
        for j in range(0, img.shape[1] - 20, 30):
            try:
                # opencv 3.1.0
                if flow.shape[-1] == 2:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                    (150, 0, 0), 2)
                else:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)

            except AttributeError:
                # opencv 2.4.8
                if flow.shape[-1] == 2:
                    cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                             (150, 0, 0), 2)
                else:
                    cv2.line(img, pt1=(j, i), pt2=(j + int(round(flow[i, j])), i), color=(150, 0, 0), thickness=1)

    plt.figure()
    plt.imshow(img)
    plt.title('velocity vector')


def flow2color(flow):
    """
        plot optical flow
        optical flow have 2 channel : u ,v indicate displacement
    """
    hsv = np.zeros(flow.shape[:2] + (3,)).astype(np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    plt.figure()
    plt.imshow(rgb)
    plt.title('optical flow')