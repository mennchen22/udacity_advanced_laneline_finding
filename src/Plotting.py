import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_results(original, output_imgage, save_file, out_gray=True):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=50)
    if out_gray:
        ax2.imshow(output_imgage, cmap='gray')
    else:
        ax2.imshow(output_imgage)
    ax2.set_title('Output Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(save_file, format='jpg')
    plt.show()


def add_polygon(img, points: np.array, color=(0, 255, 0), isClosed=True, thickness=2):
    """
    Add a polygon to the image
    :param color: RGB color
    :param img: RGB image
    :param points: [[x,y], [] ...]
    :return:
    """
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed, color, thickness=thickness)
    return img


def add_text(img, text, position=(100, 100), fontScale=1, fontColor=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    lineType = 2
    cv2.putText(img, text, position, font, fontScale, fontColor, lineType)
    return img
