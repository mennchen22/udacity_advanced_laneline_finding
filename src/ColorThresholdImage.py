import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.Plotting import plot_results


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel_x = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def image_dilation(img, kernel_size=3, iterations=1):
    """
    Image dilation
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=iterations)
    return img_dilation


def image_erosion(img, kernel_size=3, iterations=1):
    """
    Image erosion
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv2.erode(img, kernel, iterations=iterations)
    return img_dilation


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobel = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grad_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    binary_output = np.zeros_like(grad_sobel)
    binary_output[(grad_sobel >= thresh[0]) & (grad_sobel <= thresh[1])] = 1
    return binary_output


def remove_fractals(gray, kernel_size=3, iterations=1):
    # use a dilation / erosion convolution to remove fractals
    gray = image_erosion(gray, kernel_size, iterations)
    gray = image_dilation(gray, kernel_size, iterations)
    return gray


def combined_threshold(img, plot=False):
    """
    Combine different threshold convolutions
    :param image: RGB image
    :return:
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2]  # Better gray image from HSV - V channel
    # gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 2]
    gray_image = image_erosion(gray_image, 3, 1)
    gradx = abs_sobel_thresh(gray_image, orient='x', sobel_kernel=9, thresh=(20, 100))
    grady = abs_sobel_thresh(gray_image, orient='y', sobel_kernel=15, thresh=(20, 100))
    mag_binary = mag_thresh(gray_image, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray_image, sobel_kernel=15, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1))] = 1
    combined[(mag_binary == 1) & (dir_binary == 1)] = 1
    combined = image_dilation(combined, 3, 1)
    if plot:
        plot_convolution_results(img, gradx, grady, mag_binary, dir_binary, combined)
    return combined


def de_shadow_lab(img, threshold=(145, 200)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    B = lab[:, :, 2]
    binary = np.zeros_like(B)
    binary[(B > threshold[0]) & (B <= threshold[1])] = 1
    return binary


def white_line_detection_hls(img, thresh=(90, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary


def orange_lane_detection_hsv(img, thresh=(215, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    V = hsv[:, :, 2]
    binary = np.zeros_like(V)
    binary[(V > thresh[0]) & (V <= thresh[1])] = 1
    return binary


def white_lane_detection_luv(img, thresh=(215, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    L = luv[:, :, 0]
    binary = np.zeros_like(L)
    binary[(L > thresh[0]) & (L <= thresh[1])] = 1
    return binary


def pipeline_image_threshold_filter(img, plot=False):
    """
    A combination of different binary image transformation to detect lanes
    :param img: incoming image
    :param plot: Plot results as image subplot
    :return: binary image
    """
    combined_thresholds = combined_threshold(img, plot)
    remove_shadow = de_shadow_lab(img)
    orange_lane = orange_lane_detection_hsv(img)
    white_lane = white_lane_detection_luv(img)
    white_lane_2 = white_line_detection_hls(img)
    combined = np.zeros_like(combined_thresholds)

    # only add image information if less then 80% are white pixel
    white_pixel_white_thresh = 0.8 * combined.shape[0] * combined.shape[1]
    if np.sum(orange_lane == 1) < white_pixel_white_thresh:
        combined[(orange_lane == 1) & (combined_thresholds == 1)] = 1
    if np.sum(white_lane == 1) < white_pixel_white_thresh:
        combined[(white_lane == 1) & (combined_thresholds == 1)] = 1
    if np.sum(white_lane_2 == 1) < white_pixel_white_thresh:
        combined[(white_lane_2 == 1) & (combined_thresholds == 1)] = 1
    combined[remove_shadow == 1] = 1

    if plot:
        plot_pipe_results(img, combined_thresholds, remove_shadow, orange_lane, white_lane, white_lane_2, combined)
    return combined


def plot_pipe_results(original, combined_theshold, de_shadow, orange_lane, white_lan, white_lane_2, combined):
    # Plot the result
    f, axs = plt.subplots(4, 2, figsize=(24, 9 * 4))
    f.tight_layout()
    axs[0, 0].imshow(original)
    axs[0, 0].set_title('Original Image', fontsize=50)

    axs[0, 1].imshow(combined_theshold, cmap='gray')
    axs[0, 1].set_title('Combined Threshold', fontsize=50)

    axs[1, 0].imshow(de_shadow, cmap='gray')
    axs[1, 0].set_title('De Shadow', fontsize=50)

    axs[1, 1].imshow(orange_lane, cmap='gray')
    axs[1, 1].set_title('Orange Lane', fontsize=50)

    axs[2, 0].imshow(white_lan, cmap='gray')
    axs[2, 0].set_title('White LUV', fontsize=50)

    axs[2, 1].imshow(white_lane_2, cmap='gray')
    axs[2, 1].set_title('White HLS', fontsize=50)

    axs[3, 0].imshow(combined, cmap='gray')
    axs[3, 0].set_title('All combined', fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def plot_convolution_results(original, sobel_x, sobel_y, mag, dir, combined):
    # Plot the result
    f, axs = plt.subplots(3, 2, figsize=(24, 9 * 3))
    f.tight_layout()
    axs[0, 0].imshow(original)
    axs[0, 0].set_title('Original Image', fontsize=50)

    axs[0, 1].imshow(sobel_x, cmap='gray')
    axs[0, 1].set_title('Sobel X', fontsize=50)

    axs[1, 0].imshow(sobel_y, cmap='gray')
    axs[1, 0].set_title('Sobel Y', fontsize=50)

    axs[1, 1].imshow(mag, cmap='gray')
    axs[1, 1].set_title('Magnitude', fontsize=50)

    axs[2, 0].imshow(dir, cmap='gray')
    axs[2, 0].set_title('Direction', fontsize=50)

    axs[2, 1].imshow(combined, cmap='gray')
    axs[2, 1].set_title('All combined', fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def test_all_test_images():
    for pos in range(1, 7):
        test_image = f"../test_images/test{pos}.jpg"
        img = mpimg.imread(test_image)
        out_img = pipeline_image_threshold_filter(img)
        f, (axs1, axs2) = plt.subplots(1, 2, figsize=(24, 9))
        axs1.imshow(img)
        axs1.set_title('Original', fontsize=50)
        axs2.imshow(out_img, cmap='gray')
        axs2.set_title('All combined', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()


if __name__ == "__main__":
    # test_all_test_images()
    # image paths
    test_image = "../test_images/test1.jpg"
    output_image_com = "../output_images/binary_combination_threshold.jpg"
    output_image_hsl = "../output_images/hsl_threshold.jpg"
    # read image
    img = mpimg.imread(test_image)
    out_img = pipeline_image_threshold_filter(img, plot=True)
    # plot_results(img, hsl_binary, output_image_hsl, out_gray=True)
    plot_results(img, out_img, output_image_com, out_gray=True)
