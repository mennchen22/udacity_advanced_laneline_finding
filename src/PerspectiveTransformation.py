# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
import cv2
import numpy as np
import matplotlib.image as mpimg

from src.CameraCalibration import distortion_from_calibration_images, cal_undistort_2, CameraCalibration
from src.Plotting import plot_results, add_polygon

'''
### First attempt ###
 # pre tuned source positions based on straight road lane image
    image_fix_point_ratio = 0.36
    top_bar_length_left = img_size[0] / 11
    top_bar_length_right = img_size[0] / 10
    bottom_side_offset_left = img_size[0] / 6
    bottom_side_offset_right = img_size[0] / 8
    src = np.float32([
        [img_size[0] / 2 - (top_bar_length_left / 2), img_size[1] * (1 - image_fix_point_ratio)],
        [img_size[0] / 2 + (top_bar_length_right / 2), img_size[1] * (1 - image_fix_point_ratio)],
        [img_size[0] - bottom_side_offset_right, img_size[1]],
        [0 + bottom_side_offset_left, img_size[1]]
    ], dtype=np.int32)
    
### Second Attempt ###
 # pre tuned source positions based on straight road lane image
    image_fix_point_ratio = 0.40
    top_bar_length_left = img_size[0] / 30
    top_bar_length_right = img_size[0] / 29
    bottom_side_offset_left = img_size[0] / 6
    bottom_side_offset_right = img_size[0] / 8
    src = np.float32([
        [img_size[0] / 2 - (top_bar_length_left / 2), img_size[1] * (1 - image_fix_point_ratio)],
        [img_size[0] / 2 + (top_bar_length_right / 2), img_size[1] * (1 - image_fix_point_ratio)],
        [img_size[0] - bottom_side_offset_right, img_size[1]],
        [0 + bottom_side_offset_left, img_size[1]]
    ], dtype=np.int32)
'''


def road_perspective_transformation(input_image, offset=200, show_transformation_line=False, reverse=False):
    """
    make a perspective transformation from a front view lane image to birds view
    :param show_transformation_line: Add the transformation line to the image
    :param image: RGB image
    :param offset: image side pixel offset to lanes in resulting image
    :return: RGB image transformed and the input image with the transformation polygon
    """
    image = input_image.copy()
    # Grab the image shape
    img_size = (image.shape[1], image.shape[0])

    # pre tuned source positions based on straight road lane image
    image_fix_point_ratio = 0.36
    top_bar_length_left = img_size[0] / 11
    top_bar_length_right = img_size[0] / 10
    bottom_side_offset_left = img_size[0] / 6
    bottom_side_offset_right = img_size[0] / 8
    src = np.float32([
        [img_size[0] / 2 - (top_bar_length_left / 2), img_size[1] * (1 - image_fix_point_ratio)],
        [img_size[0] / 2 + (top_bar_length_right / 2), img_size[1] * (1 - image_fix_point_ratio)],
        [img_size[0] - bottom_side_offset_right, img_size[1]],
        [0 + bottom_side_offset_left, img_size[1]]
    ], dtype=np.int32)
    if show_transformation_line:
        image = add_polygon(image, src)

    # destination points with an offset added to the x axis
    dst = np.float32([[offset, 0],  # top left
                      [img_size[0] - offset, 0],  # top right
                      [img_size[0] - offset, img_size[1]],  # bottom right
                      [offset, img_size[1]]])  # bottom left

    # Given src and dst points, calculate the perspective transform matrix
    if reverse:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, img_size)

    # add the src lines to img
    image_with_lines = add_polygon(image, src)

    # Return the resulting image and matrix
    return warped, image_with_lines


if __name__ == "__main__":
    # image paths
    test_image = "../test_images/straight_lines2.jpg"
    output_image = "../output_images/perspective_transform.jpg"
    output_image_2 = "../output_images/perspective_transform_reversed.jpg"
    # first calculate the camera distortion
    camera = CameraCalibration(calibration_dir="../camera_cal/calibration*.jpg", nx=9, ny=6)
    # Load an image
    img = mpimg.imread(test_image)
    # create an undistorted image
    undistorted_img = camera.undistorted_image(img)
    # make a perspective transformation
    wrap_img, input_image = road_perspective_transformation(undistorted_img, show_transformation_line=False)
    plot_results(input_image, wrap_img, output_image, out_gray=False)
    # Result : An perspective transformation with a preset polygon over the source image can give good results
    # on the test images. This have to be tested and likely tuned or replaced with an dynamic algorithm in a
    # real case scenario.
    # now reverse
    reversed_img, _ = road_perspective_transformation(wrap_img, show_transformation_line=False, reverse=True)
    plot_results(input_image, reversed_img, output_image_2, out_gray=False)
