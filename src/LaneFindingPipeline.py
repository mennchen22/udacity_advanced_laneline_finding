import cv2
import matplotlib.image as mpimg
from src.CameraCalibration import CameraCalibration
from src.ColorThresholdImage import white_line_detection_hls
from src.LanePolynomial import RoadLineFit
from src.PerspectiveTransformation import road_perspective_transformation
from src.Plotting import plot_results, add_polygon, add_text


def add_results_to_image(original_img, polyline_img, car_offset, lane_radius):
    """
    Reverse the perspective transformation and add information as text onto the image
    :param original_img:
    :param polyline_img: Result image wraped
    :param car_offset: Car offset information
    :param lane_radius: Lane radius information
    :return: image
    """
    # print image back to original image
    unwrap_img, _ = road_perspective_transformation(polyline_img, show_transformation_line=False, reverse=True)
    # add to original image with additional data
    out_image = cv2.addWeighted(original_img, 1, unwrap_img, 0.3, 0, dtype=cv2.CV_64F)
    if car_offset is None:
        add_text(out_image, "[Car Center Offset] --- ", (30, 50))
    else:
        side = "right" if car_offset > 0 else "left"
        car_offset_text = "[Car Center Offset] {:.2f} m to the {}".format(abs(car_offset), side)
        add_text(out_image, car_offset_text, (30, 50))
    if lane_radius is None:
        add_text(out_image, "[Lane radius] ---", (30, 120))
    else:
        lane_radius_text = f"[Lane radius] {lane_radius} m"
        add_text(out_image, lane_radius_text, (30, 120))
    return out_image


if __name__ == "__main__":
    # image paths
    test_image = "../test_images/test6.jpg"
    output_image = "../output_images/lane_line_detection_pipe.jpg"
    output_image_2 = "../output_images/undistorted_image.jpg"
    # first calculate the camera distortion
    camera = CameraCalibration(calibration_dir="../camera_cal/calibration*.jpg", nx=9, ny=6)
    # Load an image
    img = mpimg.imread(test_image)
    # create an undistorted image
    undistorted_img = camera.undistorted_image(img)
    plot_results(img, undistorted_img, output_image_2, out_gray=False)
    # make a perspective transformation
    wrap_img, _ = road_perspective_transformation(undistorted_img, show_transformation_line=False)
    # create an binary image
    binary_image_wrap = white_line_detection_hls(wrap_img, thresh=(120, 255))
    # calculate segments
    road_line_fit = RoadLineFit()
    poly_img = road_line_fit.lane_line_pipe(binary_warped=binary_image_wrap)
    # calculate with a new image
    poly_img_second = road_line_fit.lane_line_pipe(binary_warped=binary_image_wrap)
    # print road line radius
    car_offset = road_line_fit.calculate_car_center_offset_m(car_center_pos=poly_img.shape[1] / 2)
    lane_radius = road_line_fit.calculate_lane_radius_m()
    # transform the polynomial back to the original
    result_img = add_results_to_image(undistorted_img, poly_img_second, car_offset, lane_radius)
    plot_results(undistorted_img, result_img, output_image, out_gray=False)
