import cv2
import matplotlib.image as mpimg
import numpy as np

from src.CameraCalibration import CameraCalibration
from src.ColorThresholdImage import white_line_detection_hls
from src.PerspectiveTransformation import road_perspective_transformation
from src.Plotting import plot_results, add_polygon


class RoadLineFit:
    def __init__(self, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
        self.left_fit_x = None
        self.right_fit_x = None
        self.plot_y = None
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None
        self.left_fit = None
        self.right_fit = None
        self.saved_left_fit = None
        self.saved_right_fit = None
        self.failed_process = False
        self.bottom_image_pixel = None
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix
        self.last_lane_radius_left = None
        self.last_lane_radius_right = None

    def reset_values(self):
        self.left_fit_x = None
        self.right_fit_x = None
        self.plot_y = None
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None
        self.left_fit = None if self.saved_left_fit is None else self.saved_left_fit
        self.right_fit = None if self.saved_right_fit is None else self.saved_right_fit
        self.failed_process = False

    def get_road_fits(self):
        return self.left_fit, self.right_fit

    def calculate_lane_radius_m(self, max_change_threshold=500, max_equality_threshold=500):
        left_fit_cr = np.polyfit(self.plot_y * self.ym_per_pix, self.left_fit_x * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.plot_y * self.ym_per_pix, self.right_fit_x * self.xm_per_pix, 2)
        poly_y_cr = self.plot_y * self.ym_per_pix
        y_eval = np.max(poly_y_cr)
        left_curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curve_rad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # print(f"[Radius Lanes] [m] Left: {left_curve_rad} | Right: {right_curve_rad}")
        # check if the lane radius differ from last result. Can restart pipeline

        if not (self.last_lane_radius_left is None):
            change = np.absolute([self.last_lane_radius_right - right_curve_rad,
                                  self.last_lane_radius_left - left_curve_rad])
            is_over_threshold = np.any(change > max_change_threshold)
            lanes_equality_over_threshold = np.absolute(
                [self.last_lane_radius_right - self.last_lane_radius_left])[0] > max_equality_threshold
            if is_over_threshold or lanes_equality_over_threshold:
                print(f"[Lane Radius] Change/Lane Equality is over given threshold. Rest pipeline")
                self.reset_values()
        else:
            self.last_lane_radius_left = left_curve_rad
            self.last_lane_radius_right = right_curve_rad

        return np.min([left_curve_rad, right_curve_rad])

    def calculate_lane_radius_pix(self):
        y_eval = np.max(self.plot_y)
        left_curve_rad = ((1 + (2 * self.left_fit[0] * y_eval + self.left_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.left_fit[0])
        right_curve_rad = ((1 + (2 * self.right_fit[0] * y_eval + self.right_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.right_fit[0])
        # print(f"[Radius Lanes] [Pix] Left: {left_curve_rad} | Right: {right_curve_rad}")
        return np.min([left_curve_rad, right_curve_rad])

    def check_lane_gap_distance(self, min_length=2.00, max_length=3.75 + 0.25):
        left_lane_pos = self.left_fit_x[-1]
        right_lane_pos = self.right_fit_x[-1]
        gap_distance = ((right_lane_pos - left_lane_pos) / 2) * self.xm_per_pix
        print(f"Lane gap {gap_distance} m")
        if (max_length < gap_distance) or (gap_distance < min_length):
            print(f"Lane doesn't fit plausible gap distance")
            self.reset_values()

    def check_result_exists(self):
        if (self.left_fit_x is None) or (self.right_fit_x is None):
            return False
        return True

    def calculate_car_center_offset_m(self, car_center_pos):
        left_lane_pos = self.left_fit_x[-1]
        right_lane_pos = self.right_fit_x[-1]
        lane_center = left_lane_pos + (right_lane_pos - left_lane_pos) / 2
        offset = (lane_center - car_center_pos) * self.xm_per_pix
        # side = "right" if offset > 0 else "left"
        # print("[Car Center Offset] {:.2f} m to the {}".format(abs(offset), side))
        return offset

    def lane_line_pipe(self, binary_warped, plot=False):
        if not (self.bottom_image_pixel is None):
            print("[Lane Line Pipeline] Take bottom lane saved part")
            binary_warped = cv2.bitwise_or(binary_warped, self.bottom_image_pixel)
        # check if we have a previous calculation
        if (self.left_fit is None) or (self.right_fit is None):
            output_image = self.lane_line_detection_from_scratch(binary_warped, plot)
        # if we have one, we search around the old line
        else:
            print("[Lane Line Pipeline] Try detection from previous lines")
            output_image = self.search_around_poly(binary_warped=binary_warped, plot_lines=plot)
            # we can retry a lane search by scratch
            if self.failed_process:
                output_image = self.lane_line_detection_from_scratch(binary_warped, plot)
        if not plot:
            output_image = np.zeros_like(output_image)
        output_image = self.add_polynomial_lines(output_image, plot_road_space=not plot)
        return output_image

    def lane_line_detection_from_scratch(self, binary_wrapped, plot, half=False):
        print("[Lane Line Pipeline] Start from scratch")
        output_image = self.find_lane_pixels(binary_warped=binary_wrapped, plot_lines=plot)
        self.failed_process = not self.fit_poly(binary_wrapped.shape)
        if plot:
            output_image = self.add_color_lines(output_image, False)
        # we can not retry a lane search
        if self.failed_process and not half:
            print("[Lane Line Pipe] Lost this line, retry with half image")
            ind = np.int(binary_wrapped.shape[1] / 2)  # get width/2 value of the image for indexing
            binary_wrapped[:, 0:ind] = 0  # set top half pixel to black
            self.lane_line_detection_from_scratch(binary_wrapped, plot, half=True)
        elif self.failed_process:
            print("[Lane Line Pipe] Lost this line in half, stop")
            self.reset_values()
        return output_image

    def find_lane_pixels(self, binary_warped, num_windows=12, margin=100, min_pix=50, plot_lines=False):
        """
        Find lane pixels based on the image side (left, right) with a histogram and sliding windows
        :param binary_warped: bird view image of lane lines
        :param num_windows: Choose the number of sliding windows
        :param margin:  Set the width of the windows +/- margin
        :param min_pix:  Set minimum number of pixels found to recenter window
        :param plot_lines: Show the window boxes and lane pixels
        :return: image
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // num_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Create an output image (RGB) to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        # Step through the windows one by one
        for window in range(num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if plot_lines:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low)
                              & (nonzeroy < win_y_high)
                              & (nonzerox >= win_xleft_low)
                              & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low)
                               & (nonzeroy < win_y_high)
                               & (nonzerox >= win_xright_low)
                               & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            # (`rightx_current` or `leftx_current`) on their mean position
            if len(good_left_inds) > min_pix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > min_pix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError as e:
            # Avoids an error if the above is not implemented fully
            print(f"[Find Lane Pixels] Failed within process: {e}")

        # Extract left and right line pixel positions
        self.left_x = nonzerox[left_lane_inds]
        self.left_y = nonzeroy[left_lane_inds]
        self.right_x = nonzerox[right_lane_inds]
        self.right_y = nonzeroy[right_lane_inds]

        return out_img

    def search_around_poly(self, binary_warped, margin=100, plot_lines=False):
        """
        Search near the old lane lines for lane pixels. Faster then the 'from scratch' approach
        :param binary_warped: Wrapped image
        :param margin: Search region around the old lanes
        :param plot_lines: Plot the margin region
        :return: image
        """
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy +
                                       self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0] * (nonzeroy ** 2) +
                                                                                  self.left_fit[1] * nonzeroy +
                                                                                  self.left_fit[
                                                                                      2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy +
                                        self.right_fit[2] - margin)) & (
                                   nonzerox < (self.right_fit[0] * (nonzeroy ** 2) +
                                               self.right_fit[1] * nonzeroy + self.right_fit[
                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        self.left_x = nonzerox[left_lane_inds]
        self.left_y = nonzeroy[left_lane_inds]
        self.right_x = nonzerox[right_lane_inds]
        self.right_y = nonzeroy[right_lane_inds]

        # Fit new polynomials and check + save success
        self.failed_process = not self.fit_poly(binary_warped.shape)

        if plot_lines:
            out_img = self.add_color_lines(binary_warped)
            window_img = np.zeros_like(out_img)

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([self.left_fit_x - margin, self.plot_y]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fit_x + margin,
                                                                            self.plot_y])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([self.right_fit_x - margin, self.plot_y]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fit_x + margin,
                                                                             self.plot_y])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            out_image = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        else:
            out_image = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        return out_image

    def fit_poly(self, img_shape):
        """
        Calculate the polynomials and the x,y values for each lane
        :param img_shape:
        :return: success
        """
        if (self.left_y is None) or (self.right_y is None) or (self.left_x is None) or (self.right_x is None):
            return False
        # Fit a second order polynomial to each with np.polyfit() #
        self.left_fit = np.polyfit(self.left_y, self.left_x, 2)
        self.right_fit = np.polyfit(self.right_y, self.right_x, 2)
        # Generate x and y values for plotting
        self.plot_y = np.linspace(0, img_shape[0] - 1, img_shape[0])
        # Calc both polynomials using ploty, left_fit and right_fit
        try:
            self.left_fit_x = self.left_fit[0] * self.plot_y ** 2 + self.left_fit[1] * self.plot_y + self.left_fit[2]
            self.right_fit_x = self.right_fit[0] * self.plot_y ** 2 + self.right_fit[1] * self.plot_y + self.right_fit[
                2]
            return True
        except ValueError:
            return False

    def add_color_lines(self, image_input, binary=True):
        """
        Add color lines based on the left and right lane indices
        :param image_input:
        :param binary: image is binary
        :return: image
        """
        if binary:
            out_img = np.dstack((image_input, image_input, image_input)) * 255
        else:
            out_img = image_input
        if (self.left_y is None) or (self.right_y is None) or (self.left_x is None) or (self.right_x is None):
            print("[Colored Lines] Cannot create image with no line data")
            return out_img
        # Color in left and right line pixels
        out_img[self.left_y, self.left_x] = [255, 0, 0]
        out_img[self.right_y, self.right_x] = [0, 0, 255]
        return out_img

    def add_polynomial_lines(self, image_input, plot_road_space=True):
        """
        Add the polynomial lines to the imae based on the calculated x,y values from fit
        :param image_input:
        :param plot_road_space: Add color between the two lines
        :return: image
        """
        if (self.plot_y is None) or (self.left_fit_x is None) or (self.right_fit_x is None):
            print("[Colored Lines] Cannot create image with no line data")
            return image_input
        if plot_road_space:
            # Fill the area between the lines
            left_line_window = np.array([np.transpose(np.vstack([self.left_fit_x, self.plot_y]))])
            right_line_window = np.array([np.flipud(np.transpose(np.vstack([self.right_fit_x, self.plot_y])))])
            road_space = np.hstack((left_line_window, right_line_window))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(image_input, np.int_([road_space]), (0, 255, 0))

        # Color in left and right line pixels
        pts_left = []
        pts_right = []
        for pos in range(len(self.plot_y)):
            pts_left.append([self.left_fit_x[pos], self.plot_y[pos]])
            pts_right.append([self.right_fit_x[pos], self.plot_y[pos]])
        out_image = add_polygon(image_input, pts_left, color=(255, 255, 0), isClosed=False, thickness=10)
        out_image = add_polygon(out_image, pts_right, color=(255, 255, 0), isClosed=False, thickness=10)

        return out_image

    def save_values(self, bin_image, bottom_pixel_copy=200):
        print("[Lane Line Pipe] Save values and bottom image section")
        max_height = bin_image.shape[1]
        self.bottom_image_pixel = np.zeros_like(bin_image)
        self.bottom_image_pixel[:, (max_height - bottom_pixel_copy):max_height] = bin_image[:, (
                                                                                                           max_height - bottom_pixel_copy):max_height]


if __name__ == "__main__":
    # image paths
    test_image = "../test_images/test1.jpg"
    output_image = "../output_images/lane_segment_detection_1.jpg"
    output_image_2 = "../output_images/lane_segment_detection_2.jpg"
    # first calculate the camera distortion
    camera = CameraCalibration(calibration_dir="../camera_cal/calibration*.jpg", nx=9, ny=6)
    # Load an image
    img = mpimg.imread(test_image)
    # create an undistorted image
    undistorted_img = camera.undistorted_image(img)
    # make a perspective transformation
    wrap_img, input_image = road_perspective_transformation(undistorted_img, show_transformation_line=False)
    # create an binary image
    binary_image_wrap = white_line_detection_hls(wrap_img, thresh=(120, 255))
    # calculate segments
    road_line_fit = RoadLineFit()
    poly_img = road_line_fit.lane_line_pipe(binary_warped=binary_image_wrap, plot=True)
    plot_results(input_image, poly_img, output_image, out_gray=True)
    # calculate with a new image
    poly_img_second = road_line_fit.lane_line_pipe(binary_warped=binary_image_wrap, plot=True)
    plot_results(input_image, poly_img_second, output_image_2, out_gray=True)
    # print road line radius
    road_line_fit.calculate_car_center_offset_m(car_center_pos=poly_img.shape[1] / 2)
    road_line_fit.calculate_lane_radius_m()
    road_line_fit.calculate_lane_radius_pix()
