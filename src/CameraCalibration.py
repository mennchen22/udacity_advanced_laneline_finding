import numpy as np
import cv2
import glob

from src.Plotting import plot_results
import matplotlib.image as mpimg


class CameraCalibration:
    """
    This class stores the calibration values once
    """
    def __init__(self, calibration_dir, nx, ny):
        # first calculate the camera distortion
        self.mtx, self.dist = distortion_from_calibration_images(calibration_dir, nx, ny, show=False)

    def undistorted_image(self, img):
        return cal_undistort_2(img, self.mtx, self.dist)


# noinspection PySimplifyBooleanCheck
def distortion_from_calibration_images(calibration_folder, nx=9, ny=6, show=False, iterations=10):
    """
    Calculate the distortion correction matrix from calibration images
    Check if enogh images are given
    :param calibration_folder: folder containing all calibration images as jgp starting with 'calibration*.jpg'
    :param nx: Number of corners in x axis on each chessboard
    :param ny: Number of corners in y axis on each chessboard
    :param show: Show corners on the chessboard as image
    :param iterations: Number if adjustment iterations of the corner finding algorithm
    :return: mtx, dist
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibration_folder)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            # add an iterative algorithm to adjust the corners even more to the chess segment corners by using
            # the OpenCv cornerSubPix function with basic parameters
            win_size = (15, 15)
            zero_zone = (-1, -1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
            for _ in range(iterations):
                corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)

            objpoints.append(objp)
            imgpoints.append(corners)

            if show:
                # Draw and display the corners
                display_img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.imshow('img', display_img)
                cv2.waitKey(500)
        else:
            print(f"[Camera Calibration] [Chessboard Files] Can not use file {fname}")
    if show:
        cv2.destroyAllWindows()

    if len(objpoints) == 0:
        raise Exception("No calibration data found")
    if len(objpoints) < 20:
        print("[WARNING] Less than 20 calibration images could be processed to calculate the distortion matrix")

    # calculate mtx and dist
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    if ret == False:
        raise Exception("Distortion matrix can not be calculated")

    return mtx, dist


def cal_undistort(img, objpoints, imgpoints):
    """
    performs the camera calibration, image distortion correction and
    returns the undistorted image
    :param img: RGB image
    :param objpoints: object points
    :param imgpoints: real image points in pixel scale
    :return: undistorted image
    """
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def cal_undistort_2(img, mtx, dist):
    """
    performs the camera calibration, image distortion correction and
    returns the undistorted image
    :param img: RGB image
    :param mtx: Transformation matrix
    :param dist:  Distortion matrix
    :return: undistorted image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


if __name__ == "__main__":
    # image paths
    test_image = "../camera_cal/calibration1.jpg"
    output_image = "../output_images/camera_calibration.jpg"
    # first calculate the camera distortion
    mtx, dist = distortion_from_calibration_images("../camera_cal/calibration*.jpg", show=False)
    img = mpimg.imread(test_image)
    # make an undistorted image
    undistorted_img = cal_undistort_2(img, mtx, dist)
    plot_results(img, undistorted_img, output_image, out_gray=False)
    # Results: The distortion matrix can be calculated by the calibration images and processed over an
    # image from the camera
