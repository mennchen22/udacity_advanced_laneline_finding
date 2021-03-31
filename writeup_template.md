## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration.jpg "Undistorted"

[image2]: ./output_images/undistorted_image.jpg "Road Transformed"

[image3]: ./output_images/binary_combination_threshold.jpg "Combined Binary Example"

[image4]: ./output_images/hsl_threshold.jpg "HSL Binary Example"

[image5]: ./output_images/perspective_transform.jpg "Warp Example"

[image6]: ./output_images/perspective_transform_reversed.jpg "Warp Example Reversed"

[image7]: ./output_images/lane_segment_detection_1.jpg "Fit Visual"

[image8]: ./output_images/lane_segment_detection_2.jpg "Output"

[image9]: ./output_images/lane_line_detection_pipe.jpg "Lane detection pipeline"

[radius_formula]: ./output_images/radius_formula.png "Radius Formula"

[video1]: ./output_videos/lane_detection_project.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

#### 1. Calculate distortion based on calibration images

_The code for this step is located in [CameraCalibration.py](./src/CameraCalibration.py)_

The process is separated within two steps. At first the program loads calibration images from a file directory (
camera_cal/) . Within the image a chessboard is detected by the OpenCv `cv2.findChessboardCorners()` function.
Identified corners within the image will be stored as (x, y, z) `imgpoints` related to `objpoints` that contains the
number of the point. For a better corner detection an iterative approach will be used to adjust the corners even more.
Therefore the `cv2.cornerSubPix()` function use a sliding window to search for better corners near the calculated ones.

Based on the given information (a list of  `imgpoints` and `objpoints`) the distortion matrix can be calculated with the
OpenCv `cv2.calibrateCamera()` function.

The second step is to use the distortion matrix to un-distort an image from this camera. Therefore, a class is
implemented to store the camera distortion values from the calibration process. A class function takes an image and
calculates the undistorted image with the `cv2.undistort()` function

An example of this process is shown with a chessboard image before and after the process.

![alt text][image1]

### Pipeline (single images)

The pipeline takes an image and process laneinformations from the image and pass them onto the picture as lane markers.
Additionally, the car position, and the lane radius is calculated. The process is divided in single tasks:

1) Distortion correction
2) Creating a binary threshold image
3) Perspective transformation
4) Calculating the polynomial from the lane lines
5) Calculate the real world lane radius and car position offset

#### 1. Distortion-corrected image.

First we use the camera calibration algorithm to calculate the distortion matrix. The image is then processed and
corrected based on the received information. An example is shown below.

![alt text][image2]

#### 2. Binary Images

_The code for this step is located in [ColorThresholdImage.py](./src/ColorThresholdImage.py)_

To detect the lane lines a binary image is used with should only represent features likely to be lane lines. Therefore,
two approaches will be compared. A combination of multiple convolution filters (like sobel) will be processed over the
image. Later on the results will be combined to a single image highlights the features within each of the filters.

The thresholds are the best practise values from the lesson exercises. An example is shown below.

![alt text][image3]

The second approach converts the image in the HSL color space. The luminosity space is used to detect highly light
reflecting objects, like lane lines. The result are more reliable than the binary convolution filters shown above. An
example is attched here.

![alt text][image4]

#### 3. Perspective transformation

_The code for this step is located in [PerspectiveTransformation.py](./src/PerspectiveTransformation.py)_
For the perspective transformation a region from a source image is projected to a destination image of the same size.
The source polygon was tested on a image of straight lane lines. The best result comes with this set of points:

```python
sr  # pre tuned source positions based on straight road lane image
image_fix_point_ratio = 0.36  # The percentage of the y axis position to be the upper top bar 
top_bar_length_left = (img_size[0] / 11) / 2
top_bar_length_right = (img_size[0] / 10) / 2
bottom_side_offset_left = img_size[0] / 6
bottom_side_offset_right = img_size[0] / 8
src = np.float32([
    [img_size[0] / 2 - (top_bar_length_left), img_size[1] * (1 - image_fix_point_ratio)],
    [img_size[0] / 2 + (top_bar_length_right), img_size[1] * (1 - image_fix_point_ratio)],
    [img_size[0] - bottom_side_offset_right, img_size[1]],
    [0 + bottom_side_offset_left, img_size[1]]
], dtype=np.int32)
```

This resulted in the following source and destination points:

| Source        | Destination   | Position  | 
|:-------------:|:-------------:| :-------------:| 
| 581.8182, 460.8      | 200. , 0.        | Top Left |
| 704. , 460.8       | 1080. , 0.      | Top Right|
| 1120. , 720.     | 1080. , 720.    | Bottom Right|
| 213.33333 , 720.    | 200. , 720.      | Bottom Laft|

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
and its warped counterpart to verify that the lines appear parallel in the warped image. Then a revert the process to 
see if the image section is recreateable.

The wrapped image:

![alt text][image5]

Reverted wrapped image:

![alt text][image6]

#### 4.  Calculating the polynomial from the lane lines

_The code for this step is located in [LanePolynomial.py](./src/LanePolynomial.py)_

For this part the perspective wrapped image is taken in the `find_lane_pixels()` function and within the first iteration a histogram is used to identify
the lane starting positions at the bottom. Then iteratively over a number of windows a slice is taken 
from both the left, and the right site, and the lane position is adjusted.
The positions are used to calculate a polynomial fit with `cv2.polyfit()`. The result is used to calculate the `y(x)` 
positions of both lane line polynomials. The result is stored so that in another image the lane could be searched within a margin
around the last lane. This is done by the function `search_around_poly()` within a `RoadLineFit` class, storing all information.

Example of a first image processed with the lane finding algorithm:

![alt text][image7]

In another step the lane line is searched near the last one. The margin boundary are shown as well:

![alt text][image8]

#### 5.  Calculate the real world lane radius and car position offset

_The code for this step is part of the lane line finding pipeline and is located in [LanePolynomial.py](./src/LanePolynomial.py)_

For the road radius calculation a meter to pixel scale is used to transform the sized 
to a real world imperial measurement. Given by the formula (image from this [tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php))

![alt text][radius_formula]

the radius can be calculated. Additionally, the bottom positions of the lane lines will be isolated and the car position 
offset based in the center of the two lanes is computes as well.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

At the end all steps are combined within a single pipeline. The reverse perspective transformation is used to print the result back on the original image. 
Additionally, the space between the lanes is filled with green and added over the road image as well. Lane radius and car offset are plotted onto the image, too.
The result is shown below:

![alt text][image9]

---

### Pipeline (video)

The pipeline is used ofer an image stream of real highway data:

Here's a [link to my video result](./output_videos/lane_detection_project.mp4)

---

### Discussion

The pipeline can detect lane lines in good lighting conditions. To avoid losing the lane in different lighting szenarios 
within a single image stream, the binary image thresholds should be more flexible. Maybe a rough lane type detection can choose 
from a preset of filters to calculate with the best binary thresholds 
for different lighting conditions.

At least the last lane lines will be used to speed up the process, but the parameters
have to be adjusted more to match with real road curvatures. Testing is needed. Additionally, 
more edge cases can be implemented to detect road loss and restart the pipeline.



