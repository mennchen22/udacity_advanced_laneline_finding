from moviepy.editor import VideoFileClip

from src.CameraCalibration import CameraCalibration
from src.ColorThresholdImage import hls_select
from src.LaneFindingPipeline import add_results_to_image
from src.LanePolynomial import RoadLineFit
from src.PerspectiveTransformation import road_perspective_transformation


class LaneDetectionPipeline:
    """
    Putting all steps together
    """
    def __init__(self):
        self.camera = CameraCalibration(calibration_dir="../camera_cal/calibration*.jpg", nx=9, ny=6)
        self.road_line_fit = RoadLineFit()

    def image_pipeline(self, input_image):
        # create an undistorted image
        undistorted_img = self.camera.undistorted_image(input_image)
        # make a perspective transformation
        wrap_img, _ = road_perspective_transformation(undistorted_img, show_transformation_line=False)
        # create an binary image
        binary_image_wrap = hls_select(wrap_img, thresh=(120, 255))
        # calculate segments
        try:
            poly_img = self.road_line_fit.lane_line_pipe(binary_warped=binary_image_wrap)
        except Exception:
            print("[Image Pipeline] Caught exception in pipeline. Reset and pass raw image")
            self.road_line_fit.reset_values()
            return undistorted_img

        # print road line radius
        car_offset = self.road_line_fit.calculate_car_center_offset_m(car_center_pos=poly_img.shape[1] / 2)
        lane_radius = self.road_line_fit.calculate_lane_radius_m()
        # transform the polynomial back to the original
        result_img = add_results_to_image(undistorted_img, poly_img, car_offset, lane_radius)
        return result_img


if __name__ == "__main__":
    # video paths
    video_input = "../challenge_video.mp4"
    video_output = "../output_videos/lane_detection_challenge.mp4"
    # create pipeline
    lane_line_pipe = LaneDetectionPipeline()
    # load video
    clip = VideoFileClip(video_input)
    white_clip = clip.fl_image(lane_line_pipe.image_pipeline)  # NOTE: this function expects color images!!
    # save video
    white_clip.write_videofile(video_output, audio=False)
