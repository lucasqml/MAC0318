# MAC0318 Intro to Robotics
# Please fill-in the fields below with every team member info
#
# Name: Luã Santilli
# NUSP: 11795492
#
# Name: Lucas Quaresma
# NUSP: 11796399
#
# Name: Thiago Guerrero
# NUSP: 11275297
#
# Any supplemental material for your agent to work (e.g. neural networks, data, etc.) should be
# uploaded elsewhere and listed down below together with a download link.
#
#
#
# ---
#
# Final Project - The Travelling Mailduck Problem
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/challenge/challenge.py assignments/challenge/examples/challenge_n
#
# Submission instructions:
#  0. Add your names and USP numbers to the file header above.
#  1. Make sure that any last change hasn't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file via e-disciplinas.

from pyglet.window import key
import numpy as np
import math
import cv2
from typing import Tuple


def exists(object: any) -> bool:
    return object is not None


class Agent:
    def __init__(self, env):
        self.env = env
        self.radius = 0.0318
        self.baseline = env.unwrapped.wheel_dist/2
        self.motor_gain = 0.68*0.0784739898632288
        self.motor_trim = 0.0007500911693361842
        self.initial_pos = env.get_position()
        self.image_height, self.image_width, _ = env.front().shape
        self.score = 0

        self.automatic = True

        self.velocity = 0.0
        self.rotation = 0.0

        # White lanes hsv color range
        self.white_lower_hsv = np.array([0, 0, 150])
        self.white_upper_hsv = np.array([179, 70, 255])

        # Yellow lane hsv color range
        self.yellow_lower_hsv = np.array([24, 70, 75])
        self.yellow_upper_hsv = np.array([30, 220, 255])

        # Duck hsv color range
        self.duck_lower_hsv = np.array([15, 200, 120])
        self.duck_upper_hsv = np.array([25, 255, 255])

        # Collision avoidance
        self.default_impulse_duration = 20
        self.remaining_impulse_duration = self.default_impulse_duration
        self.should_impulse = False
        self.detected_frames = 0

        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

    def get_pwm_control(self, v: float, w: float) -> Tuple[float, float]:
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim) * \
            (v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim) * \
            (v+w*self.baseline)/self.radius
        return V_l, V_r

    #########################################
    # Computer Vision processing

    def apply_gaussian_blurring(self, image: np.ndarray) -> np.ndarray:
        sigma = 2
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    def apply_edge_detection(self, image: np.ndarray) -> np.ndarray:
        mask_edges = cv2.Canny(image, threshold1=100, threshold2=150)
        return self.region_of_interest(mask_edges)

    def apply_openning(self, image: np.ndarray, kernel: np.ndarray, iterations_erosion: int, iterations_dilation: int) -> np.ndarray:
        ''' Erosion followed by dilation'''
        image = cv2.erode(image, kernel, iterations=iterations_erosion)
        image = cv2.dilate(image, kernel, iterations=iterations_dilation)
        return image

    def apply_rho_transform(self, mask: np.ndarray) -> np.ndarray:
        ''' Identifies lines in the given mask. '''
        lines = cv2.HoughLinesP(mask,
                                rho=5,
                                theta=2*np.pi/180,
                                threshold=100,
                                lines=np.array([]),
                                minLineLength=30,
                                maxLineGap=30)
        return lines

    def compute_edge_mask(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = self.apply_gaussian_blurring(image)
        mask_edges = self.apply_edge_detection(image)

        # Apply dilation to edges
        kernel = np.ones((3, 3), np.uint8)
        mask_edges = cv2.dilate(mask_edges, kernel, iterations=1)

        return mask_edges

    def compute_color_mask(self, image_hsv, lower_hsv, upper_hsv,
                           kernel, iterations_erosion, iterations_dilation):

        color_mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
        color_mask = self.apply_openning(
            color_mask, kernel, iterations_erosion, iterations_dilation)

        return color_mask

    def region_of_interest(self, mask_edges):
        height, width = mask_edges.shape
        mask = np.zeros_like(mask_edges)

        # only focus on bottom half of the screen
        polygon = np.array([[
            (0, height // 2),
            (width, height // 2),
            (width, height),
            (0, height),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        mask_cropped_edges = cv2.bitwise_and(mask_edges, mask)
        return mask_cropped_edges

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        #########################################
        # Find the edge mask

        edge_mask = self.compute_edge_mask(image)

        #########################################
        # Find the color masks

        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        kernel = np.ones((2, 2), np.uint8)
        yellow_mask = self.compute_color_mask(image_hsv, self.yellow_lower_hsv, self.yellow_upper_hsv,
                                              kernel, iterations_erosion=1, iterations_dilation=3)

        kernel = np.ones((15, 10), np.uint8)
        duck_mask = self.compute_color_mask(image_hsv, self.duck_lower_hsv, self.duck_upper_hsv,
                                            kernel, iterations_erosion=1, iterations_dilation=5)

        kernel = np.ones((5, 5), np.uint8)
        white_mask = self.compute_color_mask(image_hsv, self.white_lower_hsv, self.white_upper_hsv,
                                             kernel, iterations_erosion=1, iterations_dilation=1)

        return edge_mask, yellow_mask, duck_mask, white_mask

    #########################################
    # Lane and objects processing

    def compute_median_lane(self, lanes):
        if lanes is not None and len(lanes) > 0:
            lanes = np.array(lanes)
            lanes = lanes.reshape(
                lanes.shape[0] * lanes.shape[1], lanes.shape[2])
            x1, y1, x2, y2 = np.median(lanes, axis=0)
            return [int(x1), int(y1), int(x2), int(y2)]

    def find_yellow_lane(self, edge_mask: np.ndarray, yellow_mask: np.ndarray) -> np.ndarray:
        yellow_lines = self.apply_rho_transform(
            edge_mask * (yellow_mask // 255))

        yellow_lane = None
        if exists(yellow_lines):
            yellow_lane = self.compute_median_lane(yellow_lines)

        return yellow_lane

    def find_white_lanes(self, edge_mask: np.ndarray, white_mask: np.ndarray) -> np.ndarray:
        white_lines = self.apply_rho_transform(edge_mask * (white_mask // 255))

        white_left_lane = None
        white_right_lane = None
        if exists(white_lines):
            white_left_lanes = []
            white_right_lanes = []
            for line in white_lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2((y2-y1), (x2-x1))
                if x1 > (self.image_width // 2):
                    if angle > 0.1:
                        white_right_lanes.append(line)
                else:
                    white_left_lanes.append(line)
            white_left_lane = self.compute_median_lane(white_left_lanes)
            white_right_lane = self.compute_median_lane(white_right_lanes)

        return white_left_lane, white_right_lane

    def detect_duck(self, duck_mask: np.ndarray, yellow_lane: np.ndarray):
        # Detect connect components of the yellow_mask_copy
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            duck_mask, connectivity=8)

        detected_objects = []
        max_y = 0
        max_y_index = 0
        # Filter out the small components
        for i in range(1, nb_components):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1300:
                duck_mask[output == i] = 0
            else:
                if y > max_y:
                    max_y = y + h
                    max_y_index = i
                detected_objects.append((x, y, w, h))

        coliding_obj = None
        if len(detected_objects) > 0 and max_y_index > 0:
            x = stats[max_y_index, cv2.CC_STAT_LEFT]
            y = stats[max_y_index, cv2.CC_STAT_TOP]
            w = stats[max_y_index, cv2.CC_STAT_WIDTH]
            h = stats[max_y_index, cv2.CC_STAT_HEIGHT]
            coliding_obj = (x, y, w, h)

        if exists(coliding_obj):
            x, y, w, h = coliding_obj
            if exists(yellow_lane):
                x1, y1, x2, y2 = yellow_lane
                if x > x1 and (y + h) > (self.image_height // 2):
                    return coliding_obj

        return None

    def compute_midle_lane(self, yellow_lane, white_right_lane):
        if yellow_lane is not None and white_right_lane is not None:
            lx1, ly1, lx2, ly2 = yellow_lane
            rx1, ry1, rx2, ry2 = white_right_lane
            mx1 = (lx1 + lx2) / 2
            my1 = (ly1 + ly2) / 2
            mx2 = (rx1 + rx2) / 2
            my2 = (ry1 + ry2) / 2

            return [int(mx1), int(my1), int(mx2), int(my2)]
        else:
            return None

    #########################################
    # Controller

    def get_pwms(self, image: np.ndarray):
        edge_mask, yellow_mask, duck_mask, white_mask = self.preprocess(image)

        yellow_lane = self.find_yellow_lane(
            edge_mask, yellow_mask)
        white_left_lane, white_right_lane = self.find_white_lanes(
            edge_mask, white_mask)

        colliding_object = self.detect_duck(
            duck_mask, yellow_lane)

        if exists(colliding_object):
            self.detected_frames += 1

            if self.detected_frames == 20:
                self.should_impulse = True
                self.detected_frames = 0

        self.rotation = 0.0
        if exists(colliding_object):
            x, y, w, h = colliding_object
            if exists(yellow_lane):
                x1, y1, x2, y2 = yellow_lane
                if x > x1:
                    self.velocity = 0.0
                    self.rotation = 2.0
        elif self.should_impulse:
            self.velocity += 0.05
            self.rotation = -2.0
            self.remaining_impulse_duration -= 1

            if self.remaining_impulse_duration == 0:
                self.should_impulse = False
                self.remaining_impulse_duration = self.default_impulse_duration
        elif exists(white_right_lane) and exists(yellow_lane):
            lane_intersect = self.compute_midle_lane(
                yellow_lane, white_right_lane)

            x1, y1, x2, y2 = lane_intersect
            distance_y = y2 - y1
            distance_x = x2 - x1
            ang = np.rad2deg(math.atan2(distance_y, distance_x))

            factor = 0.0
            if (abs(ang) >= 20.0):
                factor = 1.0
            elif (10 <= abs(ang) < 20.0):
                factor = 0.8
            elif (8.0 <= abs(ang) < 10.0):
                factor = 0.5
            elif (5.0 <= abs(ang) < 8.0):
                factor = 0.1

            self.velocity += 0.05
            self.rotation += factor * np.sign(ang)
        elif exists(white_right_lane):
            self.rotation += 2.0
            self.velocity -= 0.05
        elif exists(white_left_lane):
            self.rotation -= 2.0
            self.velocity -= 0.05
        elif exists(yellow_lane):
            self.rotation -= 2.0
            self.velocity -= 0.05
        else:
            self.rotation -= 0.1

        self.velocity = max(self.velocity, 0.0)
        self.velocity = min(self.velocity, 0.5)

        return self.get_pwm_control(self.velocity, self.rotation)

    def send_commands(self, dt: float):
        if not self.automatic:
            self.velocity = 0
            self.rotation = 0

        velocity = rotation = 0
        if self.key_handler[key.W]:
            velocity += 0.5
        if self.key_handler[key.A]:
            rotation += 1.5
        if self.key_handler[key.S]:
            velocity -= 0.5
        if self.key_handler[key.D]:
            rotation -= 1.5
        if self.key_handler[key.P]:
            self.automatic = not self.automatic

        # acquire front camera image
        img = self.env.front()

        if self.automatic:
            pwm_left, pwm_right = self.get_pwms(img)
        else:
            pwm_left, pwm_right = self.get_pwm_control(velocity, rotation)

        _, r, _, _ = self.env.step(pwm_left, pwm_right)
        self.score += (r-self.score)/self.env.step_count
