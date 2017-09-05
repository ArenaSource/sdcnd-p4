import logging
import os
import glob
import pickle
import collections

import numpy as np
import cv2
import matplotlib.pyplot as plt


class Camera(object):

    wp_src = np.float32([
        [705, 460],
        [1078, 700],
        [244, 700],
        [580, 460]
    ])

    wp_dst = np.float32([
        [1078, 0],
        [1078, 720],
        [244, 720],
        [244, 0]
    ])

    # Define conversions in x and y from pixels space to meters
    m2py = 30 / 720  # meters per pixel in y dimension
    m2px = 3.7 / 880  # meters per pixel in x dimension

    def __init__(self, name='default'):
        self.name = name
        self._calibration_file = './camera_{}_calibration.p'.format(self.name)
        if os.path.isfile(self._calibration_file):
            self._calibration = pickle.load(open(self._calibration_file, 'rb'))
        else:
            self._calibration = None

    def calibrate(self, images, shape=(9, 6), store=True):
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Default object point
        objp = np.zeros((np.prod(shape), 3), np.float32)
        objp[:, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)

        for file in glob.glob(images):
            img = cv2.imread(file, 1)
            if img is None:
                logging.warning('Unable to open {} image file for calibration'.format(file))
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, shape, None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        assert len(imgpoints) > 0

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        self._calibration = {
            'cameraMatrix': mtx,
            'distCoeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs
        }

        if store:
            pickle.dump(self._calibration, open(self._calibration_file, 'wb'))

        return self._calibration

    def undistort(self, image):
        mtx = self._calibration['cameraMatrix']
        dist = self._calibration['distCoeffs']
        return cv2.undistort(image, mtx, dist, None, mtx)

    def imread(self, filename):
        return Image(self, cv2.imread(filename))

    def read(self, pixels):
        return Image(self, pixels)


class Image(object):

    def __init__(self, camera, pixels, undistort=True):
        self.camera = camera
        if undistort:
            self.pixels = self.camera.undistort(pixels)
        else:
            self.pixels = pixels

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def RGB(self):
        return self.copy(cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB))

    @property
    def HLS(self):
        return self.copy(cv2.cvtColor(self.pixels, cv2.COLOR_BGR2HLS))

    @property
    def H(self):
        return self.copy(self.HLS.pixels[:, :, 0])

    @property
    def L(self):
        return self.copy(self.HLS.pixels[:, :, 1])

    @property
    def S(self):
        return self.copy(self.HLS.pixels[:, :, 2])

    def copy(self, pixels=None):
        if pixels is None:
            pixels = np.copy(self.pixels)
        return Image(self.camera, pixels, undistort=False)

    def __and__(self, image):
        return self.copy(self.pixels & image.pixels)

    def __or__(self, image):
        return self.copy(self.pixels | image.pixels)

    def warp(self):
        M = cv2.getPerspectiveTransform(self.camera.wp_src, self.camera.wp_dst)
        pixels = cv2.warpPerspective(self.pixels, M, (self.shape[1], self.shape[0]), flags=cv2.INTER_LINEAR)
        return Image(self.camera, pixels, undistort=False)

    def threshold(self, lo, hi):
        assert len(self.shape) == 2, "Multichannel operation is not supported"
        binary = np.zeros_like(self.pixels)
        binary[(self.pixels >= lo) & (self.pixels <= hi)] = 1
        return self.copy(binary)

    def sobel(self, vertical=(0, 255), horizontal=(0, 255), length=(0, 255), angle=(0, np.pi/2), kernel=3):
        assert len(self.shape) == 2, "Multichannel operation is not supported"

        x = cv2.Sobel(self.pixels, cv2.CV_64F, 1, 0, ksize=kernel)
        y = cv2.Sobel(self.pixels, cv2.CV_64F, 0, 1, ksize=kernel)

        abs_x = np.absolute(x)
        abs_y = np.absolute(y)

        scaled_x = np.uint8(255 * abs_x / np.max(abs_x))
        scaled_y = np.uint8(255 * abs_y / np.max(abs_y))

        x_bin = np.zeros_like(scaled_x)
        x_bin[(scaled_x >= vertical[0]) & (scaled_x <= vertical[1])] = 1

        y_bin = np.zeros_like(scaled_y)
        y_bin[(scaled_y >= horizontal[0]) & (scaled_y <= horizontal[1])] = 1

        magnitude = np.sqrt(x ** 2 + y ** 2)
        magnitude = (magnitude / (np.max(magnitude) / 255)).astype(np.uint8)

        magnitude_bin = np.zeros_like(magnitude)
        magnitude_bin[(magnitude >= length[0]) & (magnitude <= length[1])] = 1

        direction = np.arctan2(abs_x, abs_y)
        direction_bin = np.zeros_like(direction)
        direction_bin[(direction >= angle[0]) & (direction <= angle[1])] = 1

        combined = np.zeros_like(self.pixels)
        combined[(x_bin == 1) & (y_bin == 1) & (magnitude_bin == 1) & (direction_bin == 1)] = 1
        return self.copy(combined)

    def pipeline(self):
        return (self.S.threshold(170, 255) | self.S.sobel(vertical=(20, 100), kernel=9)).warp()

    def peaks(self, ratio=0.5):
        histogram = np.sum(self.pixels[int(self.shape[0] * ratio):, :], axis=0)

        midpoint = np.int(histogram.shape[0] / 2)
        left = np.argmax(histogram[:midpoint])
        right = np.argmax(histogram[midpoint:]) + midpoint
        return left, right

    def fit(self, debug=False):

        if debug:
            # Create an output image to draw on and visualize the result
            out_img = np.dstack((self.pixels, self.pixels, self.pixels)) * 255

        leftx_base, rightx_base = self.peaks()

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(self.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.pixels.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):

            win_y_low = self.shape[0] - (window + 1) * window_height
            win_y_high = self.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if debug:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 4)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 4)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if debug:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            # Generate x and y values for plotting
            y = np.linspace(0, self.shape[0] - 1, self.shape[0])
            lx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
            rx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
            plt.plot(lx, y, color='yellow')
            plt.plot(rx, y, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)

        return left_fit, right_fit,

    def adjust(self, left_fit, right_fit, debug=False):

        binary_warped = self.pixels

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        if debug:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)

        return left_fit, right_fit

    def show(self, cmap=None):
        if len(self.shape) == 3:
            plt.imshow(self.RGB.pixels, cmap=cmap)
        else:
            plt.imshow(self.pixels, cmap='gray')


class Line(object):
    def __init__(self, finder):
        self.finder = finder
        self._fits = collections.deque([], 12)

    @property
    def fit(self):
        if len(self._fits) == 0:
            return None
        return np.mean(self._fits, axis=0)

    @fit.setter
    def fit(self, value):
        # Sanity checks
        if value is not None and self._curvature(value) > 500 and abs(self._position(value)) < 3.0:
            self._fits.append(value)

    @property
    def curvature(self):
        return self._curvature(self.fit)

    def _curvature(self, fit):
        # Generate x and y values for plotting
        y = np.linspace(0, self.finder.image.shape[0] - 1, self.finder.image.shape[0])
        x = fit[0] * y ** 2 + fit[1] * y + fit[2]
        # Fit new polynomials to x,y in world space
        fit = np.polyfit(y * self.finder.camera.m2py, x * self.finder.camera.m2px, 2)
        # Calculate the new radii of curvature
        rad = ((1 + (2 * fit[0] * np.max(y) * self.finder.camera.m2py + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
        return round(rad)

    @property
    def position(self):
        return self._position(self.fit)

    def _position(self, fit):
        y = self.finder.image.shape[0] - 1
        x = fit[0] * y ** 2 + fit[1] * y + fit[2]
        d = (x - self.finder.image.shape[1] / 2) * self.finder.camera.m2px
        return round(d, 2)


class LaneFinder(object):
    def __init__(self, camera):
        self.camera = camera
        self.left = Line(self)
        self.right = Line(self)
        self._image = None

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

        try:
            if self.left.fit is None or self.right.fit is None:
                self.left.fit, self.right.fit = self._image.pipeline().fit()
            else:
                self.left.fit, self.right.fit = self._image.pipeline().adjust(self.left.fit, self.right.fit)
        except Exception:
            self.left.fit, self.right.fit = self._image.pipeline().fit()

    @property
    def position(self):
        return -(self.left.position + self.right.position) / 2.

    @property
    def curvature(self):
        return round((self.left.curvature + self.right.curvature) / 2.)

    @property
    def output(self):
        y = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
        lx = self.left.fit[0] * y ** 2 + self.left.fit[1] * y + self.left.fit[2]
        rx = self.right.fit[0] * y ** 2 + self.right.fit[1] * y + self.right.fit[2]

        # Create an image to draw the lines on
        warped = np.zeros_like(self.image.pixels).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        lp = np.array([np.transpose(np.vstack([lx, y]))])
        rp = np.array([np.flipud(np.transpose(np.vstack([rx, y])))])
        p = np.hstack((lp, rp))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(warped, np.int_([p]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = cv2.getPerspectiveTransform(self.camera.wp_dst, self.camera.wp_src)
        unwarped = cv2.warpPerspective(warped, Minv, (self.image.shape[1], self.image.shape[0]))
        # Combine the result with the original image
        image = self.image.copy(cv2.addWeighted(self.image.pixels, 1, unwarped, 0.3, 0))
        cv2.putText(image.pixels, "Curvature: {} m".format(int(self.curvature)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image.pixels, "Position:  {0:.2f} m".format(self.position), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

    def imread(self, filename):
        self.image = self.camera.imread(filename)
        return self.image

    def run(self, pixels):
        self.image = self.camera.read(pixels)
        return self.output

