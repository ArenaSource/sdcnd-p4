import logging
import os
import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt


class Camera(object):

    wp_src = np.float32([
        [674, 440],
        [1040, 675],
        [280, 675],
        [610, 440]
    ])

    wp_dst = np.float32([
        [1040, 0],
        [1040, 720],
        [280, 720],
        [280, 0]
    ])

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

        for file in glob.glob('camera_cal/*'):
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
        return self._copy(cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB))

    @property
    def HLS(self):
        return self._copy(cv2.cvtColor(self.pixels, cv2.COLOR_BGR2HLS))

    @property
    def H(self):
        return self._copy(self.HLS.pixels[:, :, 0])

    @property
    def L(self):
        return self._copy(self.HLS.pixels[:, :, 1])

    @property
    def S(self):
        return self._copy(self.HLS.pixels[:, :, 2])

    def _copy(self, pixels=None):
        if pixels is None:
            pixels = self.pixels
        return Image(self.camera, pixels, undistort=False)

    def __and__(self, image):
        return self._copy(self.pixels & image.pixels)

    def __or__(self, image):
        return self._copy(self.pixels | image.pixels)

    def warp(self):
        M = cv2.getPerspectiveTransform(self.camera.wp_src, self.camera.wp_dst)
        pixels = cv2.warpPerspective(self.pixels, M, (self.shape[1], self.shape[0]), flags=cv2.INTER_LINEAR)
        return Image(self.camera, pixels, undistort=False)

    def threshold(self, lo, hi):
        assert len(self.shape) == 2, "Multichannel operation is not supported"
        binary = np.zeros_like(self.pixels)
        binary[(self.pixels >= lo) & (self.pixels <= hi)] = 1
        return self._copy(binary)

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
        return self._copy(combined)

    def pipeline(self):
        return (self.S.threshold(170, 255) | self.S.sobel(vertical=(20, 100), kernel=9)).warp()

    def show(self, cmap=None):
        if len(self.shape) == 3:
            plt.imshow(self.RGB.pixels, cmap=cmap)
        else:
            plt.imshow(self.pixels, cmap='gray')
        plt.show()


