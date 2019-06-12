import numpy as np
import cv2

import glob
import os


DEBUG = False


def main():
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)

    # Objects points
    objp = np.zeros((7*9, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape((-1, 2))

    # Arrays to store object points and image points
    objpoints = []
    imgpoints = []

    images = glob.glob('../Calibration_frames/*')

    for file_name in images:
        img = cv2.imread(file_name)
        img = cv2.resize(img, (500, 500))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 9))

        # If found, add objects and image points
        if ret is True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if DEBUG is True:
                img = cv2.drawChessboardCorners(img, (7, 9), corners2, ret)
                cv2.imshow('out', img)
                cv2.waitKey(500)

    ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret is not None:
        if not os.path.exists('../calibration_parameters'):
            os.mkdir('../calibration_parameters')
        else:
            if not os.path.isdir('../calibration_parameters'):
                raise NotADirectoryError("calibration_parameters already exists but it's not a directory.")

        np.save('../calibration_parameters/camera_matrix', camera_matrix)
        np.save('../calibration_parameters/distortion_coefficients', distortion_coefficients)
        np.save('../calibration_parameters/rotation_vectors', rotation_vectors)
        np.save('../calibration_parameters/translation_vectors', translation_vectors)
        print('Calibration done correctly.')
    else:
        print('Something went wrong during the computation of the calibration.')


if __name__ == '__main__':
    main()
