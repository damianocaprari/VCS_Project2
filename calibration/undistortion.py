import cv2


def undistort_img(img, camera_matrix, distortion_coefficients, alpha=0.):
    h, w = img.shape[:2]

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (h, w), alpha)
    dst = cv2.undistort(img, camera_matrix, distortion_coefficients, newCameraMatrix=new_camera_matrix)
    return dst
