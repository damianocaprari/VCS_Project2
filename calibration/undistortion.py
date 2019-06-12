import cv2


def undistort_img(img, camera_matrix, distortion_coefficients, destination_size, alpha=0.):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coefficients, (h, w), alpha, destination_size
    )
    dst = cv2.undistort(img, camera_matrix, distortion_coefficients, newCameraMatrix=new_camera_matrix)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
