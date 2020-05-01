"""
    This will help in to detect motion.
"""
# pylint: disable=E1101
import imutils
import numpy as np
import cv2

class MotionDetector:
    """
        Class which contains all the methods which will detect motion.
    """
    @classmethod
    def __init__(cls, accumweight=0.5):
        """
            Init Method of a system.
        """
        cls.accumweight = accumweight
        cls.background = None

    @classmethod
    def update(cls, image):
        """
            if the background model is None, initialize it and
            update the background model by accumulating the weighted average
        """
        if cls.background is None:
            cls.background = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, cls.background, cls.accumweight)

    @classmethod
    def detect(cls, image, tval=25):
        """
            Compute the absolute difference between the background model
            and the image passed in, then threshold the delta image

            Perform a series of erosions and dilations to remove small blobs

            Find contours in the thresholded image and initialize the
            Minimum and maximum bounding box regions for motion
        """
        delta = cv2.absdiff(cls.background.astype("uint8"), image)
        thresh = cv2.threshold(delta, tval, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        (xmin, ymin) = (np.inf, np.inf)
        (xmax, ymax) = (-np.inf, -np.inf)

        if len(contours) == 0:
            return None

        for contour in contours:
            (initx, inity, width, height) = cv2.boundingRect(contour)
            (xmin, ymin) = (min(xmin, initx), min(ymin, inity))
            (xmax, ymax) = (max(xmax, initx + width), max(ymax, inity + height))
        return (xmin, ymin, xmax, ymax)
