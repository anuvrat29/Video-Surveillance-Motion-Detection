"""
    This code will take care of surveillance.
"""
# pylint: disable=E1101
# pylint: disable=E0211
# pylint: disable=W0603
import datetime
import argparse
import threading
import imutils

import cv2
from imutils.video import VideoStream

from flask import Response, Flask, render_template
from utilities.motiondetector import MotionDetector

OUTPUT_IMAGE, LOCK, VIDEO_STREAM = None, threading.Lock(), VideoStream(src=0).start()

APP = Flask(__name__, template_folder="frontend")

class Surveillance:
    """
        This class contains all the methods of surveillance.
    """
    @classmethod
    def detect_motion(cls, frame_count):
        """
            This will detect motion of video.
        """
        global OUTPUT_IMAGE, LOCK, VIDEO_STREAM
        total, motiondetect = 0, MotionDetector(0.1)

        while True:

            video_frames = VIDEO_STREAM.read()
            video_frames = imutils.resize(video_frames, width=500)
            gray = cv2.cvtColor(video_frames, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            timestamp = datetime.datetime.now()
            cv2.putText(video_frames, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),\
                                    (10, video_frames.shape[0] - 10),\
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            if total > frame_count:
                motion = motiondetect.detect(gray)
                if motion is not None:
                    xmin, ymin, xmax, ymax = motion
                    cv2.rectangle(video_frames, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            motiondetect.update(gray)
            total = total + 1
            with LOCK:
                OUTPUT_IMAGE = video_frames.copy()

    @classmethod
    def generate(cls):
        """
            This will generate output_image.
        """
        global OUTPUT_IMAGE, LOCK

        while True:
            with LOCK:
                if OUTPUT_IMAGE is None:
                    continue
                flag, encoded_image = cv2.imencode(".jpg", OUTPUT_IMAGE)
                if not flag:
                    continue
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'\
                                                + bytearray(encoded_image) + b'\r\n'

    @APP.route("/video_feed")
    def video_feed():
        """
            This will return frame on calling UI.
        """
        return Response(Surveillance().generate(),\
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @APP.route("/")
    def index():
        """
            Index page which will be hit when you open base URL.
        """
        return render_template("index.html")

if __name__ == '__main__':
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument("-f", "--frames", type=int, default=32,\
                                        help="# of frames used to construct the background model.")
    arguments = vars(argumentparser.parse_args())

    mythread = threading.Thread(target=Surveillance().detect_motion, args=(arguments["frames"],))
    mythread.daemon = True
    mythread.start()

    APP.config["CACHE_TYPE"] = "null"
    APP.jinja_env.cache = {}
    APP.run(host="127.0.0.1", port=65000, debug=True, threaded=True, use_reloader=False)

VIDEO_STREAM.stop()
