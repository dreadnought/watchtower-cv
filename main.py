#!/usr/bin/python3

import datetime
import time
import cv2
import os
import numpy
import json

from cysystemd.daemon import notify, Notification
import signal

from object_detection import ObjectDetectionThread
from storage.backblaze_b2 import BackblazeB2
from notification import queue_notification, slack_notification
from video import VideoWriterThread

from logger import get_logger


class Main:
    def __init__(self, video_url, video_fps, storage_bucket, slack_webhook_url, logger, motion_threshold=512):
        self.video_url = video_url
        self.video_fps = video_fps
        self.capture = None
        self.video_file_thread = None
        self.object_detection = ObjectDetectionThread(storage_bucket=storage_bucket,
                                                      slack_webhook_url=slack_webhook_url,
                                                      model_config=config['model'],
                                                      logger=logger)
        self.object_detection.start()
        self.storage_bucket = storage_bucket
        self.slack_webhook_url = slack_webhook_url
        self.motion_threshold = motion_threshold
        self.logger = logger
        self.is_running = False

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def open_video_source(self):
        if self.capture is not None:
            self.logger.warn("reconnecting...")
            self.capture.release()
        self.logger.info("opening video source")
        self.capture = cv2.VideoCapture(self.video_url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.backgound_substractor = cv2.createBackgroundSubtractorMOG2(varThreshold=128, history=120,
                                                                        detectShadows=False)

    def detect_motion(self, frame):
        scale_factor = 2
        resized = cv2.resize(frame, dsize=(0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor,
                             interpolation=cv2.INTER_LINEAR)
        height, width, channels = resized.shape
        area_limit = height * width * 0.99  # more than 99% of the frame
        blur = cv2.GaussianBlur(resized, (5, 5), 0)
        fgmask = self.backgound_substractor.apply(blur)
        if fgmask is None:
            return False, frame

        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_area = 0
        scaled_contours = []
        if len(contours) == 0:
            return False, frame

        largest_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_limit:
                self.logger.info(f"area {area} skipped, close to {area_limit}")
                continue
            if area == 0.0:
                continue
            scaled_contour = scale_contour(cnt, scale=scale_factor)
            if area > max_area:
                max_area = max(max_area, area)
                largest_contour = scaled_contour
            scaled_contours.append(scaled_contour)

        if len(scaled_contours) == 0:
            return False, frame

        mask_frame = draw_contours(frame, scaled_contours, largest_contour=largest_contour)

        if max_area < self.motion_threshold:
            return False, mask_frame

        return max_area, mask_frame

    def loop(self):
        self.open_video_source()

        self.logger.info("entering loop")
        failure_counter = 0
        active_event = False
        last_motion_counter = 0
        frame_counter = 0
        event_id = 0
        self.is_running = True
        notify(Notification.READY)
        queue_notification(func=slack_notification,
                           arguments={
                               "webhook_url": self.slack_webhook_url,
                               "text": "Started"
                           },
                           logger=self.logger,
                           )
        while self.is_running:
            if failure_counter > 10:
                self.open_video_source()

            retval, frame = self.capture.read()
            if not retval:
                self.logger.warn("%s failed to receive image" % datetime.datetime.now())
                time.sleep(1)
                failure_counter += 1
                continue
            frame_counter += 1
            last_time = time.time()
            now = datetime.datetime.now()
            notify(Notification.WATCHDOG)

            time_str = now.strftime("%Y%m%d/%H%M%S")
            # todo: run motion detection only on every X frame

            if active_event and self.object_detection.is_busy:
                # self.logger.debug("fast forward, skipping motion")
                self.video_file_thread.queue_frame(frame=frame)
                continue
            # else:
            #    self.logger.debug("normal frame")

            motion, mask_frame = self.detect_motion(frame)
            failure_counter = 0

            time_diff = time.time() - last_time

            if motion is not False:
                last_motion_counter = 0
                self.logger.info(
                    "motion %s took %0.3f sec (=%0.1f/s), result %s" % (time_str, time_diff, (1 / time_diff), motion))

            else:
                last_motion_counter += 1

                if last_motion_counter > 10 * self.video_fps and active_event:
                    self.logger.info("no motion for %i frames" % last_motion_counter)
                    if self.object_detection.is_busy:
                        self.logger.info("but object detection busy, waiting...")
                    else:
                        self.logger.debug(self.object_detection.output_queue[event_id])
                        object_detection_keys = list(self.object_detection.output_queue[event_id].keys())
                        objects_detected = []
                        while len(object_detection_keys) > 0:
                            frame_id = object_detection_keys.pop(0)
                            result = self.object_detection.output_queue[event_id][frame_id]
                            if len(result) == 0:
                                continue
                            # self.logger.info(json.dumps(result, indent=2))
                            for obj in result:
                                # bear (0.456)
                                objects_detected.append("%s (%s)" % (obj[0], obj[1]))
                        if len(objects_detected) > 0:
                            self.video_file_thread.stop(upload=True,
                                                        notification_text="Event ended, detected %s" % ", ".join(
                                                            objects_detected))
                        else:
                            self.video_file_thread.stop(upload=False)
                            """
                            notification.slack_notification(webhook_url=self.slack_webhook_url,
                                                            text="Event ended without detected object")
                            """

                        self.video_file_thread = None
                        active_event = False
                        del self.object_detection.output_queue[event_id]

                        self.logger.info("===========================================")
                        continue
                elif not motion and not active_event:
                    continue

            if active_event is False:
                active_event = True
                event_id += 1
                self.video_file_thread = VideoWriterThread(video_file_name="%s" % time_str,
                                                           fps=self.video_fps,
                                                           storage_bucket=storage_bucket,
                                                           slack_webhook_url=self.slack_webhook_url,
                                                           logger=self.logger)
                self.video_file_thread.start()
                """
                res, mask_bytes = cv2.imencode('.jpg', mask_frame.astype(numpy.uint8))
                image_url = self.storage_bucket.upload_bytes(data_bytes=mask_bytes,
                                                 remote_file_name=f"{time_str}-mask.jpg"
                                                 )
                notification.slack_notification(webhook_url=self.slack_webhook_url,
                                                text=f"Event started\n<{image_url}|open>")
                """
            self.video_file_thread.queue_frame(frame=mask_frame)

            if self.object_detection.is_busy:
                # print("object detection busy")
                pass
            elif motion:
                self.logger.info("running object detection")
                self.object_detection.process_image(event_id=event_id,
                                                    frame_id=frame_counter,
                                                    frame=frame, roi=config['roi'])
        self.logger.info("Loop ended")

    def stop(self, *args):
        self.logger.info("Stopping...")
        notify(Notification.STOPPING)
        self.is_running = False
        self.object_detection.stop()
        if self.video_file_thread:
            self.video_file_thread.stop(upload=False)
        self.capture.release()


def mask_from_contours(ref_img, contours):
    mask = numpy.zeros(ref_img.shape, numpy.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def draw_contours(src_img, contours, largest_contour):
    canvas = cv2.drawContours(src_img.copy(), contours, -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return canvas


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cx_scaled, cy_scaled = int(cx * scale), int(cy * scale)
    cnt_scaled = cnt_scaled + [cx_scaled, cy_scaled]

    cnt_scaled = cnt_scaled.astype(numpy.int32)

    return cnt_scaled


with open("config.json", "r") as f:
    config = json.load(f)

logger = get_logger('debug')

if config["storage_bucket"]["provider"] == 'backblaze_b2':
    storage_bucket = BackblazeB2(application_key_id=config["storage_bucket"]["application_key_id"],
                                 application_key=config["storage_bucket"]["application_key"],
                                 bucket_name=config["storage_bucket"]["bucket_name"],
                                 logger=logger)
    storage_bucket.cleanup_bucket(max_days=14)
else:
    storage_bucket = None

m = Main(video_url=config["video_url"],
         video_fps=5,
         storage_bucket=storage_bucket,
         slack_webhook_url=config["slack_webhook_url"],
         logger=logger)
try:
    m.loop()
except KeyboardInterrupt:
    print("KeyboardInterrupt")
    m.stop()
