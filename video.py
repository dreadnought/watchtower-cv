import time
import os
import threading
import cv2
import notification

class VideoFile():
    def __init__(self, video_file_name, fps, logger):
        self.video_format = ('x264', 'mp4')

        self.file_name = "%s.%s" % (video_file_name, self.video_format[1])
        self.fps = float(fps)
        self.frame_counter = 0
        self.frame_buffer = {}
        self.logger = logger

    def start(self, first_frame):
        self.logger.info("start_video_file %s" % self.file_name)
        if not os.path.isdir(os.path.dirname(self.file_name)):
            os.mkdir(os.path.dirname(self.file_name))
        height, width, channels = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*self.video_format[0])
        self.video_file = cv2.VideoWriter(self.file_name, fourcc, self.fps, (width, height))
        self.add_frame(frame=first_frame)

    def add_frame(self, frame):
        start_time = time.time()
        self.frame_counter += 1
        self.video_file.write(frame)
        time_diff = time.time() - start_time
        if time_diff > 0.5:
            self.logger.debug("slow, adding frame took %0.3f sec (=%i/s)" % (time_diff, (1 / time_diff)))

        return self.frame_counter

    def close(self):
        self.logger.info("close_video_file %s" % self.file_name)
        self.video_file.release()
        self.video_file = None

class VideoWriterThread(threading.Thread):
    def __init__(self, video_file_name, fps, storage_bucket, slack_webhook_url, logger):
        threading.Thread.__init__(self, name="VideoWriterThread")
        self.logger = logger
        self.storage_bucket = storage_bucket
        self.slack_webhook_url = slack_webhook_url
        self.input_queue = []
        self.event = threading.Event()
        self.video = VideoFile(video_file_name=video_file_name,
                               fps=fps,
                               logger=logger)
        self.finished = False
        self.upload = False
        self.notification_text = False

    def stop(self, upload=False, notification_text=False):
        self.logger.warn("VideoWriterThread stopping")
        self.upload = upload
        self.logger.info(notification_text)
        self.notification_text = notification_text
        self.input_queue.append(None)
        self.event.set()

    def run(self):
        self.logger.info("VideoWriterThread started %s" % self.video.file_name)
        self.is_running = True
        while True:
            if len(self.input_queue) == 0:
                # wait for the next frame
                self.event.wait()
                continue

            frame = self.input_queue.pop(0)
            if frame is None:
                # end of video
                break

            if self.video.frame_counter == 0:
                self.video.start(first_frame=frame)
            else:
                self.video.add_frame(frame=frame)

            if len(self.input_queue) > 10:
                self.logger.warn("%s frames in queue" % len(self.input_queue))
            self.event.clear()

        self.video.close()
        if self.upload:
            self.logger.info("Uploading %s" % self.video.file_name)
            upload_ok = False
            try:
                video_url = self.storage_bucket.upload_file(local_file_name=self.video.file_name,
                                                            remote_file_name=self.video.file_name)
                upload_ok = True
            except Exception as e:
                self.logger.error("upload failed")
                self.logger.error(e)

            if upload_ok:
                file_size = round(os.path.getsize(self.video.file_name) / (1024 ** 2), 2)
                notification.slack_notification(webhook_url=self.slack_webhook_url,
                                                text=f"{self.notification_text}\n<{video_url}|open ({file_size} MB)>")
            else:
                notification.slack_notification(webhook_url=self.slack_webhook_url,
                                                text="video upload failed")
        try:
            os.unlink(self.video.file_name)
        except FileNotFoundError:
            self.logger.info(f"failed to delete {self.video.file_name}, file is already gone")

        self.logger.info("VideoWriterThread stopped %s" % self.video.file_name)
        self.finished = True

    def queue_frame(self, frame):
        self.input_queue.append(frame)
        self.event.set()
