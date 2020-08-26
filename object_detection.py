import datetime
import cv2 as cv
import numpy as np
import os.path
import time
import os
import threading
import sys

from notification import slack_notification, queue_notification

default_model_config = {
    'type': 'darknet',
    'classes_file': 'models/coco.names',
    'model_configuration_file': 'models/yolov3.cfg',
    'model_weights_file': 'models/yolov3.weights',
    'confidence_threshold': 0.3,
    'nms_threshold': 0.4,
    'input_width': 416,
    'input_height': 416,
    'cuda': False,
    'ignored_classes': ['banana', 'laptop']
}


class ObjectDetection():
    def __init__(self, logger, model_config):
        self.logger = logger
        self.net = None
        self.classes = None
        self.model_config = model_config
        self.load_model()

    def load_model(self):
        with open(self.model_config['classes_file'], 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        if self.model_config['type'] == 'darknet':
            self.net = cv.dnn.readNetFromDarknet(self.model_config['model_configuration_file'],
                                                 self.model_config['model_weights_file'])
        else:
            self.logger.error('Unkown model type %s' % self.model_config['type'])
            sys.exit(1)
        if self.model_config['cuda']:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    def process_image_file(self, file_name):
        if not os.path.isfile(file_name):
            print("Input image file ", file_name, " doesn't exist")
            return False
        image = cv.imread(file_name)
        return self.process_image(frame=image)

    def process_image(self, frame, roi=None):
        start_time = time.time()
        if roi is not None:
            # look at a portion of the frame
            frame = frame[roi['y_start']:roi['y_end'], roi['x_start']:roi['x_end']]

        blob = cv.dnn.blobFromImage(frame, 1 / 255,
                                    (self.model_config['input_width'], self.model_config['input_height']), [0, 0, 0], 1,
                                    crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_outputs_names(self.net))
        detected_objects = self.postprocess(frame, outs)

        self.logger.info(
            "Object detection took %0.3f sec, found %s" % (time.time() - start_time, len(detected_objects)))
        return detected_objects

    @staticmethod
    def get_outputs_names(net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def draw_predictions(self, classId, conf, left, top, right, bottom, frame):
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        class_name = self.classes[classId].title()
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (class_name, label)
        self.logger.debug("%s %s%%" % (class_name, round(conf * 100, 2)))

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []

        num_detected = 0
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                class_name = self.classes[classId]
                if class_name in self.model_config['ignored_classes']:
                    # print('\t%s ignored' % class_name)
                    continue
                if confidence < self.model_config['confidence_threshold']:
                    continue
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                num_detected += 1

        if num_detected == 0:
            return {}

        indices = cv.dnn.NMSBoxes(boxes,
                                  confidences,
                                  self.model_config['confidence_threshold'],
                                  self.model_config['nms_threshold'])
        detected_objects = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            classId = classIds[i]
            self.draw_predictions(classId, confidences[i], left, top, left + width, top + height, frame)
            detected_objects.append([self.classes[classId], round(confidences[i], 3)])
        return detected_objects


class ObjectDetectionThread(threading.Thread):
    def __init__(self, storage_bucket, slack_webhook_url, model_config, logger):
        threading.Thread.__init__(self)
        self.storage_bucket = storage_bucket
        self.slack_webhook_url = slack_webhook_url
        self.logger = logger
        self.is_running = False
        self.is_busy = False
        self.input_queue = []
        self.output_queue = {}
        self.event = threading.Event()
        self.object_detection = ObjectDetection(model_config=model_config, logger=logger)

    def stop(self):
        self.logger.warn("ObjectDetectionThread stopping")
        self.is_running = False
        self.event.set()

    def new_event(self, event_id):
        return {"id": event_id,
                "object_detected": False}

    def run(self):
        self.logger.info("ObjectDetectionThread started")
        self.is_running = True
        event_meta = self.new_event(event_id=-1)
        while self.is_running:
            # self.logger.debug("ObjectDetectionThread run")
            if len(self.input_queue) == 0:
                self.event.wait()
                continue

            self.is_busy = True

            job = self.input_queue.pop(0)

            if event_meta["id"] != job["event_id"]:
                event_meta = self.new_event(event_id=job['event_id'])

            image_meta = self.object_detection.process_image(frame=job['frame'], roi=job['roi'])
            self.output_queue[job['event_id']][job['frame_id']] = image_meta

            if event_meta['object_detected'] == False and len(image_meta) > 0:
                event_meta['object_detected'] = True
                now = datetime.datetime.now()
                time_str = now.strftime('%Y%m%d/%H%M%S')
                res, mask_bytes = cv.imencode('.jpg', job["frame"].astype(np.uint8))
                upload_ok = False
                if self.storage_bucket:
                    try:
                        image_url = self.storage_bucket.upload_bytes(data_bytes=mask_bytes,
                                                                     remote_file_name=f"{time_str}-object.jpg"
                                                                     )
                        upload_ok = True
                    except Exception as e:
                        self.logger.error("image upload failed")
                        self.logger.error(e)
                        image_url = None
                    if upload_ok:
                        notification_text = f"Detected {image_meta}, Image\n<{image_url}|open>"
                    else:
                        notification_text = f"Detected {image_meta}, upload failed"
                else:
                    notification_text = f"Detected {image_meta}"

                if self.slack_webhook_url:
                    queue_notification(func=slack_notification,
                                       arguments={
                                           "webhook_url": self.slack_webhook_url,
                                           "text": notification_text
                                       },
                                       logger=self.logger,
                                       )

            self.is_busy = False
            self.event.clear()
        self.logger.info("ObjectDetectionThread stopped")

    def process_image(self, event_id, frame_id, frame, roi=None):
        if event_id not in self.output_queue:
            self.output_queue[event_id] = {}
        self.input_queue.append({
            "event_id": event_id,
            "frame_id": frame_id,
            "frame": frame,
            "roi": roi
        })
        self.event.set()
