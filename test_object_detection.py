import sys
import time
import cv2 as cv

from object_detection import ObjectDetection, ObjectDetectionThread, default_model_config
from logger import get_logger


def get_test_file():
    test_image_url = "https://c7.staticflickr.com/7/6014/6204014339_8290a9894d_o.jpg"
    import tempfile
    import urllib3
    http = urllib3.PoolManager()
    response = http.request(method='GET', url=test_image_url)

    tf = tempfile.NamedTemporaryFile("w+b")
    tf.write(response.data)
    return tf


def test_object_detection(test_logger):
    od = ObjectDetection(model_config=default_model_config,
                         logger=test_logger)
    tf = get_test_file()

    result = od.process_image_file(tf.name)
    tf.close()
    summary = {}
    for object_name, rating in result:
        if object_name not in summary:
            summary[object_name] = 0
        summary[object_name] += 1

    expected_summary = {'person': 8, 'chair': 8, 'tvmonitor': 1}
    if summary != expected_summary:
        test_logger.error("Summary doesn't match expected result")
        test_logger.error("Found: %s" % summary)
        test_logger.error("Expected: %s" % expected_summary)
        sys.exit(1)

def test_thread(test_logger):
    thread = ObjectDetectionThread(logger=test_logger,
                                   storage_bucket=None,
                                   slack_webhook_url=None,
                                   model_config=default_model_config)
    thread.start()
    tf = get_test_file()
    test_event_id = 100
    test_frame_id = 200
    num_tests = 3
    for x in range(0,num_tests):
        test_frame = cv.imread(tf.name)
        thread.process_image(event_id=test_event_id, frame_id=test_frame_id+x,frame=test_frame)
    # wait for max. 5 seconds
    ok = False
    for x in range(1, 50):
        time.sleep(x / 10.0)
        if len(thread.output_queue) >= 1 and len(thread.output_queue[test_event_id]) == num_tests:
            ok = True
            break

    test_logger.info(thread.output_queue)
    thread.stop()

    if ok is False:
        test_logger.error("output incorrect")
        test_logger.error(thread.output_queue)
        sys.exit(1)


test_logger = get_logger()
test_object_detection(test_logger=test_logger)
test_thread(test_logger=test_logger)