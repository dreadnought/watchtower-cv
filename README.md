# watchtower-cv
A video surveillance service with motion and object detection based on [OpenCV](https://opencv.org/) and [Darknet](https://github.com/AlexeyAB/darknet). 

It tries to reduce false positives by doing object detection on frames that show motion. When it detects an interesting object, it uploads a video clip to a file storage provider (only Backblaze B2 currently) and sends a notification (only Slack currently). By having different detection layers, it tries to keep the number of frames low which have to be processed by time-consuming functions. This way it can run on low-power single-board computers.
