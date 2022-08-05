# Object Detection

Object detection is a computer vision technique for locating instances of objects in images or videos. Object detection algorithms typically leverage machine learning or deep learning to produce meaningful results.

<p align="center" width="100%">
    <img width="40%" src="https://user-images.githubusercontent.com/76533398/182803919-b7858dfa-7d64-4ea3-9627-03dd412c82cb.png">
</p>

The state-of-the-art methods can be categorized into two main types: one-stage methods and two stage-methods. One-stage methods prioritize inference speed, and example models include YOLO, SSD and RetinaNet. Two-stage methods prioritize detection accuracy, and example models include Faster R-CNN, Mask R-CNN and Cascade R-CNN.

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/76533398/182804454-df8b0ea6-bb18-4ad3-9a77-af4c4a54658a.png">
</p>

# YOLO (You Only Look Once) Algorithm

Yolo is a state-of-the-art, real-time object detection system. This algorithm is popular because of its speed and accuracy. It has been used in various applications to detect traffic signals, people, parking meters, and animals.

Biggest advantages:

- Speed (45 frames per second — better than realtime)
- Network understands generalized object representation (This allowed them to train the network on real world images and predictions on artwork was still fairly accurate).
- Faster version (with smaller architecture) — 155 frames per sec but is less accurate.
- Open source: https://pjreddie.com/darknet/yolo/

### Pre-requisites before learning Yolo:
<i>(You can skip the PyTorch implementations in the following videos for now)</i>

- [Intersection over Union](https://www.youtube.com/watch?v=XXYG5ZWtjj0&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=2)
- [Non-Max Suppression](https://www.youtube.com/watch?v=YDkjWEN8jNA&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=3&t=14s)
- [Mean Average Precision](https://www.youtube.com/watch?v=FppOzcDvaDI&t=1186s)

### [YOLO Tutorial](https://www.youtube.com/watch?v=MhftoBaoZpg)
### [Training YOLO on custom data](https://www.youtube.com/watch?v=XNRzZkZ-Byg)
### [YOLO Reading Guide](https://towardsdatascience.com/the-practical-guide-for-object-detection-with-yolov5-algorithm-74c04aac4843)

# Task

<p align="center" width="100%">
    <img width="30%" src="https://user-images.githubusercontent.com/76533398/183144355-91424024-76b6-44af-96fa-d709b0a9afde.jpeg">
</p>
