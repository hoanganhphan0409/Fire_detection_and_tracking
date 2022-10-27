# Fire detection and tracking
## Introduction
In this project i will use two algorithms: YOLOv5 and tracking(norfair). First, our model will detect the position of the fire in each frame of video with YOLOv5. After detecting it, we will use a tracking algorithm (norfair) to observe the position of each fire. If the fire has gone over the custom thresholds, the fire will warn users about a potential fire. Otherwise, the system will not warn the user.

<img width="720" alt="Untitled1" src="https://user-images.githubusercontent.com/89737507/198232986-69f35347-7d1c-48c2-adfe-d20a5f8ae07b.png">

I use Pygame library to develop GUI for the program

<img width="960" alt="Untitled" src="https://user-images.githubusercontent.com/89737507/198233613-ff137f05-1c7d-40b5-a541-d3208344e18d.png">

There are 3 options: 
- You can use laptop camera for real time
- You can use the phone camera for real time (Using DroidCam App)
- You can upload your own video

## Run code
Start GUI:
```python
python menu_trail.py
```
## Reference
https://github.com/spacewalk01/yolov5-fire-detection

https://github.com/tryolabs/norfair

https://github.com/ultralytics/yolov5

https://pygame-menu.readthedocs.io/en/4.2.8/
