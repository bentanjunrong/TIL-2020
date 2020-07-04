# Module files to change 
1. YoloV4 python's `reduce bbox candidates()` threshold (reduce to 0.1) (utility/predict.py)

# Parts/Functions to test in arena
## [Binary classifier](robomaster/utils/frame_processing.py)
1. Need to check if it works in arena lighting, otherwise revert to hardcoded angles for search phase.
2. Need to check if cv2 frames are properly read and converted

## [Search and Rescue script](robomaster/search_rescue.py)
1. Any functions that use either object detector or binary classifier results. These include:
   1. `lock_on_loop()` > binary classifier
   2. `search_loop()` > object detector
   3. `rescue_loop()` > object detector

**ENSURE ALL TEST CODE IS COMMENTED OUT**


# Constant variables to fine tune
## Path Navigator
1. `start_disp`, `start_loc`, `end_loc`

## [Object detector](robomaster/utils/object_detector.py)
1. Height Thresh (should be more than 30%, untested)
## [Search and Rescue script](robomaster/search_rescue.py)
1. `dist thresh` and `turn_const` for object-detection-based centering
2. `boundaries`





# Robot Related Info
## EP Master

### API Reference
https://robomaster-dev.readthedocs.io/en/latest/sdk/protocol_api.html#

### Issues
- Moving along x-axis does not brake properly. Results in overshooting ALL THE TIME.

## Tello

### Issues
- Unstable when still
- Find IP address and connect with `socket.bind`.
  - Figure out how to disconnect properly (press and hold 5s)
- Camera periscope mirror missing.