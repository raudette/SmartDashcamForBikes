# SmartDashcamForBikes

## Requirements

- Requires a [Luxonis Oak Camera](https://store.opencv.ai/), which a camera with a USB interface and an on-board vision processing unit.  It will likely work with an Intel Neural Compute stick

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Main.py - SmartDashcamForBikes

Arguments:
-cam, Use camera as source
-vid <filename>, Specify test file as source
-stream, Stream video over network on port 8090
-record, Record output to file (only works if using camera as source)
-preview, Onscreen Preview

Sample Usage:

```
python3 main.py -vid samplevideo.mp4
```

To see the streamed frames, open [localhost:8090](http://localhost:8090).  It does not work with Firefox.  Tested with Chrome and the [IPCams]](https://apps.apple.com/ca/app/ipcams-ip-camera-viewer/id1045600272) app on iOS.