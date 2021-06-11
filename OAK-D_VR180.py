#!/usr/bin/env python3
import argparse
import threading
import time
from pathlib import Path
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO 
from socketserver import ThreadingMixIn
from time import sleep
from PIL import Image
import subprocess as sp
import cv2
import depthai as dai
import skvideo.io
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str,
                    help="Set filename")
parser.add_argument('-s', '--stream', action="store_true",
                    help="Stream output over network, to preview on Google Cardboard")
parser.add_argument('-p', '--preview', action="store_true",
                    help="Display output on screen")
args = parser.parse_args()


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, 'frametosend'):
                image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                stream_file = BytesIO()
                image.save(stream_file, 'JPEG')
                self.wfile.write("--jpgboundary".encode())
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

if args.stream:
    print("Streaming video on http://localhost:8090/ - works with Chrome browser or IPCams app on iOS")
    # start MJPEG HTTP Server
    server_HTTP = ThreadedHTTPServer(('', 8090), VideoStreamHandler)
    th2 = threading.Thread(target=server_HTTP.serve_forever)
    th2.daemon = True
    th2.start()

## End web streaming stuff

# Start defining a pipeline
pipeline = dai.Pipeline()

# Image is 1280 wide.  Define offset from center for which 1280/2=640px to use
# Value must be between 0 and 640
# I suggest 320 to take the middle of each frame
# In preview mode, you can actually tweak this with the 'a' (increase offset by 10) and b (decrease by 10) keys
# You may want to change depending on how far away your subject is
offset=320

# Define a source - two mono (grayscale) cameras
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
cam_left.setFps(30)

cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
cam_right.setFps(30)

# Create outputs
xout_left = pipeline.createXLinkOut()
xout_left.setStreamName('left')
cam_left.out.link(xout_left.input)
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName('right')
cam_right.out.link(xout_right.input)


def seq(packet):
    return packet.getSequenceNum()


# https://stackoverflow.com/a/10995203/5494277
def has_keys(obj, keys):
    return all(stream in obj for stream in keys)


class PairingSystem:
    allowed_instances = [1, 2]  # Left (1) & Right (2)

    def __init__(self):
        self.seq_packets = {}
        self.last_paired_seq = None

    def add_packet(self, packet):
        if packet is not None and packet.getInstanceNum() in self.allowed_instances:
            seq_key = seq(packet)
            self.seq_packets[seq_key] = {
                **self.seq_packets.get(seq_key, {}),
                packet.getInstanceNum(): packet
            }

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.allowed_instances):
                results.append(self.seq_packets[key])
                self.last_paired_seq = key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]


writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    device.startPipeline()

    # Output queue will be used to get the rgb frames from the output defined above
    q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    ps = PairingSystem()

    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        ps.add_packet(q_left.tryGet())
        ps.add_packet(q_right.tryGet())

        for synced in ps.get_pairs():
            raw_left = synced[1]
            raw_right = synced[2]

            frame_left = raw_left.getData().reshape((raw_left.getHeight(), raw_left.getWidth())).astype(np.uint8)
            frame_right = raw_right.getData().reshape((raw_right.getHeight(), raw_right.getWidth())).astype(np.uint8)
            half_frame_left = frame_left[0:720, 640-offset:1280-offset]
            half_frame_right = frame_right[0:720, 0+offset:640+offset]
            combined_frame = np.concatenate((half_frame_left, half_frame_right ), axis=1)

            if args.preview: 
                cv2.imshow("combined", combined_frame)
            if args.stream:
                server_HTTP.frametosend = combined_frame
            writer.writeFrame(combined_frame)

        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('a'):
            offset=offset+10
            if (offset>640): 
                offset=640;
            print("Current Offset: " + str(offset))
        if cv2.waitKey(1) == ord('s'):
            offset=offset-10;
            if (offset<0):
                offset=0 
            print("Current Offset: " + str(offset))

writer.close()