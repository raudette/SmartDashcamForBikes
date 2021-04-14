# Source code adapted from the following samples
# https://github.com/luxonis/depthai-experiments/tree/master/gen2-license-plate-recognition
# https://github.com/luxonis/depthai-experiments/tree/master/mjpeg-streaming

import argparse
import threading
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument('-stream', '--stream', action="store_true",
                    help="Stream output over network")
parser.add_argument('-record', '--record', action="store_true",
                    help="Record output to video file")
parser.add_argument('-np', '--nopreview', action="store_true",
                    help="record output")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError(
        "No source selected. Use either \"-cam\" to run on RGB camera as a source or \"-vid <path>\" to run on video"
    )

debug = not args.no_debug


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2020_1)

    if args.camera:
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(672, 384)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Vehicle Detection Neural Network...")
    veh_nn = pipeline.createMobileNetDetectionNetwork()
    veh_nn.setConfidenceThreshold(0.5)
    veh_nn.setBlobPath(str(Path("models/vehicle-detection-adas-0002.blob").resolve().absolute()))
    veh_nn.input.setQueueSize(1)
    veh_nn.input.setBlocking(False)
    veh_nn_xout = pipeline.createXLinkOut()
    veh_nn_xout.setStreamName("veh_nn")
    veh_nn.out.link(veh_nn_xout.input)
    veh_pass = pipeline.createXLinkOut()
    veh_pass.setStreamName("veh_pass")
    veh_nn.passthrough.link(veh_pass.input)

    if args.camera:
        cam.preview.link(veh_nn.input)
    else:
        veh_xin = pipeline.createXLinkIn()
        veh_xin.setStreamName("veh_in")
        veh_xin.out.link(veh_nn.input)
    attr_nn = pipeline.createNeuralNetwork()
    attr_nn.setBlobPath(str((Path(__file__).parent / Path('models/vehicle-attributes-recognition-barrier-0039.blob')).resolve().absolute()))
    attr_nn.input.setBlocking(False)
    attr_nn.setNumInferenceThreads(2)
    attr_nn.input.setQueueSize(1)
    attr_xout = pipeline.createXLinkOut()
    attr_xout.setStreamName("attr_nn")
    attr_nn.out.link(attr_xout.input)
    attr_pass = pipeline.createXLinkOut()
    attr_pass.setStreamName("attr_pass")
    attr_nn.passthrough.link(attr_pass.input)
    attr_xin = pipeline.createXLinkIn()
    attr_xin.setStreamName("attr_in")
    attr_xin.out.link(attr_nn.input)

    print("Pipeline created.")
    return pipeline


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if not args.camera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


running = True
license_detections = []
vehicle_detections = []
rec_results = []
attr_results = []
frame_det_seq = 0
frame_seq_map = {}
veh_last_seq = 0
lic_last_seq = 0

if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    fps = FPSHandler(cap)


def lic_thread(det_queue, det_pass, rec_queue):
    global license_detections, lic_last_seq

    while running:
        try:
            in_det = det_queue.get().detections
            in_pass = det_pass.get()

            orig_frame = frame_seq_map.get(in_pass.getSequenceNum(), None)
            if orig_frame is None:
                continue

            license_detections = [detection for detection in in_det if detection.label == 2]

            for detection in license_detections:
                bbox = frame_norm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cropped_frame = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                tstamp = time.monotonic()
                img = dai.ImgFrame()
                img.setTimestamp(tstamp)
                img.setType(dai.RawImgFrame.Type.BGR888p)
                img.setData(to_planar(cropped_frame, (94, 24)))
                img.setWidth(94)
                img.setHeight(24)
                rec_queue.send(img)

            fps.tick('lic')
        except RuntimeError:
            continue


def veh_thread(det_queue, det_pass, attr_queue):
    global vehicle_detections, veh_last_seq

    while running:
        try:
            vehicle_detections = det_queue.get().detections
            in_pass = det_pass.get()

            orig_frame = frame_seq_map.get(in_pass.getSequenceNum(), None)
            if orig_frame is None:
                continue
                
            veh_last_seq = in_pass.getSequenceNum()

            for detection in vehicle_detections:
                bbox = frame_norm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cropped_frame = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                tstamp = time.monotonic()
                img = dai.ImgFrame()
                img.setTimestamp(tstamp)
                img.setType(dai.RawImgFrame.Type.BGR888p)
                img.setData(to_planar(cropped_frame, (72, 72)))
                img.setWidth(72)
                img.setHeight(72)
                attr_queue.send(img)

            fps.tick('veh')
        except RuntimeError:
            continue


def attr_thread(q_attr, q_pass):
    global attr_results

    while running:
        try:
            attr_data = q_attr.get()
            attr_frame = q_pass.get().getCvFrame()
        except RuntimeError:
            continue

        colors = ["white", "gray", "yellow", "red", "green", "blue", "black"]
        types = ["car", "bus", "truck", "van"]

        in_color = np.array(attr_data.getLayerFp16("color"))
        in_type = np.array(attr_data.getLayerFp16("type"))

        color = colors[in_color.argmax()]
        color_prob = float(in_color.max())
        type = types[in_type.argmax()]
        type_prob = float(in_type.max())

        attr_results = [(attr_frame, color, type, color_prob, type_prob)] + attr_results[:9]

        fps.tick_fps('attr')


with dai.Device(create_pipeline()) as device:
    print("Starting pipeline...")
    device.startPipeline()
    if args.camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        veh_in = device.getInputQueue("veh_in")
    attr_in = device.getInputQueue("attr_in")
    veh_nn = device.getOutputQueue("veh_nn", 1, False)
    veh_pass = device.getOutputQueue("veh_pass", 1, False)
    attr_nn = device.getOutputQueue("attr_nn", 1, False)
    attr_pass = device.getOutputQueue("attr_pass", 1, False)
    veh_t = threading.Thread(target=veh_thread, args=(veh_nn, veh_pass, attr_in))
    veh_t.start()
    attr_t = threading.Thread(target=attr_thread, args=(attr_nn, attr_pass))
    attr_t.start()


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        global frame_det_seq

        if args.video:
            read_correctly, frame = cap.read()
            if read_correctly:
                frame_seq_map[frame_det_seq] = frame
                frame_det_seq += 1
            return read_correctly, frame
        else:
            in_rgb = cam_out.get()
            frame = in_rgb.getCvFrame()
            frame_seq_map[in_rgb.getSequenceNum()] = frame

            return True, frame


    try:
        while should_run():
            read_correctly, frame = get_frame()

            if not read_correctly:
                break
                
            for map_key in list(filter(lambda item: item <= min(lic_last_seq, veh_last_seq), frame_seq_map.keys())):
                del frame_seq_map[map_key]

            fps.next_iter()

            if not args.camera:
                tstamp = time.monotonic()
                veh_frame = dai.ImgFrame()
                veh_frame.setData(to_planar(frame, (300, 300)))
                veh_frame.setTimestamp(tstamp)
                veh_frame.setSequenceNum(frame_det_seq)
                veh_frame.setType(dai.RawImgFrame.Type.BGR888p)
                veh_frame.setWidth(300)
                veh_frame.setHeight(300)
                veh_frame.setData(to_planar(frame, (672, 384)))
                veh_frame.setWidth(672)
                veh_frame.setHeight(384)
                veh_in.send(veh_frame)

            if debug:
                debug_frame = frame.copy()
                leftcar = False
                rightcar = False
                for detection in license_detections + vehicle_detections:
                    bbox = frame_norm(debug_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    # Here is where I check if a car is detected on the right side or left side
                    if (bbox[0] > 336): 
                        rightcar = True

                    if (bbox[2] < 336):
                        leftcar = True
                debug_frame = cv2.flip(debug_frame,1)

                if rightcar:
                    cv2.circle(debug_frame,(60,320), 20, (2,210,238), -1)
                    cv2.putText(debug_frame, f"CAUTION", (25, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2,210,238))

                if leftcar:
                    cv2.circle(debug_frame,(612,320), 20, (2,210,238), -1)
                    cv2.putText(debug_frame, f"CAUTION", (580, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2,210,238))

                cv2.imshow("rgb", debug_frame)
                attr_stacked = None

                for attr_img, attr_color, attr_type, color_prob, type_prob in attr_results:
                    attr_placeholder_img = np.zeros((72, 200, 3), np.uint8)
                    cv2.putText(attr_placeholder_img, attr_color, (15, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, attr_type, (15, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, f"{int(color_prob * 100)}%", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(attr_placeholder_img, f"{int(type_prob * 100)}%", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    attr_combined = np.hstack((attr_img, attr_placeholder_img))

                    if attr_stacked is None:
                        attr_stacked = attr_combined
                    else:
                        attr_stacked = np.vstack((attr_stacked, attr_combined))

                    
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    running = False

attr_t.join()
veh_t.join()
print("FPS: {:.2f}".format(fps.fps()))
if not args.camera:
    cap.release()
