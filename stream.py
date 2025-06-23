import cv2
import time
import queue
import threading
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
from deepface import DeepFace

from processor import FrameProcessor
from utils import ImageUtils


class StreamReader:
    """
    This class handles reading RTSP streams in a dedicated thread.
    """

    def __init__(self, rtsp_url, stream_id, frame_queue):
        self.rtsp_url = rtsp_url
        self.stream_id = stream_id
        self.frame_queue = frame_queue
        self.stop_flag = False

    def read_rtsp_stream(self):
        """
        Continuously reads from the RTSP stream,
        placing frames into frame_queue.
        """
        pipeline = (
            f"rtspsrc location={self.rtsp_url} latency=0 ! "
            "decodebin ! videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        print(f"Stream {self.stream_id}: Reading started.")

        while not self.stop_flag:
            if not cap.isOpened():
                print(f"Stream {self.stream_id}: Unable to open RTSP stream.")
                cap.release()
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                time.sleep(1)
                print(f"Stream {self.stream_id}: Reading started again.")
                continue

            ret, frame = cap.read()
            if not ret:
                print(f"Stream {self.stream_id}: Frame not received.")
                cap.release()
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                time.sleep(1)
                print(f"Stream {self.stream_id}: Reading started again.")
                continue

            if self.frame_queue.full():
                # Drop oldest frame if queue is full
                try:
                    _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((self.stream_id, frame))

        # Cleanup
        cap.release()
        print(f"Stream {self.stream_id}: Stopped reading.")


class MultiStreamProcessor:
    """
    Encapsulates handling multiple streams, writing videos, and combining frames.
    """

    def __init__(self, rtsp_urls, pose_model_path, patch_model_ckpt, device="cuda:0"):
        self.rtsp_urls = rtsp_urls
        self.device = device
        self.num_classes = 5

        # Load pose (YOLO)
        self.pose_model = YOLO(pose_model_path)

        # Load patch classifier
        self.patch_model = models.efficientnet_b7(pretrained=False)
        if hasattr(self.patch_model, 'classifier'):
            in_features = self.patch_model.classifier[1].in_features
            self.patch_model.classifier[1] = nn.Linear(in_features, self.num_classes)
        self.patch_model.load_state_dict(torch.load(patch_model_ckpt))
        self.patch_model = self.patch_model.to(self.device)
        self.patch_model.eval()

        # Initialize frame queues for each stream
        self.frame_queues = [queue.Queue(maxsize=1) for _ in self.rtsp_urls]
        self.threads = []
        self.n_frames = [0] * len(self.rtsp_urls)
        self.last_frames = [None] * len(self.rtsp_urls)

        # Video writers for each single stream
        self.single_stream_outputs = []
        self.single_frame_sizes = []

        # Combined output
        self.combined_output = None

        # Track detection times for known individuals
        self.last_times = {
            'Andrelania': [datetime.now(), timedelta(seconds=0)],
            'Eduarda': [datetime.now(), timedelta(seconds=0)],
            'Eduardo': [datetime.now(), timedelta(seconds=0)],
            'Fabio': [datetime.now(), timedelta(seconds=0)],
            'Patricia': [datetime.now(), timedelta(seconds=0)],
            'Suzana': [datetime.now(), timedelta(seconds=0)]
        }

        # Create a dedicated FrameProcessor instance
        self.frame_processor = FrameProcessor(
            pose_model=self.pose_model,
            patch_model=self.patch_model,
            device=self.device,
            last_times=self.last_times
        )

    def start_streams(self):
        """
        Start each RTSP stream in its dedicated thread.
        """
        for idx, url in enumerate(self.rtsp_urls):
            stream_reader = StreamReader(url, idx, self.frame_queues[idx])
            t = threading.Thread(target=stream_reader.read_rtsp_stream, daemon=True)
            self.threads.append((t, stream_reader))
            t.start()

        # Retrieve initial frames for shape references
        for i in range(len(self.rtsp_urls)):
            while self.frame_queues[i].empty():
                time.sleep(0.1)
            stream_id, frame = self.frame_queues[i].get()
            self.last_frames[i] = frame

        # Initialize video writers for single streams
        for i, lf in enumerate(self.last_frames):
            if lf is not None:
                h, w = lf.shape[:2]
                out = cv2.VideoWriter(
                    f"cam{i+1}.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    20,
                    (w, h)
                )
                self.single_stream_outputs.append(out)
                self.single_frame_sizes.append((h, w))
            else:
                self.single_stream_outputs.append(None)
                self.single_frame_sizes.append((0, 0))

        # Create combined output
        ref_h, ref_w = self.last_frames[0].shape[:2]
        self.combined_output = cv2.VideoWriter(
            "combined_view.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            20,
            (ref_w * 2, ref_h * 2)  # 2x2 grid
        )

    def get_combined_frame(self):
        """
        Pull frames from each queue, process them, and combine into a grid.
        Returns: combined_frame, last_times, last_detect_str
        """
        frames = []
        for idx, frame_queue in enumerate(self.frame_queues):
            try:
                stream_id, frame = frame_queue.get_nowait()
                frame, self.last_times, _ = self.frame_processor.process_image(frame, self.n_frames[idx], stream_id)
                self.n_frames[idx] += 1
                self.last_frames[idx] = frame
            except queue.Empty:
                frame = self.last_frames[idx]
            frames.append(frame)

        # If any frame is None, cannot combine
        if any(f is None for f in frames):
            return None, self.last_times, ""

        # Write single-stream videos
        for i, (out_writer, size) in enumerate(zip(self.single_stream_outputs, self.single_frame_sizes)):
            if out_writer is not None and frames[i] is not None:
                out_writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))

        # Resize frames if needed for consistency
        ref_h, ref_w = frames[0].shape[:2]
        resized_frames = []
        for f in frames:
            h, w = f.shape[:2]
            if (h, w) != (ref_h, ref_w):
                resized_frames.append(cv2.resize(f, (ref_w, ref_h)))
            else:
                resized_frames.append(f)

        # Make 2x2 grid
        grid_rows, grid_cols = 2, 2
        while len(resized_frames) < grid_rows * grid_cols:
            placeholder = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
            resized_frames.append(placeholder)

        row_images = []
        for i in range(0, grid_rows * grid_cols, grid_cols):
            row = np.hstack(resized_frames[i : i + grid_cols])
            row_images.append(row)
        combined_frame = np.vstack(row_images)

        # Write combined output
        self.combined_output.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

        # Collect detection times
        last_detect_str = ImageUtils.get_strcounter(self.last_times)
        return combined_frame, self.last_times, last_detect_str

    def stop_all(self):
        """
        Release all resources and stop streaming threads.
        """
        for t, rdr in self.threads:
            rdr.stop_flag = True
        for t, _ in self.threads:
            t.join()

        if self.combined_output is not None:
            self.combined_output.release()

        for writer in self.single_stream_outputs:
            if writer is not None:
                writer.release()