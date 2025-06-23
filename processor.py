import cv2
import time
import torch

from utils import ImageUtils


class FrameProcessor:
    """
    Encapsulates the logic for processing a single frame (i.e., detection, cropping).
    """

    def __init__(self, pose_model, patch_model, device, last_times):
        self.pose_model = pose_model     # e.g., YOLO pose
        self.patch_model = patch_model   # e.g., Person classifier
        self.device = device
        self.last_times = last_times     # dict: {name: [datetime, timedelta]}
    
    def process_image(self, frame, n_frames, cam_number):
        start = time.time()
        H, W = frame.shape[:-1]
        image_points = frame.copy()
        image_points = cv2.cvtColor(image_points, cv2.COLOR_BGR2RGB)

        # Pre-processing
        frame = ImageUtils.increase_brightness(frame, value=70)
        frame = ImageUtils.increase_contrast(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose detection
        with torch.no_grad():
            results = self.pose_model(frame, verbose=False)
        xyn = results[0].keypoints.xyn

        # Parse coordinates
        (Shoulder_coords, Elbow_coords,
         Nose_coords, Lear_coords, Rear_coords) = ImageUtils.get_coordinates(xyn, H, W)

        draw_scoords = []
        draw_fcoords = []
        shoulder_names = []
        names_faces_list = []
        all_names_faces = []

        # Shoulder detection & classification
        for Shoulder_xy, Elbow_xy in zip(Shoulder_coords, Elbow_coords):
            patch, x_max, y_max, x1, x2, y1, y2 = ImageUtils.crop_shoulder(
                frame, Elbow_xy, Shoulder_xy, H, W, alpha=0.9, beta=0.3
            )
            if patch is not None:
                name = ImageUtils.predict_name(self.patch_model, patch, self.device)
                draw_scoords.append([x_max, y_max, x1, x2, y1, y2])
                shoulder_names.append(name)

        # Face detection & recognition
        for Nose_xy, Lear_xy, Rear_xy in zip(Nose_coords, Lear_coords, Rear_coords):
            names_faces, coords_faces = ImageUtils.crop_face(
                image_points, Nose_xy, Lear_xy, Rear_xy, H, W, alpha=2., beta=2.
            )
            names_faces_list.append(names_faces)
            draw_fcoords.append(coords_faces)
            all_names_faces += names_faces

        # Evaluate times
        for n in set(all_names_faces + shoulder_names):
            self.last_times = ImageUtils.eval_time(n, self.last_times, time.time() - start)

        # Draw shoulders
        for n, coords in zip(shoulder_names, draw_scoords):
            image_points = ImageUtils.draw_shoulder(image_points, n, *coords, last_times=self.last_times)

        # Draw faces
        for n, coords in zip(names_faces_list, draw_fcoords):
            image_points = ImageUtils.draw_face(image_points, coords, n, self.last_times)

        # Combine face & shoulder names => coordinates
        names_coords = {
            'Andrelania': [],
            'Eduarda': [],
            'Eduardo': [],
            'Fabio': [],
            'Patricia': [],
            'Suzana': []
        }
        for name, coords in zip(shoulder_names, draw_scoords):
            if name != 'Unknown':
                names_coords[name].append(coords[:2])

        for names, coords in zip(names_faces_list, draw_fcoords):
            for n, c in zip(names, coords):
                if n != 'Unknown':
                    names_coords[n].append(c[:2])

        return image_points, self.last_times, names_coords