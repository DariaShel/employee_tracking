import cv2
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
from deepface import DeepFace


class ImageUtils:
    names_labels = {
        0: 'Eduardo',
        1: 'Fabio',
        2: 'Suzana',
        3: 'Eduarda',
        4: 'Unknown'
    }

    @staticmethod
    def rescale_frame(frame, percent=70):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def add_text_to_image(
        image_rgb: np.ndarray,
        label: str,
        top_left_xy: tuple = (0, 0),
        font_scale: float = 2,
        font_thickness: float = 3,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_color_rgb: tuple = (0, 255, 0),
        bg_color_rgb: tuple = None,
        outline_color_rgb: tuple = None,
        line_spacing: float = 1,
    ):
        """
        Draw multiline text on an image.
        """
        OUTLINE_FONT_THICKNESS = 3 * font_thickness
        im_h, im_w = image_rgb.shape[:2]

        for line in label.splitlines():
            x, y = top_left_xy
            if outline_color_rgb is None:
                get_text_size_font_thickness = font_thickness
            else:
                get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

            (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
                line,
                font_face,
                font_scale,
                get_text_size_font_thickness,
            )
            line_height = line_height_no_baseline + baseline

            if bg_color_rgb is not None and line:
                if im_h - (y + line_height) <= 0:
                    sz_h = max(im_h - y, 0)
                else:
                    sz_h = line_height

                if im_w - (x + line_width) <= 0:
                    sz_w = max(im_w - x, 0)
                else:
                    sz_w = line_width

                if sz_h > 0 and sz_w > 0:
                    bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                    bg_mask[:, :] = np.array(bg_color_rgb)
                    image_rgb[y : y + sz_h, x : x + sz_w] = bg_mask

            if outline_color_rgb is not None:
                image_rgb = cv2.putText(
                    image_rgb,
                    line,
                    (x, y + line_height_no_baseline),
                    font_face,
                    font_scale,
                    outline_color_rgb,
                    OUTLINE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )

            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),
                font_face,
                font_scale,
                font_color_rgb,
                font_thickness,
                cv2.LINE_AA,
            )
            top_left_xy = (x, y + int(line_height * line_spacing))

        return image_rgb

    @staticmethod
    def increase_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def increase_contrast(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img

    @staticmethod
    def distanceCalculate(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    @staticmethod
    def draw_face(img, coords=[], names=[], last_times={}):
        if names and coords:
            for name, coord in zip(names, coords):
                x, y, w, h = coord
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                counter = str(last_times[name][1]).split('.')[0]
                img = ImageUtils.add_text_to_image(
                    img,
                    f'{name}\n{counter}',
                    font_scale=2,
                    font_thickness=3,
                    top_left_xy=(x + w, y + h)
                )
        return img

    @staticmethod
    def draw_shoulder(img, name, x_text, y_text, x1, x2, y1, y2, last_times):
        if x_text is not None and y_text is not None and name is not None:
            if name != 'Unknown':
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                counter = str(last_times[name][1]).split('.')[0]
                img = ImageUtils.add_text_to_image(
                    img,
                    f'{name}\n{counter}',
                    font_scale=2,
                    font_thickness=3,
                    top_left_xy=(x_text, y_text)
                )
            else:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img

    @staticmethod
    def crop_face(img, nose, Lear, Rear, H, W, alpha=1.5, beta=2.):
        """
        Crop a face from the frame using the nose and ear coordinates.
        Then run DeepFace.find to match with database.
        """
        Ldist = 0
        Rdist = 0
        if Lear is not None:
            Ldist = ImageUtils.distanceCalculate(
                (nose[0] * W, nose[1] * H),
                (Lear[0] * W, Lear[1] * H)
            )
        if Rear is not None:
            Rdist = ImageUtils.distanceCalculate(
                (nose[0] * W, nose[1] * H),
                (Rear[0] * W, Rear[1] * H)
            )

        max_dist = Ldist if Ldist > Rdist else Rdist
        if Lear is not None and Rear is not None:
            x1 = int(nose[0] * W - max_dist * alpha)
            x2 = int(nose[0] * W + max_dist * alpha)
        elif Lear is not None:
            if nose[0] > Lear[0]:
                x1 = int(Lear[0] * W - max_dist)
                x2 = int(Lear[0] * W + max_dist * alpha)
            else:
                x1 = int(Lear[0] * W - max_dist * alpha)
                x2 = int(Lear[0] * W + max_dist)
        else:
            if nose[0] > Rear[0]:
                x1 = int(Rear[0] * W - max_dist)
                x2 = int(Rear[0] * W + max_dist * alpha)
            else:
                x1 = int(Rear[0] * W - max_dist * alpha)
                x2 = int(Rear[0] * W + max_dist)
        y1 = int(nose[1] * H - max_dist * beta)
        y2 = int(nose[1] * H + max_dist * beta)
        x1 = max(x1, 0)
        x2 = min(x2, W)
        y1 = max(y1, 0)
        y2 = min(y2, H)

        try:
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            dfs = DeepFace.find(
                img_path=crop_img,
                db_path="./staff",
                enforce_detection=False,
                silent=True,
            )
            names = []
            xywh_coords = []
            for df in dfs:
                if len(df) == 0:
                    continue
                name = df['identity'].iloc[0]
                xywh = df[['source_x', 'source_y', 'source_w', 'source_h']].iloc[0].tolist()
                xywh[0] += x1
                xywh[1] += y1
                xywh_coords.append(xywh)
                name = Path(name).relative_to('staff').parent
                names.append(str(name))
            return names, xywh_coords
        except:
            return [], []

    @staticmethod
    def crop_shoulder(img, Ecoord, Scoord, H, W, alpha, beta):
        """
        Crop a region around shoulder to elbow, as in the original code.
        """
        dist = ImageUtils.distanceCalculate(
            (Ecoord[0] * W, Ecoord[1] * H),
            (Scoord[0] * W, Scoord[1] * H)
        )
        h = alpha * dist
        xShoulder = int(Scoord[0] * W)
        yShoulder = int(Scoord[1] * H)
        xElbow = int(Ecoord[0] * W)
        yElbow = int(Ecoord[1] * H)

        # The big chunk: deciding corners for x1, y1, x2, y2
        if xElbow > xShoulder:
            xdiff = xElbow - xShoulder
            if yElbow > yShoulder:
                ydiff = yElbow - yShoulder
                if xdiff < ydiff:
                    x1 = xShoulder + xdiff * beta - h / 2
                    y1 = yShoulder
                    x2 = x1 + h
                    y2 = y1 + h
                else:
                    x1 = xShoulder
                    y1 = yShoulder + ydiff * beta - h / 2
                    x2 = x1 + h
                    y2 = y1 + h
            else:
                ydiff = yShoulder - yElbow
                if xdiff < ydiff:
                    x1 = xShoulder + xdiff * beta - h / 2
                    y1 = yShoulder
                    x2 = x1 + h
                    y2 = y1 - h
                else:
                    x1 = xShoulder
                    y1 = yShoulder - ydiff * beta + h / 2
                    x2 = x1 + h
                    y2 = y1 - h
        else:
            xdiff = xShoulder - xElbow
            if yElbow > yShoulder:
                ydiff = yElbow - yShoulder
                if xdiff < ydiff:
                    x1 = xShoulder - xdiff * beta + h / 2
                    y1 = yShoulder
                    x2 = x1 - h
                    y2 = y1 + h
                else:
                    x1 = xShoulder
                    y1 = yShoulder + ydiff * beta - h / 2
                    x2 = x1 - h
                    y2 = y1 + h
            else:
                ydiff = yShoulder - yElbow
                if xdiff < ydiff:
                    x1 = xShoulder - xdiff * beta + h / 2
                    y1 = yShoulder
                    x2 = x1 - h
                    y2 = y1 - h
                else:
                    x1 = xShoulder
                    y1 = yShoulder - ydiff * beta + h / 2
                    x2 = x1 - h
                    y2 = y1 - h

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if x1 > x2:
            x_max = x1
            if y1 > y2:
                crop_img = img[y2:y1, x2:x1]
                y_max = y1
            else:
                crop_img = img[y1:y2, x2:x1]
                y_max = y2
        else:
            x_max = x2
            if y1 > y2:
                y_max = y1
                crop_img = img[y2:y1, x1:x2]
            else:
                y_max = y2
                crop_img = img[y1:y2, x1:x2]
        try:
            crop_img = cv2.resize(crop_img, (64, 64), interpolation=cv2.INTER_LINEAR)
            return crop_img, x_max, y_max, x1, x2, y1, y2
        except:
            return None, None, None, x1, x2, y1, y2

    @staticmethod
    def predict_name(model, patch, device):
        patch_pil = Image.fromarray(patch)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        patch_tensor = transform(patch_pil)
        with torch.no_grad():
            label = model(patch_tensor.unsqueeze(0).to(device)).argmax().item()
        name = ImageUtils.names_labels[label]
        return name

    @staticmethod
    def eval_time(name, last_times, delay):
        """
        Adds the tracked detection time to the person's name.
        """
        for k in last_times.keys():
            if k == name:
                last_times[k][1] += (
                    datetime.now() - last_times[k][0] - timedelta(seconds=delay)
                )
                last_times[k][0] = datetime.now()
        return last_times

    @staticmethod
    def get_coordinates(xyn, H, W):
        """
        Pulls out the relevant coordinates from YOLO pose detection.
        """
        Lshoulder_idx = 5
        Rshoulder_idx = 6
        Lelbow_idx = 7
        Relbow_idx = 8
        Nose_idx = 0
        Lear_idx = 3
        Rear_idx = 4

        Shoulder_coords = []
        Elbow_coords = []
        Nose_coords = []
        Lear_coords = []
        Rear_coords = []
        for coord in xyn:
            if len(coord) == 0:
                continue
            Lshoulder = coord[Lshoulder_idx]
            Rshoulder = coord[Rshoulder_idx]
            Lelbow = coord[Lelbow_idx]
            Relbow = coord[Relbow_idx]
            Nose = coord[Nose_idx]
            Lear = coord[Lear_idx]
            Rear = coord[Rear_idx]
            distL = ImageUtils.distanceCalculate(
                (Lelbow[0] * W, Lelbow[1] * H),
                (Lshoulder[0] * W, Lshoulder[1] * H)
            )
            distR = ImageUtils.distanceCalculate(
                (Relbow[0] * W, Relbow[1] * H),
                (Rshoulder[0] * W, Rshoulder[1] * H)
            )
            if distR == 0 and distL == 0:
                continue

            # Pick the side with bigger elbow-shoulder distance
            if distR > distL and Rshoulder.sum() > 0 and Relbow.sum() > 0:
                Shoulder_coords.append(Rshoulder)
                Elbow_coords.append(Relbow)
            elif distR < distL and Lshoulder.sum() > 0 and Lelbow.sum() > 0:
                Shoulder_coords.append(Lshoulder)
                Elbow_coords.append(Lelbow)

            if Nose.sum() != 0:
                if Lear.sum() != 0 and Rear.sum() != 0:
                    Nose_coords.append(Nose)
                    Lear_coords.append(Lear)
                    Rear_coords.append(Rear)
                elif Lear.sum() != 0 and Rear.sum() == 0:
                    Nose_coords.append(Nose)
                    Lear_coords.append(Lear)
                    Rear_coords.append(None)
                elif Lear.sum() == 0 and Rear.sum() != 0:
                    Nose_coords.append(Nose)
                    Lear_coords.append(None)
                    Rear_coords.append(Rear)

        return Shoulder_coords, Elbow_coords, Nose_coords, Lear_coords, Rear_coords

    @staticmethod
    def get_strcounter(last_times):
        """
        Returns string summarizing how long each person was detected.
        """
        last_times_str = ''
        for k in last_times.keys():
            time_count = str(last_times[k]).split('.')[0]
            s = f'{k} detecting: {time_count}\n'
            last_times_str += s
        return last_times_str