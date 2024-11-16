import math
import os
from typing import Tuple, Union

import cv2
import numpy as np
import pytesseract
import torch
import torch.nn.functional as F
from deskew import determine_skew

from src.consts.preprocessor_consts import ANGLES, ROTATIONS, FIRST_MODEL_PATH, SECOND_MODEL_PATH
from src.consts.path_consts import TABLES_PATH
from src.preprocessing.utils.crop_merge_image import stride_integral
from src.preprocessing.utils.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L
from src.preprocessing.utils.utils import convert_state_dict


class Preprocessor:
    def __init__(self, session_id: str):
        self.to_process_images = [f"{TABLES_PATH}{session_id}/{i}" for i in os.listdir(f"{TABLES_PATH}{session_id}")]

        self.model1_path = FIRST_MODEL_PATH
        self.model2_path = SECOND_MODEL_PATH

        self.cuda_is_available = torch.cuda.is_available()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # eval the models
        model1 = UNext_full_resolution_padding(num_classes=3,
                                               input_channels=3,
                                               img_size=512).to(self.device)

        state = convert_state_dict(torch.load(self.model1_path,
                                              map_location=self.device)['model_state'])
        model1.load_state_dict(state)
        model2 = UNext_full_resolution_padding_L_py_L(num_classes=3,
                                                      input_channels=6,
                                                      img_size=512).to(self.device)
        state = convert_state_dict(torch.load(self.model2_path,
                                              map_location=self.device)['model_state'])
        model2.load_state_dict(state)

        model1.eval()
        model2.eval()

        self.model1 = model1
        self.model2 = model2

    @staticmethod
    def _norm_denoise(img: np.ndarray) -> np.ndarray:
        img_normalized = cv2.normalize(img, np.zeros((img.shape[0], img.shape[1])),
                                       0, 255, cv2.NORM_MINMAX)
        img_denoised = cv2.fastNlMeansDenoising(img_normalized, None, 20, 7, 15)
        return img_denoised

    @staticmethod
    def _rotate(
            image: np.ndarray, angle: float,
            background: Union[int, Tuple[int, int, int]]
    ) -> np.ndarray:

        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(255, 255, 255))

    def _deskewing(self, img: np.ndarray) -> np.ndarray:
        angle = determine_skew(img)
        rotated = self._rotate(img, angle, (0, 0, 0))
        return rotated

    @staticmethod
    def _define_angle_90n(image_path: str) -> int:
        results = pytesseract.image_to_osd(image_path, config='--psm 0 -c min_characters_to_try=3')  # in RGB format
        angle = int(results.split('\n')[2].split()[-1])
        return angle

    def _document_like(self) -> np.ndarray:
        for image_path in self.to_process_images:
            im_org = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            im_org, padding_h, padding_w = stride_integral(im_org)
            h, w = im_org.shape[:2]
            im = cv2.resize(im_org, (512, 512))
            im = im_org
            with torch.no_grad():
                im = torch.from_numpy(im.transpose(2, 0, 1) / 255).unsqueeze(0)
                im = im.float().to(self.device)

                im_org = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0)
                im_org = im_org.float().to(self.device)

                shadow = self.model1(im)
                shadow = F.interpolate(shadow, (h, w))

                model1_im = torch.clamp(im_org / shadow, 0, 1)
                pred, _, _, _ = self.model2(torch.cat((im_org, model1_im), 1))

                shadow = shadow[0].permute(1, 2, 0).data.cpu().numpy()
                shadow = (shadow * 255).astype(np.uint8)
                shadow = shadow[padding_h:, padding_w:]

                model1_im = model1_im[0].permute(1, 2, 0).data.cpu().numpy()
                model1_im = (model1_im * 255).astype(np.uint8)
                model1_im = model1_im[padding_h:, padding_w:]

                pred = pred[0].permute(1, 2, 0).data.cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                pred = pred[padding_h:, padding_w:]

            yield image_path, pred

    def fully_preprocess(self):
        for image_path, img in self._document_like():
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            preprocessed = self._norm_denoise(self._deskewing(gray_img))  # maybe we won't apply 1st function

            cv2.imwrite(image_path, preprocessed)

            img2rotate = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            angle = self._define_angle_90n(image_path)
            if angle and angle in ANGLES:
                rotated = cv2.rotate(img2rotate, ROTATIONS[ANGLES.index(angle)])
                cv2.imwrite(image_path, rotated)