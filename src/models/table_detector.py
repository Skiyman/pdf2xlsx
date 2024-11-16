import json
import os

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.consts.model_consts import MODEL_CONFIDENCE, MODEL_THRESHOLD, MODEL_NMS_AGNOSTIC
from src.consts.path_consts import TABLES_PATH, WEIGHTS_PATH


class TableDetector:
    def __init__(self, images_path: list[str], session_id: str) -> None:
        self.images_path = images_path
        self.filenames = session_id
        self.tables_path = f"{TABLES_PATH}{self.filenames}/"

        self.model = self._load_model()
        self.predict_results = {}
        self.table_count = 0
        self.table_data = []

    @staticmethod
    def _load_model():
        model = YOLO(WEIGHTS_PATH + "best_2.pt")

        # # set model parameters
        # In this way parameters don't do anything
        # model.overrides['conf'] = MODEL_CONFIDENCE
        # model.overrides['iou'] = MODEL_THRESHOLD
        # model.overrides['agnostic_nms'] = MODEL_NMS_AGNOSTIC

        return model

    @staticmethod
    def _preprocessing_image(image: Image.Image) -> Image.Image:
        """
        :param image: Image.Image
        :return: Image.Image
        """

        image = np.array(image)
        norm_img = np.zeros((image.shape[0], image.shape[1]))

        img_normalized = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        img_denoised = cv2.fastNlMeansDenoising(img_normalized, None, 20, 7, 15)

        img_denoised = Image.fromarray(img_denoised)

        return img_denoised

    def make_predict(self) -> dict:
        for image in self.images_path:
            results = self.model.predict(
                image,
                verbose=False,
                conf=MODEL_CONFIDENCE,
                iou=MODEL_THRESHOLD,
                agnostic_nms=MODEL_NMS_AGNOSTIC
            )

            self.predict_results[image] = json.loads(results[0].to_json())

        return self.predict_results

    def _crop_table(self, table_data: dict, image_path: str, table_type: str) -> str:
        image = Image.open(image_path)
        table_coordinates = (
            table_data["box"]["x1"],
            table_data['box']["y1"],
            table_data["box"]["x2"],
            table_data["box"]["y2"]
        )
        table = image.crop(table_coordinates)

        if table_type == "rotated_left":
            table = table.rotate(-90, expand=True)

        table = self._preprocessing_image(table)

        table_path = f"{self.tables_path}{self.table_count}.jpg"
        table.save(table_path)

        return table_path

    def crop_images(self) -> list:
        os.makedirs(self.tables_path, exist_ok=True)
        for image in self.predict_results:
            if image:
                image_data = {
                    "path": image,
                    "data": []
                }
                for table in sorted(self.predict_results[image], key=lambda value: value["box"]["y1"]):
                    # self.table_paths[image].append({self.crop_table(table, image)})
                    image_data["data"].append({
                        "path": self._crop_table(table, image, table["name"]),
                        "table_class": table["name"],
                        "confidence": table["confidence"]
                    })
                    self.table_count += 1

                self.table_data.append(image_data)

        return self.table_data
