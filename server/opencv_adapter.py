
from openrtist_adapter import OpenrtistAdapter
import numpy as np
import os
import logging
import cv2

logger = logging.getLogger(__name__)


STARTUP_ONES_SIZE = (360, 240, 3)


class OpenCVAdapter(OpenrtistAdapter):
    def __init__(self, cpu_only, default_style):
        super().__init__(default_style)

        self.cpu_only = cpu_only

        models_dir = "models"
        self.path = os.path.join(os.getcwd(), "..", models_dir)
        # self._update_model_style(default_style)

        # for name in os.listdir(self.path):
        #     if name.endswith(".model"):
        #         self.add_supported_style(name[:-6])
        self.add_supported_style('Invert')

    def set_style(self, new_style):
        super().set_style(new_style)

    def preprocessing(self, img):
        # content_image = self.content_transform(img)
        # if not self.cpu_only:
        #     content_image = content_image.cuda()
        # content_image = content_image.unsqueeze(0)
        # return Variable(content_image)
        return img

    def inference(self, preprocessed):
        # output = self.style_model(preprocessed)
        # return output.data[0].clamp(0, 255).cpu().numpy()
        return preprocessed

    def postprocessing(self, post_inference):
        # return post_inference.transpose(1, 2, 0)
        return post_inference

    def process(self, img):
        return cv2.bitwise_not(img)

    def _update_model_style(self, new_style):
        model = os.path.join(self.path, "{}.model".format(new_style))
