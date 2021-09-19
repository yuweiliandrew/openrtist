import cv2
import numpy as np
import logging
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import openrtist_pb2
import os
from io import BytesIO

logger = logging.getLogger(__name__)

from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json

class GanEngine(cognitive_engine.Engine):
    SOURCE_NAME = "openrtist"

    def __init__(self, compression_params, adapter):
        self.compression_params = compression_params
        self.adapter = adapter

        # The waterMark is of dimension 30x120
        wtr_mrk4 = cv2.imread("../wtrMrk.png", -1)

        # The RGB channels are equivalent
        self.mrk, _, _, mrk_alpha = cv2.split(wtr_mrk4)

        self.alpha = mrk_alpha.astype(float) / 255

        # TODO support server display

        logger.info("FINISHED INITIALISATION")

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        extras = cognitive_engine.unpack_extras(openrtist_pb2.Extras, input_frame)

        new_style = False
        send_style_list = False
        
        if extras.style == "?":
            new_style = True
            send_style_list = True
        elif extras.style != self.adapter.get_style():
            self.adapter.set_style(extras.style)
            logger.info("New Style: %s", extras.style)
            new_style = True

        style = self.adapter.get_style()

        # Preprocessing steps used by both engines
        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        orig_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # It is possible that no face is detected and style is None, if so bypass processing
        if style:
            image = self.process_image(orig_img)
        else:
            image = orig_img


        image = image.astype("uint8")
        image = self._apply_watermark(image)

        _, jpeg_img = cv2.imencode(".jpg", image, self.compression_params)
        img_data = jpeg_img.tostring()

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data

        extras = openrtist_pb2.Extras()
        
        if style:
            extras.style = style

        if new_style:
            extras.style_image.value = self.adapter.get_style_image()
        if send_style_list:
            for k, v in self.adapter.get_all_styles().items():
                extras.style_list[k] = v

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.results.append(result)
        result_wrapper.extras.Pack(extras)

        return result_wrapper

    def process_image(self, image):
        # preprocessed = self.adapter.preprocessing(image)
        # post_inference = self.inference(preprocessed)
        # img_out = self.adapter.postprocessing(post_inference)
        img_out = self.adapter.process(image)
        return img_out

    def inference(self, preprocessed):
        """Allow timing engine to override this"""
        return self.adapter.inference(preprocessed)

    def _apply_watermark(self, image):
        img_mrk = image[-30:, -120:]  # The waterMark is of dimension 30x120
        img_mrk[:, :, 0] = (1 - self.alpha) * img_mrk[:, :, 0] + self.alpha * self.mrk
        img_mrk[:, :, 1] = (1 - self.alpha) * img_mrk[:, :, 1] + self.alpha * self.mrk
        img_mrk[:, :, 2] = (1 - self.alpha) * img_mrk[:, :, 2] + self.alpha * self.mrk
        image[-30:, -120:] = img_mrk
        # img_out = image.astype("uint8")
        img_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return img_out
