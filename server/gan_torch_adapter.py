
from openrtist_adapter import OpenrtistAdapter
from cut_gan.models import create_model
from cut_gan.options.test_options import TestOptions
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import os
import logging
import torch

logger = logging.getLogger(__name__)


STARTUP_ONES_SIZE = (360, 240, 3)

class GanTorchAdapter(OpenrtistAdapter):
    def __init__(self, cpu_only, default_style):
        super().__init__(default_style)

        self.cpu_only = cpu_only
        self.content_transform = transforms.Compose([transforms.ToTensor()])

        self.opt = TestOptions().parse()
        self.opt.num_threads = 0   # test code only supports num_threads = 0
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.opt.model = 'cut'
        self.opt.name = 'cityscapes_cut_pretrained'
        self.opt.checkpoints_dir = '../models/cut_gan_models'

        # models_dir = "models/cycle_gan_models"
        self.path = os.path.join(os.getcwd(), self.opt.checkpoints_dir)
        self.style_model = create_model(self.opt)
        self.style_model.setup(self.opt)
        # self._update_model_style(None)

        for name in os.listdir(self.path):
            if name.endswith("_cut_pretrained"):
                self.add_supported_style(name[:-15])

    def set_style(self, new_style):
        if super().set_style(new_style):
            self._update_model_style(new_style)

    def preprocessing(self, img):
        content_image = self.content_transform(img)
        if not self.cpu_only:
            content_image = content_image.cuda()
        content_image = content_image.unsqueeze(0)
        self.style_model.set_input(Variable(content_image))

    def inference(self):
        self.style_model.test()
    
    def postprocessing(self):
        visuals = self.style_model.get_current_visuals()
        im_data = list(visuals.values())[1]
        img = tensor2im(im_data)
        return img

    def _update_model_style(self, new_style):
        self.opt.name = new_style + "_cut_pretrained"
        self.style_model.setup(self.opt)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
