import mxnet as mx
import ConfigParser
import os
from PIL import Image
import numpy as np
import cv2
import time
class Pipeline:

    def __init__(self, config_path):

        config = ConfigParser.ConfigParser()
        config.read(config_path)

        # model base folder
        base_folder = os.path.split(os.path.abspath(config_path))[0]

        # prefix:'stereo' or 'flow'
        self.model_type = config.get('model', 'model_prefix')
        self.need_preprocess = config.getboolean('model', 'need_preprocess')
        if self.model_type not in ['stereo', 'flow']:
            raise ValueError('model prefix must be "stereo" or "flow"')

        self.ctx = mx.gpu(int(config.get('model', 'ctx')))
        model_path = os.path.join(base_folder, self.model_type)
        self.model = self.load_model(model_path)

    def load_model(self,model_path):
        net, arg_params, aux_params = mx.model.load_checkpoint(model_path,0)
        new_arg_params = {}
        for k, v in arg_params.items():
            if k != 'img1' and k != 'img2' and not k.startswith('stereo'):
                new_arg_params[k] = v
        print new_arg_params.keys()
        model = mx.model.FeedForward(ctx=self.ctx,
                                     symbol=net,
                                     arg_params=new_arg_params,
                                     aux_params=aux_params,
                                     numpy_batch_size=1)
        return model

    def transform(self, im, pixel_means):
        im = cv2.resize(im, (768, 384))
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
        for i in range(3):
            im_tensor[0, i, :, :] = im[:, :, 2 - i] * 0.0039216 - pixel_means[2 - i]
        return im_tensor

    def preprocess_img(self, img1,img2):

        if isinstance(img1, Image.Image):
            img1 = np.asarray(img1)
        if isinstance(img2, Image.Image):
            img2 = np.asarray(img2)

        self.original_shape = img1.shape[:2]
        img1 = self.transform(img1, np.array([0.411451, 0.432060, 0.450141]))
        img2 = self.transform(img2, np.array([0.411451, 0.432060, 0.450141]))
        return img1,img2

    def process(self,img1,img2):
        if self.need_preprocess:
            img1,img2 = self.preprocess_img(img1,img2)
        batch = mx.io.NDArrayIter(data = {'img1':img1,'img2':img2})
        pred = self.model.predict(batch)[0][0].transpose(1,2,0)
        pred = cv2.resize(pred,(self.original_shape[1],self.original_shape[0]))

        if self.model_type == 'stereo':
            pred = pred * (self.original_shape[1]/768.0)
        elif self.model_type == 'flow':
            pred[:,:,0]  = pred[:,:,0] * (self.original_shape[1]/768.0)
            pred[:,:,1] =  pred[:,:,1] * (self.original_shape[0]/384.0)

        return pred




