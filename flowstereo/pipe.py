import mxnet as mx
import ConfigParser
import os
from PIL import Image
import numpy as np
import cv2

class Pipeline:

    def __init__(self, config_path):

        config = ConfigParser.ConfigParser()
        config.read(config_path)

        # model base folder
        base_folder = os.path.split(os.path.abspath(config_path))[0]

        # prefix:'stereo' or 'flow'
        self.model_type = config.get('model', 'model_prefix')
        if self.model_type not in ['stereo', 'flow']:
            raise ValueError('model prefix must be "stereo" or "flow"')

        self.ctx = mx.gpu(int(config.get('model', 'ctx')))
        model_path = os.path.join(base_folder, self.model_type)
        self.model = self.load_model(model_path)

    def load_model(self,model_path):
        net, arg_params, aux_params = mx.model.load_checkpoint(model_path,0)
        model = mx.model.FeedForward(ctx=self.ctx,
                                     symbol=net,
                                     arg_params=arg_params,
                                     aux_params=aux_params,
                                     numpy_batch_size=1)
        return model

    def preprocess_img(self, img1,img2):

        if isinstance(img1, Image.Image):
            img1 = np.asarray(img1)
        if isinstance(img2, Image.Image):
            img2 = np.asarray(img2)

        self.original_shape = img1.shape[:2]

        img1 = (img1 * 0.0039216) - np.array([0.411451, 0.432060, 0.450141])
        img2 = (img2 * 0.0039216) - np.array([0.410602, 0.431021, 0.448553])

        img1 = cv2.resize(img1,(768,384))
        img2 = cv2.resize(img2,(768,384))
        img1 = np.expand_dims(img1,0).transpose(0,3,1,2)
        img2 = np.expand_dims(img2,0).transpose(0,3,1,2)
        return img1,img2

    def process(self,img1,img2):
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




