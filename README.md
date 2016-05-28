# flowstereo-predictor
Predict optical flow and stereo seperately.

## Installation
``` python
python setup.py install --user
```

## Model
This project use [Dropbox](https://www.dropbox.com/s/ekriap1abdc9yu5/model.zip?dl=0) to manage model files. 

## Model Version
Current model version is 1.0
 

## Usage
We want to make sure that APIs of car group are consistency.So this usage is almost the same as [carpipeline's](https://github.com/TuSimple/erya-fuyi-car).
You only need to do is putting .config file and model file together,and provide path of config file. 

``` python
import  flowstereo.pipe as pipe
import  flowstereo.util as util
from PIL import Image
import matplotlib.pyplot as plt
import sys

# load model 
config_path = './flowstereo/model.config'
piper = pipe.Pipeline(config_path)

# stereo: img1 is left image ,img2 is right image
# optical flow: img1 is the first frame,img2 is the second frame.

# load data 
img1 = Image.open(sys.argv[1])
img2 = Image.open(sys.argv[2])

# predict
ret = piper.process(img1,img2)

# plot result
if  piper.model_type == 'flow':
    util.plot_velocity_vector(ret)
    util.flow2color(ret)
    
elif piper.model_type == 'stereo':
    plt.imshow(ret)

```

##Config example
- model_prefix :  flow or stereo 
- ctx : indicate which gpu you want to use.
``` config
[model]
model_prefix = flow  
ctx = 0
```
