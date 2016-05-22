# flowstereo-predictor
Optical flow and stereo predictor

## Installation
``` python
python setup.py install --user
```

## Model
This project use **Aliyun** to manage model files. 

## Model Version
Current model version is 1.0
 

## Usage
``` python
import  flowstereo.pipe as pipe
import  flowstereo.util as util
from PIL import Image
import matplotlib.pyplot as plt
import sys


config_path = './flowstereo/model.config'
piper = pipe.Pipeline(config_path)

# stereo: img1 is left image ,img2 is right image
# optical flow: img1 is the first frame,img2 is the scond frame.

img1 = Image.open(sys.argv[1])
img2 = Image.open(sys.argv[2])

ret = piper.process(img1,img2)

# plot result

if  piper.model_type == 'flow':
    util.plot_velocity_vector(ret)
    util.flow2color(ret)
    
elif piper.model_type == 'stereo':
    plt.imshow(ret)

```

##Config example
``` config
[model]
model_prefix = flow  
ctx = 0
```