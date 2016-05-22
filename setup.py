from setuptools import setup, find_packages

setup(
	    name='flowstereo-predictor',
	    version='1.0',
	    author='Xu Dong',
	    description=('Optical flow and stereo predictor'),
	    install_requires=['numpy', 'mxnet', 'Pillow'],
	    url='http://github.com/TuSimple/flowstereo-predictor',
	    packages=find_packages()
        )