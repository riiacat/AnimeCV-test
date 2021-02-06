from setuptools import setup, find_namespace_packages

requirements = """bounding-box
numpy
opencv-python
pillow
pycocotools
pyyaml
requests
torch
torchvision
tensorboardX
tqdm
webcolors
timm @ git+https://github.com/rwightman/pytorch-image-models.git
"""

setup(
    name="animecv",
    version="0.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements
)