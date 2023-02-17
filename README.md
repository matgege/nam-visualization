# Neural Additive Models - Visualization Tool

![PyPI Python Version](https://img.shields.io/badge/python-3.9-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2004.13912-b31b1b.svg)](https://arxiv.org/abs/2004.13912)
![GitHub license](https://img.shields.io/github/license/matgege/nam-visualization)

[Neural Additive Models (NAMs)](https://neural-additive-models.github.io/)([Agarwal et al. 2020](https://arxiv.org/abs/2004.13912) combine some of the expressivity of DNNs with the inherent intelligibility of generalized additive models. NAMs learn a linear combination of neural networks that each attend to a single input feature. These networks are trained jointly and can learn arbitrarily complex relationships between their input feature and the output.

In this visualization approach, feature pair heatmaps (2D-heatmaps) and their corresponding feature maps (shape functions) are visualized in a [Dash](https://plotly.com/dash/) app.

![iris_heatmaps](https://github.com/matgege/nam-visualization/blob/main/iris_heatmaps.png)
![iris_feature_maps](https://github.com/matgege/nam-visualization/blob/main/iris_feature_maps.png)

As stated in the [NAM paper](https://arxiv.org/abs/2004.13912), the shape function and the normalized data density are plotted on the same graph.
The normalized data density is visualized in the form of red bars.
The darker the shade of red, the more data there is in that region. This allows us to know when the
model had adequate training data to learn appropriate shape functions.

## Installation
### Installation via pip
Create a virtual environment using this guideline:
[Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

### Installation via conda
Create a virtual environment using this guideline:
[Creating an environment with commands (conda)](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---
### PyTorch versions
Check your system, if it's CUDA capable or not, and use the corresponding version of PyTorch.

[PyTorch versions with or without CUDA support.](https://pytorch.org/get-started/locally/)

**If necessary**, change the pip statement below with respect to your PyTorch version (torch).

---
After that, install the needed packages via this pip statement:
```python
pip install numpy pandas torch torchmetrics scikit-learn plotly dash dash_daq dash-extensions pyautogui
```

## Usage
1. Download `app.py` from this repository.
2. Activate your virtual environment.
3. Go to the directory where `app.py` is located. 
4. Run the following command in the command line:
```python
python app.py
```

## References
[A simple implementation of the Neural Additive Model by Agarwal et al. in PyTorch.](https://github.com/CursedSeraphim/NAM-torch)

[Neural Additive Models (Google Research)](https://github.com/AmrMKayid/nam)

### Python Packages References
[Numpy](https://numpy.org/)

[Pandas](https://pandas.pydata.org/)

[PyTorch](https://pytorch.org/)

[TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/)

[scikit-learn](https://scikit-learn.org/stable/)

[Plotly](https://plotly.com/python/)

[Dash](https://dash.plotly.com/)

[PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)
