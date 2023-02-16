# Neural Additive Models - Visualization
[Neural Additive Models (NAMs)](https://neural-additive-models.github.io/)([Agarwal et al. 2020](https://arxiv.org/abs/2004.13912)) combine some of the expressivity of DNNs with the inherent intelligibility of generalized additive models. NAMs learn a linear combination of neural networks that each attend to a single input feature. These networks are trained jointly and can learn arbitrarily complex relationships between their input feature and the output.

In this visualization approach, feature pair heatmaps (2D-heatmaps) and their corresponding feature maps are visualized in a [Dash](https://plotly.com/dash/) app.

## Installation
### Installation via pip
Create a virtual environment using this guidline:
[Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

### Installation via conda
Create a virtual environment using this guidline:
[Creating an environment with commands (conda)](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

After that, install the needed packages via this pip command:
```python
pip install numpy pandas torch torchmetrics scikit-learn plotly dash dash_daq dash-extensions pyautogui
```

## Usage
Activate your virtual environment and run the following command in the command line:
```python
python app.py
```
