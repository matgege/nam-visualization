import pandas as pd
import torch as th
import numpy as np
import base64
import io
import os
import torch.nn.functional as F
import torch.optim as optim
import plotly.express as px
import dash_daq as daq
import pyautogui
import webbrowser
from torch.nn.parameter import Parameter
from torchmetrics import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from math import comb
from dash import dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, ServersideOutput, ServersideOutputTransform
from plotly.subplots import make_subplots

webbrowser.open('http://127.0.0.1:8050')
device = th.device("cuda" if th.cuda.is_available() else "cpu")
seed = 2
th.manual_seed(seed)
np.random.seed(seed)


# define the Neural Additive Model
class NAM(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1,
                 activation_f="ELU", class_or_reg="Classification"):
        # for each input dimension, we have an individual network with num_layers layers
        # the output of the overall network is the sum of the outputs of the individual networks,
        # fed into a softmax for classification
        super(NAM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.submodules = th.nn.ModuleList()
        self.class_or_reg = class_or_reg
        # initialize the submodules and make sure they accept input dimension = 1 for the first layer and
        # hidden_dim for all other layers. they should have num_layers layers and output_dim output dimensions
        # also use dropout with p=0.5
        if activation_f == "ELU":
            activation = th.nn.ELU()
        else:
            activation = ExU(hidden_dim, hidden_dim)
        for i in range(input_dim):
            # initialize the submodule
            submodule = th.nn.Sequential()
            for l in range(num_layers):
                if l == 0:
                    submodule.add_module(f"linear_{l}", th.nn.Linear(1, hidden_dim))
                else:
                    submodule.add_module(f"linear_{l}", th.nn.Linear(hidden_dim, hidden_dim))
                submodule.add_module(f"ELU_{l}", activation)
                submodule.add_module(f"dropout_{l}", th.nn.Dropout(0.5))
            # each subnetwork has a final linear layer to output the final output
            submodule.add_module(f"linear_{num_layers}", th.nn.Linear(hidden_dim, output_dim))
            # add the submodule to the list of submodules
            self.submodules.append(submodule)

    def forward(self, x):
        """
        The forward pass passes each input dimension through the corresponding submodule and sums over their outputs.
        The output is then fed into a softmax if classification is selected.
        """
        # initialize the output
        output = th.zeros(x.shape[0], self.output_dim).to(device)
        # for each input dimension, pass it through the corresponding submodule and add the output to the overall output
        for i in range(self.input_dim):
            output += self.submodules[i](x[:, i].unsqueeze(1))
        if self.class_or_reg == "Classification":
            # return the softmax of the output
            return th.nn.functional.softmax(output, dim=1)
        else:
            return output

    @staticmethod
    def init_weights(m):
        if type(m) == th.nn.Linear:
            th.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # output what each submodule predicts for each input between 0 and 1 for a given resolution
    def get_feature_maps(self, resolution=100):
        # initialize the output
        output = th.zeros(resolution, self.input_dim, self.output_dim).to(device)
        # for each input dimension, pass it through the corresponding submodule and add the output to the overall output
        for i in range(self.input_dim):
            for j in range(resolution):
                output[j, i] = self.submodules[i](th.tensor([[j / resolution]]).to(device))
        # return output as numpy array
        return np.moveaxis(output.cpu().detach().numpy(), 0, -1)


class ExU(th.nn.Module):
    """
    The ExU activation function as describe in the NAM paper.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(th.Tensor(in_features, out_features))
        self.bias = Parameter(th.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Page(4): initializing the weights using a normal distribution
        #          N(x; 0:5) with x 2 [3; 4] works well in practice.
        th.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        th.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(self, inputs: th.Tensor, n: int = 1) -> th.Tensor:
        output = (inputs - self.bias).matmul(th.exp(self.weights))
        # ReLU activations capped at n (ReLU-n)
        output = F.relu(output)
        output = th.clamp(output, 0, n)
        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


def create_data_from_csv(data, target, normalize, class_or_reg, features=0):
    """
    Creates the data from the given .csv file with various options
    """
    scaler = preprocessing.MinMaxScaler()
    df = pd.DataFrame.from_dict(data)
    target_names = df[target].unique()
    y = pd.factorize(df[target])[0] if class_or_reg == "Classification" else df[target].to_numpy(dtype="float32")
    # if integers should also be excluded from categorical inclusion, simply add to the exclude statement below
    # "int64", "int32", "int"
    categorical_col = df.select_dtypes(exclude=["float64", "float32", "float"])
    encoders = {}
    # encode all selected categorical columns and save their encoders
    for col in categorical_col.columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    # if features haven been selected by the user, just use them, otherwise use all features without the target
    if features:
        feature_names = features
        X = df[features].to_numpy(dtype="float32")
    else:
        X = df.loc[:, df.columns != target]
        feature_names = df.columns[df.columns != target]

    X = np.array(X, dtype="float32")
    # store the original range of the input dimensions
    input_ranges = np.zeros((X.shape[1], 2))
    if normalize:
        for i in range(X.shape[1]):
            input_ranges[i, 0] = np.min(X[:, i])
            input_ranges[i, 1] = np.max(X[:, i])
            X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))
        # normalize the data
        X = scaler.fit_transform(X)
    else:
        for i in range(X.shape[1]):
            input_ranges[i, 0] = np.min(X[:, i])
            input_ranges[i, 1] = np.max(X[:, i])

    return X, y, feature_names, target_names, input_ranges, encoders


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


def create_nam_model(input_dim, hidden_dim, output_dim, class_or_reg, num_layers=2, activation_f="ELU"):
    model = NAM(input_dim, hidden_dim, output_dim, num_layers, activation_f, class_or_reg)
    model.to(device)
    model.apply(model.init_weights)
    return model


def define_optimizer(optimizer, model_para, lr=4e-3, momentum=0):
    return optimizer(model_para, lr) if momentum == 0 else optimizer(model_para, lr, momentum)


def define_loss_function(loss_fn):
    return loss_fn


def train(model, X, y, optimizer, loss_fn, class_or_reg):
    # set model to training mode
    model.train()
    # convert data to tensors
    X = th.tensor(X, dtype=th.float32).to(device)
    if class_or_reg == "Classification":
        y = th.tensor(y, dtype=th.long).to(device)
    else:
        y = th.tensor(y, dtype=th.float32).to(device).unsqueeze(1)
    # zero the gradients
    optimizer.zero_grad()
    # forward pass
    output = model(X)
    # compute loss
    loss = loss_fn(output, y)
    # backpropagation
    loss.backward()
    # update parameters
    optimizer.step()
    return loss.item()


def evaluate(model, X, y, loss_fn, class_or_reg):
    # set model to evaluation mode
    model.eval()
    # convert data to tensors
    X = th.tensor(X, dtype=th.float32).to(device)
    if class_or_reg == "Classification":
        y = th.tensor(y, dtype=th.long).to(device)
    else:
        y = th.tensor(y, dtype=th.float32).to(device).unsqueeze(1)
    # forward pass
    output = model(X)
    # compute loss
    loss = loss_fn(output, y)
    # compute accuracy
    if class_or_reg == "Classification":
        accuracy = (output.argmax(dim=1) == y).sum().item() / y.shape[0]
        return loss.item(), accuracy
    else:
        # compute the mean squared error
        mean_squared_error = MeanSquaredError()
        mse = mean_squared_error(output, y)
        return loss.item(), mse.detach().numpy()


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, optimizer, loss_fn, class_or_reg, epochs=1000):
    # train epochs
    for epoch in range(epochs):
        # train
        train_loss = train(model, X_train, y_train, optimizer, loss_fn, class_or_reg)
        # evaluate
        test_loss, test_accuracy = evaluate(model, X_test, y_test, loss_fn, class_or_reg)
        # print
        if epoch % 100 == 0:
            if class_or_reg == "Classification":
                print(
                    f'Epoch {epoch + 1}: train loss = {train_loss:.4f}, test loss = {test_loss:.4f},'
                    f' test accuracy = {test_accuracy:.4f}'
                )
            else:
                print(
                    f'Epoch {epoch + 1}: train loss = {train_loss:.4f}, test loss = {test_loss:.4f},'
                    f' test MSE = {test_accuracy:.4f}'
                )

    # print the final test accuracy
    if class_or_reg == "Classification":
        print(f'Final test accuracy: {test_accuracy:.4f}')
    else:
        print(f'Final test MSE: {test_accuracy:.4f}')

    return model, test_accuracy


def create_dataframe(X, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    return df


def prediction_from_grid(model, grid):
    model_output = model(th.tensor(grid, dtype=th.float32).to(device))
    return model_output.cpu().detach().numpy()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def feature_grid(df, col_names, pos, resolution, encoder_dict, input_ranges):
    """
    This feature_grid method is only usable for datasets with maximum 32 features.
    It uses less RAM than feature_grid_total, but it is also slower than feature_grid_total.
    """
    col_list = []
    col_names_iter = iter(col_names)
    col_name1 = next(col_names_iter)
    col_name2 = next(col_names_iter)
    for i in range(len(df.columns)):
        if df.columns[i] == col_name1:
            if col_name1 in encoder_dict:
                num = int(input_ranges[1].iloc[i]) + 1
                col_list.append(np.linspace(df[col_name1].min(), df[col_name1].max(), num))
            else:
                col_list.append(np.linspace(df[col_name1].min(), df[col_name1].max(), resolution))
        elif df.columns[i] == col_name2:
            if col_name2 in encoder_dict:
                num = int(input_ranges[1].iloc[i]) + 1
                col_list.append(np.linspace(df[col_name2].min(), df[col_name2].max(), num))
            else:
                col_list.append(np.linspace(df[col_name2].min(), df[col_name2].max(), resolution))
        else:
            col_list.append(df.iloc[pos][df.columns[i]])
    return np.c_[[n.ravel() for n in np.meshgrid(*col_list)]].T


def feature_grid_above_32(df, col_names, pos, resolution, encoder_dict, input_ranges):
    """
    This feature_grid method is meant for datasets above 32 features.
    It uses less RAM than feature_grid_total, but it is also slower than feature_grid_total and feature_gird.
    """
    grid = []
    col_list = []
    col_names_iter = iter(col_names)
    col_name1 = next(col_names_iter)
    col_name2 = next(col_names_iter)
    col_name1_pos = np.where(df.columns == col_name1)
    col_name2_pos = np.where(df.columns == col_name2)
    if col_name1 in encoder_dict:
        num = int(input_ranges[1].iloc[col_name1_pos]) + 1
        col_values1 = np.linspace(df[col_name1].min(), df[col_name1].max(), num)
    else:
        col_values1 = np.linspace(df[col_name1].min(), df[col_name1].max(), resolution)
    if col_name2 in encoder_dict:
        num = int(input_ranges[1].iloc[col_name2_pos]) + 1
        col_values2 = np.linspace(df[col_name2].min(), df[col_name2].max(), num)
    else:
        col_values2 = np.linspace(df[col_name2].min(), df[col_name2].max(), resolution)

    for k in range(len(df.columns)):
        if df.columns[k] == col_name1:
            col_list.append(col_values1[0])
        elif df.columns[k] == col_name2:
            col_list.append(col_values2[0])
        else:
            col_list.append(df.iloc[pos][df.columns[k]])
    col_list = np.array(col_list)
    for i in col_values1:
        col_list_c = col_list.copy()
        col_list_c[col_name1_pos] = i
        for j in col_values2:
            col_list_cc = col_list_c.copy()
            col_list_cc[col_name2_pos] = j
            grid.append(col_list_cc)
    return np.array(grid)


def feature_grid_total(df, feature_names, resolution, encoder_dict, input_ranges):
    """
    This feature grid method is faster than feature_grid and feature_grid_above_32 and can be used with every dataset.
    It uses more RAM than feature_grid and feature_grid_above_32, but it is also faster than both other methods.
    """
    col_list = []
    feature_names_iter = iter(feature_names)
    feature_name1 = next(feature_names_iter)
    feature_name2 = next(feature_names_iter)
    position = {feature_name1: None, feature_name2: None}
    ca_range = {}

    for i in range(len(df.columns)):
        if df.columns[i] == feature_name1:
            if feature_name1 in encoder_dict:
                num = int(input_ranges[1].iloc[i]) + 1
                col_list.append(np.linspace(df[feature_name1].min(), df[feature_name1].max(), num))
                ca_range[feature_name1] = num
            else:
                col_list.append(np.linspace(df[feature_name1].min(), df[feature_name1].max(), resolution))
            position[feature_name1] = i
        elif df.columns[i] == feature_name2:
            if feature_name2 in encoder_dict:
                num = int(input_ranges[1].iloc[i]) + 1
                col_list.append(np.linspace(df[feature_name2].min(), df[feature_name2].max(), num))
                ca_range[feature_name2] = num
            else:
                col_list.append(np.linspace(df[feature_name2].min(), df[feature_name2].max(), resolution))
            position[feature_name2] = i

    features = np.c_[[n.ravel() for n in np.meshgrid(*col_list)]].T
    feature1 = np.repeat(features[:, 0], len(df))
    feature2 = np.repeat(features[:, 1], len(df))
    full_grid = np.array
    for i in range(len(df.columns)):
        if position[feature_name1] == i:
            if i == 0:
                full_grid = feature1
            else:
                full_grid = np.c_[full_grid, feature1]
        elif position[feature_name2] == i:
            if i == 0:
                full_grid = feature2
            else:
                full_grid = np.c_[full_grid, feature2]
        else:
            if i == 0:
                if len(ca_range.keys()) == 0:
                    full_grid = np.tile(df[df.columns[i]], resolution ** 2)
                elif len(ca_range.keys()) == 1:
                    if feature_name1 in ca_range:
                        multiplier = ca_range[feature_name1]
                    else:
                        multiplier = ca_range[feature_name2]
                    full_grid = np.tile(df[df.columns[i]], resolution * multiplier)
                else:
                    full_grid = np.tile(df[df.columns[i]], ca_range[feature_name1] * ca_range[feature_name2])
            else:
                if len(ca_range.keys()) == 0:
                    full_grid = np.c_[full_grid, np.tile(df[df.columns[i]], resolution ** 2)]
                elif len(ca_range.keys()) == 1:
                    if feature_name1 in ca_range:
                        multiplier = ca_range[feature_name1]
                    else:
                        multiplier = ca_range[feature_name2]
                    full_grid = np.c_[full_grid, np.tile(df[df.columns[i]], resolution * multiplier)]
                else:
                    full_grid = np.c_[
                        full_grid, np.tile(df[df.columns[i]], ca_range[feature_name1] * ca_range[feature_name2])]
    return full_grid


def mean_prediction_from_grids_less_RAM(df, model, feature_names, resolution, encoder_dict, input_ranges):
    """
    Returns the mean prediction from all grids.
    Which will be slower than mean_prediction_from_grids, but uses less RAM.
    """
    model_output_list = []
    if len(df.columns) <= 32:
        feature_g = feature_grid
    else:
        feature_g = feature_grid_above_32
    for i in range(len(df)):
        grid = feature_g(df, feature_names, i, resolution, encoder_dict, input_ranges)
        model_output = prediction_from_grid(model, grid)
        model_output_list.append(model_output)
    x_feature = np.where(grid[0] - grid[1] != 0)
    if df.columns[x_feature] != feature_names[0]:
        feature_names = np.flip(feature_names)
    model_output_array = np.array(model_output_list)
    return np.mean([output for output in model_output_array], axis=0), feature_names


def mean_prediction_from_grids(df, model, feature_names, resolution, target_names, encoder_dict, input_ranges):
    """
    Returns the mean prediction from all grids.
    Which will be faster than mean_prediction_from_grids_less_RAM, but uses more RAM.
    """
    grid = feature_grid_total(df, feature_names, resolution, encoder_dict, input_ranges)
    model_output = prediction_from_grid(model, grid)
    if target_names:
        return np.mean([output for output in model_output.reshape(-1, len(df), len(target_names))],
                       axis=1), feature_names
    else:
        return np.mean([output for output in model_output.reshape(-1, len(df), 1)], axis=1), feature_names


def heatmap_from_features(input_ranges, all_feature_names, resolution, model_prediction, class_num, feature_names,
                          encoder_dict, class_or_reg, target):
    """
    Creates a heatmap from two given features.
    """
    cla = True if class_or_reg == "Classification" else False
    x_pos = np.where(np.array(all_feature_names) == feature_names[0])[0][0]
    x_min = np.min(input_ranges.iloc[x_pos])
    x_max = np.max(input_ranges.iloc[x_pos])
    if feature_names[0] in encoder_dict:
        x_range = encoder_dict[feature_names[0]].inverse_transform(np.arange(input_ranges[1].iloc[x_pos]+1, dtype="int"))
    else:
        x_range = np.linspace(x_min, x_max, resolution)
    y_pos = np.where(np.array(all_feature_names) == feature_names[1])[0][0]
    y_min = np.min(input_ranges.iloc[y_pos])
    y_max = np.max(input_ranges.iloc[y_pos])
    if feature_names[1] in encoder_dict:
        y_range = encoder_dict[feature_names[1]].inverse_transform(np.arange(input_ranges[1].iloc[y_pos]+1, dtype="int"))
    else:
        y_range = np.linspace(y_min, y_max, resolution)
    prediction = model_prediction[:, class_num].reshape(len(y_range), len(x_range))
    fig = px.imshow(prediction,
                    labels=dict(x=f'{feature_names[0]}', y=f'{feature_names[1]}', color="Prediction" if cla else target)
                    , x=x_range, y=y_range)
    return fig.data[0]


def plot_heatmap_feature_pair(df, model, input_ranges, feature_names, all_target_names, target_names, all_feature_names,
                              resolution, class_or_reg, target, height_subplot, width_subplot, encoder_dict, RAM_option,
                              num_columns=4):
    """
    Returns the heatmap for a given feature pair.
    """
    cla = True if class_or_reg == "Classification" else False
    rows = 1
    if cla:
        if len(target_names) > num_columns:
            multiple = int(np.ceil(len(target_names) / num_columns))
            rows = rows * multiple
        else:
            num_columns = len(target_names)

    subplot_titles = [f'{x}' for x in target_names] if cla else ""
    if RAM_option:
        mp = mean_prediction_from_grids(df, model, [feature_names[0], feature_names[1]], resolution,
                                        all_target_names if cla else 0, encoder_dict, input_ranges)
    else:
        mp = mean_prediction_from_grids_less_RAM(df, model, [feature_names[0], feature_names[1]], resolution,
                                                 encoder_dict, input_ranges)
    fig = make_subplots(rows=rows, cols=num_columns if cla else 1, x_title=f'{mp[1][0]}', y_title=f'{mp[1][1]}',
                        subplot_titles=subplot_titles)

    if cla:
        for j in range(len(target_names)):
            row = int(np.ceil((j + 1) / num_columns))
            col = j % num_columns + 1
            pos = np.where(np.array(all_target_names) == target_names[j])[0][0]
            fig.add_trace(
                heatmap_from_features(input_ranges, all_feature_names, resolution, mp[0], pos, mp[1], encoder_dict,
                                      class_or_reg, target)
                , row=row, col=col)
    else:
        fig.add_trace(
            heatmap_from_features(input_ranges, all_feature_names, resolution, mp[0], 0, mp[1], encoder_dict,
                                  class_or_reg, target)
            , row=1, col=1)
    text = f"Classification on {target}" if cla else f"Regression on {target}"
    if width_subplot != 0:
        fig.update_layout(
            height=height_subplot * rows, width=width_subplot * num_columns if cla else width_subplot,
            coloraxis_colorbar=dict(title="Prediction" if cla else target),
            title={'text': text, 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    else:
        fig.update_layout(
            height=height_subplot * rows,
            coloraxis_colorbar=dict(title="Prediction" if cla else target),
            title={'text': text, 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    return fig


def plot_feature_heatmaps(df, model, input_ranges, features_names, all_target_names, target_names,
                          all_feature_names, resolution, class_or_reg, target, num_heatmaps, height_subplot,
                          width_subplot, encoder_dict, RAM_option, num_columns=4):
    """
    Automatically returns the most useful heatmaps filter by their variance.
    """
    cla = True if class_or_reg == "Classification" else False

    def subplot():
        row = int((i * multiple if cla else 0) + np.ceil((j + 1) / num_columns))
        col = j % num_columns + 1
        if RAM_option:
            mp = mean_prediction_from_grids(df, model, [features[j][0], features[j][1]], resolution,
                                            all_target_names if cla else 0, encoder_dict, input_ranges)
        else:
            mp = mean_prediction_from_grids_less_RAM(df, model, [features[j][0], features[j][1]], resolution,
                                                     encoder_dict, input_ranges)
        if cla:
            if isinstance(target_names[i], str):
                mask = np.array(all_target_names, dtype="str") == target_names[i]
            else:
                mask = np.array(all_target_names, dtype="float32") == float(target_names[i])
        fig.add_trace(
            heatmap_from_features(input_ranges, all_feature_names, resolution, mp[0], np.where(mask)[0] if cla else 0,
                                  mp[1], encoder_dict, class_or_reg, target), row=row, col=col
        )

        fig.update_xaxes(title=f'{mp[1][0]}', row=row, col=col)
        fig.update_yaxes(title=f'{mp[1][1]}', row=row, col=col)

    lower = 0
    upper = 1
    heatmap_var = {}
    for i in range(comb(len(features_names), 2)):
        if upper == len(features_names):
            lower += 1
            upper = lower + 1
        # for the filtering of the heatmaps only a resolution of 10 is used, to be more efficient.
        # if you have the doubt that the filter is in some cases weird, increase the resolution, this might help.
        if RAM_option:
            mp = mean_prediction_from_grids(df, model, [features_names[lower], features_names[upper]],
                                            10, all_target_names if cla else 0, encoder_dict, input_ranges)
        else:
            mp = mean_prediction_from_grids_less_RAM(df, model, [features_names[lower], features_names[upper]], 20,
                                                     encoder_dict, input_ranges)
        for j in range(len(target_names if cla else target)):
            if cla:
                mask = np.array(all_target_names) == target_names[j]
                heatmap_var[f'{mp[1][0]},{mp[1][1]},{target_names[j]}'] = mp[0][:, np.where(mask)[0]].var()
            else:
                heatmap_var[f'{mp[1][0]},{mp[1][1]},{target}'] = mp[0].var()
        upper += 1

    rows = len(target_names) if cla else 1
    multiple = 1
    repeats = num_heatmaps
    if num_heatmaps > num_columns:
        multiple = int(np.ceil(num_heatmaps / num_columns))
        rows = rows * multiple
        repeats = num_heatmaps * multiple
    else:
        num_columns = num_heatmaps

    heat_sorted = sorted(heatmap_var, key=heatmap_var.get, reverse=True)
    subplot_titles = [f'{name}' for name in np.repeat(target_names if cla else "", repeats)]
    fig = make_subplots(rows=rows, cols=num_columns, subplot_titles=subplot_titles)

    for i in range(len(target_names if cla else target)):
        features = [k.split(",")[:-1] for k in heat_sorted if
                    k.split(",")[-1] == str(target_names[i] if cla else target)][:num_heatmaps]
        for j in range(len(features)):
            subplot()

    text = f"Classification on {target}" if cla else f"Regression on {target}"
    if width_subplot != 0:
        fig.update_layout(height=height_subplot * rows, width=width_subplot * num_columns,
                          coloraxis_colorbar=dict(title="Prediction" if cla else target),
                          title={'text': text, 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    else:
        fig.update_layout(height=height_subplot * rows,
                          coloraxis_colorbar=dict(title="Prediction" if cla else target),
                          title={'text': text, 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    return fig, heat_sorted


def plot_feature_maps(model, input_ranges, all_feature_names, all_target_names, feature_names, target_names, resolution,
                      class_or_reg, densities, target, num_heatmaps, heatmaps_sorted, height_subplot, width_subplot,
                      encoder_dict, num_columns=4):
    """
    Returns the corresponding feature maps, which are related to the heatmaps in the method plot_feature_heatmaps.
    """
    cla = True if class_or_reg == "Classification" else False

    def subplot(feature, target_name):
        row = int((i * multiple if cla else 0) + np.ceil((j + 1) / num_columns))
        col = j % num_columns + 1
        x_pos = np.where(np.array(all_feature_names) == feature)[0][0]
        x_min = np.min(input_ranges.iloc[x_pos])
        x_max = np.max(input_ranges.iloc[x_pos])
        if feature in encoder_dict:
            x_range = encoder_dict[feature].inverse_transform(np.arange(input_ranges[1].iloc[x_pos] + 1, dtype="int"))
        else:
            x_range = np.linspace(x_min, x_max, resolution)
        x_ind = np.where(np.array(all_feature_names) == feature)[0][0]
        if cla:
            if isinstance(target_name, str):
                y_ind = np.where(np.array(all_target_names, dtype="str") == target_name)[0][0]
            else:
                y_ind = np.where(np.array(all_target_names, dtype="float32") == float(target_name))[0][0]
        else:
            y_ind = 0

        if feature in encoder_dict:
            y_index = [find_nearest(np.linspace(0, len(x_range) - 1, resolution), x) for x in np.arange(len(x_range))]
            y_value = feature_maps[x_ind, y_ind, :][y_index]
        else:
            y_value = feature_maps[x_ind, y_ind, :]

        fig.add_trace(
            px.line(x=x_range, y=y_value, labels={"x": feature}).data[0], row=row, col=col
        )
        for param in densities[y_ind][x_ind] if cla else densities[x_ind]:
            opacity, start_x, end_x = param
            y0 = np.min(feature_maps[x_ind, y_ind, :])
            y1 = np.max(feature_maps[x_ind, y_ind, :])
            scale = np.abs(y1 - y0) * 0.05
            y0 = y0 - scale if y0 < 0 else 0
            y1 = y1 + scale if y1 > 0 else 0
            fig.add_shape(type="rect", x0=start_x, x1=end_x, y0=y0, y1=y1, fillcolor="red", opacity=opacity,
                          layer="below", row=row, col=col, line=dict(width=0)
                          )
        fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0, layer="above", row=row, col=col,
                      line=dict(width=1.5, dash="dash", color="black"))
        fig.update_xaxes(title=f'{feature}', row=row, col=col)

    all_used_features = []
    for i in range(len(target_names) if cla else 1):
        features = [k.split(",")[:-1] for k in heatmaps_sorted if
                    k.split(",")[-1] == str(target_names[i] if cla else target)][:num_heatmaps]
        for j in range(len(features)):
            all_used_features.extend(features)

    unique_features = np.unique(all_used_features)
    cols = len(unique_features)
    rows = len(target_names) if cla else 1
    multiple = 1

    repeats = cols
    if cols > num_columns:
        multiple = int(np.ceil(cols / num_columns))
        rows = rows * multiple
        repeats = num_columns * multiple
    else:
        num_columns = cols

    subplot_titles = [f'{name}' for name in np.repeat(target_names if cla else "", repeats)]
    fig = make_subplots(rows=rows, cols=num_columns, subplot_titles=subplot_titles)
    feature_maps = model.get_feature_maps(resolution)

    for i in range(len(target_names) if cla else 1):
        for j in range(cols):
            subplot(unique_features[j], target_names[i] if cla else target)

    text = f"Classification on {target}" if cla else f"Regression on {target}"
    if width_subplot != 0:
        fig.update_layout(height=height_subplot * rows, width=width_subplot * num_columns,
                          title={'text': text, 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    else:
        fig.update_layout(height=height_subplot * rows,
                          title={'text': text, 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    return fig


def plot_feature_maps_pair(model, input_ranges, all_feature_names, all_target_names, feature_names, target_names,
                           resolution, class_or_reg, densities, target, height_subplot, width_subplot, encoder_dict,
                           num_columns=4):
    """
    Returns the corresponding feature maps, which are related to the heatmap in plot_heatmap_feature_pair.
    """
    cla = True if class_or_reg == "Classification" else False

    def calc_x_values():
        x_pos = np.where(np.array(all_feature_names) == feature_names[i])[0][0]
        x_min = np.min(input_ranges.iloc[x_pos])
        x_max = np.max(input_ranges.iloc[x_pos])
        if feature_names[i] in encoder_dict:
            x_range = encoder_dict[feature_names[i]].inverse_transform(np.arange(input_ranges[1].iloc[x_pos] + 1, dtype="int"))
        else:
            x_range = np.linspace(x_min, x_max, resolution)

        return x_range, x_min, x_max, x_pos

    feature_maps = model.get_feature_maps(resolution)
    if cla:
        cols = len(target_names)
        rows = len(feature_names)
        multiple = 1

        if cols > num_columns:
            multiple = int(np.ceil(cols / num_columns))
            rows = rows * multiple
        else:
            num_columns = len(target_names)

    if cla:
        fig = make_subplots(rows=rows, cols=num_columns)
        for i in range(len(feature_names)):
            x_range, x_min, x_max, x_pos = calc_x_values()
            for j in range(len(target_names)):
                row = int(i * multiple + np.ceil((j + 1) / num_columns))
                col = j % num_columns + 1
                y_pos = np.where(np.array(all_target_names) == target_names[j])[0][0]
                if feature_names[i] in encoder_dict:
                    y_index = [find_nearest(np.linspace(0, len(x_range) - 1, resolution), x) for x in
                               np.arange(len(x_range))]
                    y_value = feature_maps[x_pos, y_pos, :][y_index]
                else:
                    y_value = feature_maps[x_pos, y_pos, :]
                fig.add_trace(
                    px.line(x=x_range, y=y_value, labels={"x": feature_names[i]}).data[0]
                    , row=row, col=col)
                for param in densities[y_pos][x_pos]:
                    opacity, start_x, end_x = param
                    y0 = np.min(feature_maps[x_pos, y_pos, :])
                    y1 = np.max(feature_maps[x_pos, y_pos, :])
                    scale = np.abs(y1 - y0) * 0.05
                    y0 = y0 - scale if y0 < 0 else 0
                    y1 = y1 + scale if y1 > 0 else 0
                    fig.add_shape(type="rect", x0=start_x, x1=end_x, y0=y0, y1=y1, fillcolor="red", opacity=opacity,
                                  layer="below", row=row, col=col, line=dict(width=0)
                                  )
                fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0, layer="above", row=row, col=col,
                              line=dict(width=1.5, dash="dash", color="black"))
                fig.update_xaxes(title=f'{feature_names[i]}', row=row, col=col)
                fig.add_annotation(xref="x domain", yref="y domain", x=0.5, y=1.1, showarrow=False,
                                   text=target_names[j], row=row, col=col, font={"size": 17})
    else:
        fig = make_subplots(rows=1, cols=2)
        for i in range(len(feature_names)):
            x_range, x_min, x_max, x_pos = calc_x_values()
            if feature_names[i] in encoder_dict:
                y_index = [find_nearest(np.linspace(0, len(x_range)-1, resolution), x) for x in np.arange(len(x_range))]
                y_value = feature_maps[x_pos, 0, :][y_index]
            else:
                y_value = feature_maps[x_pos, 0, :]
            fig.add_trace(px.line(x=x_range, y=y_value, labels={"x": feature_names[i]}).data[0], row=1, col=i + 1)
            for param in densities[x_pos]:
                opacity, start_x, end_x = param
                fig.add_vrect(start_x, end_x, fillcolor="red", opacity=opacity,
                              layer="below", line_width=0, row=1, col=i + 1)
            fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0, layer="above", row=1, col=i + 1,
                          line=dict(width=1.5, dash="dash", color="black"))
            fig.update_xaxes(title=f'{feature_names[i]}', row=1, col=i + 1)

    text = f"Classification on {target}" if cla else f"Regression on {target}"
    if width_subplot != 0:
        fig.update_layout(title={'text': text, 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                          height=height_subplot * rows if cla else height_subplot,
                          width=width_subplot * num_columns if cla else width_subplot * 2)
    else:
        fig.update_layout(title={'text': text, 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                          height=height_subplot * rows if cla else height_subplot)
    return fig


def calculate_density_parameter(data, feature_names, target_names, target, class_or_reg, n_blocks=20):
    """
    Returns the density for each used feature.
    """
    df = pd.DataFrame.from_dict(data)
    densities = []
    if class_or_reg == "Classification":
        for target_name in target_names:
            collect_density = []
            mask = df[target] == target_name
            for feature in feature_names:
                if isinstance(df[feature][0], str):
                    single_feature_data = pd.factorize(df[feature][mask])[0]
                    unique_feat_data = pd.factorize(df[feature][mask].unique())[0]
                else:
                    single_feature_data = df[feature][mask]
                    unique_feat_data = df[feature][mask].unique()
                x_n_blocks = min(n_blocks, len(unique_feat_data))
                min_x = np.min(unique_feat_data)
                max_x = np.max(unique_feat_data)
                segments = (max_x - min_x) / x_n_blocks
                density = np.histogram(single_feature_data, bins=x_n_blocks)
                normed_density = density[0] / np.max(density[0])
                rect_params = []
                for p in range(x_n_blocks):
                    start_x = min_x + segments * p
                    end_x = min_x + segments * (p + 1)
                    d = min(1.0, 0.01 + normed_density[p])
                    rect_params.append((d, start_x, end_x))
                collect_density.append(rect_params)
            densities.append(collect_density)
    else:
        for feature in feature_names:
            if isinstance(df[feature][0], str):
                single_feature_data = pd.factorize(df[feature])[0]
                unique_feat_data = pd.factorize(df[feature].unique())[0]
            else:
                single_feature_data = df[feature]
                unique_feat_data = df[feature].unique()
            x_n_blocks = min(n_blocks, len(unique_feat_data))
            min_x = np.min(unique_feat_data)
            max_x = np.max(unique_feat_data)
            segments = (max_x - min_x) / x_n_blocks
            density = np.histogram(single_feature_data, bins=x_n_blocks)
            normed_density = density[0] / np.max(density[0])
            rect_params = []
            for p in range(x_n_blocks):
                start_x = min_x + segments * p
                end_x = min_x + segments * (p + 1)
                d = min(1.0, 0.01 + normed_density[p])
                rect_params.append((d, start_x, end_x))
            densities.append(rect_params)
    return densities


def permutation_importance(model, X, y, loss_fn, num_rep, class_or_reg):
    """
    Returns the permutation importance's and their standard deviations for given data.
    """
    cla = True if class_or_reg == "Classification" else False
    np.random.seed(seed)
    th.manual_seed(seed)
    ref_acc = evaluate(model, X, y, loss_fn, class_or_reg)[1]
    feature_importance_ = []
    importance_std = []
    for i in range(X.shape[1]):
        per_score = []
        for j in range(num_rep):
            X_c = X.copy()
            per = np.random.permutation(X[:, i])
            X_c[:, i] = per
            per_acc = evaluate(model, X_c, y, loss_fn, class_or_reg)[1]
            per_score.append(per_acc)
        feature_importance_.append(np.array(per_score).mean())
        importance_std.append(np.array(per_score).std())
    per_im = ref_acc - np.array(feature_importance_) if cla else np.array(feature_importance_) - ref_acc
    return per_im, np.array(importance_std)


def find_important_features(model, X, y, loss_fn, num_rep, feature_names, im_threshold, class_or_reg):
    """
    Filters the permutation importance's for the most useful ones.
    """
    per_im, std_im = permutation_importance(model, X, y, loss_fn, num_rep, class_or_reg)
    per_importance = []
    per_im_sorted = []
    std_im_sorted = []
    for i in per_im.argsort()[::-1]:
        if len(per_im_sorted) == im_threshold:
            break
        else:
            per_im_sorted.append(per_im[i])
            per_importance.append(feature_names[i])
            std_im_sorted.append(std_im[i])
    return np.array(per_im_sorted), np.array(per_importance), np.array(std_im_sorted)


def plot_permutation_importance(per_im_sorted, per_importance, std_error, class_or_reg):
    """
    Returns the plots of the permutation importance's.
    """
    cla = True if class_or_reg == "Classification" else False
    feature_importance_ = pd.Series(per_im_sorted, index=per_importance)
    fig = px.bar(title='Feature importance using permutation on full model', x=per_importance, y=feature_importance_,
                 error_y=std_error,
                 labels={'x': 'Feature', 'y': 'Mean accuracy decrease' if cla else "Mean squared error decrease"})
    fig.update_traces(marker_color='#2ca02c')
    return fig


# This section is the Dash part, which is responsible for the web browser interface
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DashProxy(__name__, transforms=[ServersideOutputTransform()], external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Neural Additive Models - NAMs', style={'textAlign': 'center'}),

    html.H3('Explainable Artificial Intelligence', style={'textAlign': 'center'}),

    html.H5('Select a .csv file from your System', style={'textAlign': 'center'}),
    html.H5('The dataset has to be free of NaN values or other kinds of missing values -->'
            ' it should be properly preprocessed!', style={'textAlign': 'center'}),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '98%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    dcc.Loading(html.Div(id='output-data-upload'), type="circle"),

    html.H4("If something goes wrong, refresh the page or kill the application:"),
    html.Button(id="refresh", n_clicks=0, children='Refresh Page',
                style={"background-color": "#ff6600", "height": "50px"}),
    html.Button(id="kill", n_clicks=0, children='Kill Application',
                style={"background-color": "#ff0000", "height": "50px"}),

    html.Hr(),  # horizontal line

    html.Div([
        html.H6("Normalize data:"),
        daq.BooleanSwitch(id='normalize', on=True, label="Normalize data")
    ]),

    html.H6("Classification or Regression:"),
    dcc.Dropdown(["Classification", "Regression"], "Classification", id="class_or_reg"),

    html.H6('Select target:'),
    dcc.Dropdown(id='s_target'),

    html.Div([
        html.H6("Use only select features:"),
        daq.BooleanSwitch(id='sel_features', on=False, label="Select features"),
        html.Div(id='use_features', children="")
    ]),

    html.H6("Select activation function:"),
    dcc.Dropdown(["ELU", "ExU"], "ELU", id="activation_f"),

    html.H6("Select optimizer:"),
    dcc.Dropdown(["Adam", "SGD"], "Adam", id="optimizer_c"),
    html.Div(id="momentum_input"),

    html.H6("Learning rate of the optimizer:"),
    "Learning rate: ",
    dcc.Input(id='learning_rate', value=4e-3, type='number'),

    html.H6("Number of hidden layers:"),
    "Hidden layers: ",
    dcc.Input(id='hidden_layers', value=10, type='number'),

    html.H6("Number of layers for each individual network:"),
    "Number of layers: ",
    dcc.Input(id='num_layers', value=2, type='number'),

    html.H6("Number of training epochs for the NAM."),
    "Epochs: ",
    dcc.Input(id='epochs', value=1000, type='number'),

    html.H4("Run the Neural Additive Model:"),
    html.Button(id='run_model', n_clicks=0, children='Run model', style={"background-color": "#00ff00"}),

    dcc.Loading(html.Div(id="test_accuracy"), type="dot"),

    html.Hr(),  # horizontal line

    html.H4("Show heatmaps and feature-maps:"),
    html.Div([
        html.H6('Select target classes for the plots:'),
        daq.BooleanSwitch(id='select_t_class', on=False, label="Select target classes"),
        html.Div(id='use_t_class', children="")
    ]),
    html.H6("Choose between RAM intensive & faster or less RAM intensive & slower"),
    dcc.Dropdown(["RAM intensive & faster", "less RAM intensive & slower"], "RAM intensive & faster", id="RAM"),
    html.H6("Amount of columns for the subplots:"),
    "Columns: ",
    dcc.Input(id='num_columns', value=4, type='number'),
    html.H6('Number of most useful heatmaps for each selected class (minimum input 1):'),
    "Heatmaps: ",
    dcc.Input(id='num_heatmaps', value=3, type='number'),
    html.H6("Height of each subplot in pixel:"),
    "Height: ",
    dcc.Input(id='height_subplot', value=400, type='number'),
    html.H6("Width of each subplot in pixel (if set to zero, auto-width is enabled):"),
    "Width: ",
    dcc.Input(id='width_subplot', value=0, type='number'),  # if set to zero, auto-width is enabled
    html.H6("Resolution of the plots (minimum input 2):"),
    "Resolution: ",
    dcc.Input(id='resolution', value=51, type='number'),
    html.Button(id='plot_heatmaps', n_clicks=0, children='Show heatmaps', style={"background-color": "#00ccff"}),

    html.H6("Most useful heatmaps for each selected class:"),
    dcc.Loading(dcc.Graph(id='heatmaps'), type="cube"),
    html.H6("Feature-maps for each selected class:"),
    dcc.Loading(dcc.Graph(id='feature_maps'), type="graph"),

    html.Hr(),  # horizontal line

    html.H4("Calculate feature importance on full model:"),
    html.H6('Find most useful features using permutation feature importance (minimum input 2):'),
    dcc.Input(id="im_threshold", value=10, type="number"),
    html.Button(id="plot_feature_importance", n_clicks=0, children='Calculate feature importance',
                style={"background-color": "#00e300"}),

    dcc.Loading(dcc.Graph(id='permutation_importance'), type="cube"),

    html.Hr(),  # horizontal line

    html.H3('Choose your own feature combinations for the heatmaps and feature-maps:'),
    html.H6('Select two important features:'),
    dcc.Dropdown(id="important_features", multi=True),
    html.Br(),
    html.Button(id='select_features', n_clicks=0, children='Select Features', style={"background-color": "#cce6ff"}),

    html.H6("Manual heatmaps:"),
    dcc.Loading(dcc.Graph(id='heatmap'), type="cube"),

    html.H6("Manual feature-maps:"),
    dcc.Loading(dcc.Graph(id='feat_maps'), type="graph"),

    dcc.Store(id="type_optimizer"),
    dcc.Store(id="type_activation"),
    dcc.Store(id="num_refresh"),
    dcc.Store(id="num_kills"),
    dcc.Store(id="RAM_option"),
    dcc.Store(id="input_ranges"),
    dcc.Store(id="dataframe"),
    dcc.Store(id="splitted_data", data="data"),
    dcc.Store(id="feature_names", data="data"),
    dcc.Store(id="target_names", data="data"),
    dcc.Store(id="csv_data"),
    dcc.Store(id="s_features"),
    dcc.Store(id="densities"),
    dcc.Store(id="encoder_dict"),
    dcc.Store(id="trained_model")
])


def parse_contents(contents, filename):
    """
    Parses the given csv file.
    """
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    data = df.copy()
    if "Id" in data.columns:
        data.drop(columns=["Id"], inplace=True)
    elif "Index" in data.columns:
        data.drop(columns=["Index"], inplace=True)
    elif "index" in data.columns:
        data.drop(columns=["index"], inplace=True)
    elif "id" in data.columns:
        data.drop(columns=["id"], inplace=True)
    return html.Div([
        html.H5(filename),

        dash_table.DataTable(
            df.head().to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr()  # horizontal line
    ]), data.to_dict('records'), data.columns


@app.callback(Output('output-data-upload', 'children'),
              Output("csv_data", "data"),
              Output("s_features", "data"),
              Output("s_target", "options"),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    """
    Updates the output --> stores the data and displays the head of the given .csv file
    """
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children[0][0], children[0][1], children[0][2], children[0][2]
    else:
        raise PreventUpdate


@app.callback(
    Output("num_refresh", "data"),
    Input("refresh", "n_clicks")
)
def refresh_page(n_clicks):
    """
    Refreshes the page
    """
    if n_clicks:
        pyautogui.hotkey('f5')  # Simulates F5 key press = page refresh
        return n_clicks


@app.callback(
    Output("num_kills", "data"),
    Input("kill", "n_clicks")
)
def kill_application(n_clicks):
    """
    Kills the application
    """
    if n_clicks:
        os.kill(os.getpid(), 9)
        return n_clicks


@app.callback(
    Output("type_optimizer", "data"),
    Output("momentum_input", "children"),
    Input("optimizer_c", "value")
)
def select_optimizer(optimizer_c):
    """
    Interface for selection of the optimizer.
    """
    if optimizer_c == "SGD":
        return optimizer_c, html.Div([html.H6('Momentum for SGD:'), dcc.Input(value=0.9, type="number")])
    else:
        return optimizer_c, html.Div([])


@app.callback(
    Output("type_activation", "data"),
    Input("activation_f", "value")
)
def select_activation(activation_f):
    """
    Interface for selection of the activation function.
    """
    return activation_f


@app.callback(
    Output("use_t_class", "children"),
    Input('s_target', 'value'),
    Input("select_t_class", "on"),
    State('csv_data', 'data')
)
def select_target_classes(target, on, data):
    """
    Interface for selection of the wanted target classes.
    """
    if target is not None:
        if bool(on):
            df = pd.DataFrame.from_dict(data)
            target_names = df[target].unique()
            return html.Div([html.H6('Select target classes:'),
                             dcc.Dropdown(options=target_names, multi=True)])
        else:
            return html.Div([])
    else:
        raise PreventUpdate


@app.callback(
    Output('use_features', 'children'),
    Input('sel_features', 'on'),
    Input('s_target', 'value'),
    State("s_features", "data")
)
def select_features(on, target, features):
    """
    Interface for selection of the wanted features.
    """
    if target is not None:
        if bool(on):
            features.remove(target)
            return html.Div([html.H6('Select features:'),
                             dcc.Dropdown(options=features, multi=True)])
        else:
            return html.Div(
                [html.H6("The whole table without the target was selected as features.", style={'color': '#DAAD35'})])
    else:
        raise PreventUpdate


@app.callback(
    Output("input_ranges", "data"),
    Output("dataframe", "data"),
    Output("splitted_data", "data"),
    Output("feature_names", "data"),
    Output("target_names", "data"),
    Output("test_accuracy", "children"),
    Output("densities", "data"),
    ServersideOutput("encoder_dict", "data"),
    ServersideOutput("trained_model", "data"),
    Input('run_model', 'n_clicks'),
    State('epochs', 'value'),
    State('csv_data', 'data'),
    State('s_target', 'value'),
    State('normalize', 'on'),
    State("use_features", "children"),
    State("learning_rate", "value"),
    State("hidden_layers", "value"),
    State("num_layers", "value"),
    State("type_optimizer", "data"),
    State("momentum_input", "children"),
    State("type_activation", "data"),
    State("class_or_reg", "value")
)
def run_model(n_clicks, epochs, data, target, normalize, use_features, learning_rate, hidden_layers, num_layers,
              type_optimizer, momentum_input, activation, class_or_reg):
    """
    Method for running the model and calculating the densities of the features.
    """
    if n_clicks:
        normalize = bool(normalize)
        if len(use_features["props"]["children"]) == 1:
            X, y, feature_names, target_names, input_ranges, encoder_dict = create_data_from_csv(data, target, normalize
                                                                                                 , class_or_reg)
        else:
            X, y, feature_names, target_names, input_ranges, encoder_dict = create_data_from_csv(data, target, normalize
                                                                                                 , class_or_reg,
                                                                                                 use_features["props"][
                                                                                                     "children"][1][
                                                                                                     "props"]["value"])
        X_train, X_test, y_train, y_test = split_data(X, y)
        output_dim = len(target_names) if class_or_reg == "Classification" else 1
        model = create_nam_model(input_dim=X_train.shape[1], hidden_dim=hidden_layers, output_dim=output_dim,
                                 class_or_reg=class_or_reg, num_layers=num_layers, activation_f=activation)

        if len(momentum_input["props"]["children"]) != 0:
            momentum = momentum_input["props"]["children"][1]["props"]["value"]
        else:
            momentum = 0

        if type_optimizer == "Adam":
            optimizer = define_optimizer(optim.Adam, model.parameters(), learning_rate)
        else:
            optimizer = define_optimizer(optim.SGD, model.parameters(), learning_rate, momentum)

        loss_fn = define_loss_function(F.cross_entropy) if class_or_reg == "Classification" else define_loss_function(
            F.mse_loss)
        trained_model, test_accuracy = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, optimizer,
                                                                loss_fn, class_or_reg, epochs)
        densities = calculate_density_parameter(data, feature_names, target_names, target, class_or_reg)
        df = create_dataframe(X, feature_names)
        splitted_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

        test_acc_output = html.Div(
            [html.H4("Test Accuracy:") if class_or_reg == "Classification" else html.H4("Test Mean Squared Error:"),
             daq.LEDDisplay(value=f'{test_accuracy:.4f}')])

        return input_ranges, df.to_dict(), splitted_data, feature_names, target_names, test_acc_output, densities, encoder_dict, trained_model
    else:
        raise PreventUpdate


@app.callback(
    Output("heatmaps", "figure"),
    Output("feature_maps", "figure"),
    Input("plot_heatmaps", "n_clicks"),
    State("dataframe", "data"),
    State("input_ranges", "data"),
    State("class_or_reg", "value"),
    State("target_names", "data"),
    State("feature_names", "data"),
    State("resolution", "value"),
    State("s_target", "value"),
    State('use_t_class', 'children'),
    State('num_heatmaps', 'value'),
    State("densities", "data"),
    State("num_columns", "value"),
    State("height_subplot", "value"),
    State("width_subplot", "value"),
    State("encoder_dict", "data"),
    State("trained_model", "data"),
    State("RAM", "value")
)
def plot_heatmaps(n_clicks, data, input_ranges, class_or_reg, target_names, feature_names, resolution, target,
                  target_classes, num_heatmaps, densities, num_columns, height_subplot, width_subplot, encoder_dict,
                  trained_model, RAM_option):
    """
    Method for plotting the auto selection heatmaps and their feature maps.
    """
    if n_clicks:
        df = pd.DataFrame.from_dict(data)
        RAM_option = True if RAM_option == "RAM intensive & faster" else False
        if len(target_classes["props"]["children"]) == 0:
            target_classes = target_names
        else:
            target_classes = target_classes["props"]["children"][1]["props"]["value"]
        input_ranges_ = pd.DataFrame(input_ranges)
        auto_heatmap, heatmaps_sorted = plot_feature_heatmaps(df, trained_model, input_ranges_, feature_names,
                                                              target_names, target_classes, feature_names, resolution,
                                                              class_or_reg, target, num_heatmaps, height_subplot,
                                                              width_subplot, encoder_dict, RAM_option, num_columns)
        plot_feat_maps = plot_feature_maps(trained_model, input_ranges_, feature_names, target_names, feature_names,
                                           target_classes, resolution, class_or_reg, densities, target, num_heatmaps,
                                           heatmaps_sorted, height_subplot, width_subplot, encoder_dict, num_columns)
        return auto_heatmap, plot_feat_maps
    else:
        raise PreventUpdate


@app.callback(
    Output("permutation_importance", "figure"),
    Output("important_features", "options"),
    Input("plot_feature_importance", "n_clicks"),
    State("class_or_reg", "value"),
    State("feature_names", "data"),
    State("im_threshold", "value"),
    State("splitted_data", "data"),
    State("trained_model", "data")
)
def feature_importance(n_clicks, class_or_reg, feature_names, im_threshold, splitted_data, trained_model):
    """
    Method for plotting the feature_importance's of the features.
    """
    if n_clicks:
        loss_fn = define_loss_function(F.cross_entropy) if class_or_reg == "Classification" else define_loss_function(
            F.mse_loss)
        X_test = np.array(splitted_data["X_test"])
        y_test = np.array(splitted_data["y_test"])

        per_im_sorted, imp_features, std_error = find_important_features(trained_model, X_test, y_test, loss_fn,
                                                                         30, feature_names, im_threshold, class_or_reg)
        permutation_imp = plot_permutation_importance(per_im_sorted, imp_features, std_error, class_or_reg)
        return permutation_imp, imp_features
    else:
        raise PreventUpdate


@app.callback(
    Output("heatmap", "figure"),
    Output("feat_maps", "figure"),
    Input('select_features', 'n_clicks'),
    State('important_features', 'value'),
    State('use_t_class', 'children'),
    State('dataframe', 'data'),
    State('target_names', 'data'),
    State('input_ranges', 'data'),
    State('feature_names', 'data'),
    State('resolution', 'value'),
    State("class_or_reg", "value"),
    State("densities", "data"),
    State("s_target", "value"),
    State("num_columns", "value"),
    State("height_subplot", "value"),
    State("width_subplot", "value"),
    State("trained_model", "data"),
    State("encoder_dict", "data"),
    State("RAM", "value")
)
def plot_heatmap_pair(n_clicks, important_features, target_classes, data, target_names, input_ranges,
                      all_feature_names, resolution, class_or_reg, densities, target, num_columns, height_subplot,
                      width_subplot, trained_model, encoder_dict, RAM_option):
    """
    Method for plotting the manual selected feature pair heatmap and their feature maps.
    """
    if n_clicks:
        df = pd.DataFrame.from_dict(data)
        RAM_option = True if RAM_option == "RAM intensive & faster" else False
        f1_pos = df.columns.get_loc(important_features[0])
        f2_pos = df.columns.get_loc(important_features[1])
        if f1_pos > f2_pos:
            important_features = np.flip(important_features)
        input_ranges = pd.DataFrame.from_dict(input_ranges)
        if len(target_classes["props"]["children"]) == 0:
            target_classes = target_names
        else:
            target_classes = target_classes["props"]["children"][1]["props"]["value"]
        heatmap = plot_heatmap_feature_pair(df, trained_model, input_ranges, important_features, target_names,
                                            target_classes, all_feature_names, resolution, class_or_reg, target,
                                            height_subplot, width_subplot, encoder_dict, RAM_option, num_columns)
        plot_feat_maps = plot_feature_maps_pair(trained_model, input_ranges, all_feature_names, target_names,
                                                important_features, target_classes, resolution, class_or_reg,
                                                densities, target, height_subplot, width_subplot, encoder_dict,
                                                num_columns)
        return heatmap, plot_feat_maps
    else:
        raise PreventUpdate


# if debug is set to True the application can be debugged in the web browser --> developer mode
if __name__ == '__main__':
    app.run_server(debug=False)
