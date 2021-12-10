import os

import plotly
import plotly.graph_objs as go
import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


def draw_graph(value_array: np.ndarray, x_name: str, y_name: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i + 1 for i in range(len(value_array))], y=value_array))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title=title,
                      xaxis_title=x_name,
                      yaxis_title=y_name,
                      margin=dict(l=0, r=0, t=30, b=0))

    fig.show()
    return fig


def save_param_to_html(fig: go.Figure, path_to_save: str, param_name: str):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plotly.io.write_html(fig=fig, file=os.path.join(path_to_save, param_name + ".html"))


def draw_conf_matrix(confusion_matrix: np.ndarray, cls_names: list[str]) -> go.Figure:
    x = [cls_names[i] for i in range(len(cls_names))]
    y = x
    z_text = [[str(x) for x in y] for y in confusion_matrix]
    fig = ff.create_annotated_heatmap(confusion_matrix, x=y, y=y, annotation_text=z_text, colorscale='Viridis')
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title="confusion_matrix",
                      xaxis_title="predicted value",
                      yaxis_title="real value",
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.update_layout(margin=dict(t=50, l=200))
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 9
    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()
    return fig


def draw_roc_curves(roc_data_list: list[(list[float], list[float], float)], cls_names: list[str]) -> go.Figure:
    fig = go.Figure()
    for i in range(len(roc_data_list)):
        fpr, tpr, auc = roc_data_list[i]
        _draw_roc_curve(fpr, tpr, auc, cls_names[i], fig)
    fig.add_trace(go.Scatter(name="bisector", x=[i / 100 for i in range(101)], y=[i / 100 for i in range(101)],
                             mode="markers"))
    fig.update_traces(showlegend=True)
    fig.show()
    return fig


def _draw_roc_curve(fpr: list[float], tpr: list[float], auc: float, cls_name: str, fig: go.Figure):
    fig.add_trace(go.Scatter(name=cls_name + " (auc = " + str(auc) + ")", x=fpr, y=tpr))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title="roc curve",
                      xaxis_title="fpr",
                      yaxis_title="tpr",
                      margin=dict(l=0, r=0, t=30, b=0))


def draw_roc_curve(fpr: list[float], tpr: list[float], auc: float, cls_name: str) -> go.Figure:
    fig = go.Figure()
    _draw_roc_curve(fpr, tpr, auc, cls_name, fig)
    fig.update_traces(showlegend=True)
    fig.add_trace(go.Scatter(name="bisector", x=[i / 100 for i in range(101)], y=[i / 100 for i in range(101)],
                             mode="markers"))
    fig.show()
    return fig


def draw_metric_bar(cls_names: list[str], value: list[float], metric_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cls_names, y=value))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title=metric_name,
                      xaxis_title="classes",
                      yaxis_title="value of " + metric_name,
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.show()
    return fig
