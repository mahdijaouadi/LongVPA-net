
import plotly.graph_objects as go
import io
from PIL import Image
import numpy as np
import torch
import os

from config.settings import (
    CHART_WIDTH,
    CHART_HEIGHT,
    CHART_SCALE,
    CHART_CROP_COORDS,
    CHART_RESIZE_DIMS,
    TORCH_FLOAT16
)

class ChartGenerator:
    def __init__(self):
        pass

    def create_chart(self, data):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        ))
        fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            font=dict(color='white'),
            showlegend=False
        )

        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=False)
            )
        )
        fig.update_layout(yaxis=dict(range=[data['low'].min(), data['high'].max()]))
        fig.update_layout(xaxis=dict(range=[data.index.min(), data.index.max()]))
        img_bytes = fig.to_image(format='jpg', width=CHART_WIDTH, height=CHART_HEIGHT, scale=CHART_SCALE)
        img = Image.open(io.BytesIO(img_bytes))
        return img

    def img_preprocess(self, img):
        img = np.array(img)
        top, bottom, left, right = CHART_CROP_COORDS
        img = img[top:bottom, left:right]
        img = Image.fromarray(img)
        img = img.resize(CHART_RESIZE_DIMS)
        img = np.array(img)
        img = (img / 255.0)
        img = torch.from_numpy(img).to(torch.float16)
        img = img.permute(2, 0, 1)
        return img
