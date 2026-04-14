import copy
import os
from functools import partial
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco import CocoDetection


def label_colormap(n_label=256, value=None):

    def bitget(byteval, idx):
        shape = byteval.shape + (8,)
        return np.unpackbits(byteval).reshape(shape)[..., -1 - idx]

    i = np.arange(n_label, dtype=np.uint8)
    r = np.full_like(i, 0)
    g = np.full_like(i, 0)
    b = np.full_like(i, 0)

    i = np.repeat(i[:, None], 8, axis=1)
    i = np.right_shift(i, np.arange(0, 24, 3)).astype(np.uint8)
    j = np.arange(8)[::-1]
    r = np.bitwise_or.reduce(np.left_shift(bitget(i, 0), j), axis=1)
    g = np.bitwise_or.reduce(np.left_shift(bitget(i, 1), j), axis=1)
    b = np.bitwise_or.reduce(np.left_shift(bitget(i, 2), j), axis=1)

    cmap = np.stack((r, g, b), axis=1).astype(np.uint8)

    if value is not None:
        hsv = cv2.cvtColor(cmap.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return cmap


def generate_color_palette(n: int, contrast: bool = False):
    colors = label_colormap(n)
    hsv_colors = cv2.cvtColor(colors[None], cv2.COLOR_RGB2HSV)[0]

    if not contrast:
        return colors

    dark_colors = hsv_colors.copy()
    dark_colors[:, -1] //= 2
    light_colors = dark_colors.copy()
    light_colors[:, -1] += 128

    dark_colors = cv2.cvtColor(dark_colors[None], cv2.COLOR_HSV2RGB)[0]
    light_colors = cv2.cvtColor(light_colors[None], cv2.COLOR_HSV2RGB)[0]
    return colors, light_colors, dark_colors


def plot_bounding_boxes_on_image_cv2(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray = None,
    classes: List[str] = None,
    show_conf: float = 0.5,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
    colors: List[Tuple[int]] = None,  # 添加colors参数
):
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        color = colors[i] if colors is not None else (255, 255, 255)  # 使用传递的颜色
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, box_thick)
        if scores is not None and scores[i] >= show_conf:
            label = classes[labels[i]] if classes is not None else f"ID {labels[i]}"
            cv2.putText(
                image,
                label,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_font_color if text_font_color is not None else (255, 255, 255),
                2,
            )
    return image


from PIL import Image, ImageDraw, ImageFont

def plot_bounding_boxes_on_image_pil(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray = None,
    classes: List[str] = None,
    show_conf: float = 0.5,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
    colors: List[Tuple[int]] = None  # 添加colors参数
):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # 可以自定义字体

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        color = colors[i] if colors is not None else (255, 255, 255)  # 使用传递的颜色
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=box_thick)

        if scores is not None and scores[i] >= show_conf:
            label = classes[labels[i]] if classes is not None else f"ID {labels[i]}"
            draw.text(
                (xmin, ymin - 10),
                label,
                fill=text_font_color if text_font_color is not None else (255, 255, 255),
                font=font
            )

    return np.array(image)


def plot_bounding_boxes_on_image(
        image: np.ndarray,
        boxes: Union[np.ndarray, List[float]],
        labels: Union[np.ndarray, List[int]],
        scores: Union[np.ndarray, List[float]] = None,
        classes: List[str] = None,
        show_conf: float = 0.5,
        font_scale: float = 1.0,
        box_thick: int = 3,
        fill_alpha: float = 0.2,
        text_box_color: Tuple[int] = (255, 255, 255),
        text_font_color: Tuple[int] = None,
        text_alpha: float = 0.5,
        backend="pil",
        colors: List[Tuple[int]] = None  # 添加colors参数
):
    if backend == "pil":
        plot_fn = plot_bounding_boxes_on_image_pil
    elif backend == "cv2":
        plot_fn = plot_bounding_boxes_on_image_cv2
    else:
        raise ValueError("Only 'pil' and 'cv2' backend are supported")

    return plot_fn(
        image=image,
        boxes=boxes,
        labels=labels,
        scores=scores,
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
        colors=colors,  # 确保传递colors给plot_fn
    )


def visualize_coco_bounding_boxes(
    data_loader: DataLoader,
    show_conf: float = 0.0,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
):

    assert data_loader.batch_size in (
        None, 1
    ), "batch_size of DataLoader for visualization must be 1"
    assert isinstance(
        data_loader.dataset, CocoDetection
    ), "Only CocoDetection dataset is supported"
    os.makedirs(show_dir, exist_ok=True)
    dataset: CocoDetection = data_loader.dataset
    cat_ids = list(range(max(dataset.coco.cats.keys()) + 1))
    classes = tuple(dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)

    data_loader.collate_fn = partial(
        _visualize_batch_in_coco,
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
        dataset=dataset,
        show_dir=show_dir,
    )
    [None for _ in tqdm(data_loader)]


def _visualize_batch_in_coco(
    batch: Tuple[np.ndarray, dict],
    dataset: CocoDetection,
    classes: List[str],
    show_conf: float = 0.0,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
):
    image, output = batch[0]
    # plot bounding boxes on image
    image = image.numpy().transpose(1, 2, 0)
    image = plot_bounding_boxes_on_image(
        image=image,
        boxes=output["boxes"],
        labels=output["labels"],
        scores=output.get("scores", None),
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
        colors=output.get("colors", None)  # 传递颜色信息
    )

    image_name = dataset.coco.loadImgs([output["image_id"]])[0]["file_name"]
    # cv2.imwrite save image with BGR format, convert RGB to BGR with image[:, :, ::-1]
    cv2.imwrite(os.path.join(show_dir, os.path.basename(image_name)), image[:, :, ::-1])
