import argparse
import contextlib
import io
import json
import logging
import os
import tempfile
import cv2
import numpy as np
import time
from typing import Dict

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import accelerate
import torch
from accelerate import Accelerator
from pycocotools.coco import COCO
from terminaltables import AsciiTable
from torch.utils import data

from datasets.coco import CocoDetection
from util.coco_eval import CocoEvaluator, loadRes
from util.coco_utils import get_coco_api_from_dataset
from util.collate_fn import collate_fn
from util.engine import evaluate_acc
from util.lazy_load import Config
from util.logger import setup_logger
from util.misc import fixed_generator, seed_worker
from util.utils import load_checkpoint, load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Test on a datasets.")
    parser.add_argument("--coco-path", type=str, required=True)
    parser.add_argument("--subset", type=str, default="val")   
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--result", type=str, default="result.json")
    parser.add_argument("--show-dir", type=str, default="visualization_output")
    parser.add_argument("--show-conf", type=float, default=0.5)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--box-thick", type=int, default=1)
    parser.add_argument("--fill-alpha", type=float, default=0.2)
    parser.add_argument("--text-box-color", type=int, nargs="+", default=(255, 255, 255))
    parser.add_argument("--text-font-color", type=int, nargs="+", default=None)
    parser.add_argument("--text-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

def create_test_data_loader(dataset, accelerator=None, **kwargs):
    data_loader = data.DataLoader(
        dataset,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=fixed_generator(),
        **kwargs,
    )
    if accelerator:
        data_loader = accelerator.prepare_data_loader(data_loader)
    return data_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / (1024 * 1024)

def test_on_dataset():
    args = parse_args()

    logging.basicConfig(filename='./test.log', level=logging.INFO)
    logger = logging.getLogger()
    logger.info("Starting test...")

    accelerator = Accelerator(cpu=args.model_config is None)
    accelerate.utils.set_seed(args.seed, device_specific=False)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    dataset = CocoDetection(
        img_folder=f"{args.coco_path}/{args.subset}",
        ann_file=f"{args.coco_path}/annotations/{args.subset}.json",
        transforms=None,
        filter_empty_img=args.subset == "train",
    )
    data_loader = create_test_data_loader(
        dataset,
        accelerator=accelerator,
        batch_size=1,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    if args.model_config:
        model = Config(args.model_config).model.eval()
        checkpoint = load_checkpoint(args.checkpoint)
        if isinstance(checkpoint, Dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        load_state_dict(model, checkpoint)
        model = accelerator.prepare_model(model)

        num_params_mb = count_parameters(model)
        logger.info(f"Model parameters: {num_params_mb:.2f} MB")

        start_time = time.time()
        coco_evaluator = evaluate_acc(model, data_loader, 0, accelerator)
        end_time = time.time()

        total_time = end_time - start_time
        fps = len(data_loader) / total_time if total_time > 0 else 0
        logger.info(f"FPS: {fps:.2f}")

        if args.result is None:
            temp_file = tempfile.NamedTemporaryFile()
            args.result = temp_file.name

        with open(args.result, "w") as f:
            det_results = coco_evaluator.predictions["bbox"]
            f.write(json.dumps(det_results))
            logger.info(f"Detection results are saved into {args.result}")

    coco = get_coco_api_from_dataset(data_loader.dataset)

    if args.model_config is None or args.show_dir and accelerator.is_main_process:
        coco_dt = loadRes(COCO(f"{args.coco_path}/annotations/{args.subset}.json"), args.result)

    if args.model_config is None and accelerator.is_main_process:
        coco_evaluator = CocoEvaluator(coco, ["bbox"])
        coco_evaluator.coco_eval["bbox"].cocoDt = coco_dt
        coco_evaluator.coco_eval["bbox"].evaluate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
        logger.info(redirect_string.getvalue())

    cat_names = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
    table_data = [["class", "imgs", "gts", "recall", "ap"]]

    bbox_coco_eval = coco_evaluator.coco_eval["bbox"]
    for cat_idx, cat_name in enumerate(cat_names):
        cat_id = coco.getCatIds(catNms=cat_name)
        num_img_id = len(coco.getImgIds(catIds=cat_id))
        num_ann_id = len(coco.getAnnIds(catIds=cat_id))
        row_data = [cat_name, num_img_id, num_ann_id]
        row_data += [f"{bbox_coco_eval.eval['recall'][0, cat_idx, 0, 2].item():.3f}"]
        row_data += [f"{bbox_coco_eval.eval['precision'][0, :, cat_idx, 0, 2].mean().item():.3f}"]
        table_data.append(row_data)

    cat_recall = coco_evaluator.coco_eval["bbox"].eval["recall"][0, :, 0, 2]
    valid_cat_recall = cat_recall[cat_recall >= 0]
    mean_recall = valid_cat_recall.sum() / max(len(valid_cat_recall), 1)
    cat_ap = coco_evaluator.coco_eval["bbox"].eval["precision"][0, :, :, 0, 2]
    valid_cat_ap = cat_ap[cat_ap >= 0]
    mean_ap50 = valid_cat_ap.sum() / max(len(valid_cat_ap), 1)
    mean_data = ["mean results", "", "", f"{mean_recall:.3f}", f"{mean_ap50:.3f}"]
    table_data.append(mean_data)

    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    logger.info("\n" + table.table)

    CATEGORY_NAMES = {
        1: "crazing",
        2: "inclusion",
        3: "patches",
        4: "pitted_surface",
        5: "rolled-in_scale",
        6: "scratches"
    }
    COLORS = {
        1: (0, 0, 255),
        2: (0, 255, 0),
        3: (0, 255, 255),
        4: (255, 0, 0),
        5: (255, 0, 255),
        6: (255, 255, 0),
    }

    def save_visualization(image_path, bboxes, save_path):
        img = cv2.imread(image_path)
        for bbox in bboxes:
            x_min, y_min, width, height = bbox['bbox']
            score = bbox['score']
            category_id = bbox['category_id']

            x_max = x_min + width
            y_max = y_min + height

            color = COLORS.get(category_id, (255, 255, 255))

            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 1)

            text_background = (int(x_min), int(y_min) - 20, int(x_max), int(y_min))
            cv2.rectangle(img, (text_background[0], text_background[1]), (text_background[2], text_background[3]),
                          color, -1)

            cv2.putText(img, f' {score:.2f}',
                        (int(x_min), int(y_min) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        cv2.imwrite(save_path, img)

    if args.show_dir and accelerator.is_main_process:
        accelerator.state.device = "cpu"
        dataset.coco = coco_dt
        data_loader = create_test_data_loader(
            dataset, accelerator=accelerator, batch_size=1, num_workers=args.workers
        )

        os.makedirs(args.show_dir, exist_ok=True)

        with open(args.result) as f:
            det_results = json.load(f)
        results_by_image_id = {}
        for res in det_results:
            img_id = res['image_id']
            if img_id not in results_by_image_id:
                results_by_image_id[img_id] = []
            results_by_image_id[img_id].append(res)

        for img_id, bboxes in results_by_image_id.items():
            img_info = dataset.coco.loadImgs(img_id)[0]
            img_path = os.path.join(args.coco_path, args.subset, img_info['file_name'])

            bboxes = [bbox for bbox in bboxes if bbox['score'] >= args.show_conf]
            save_visualization(img_path, bboxes, os.path.join(args.show_dir, img_info['file_name']))
            logger.info(f'Saved visualization to {os.path.join(args.show_dir, img_info["file_name"])}')


if __name__ == "__main__":
    test_on_dataset()
