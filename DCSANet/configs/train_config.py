from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

num_epochs = 20   # train epochs
batch_size = 16    # total_batch_size = #GPU x batch_size
num_workers = 8   # workers for pytorch DataLoader
pin_memory = True # Keep pin_memory for PyTorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm
output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

coco_path = "datasets/coco" 
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train",
    ann_file=f"{coco_path}/annotations/train.json",
    transforms=presets.flip_resize,  # see transforms/presets to choose a transform
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val",
    ann_file=f"{coco_path}/annotations/val.json",
    transforms=None,  # the eval_transform is integrated in the model
)

model_path = "configs/networks/proposed.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune
resume_from_checkpoint = None

learning_rate = 0.0005
optimizer = optim.AdamW(lr=learning_rate, weight_decay=0.05) 
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)