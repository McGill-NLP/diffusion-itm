# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import json
import argparse
import logging
import math
import os
import time
import random
from pathlib import Path
from typing import Optional
import cv2

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionImg2ImgPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from datasets_loading import get_dataset
from utils import evaluate_scores

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vanilla_finetuning",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument('--neg_prob', type=float, default=1.0, help='The probability of sampling a negative image.')
    parser.add_argument('--neg_loss_factor', type=float, default=1.0)
    parser.add_argument('--task', type=str, default='mscoco')
    parser.add_argument('--hard_neg', action='store_true')
    parser.add_argument('--relativistic', action='store_true')
    parser.add_argument('--unhinged', action='store_true')
    parser.add_argument('--neg_img', action='store_true')
    parser.add_argument('--mixed_neg', action='store_true')

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def score_batch(i, args, batch, model, img_retrieval=False):
    """
    Takes a batch of images and captions and returns a score for each image-caption pair.
    """

    imgs, texts = batch[0], batch[1]
    _, imgs_resize = imgs[0], imgs[1]

    batchsize = imgs_resize[0].shape[0]
    scores = []
    for txt_idx, text in enumerate(texts):
        for img_idx, resized_img in enumerate(imgs_resize):
            if len(resized_img.shape) == 3:
                resized_img = resized_img.unsqueeze(0)
            
            print(f'Batch {i}, Text {txt_idx}, Image {img_idx}')
            dists = model(prompt=list(text), image=resized_img, scoring=True, guidance_scale=0.0, sampling_steps=4, unconditional=img_retrieval)
            dists = dists.to(torch.float32)
            dists = dists.mean(dim=1)
            dists = -dists
            scores.append(dists)

    scores = torch.stack(scores).permute(1, 0) if batchsize > 1 else torch.stack(scores).unsqueeze(0)
    return scores

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        wandb.init(project='vanilla_finetuning_2.1', settings=wandb.Settings(start_method="fork"))


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo_name = create_repo(repo_name, exist_ok=True)
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    # freeze parameters of models to save more memory
    # unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

        # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )

    unet.set_attn_processor(lora_attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = AttnProcsLayers(unet.attn_processors)


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # add unet parameters to the optimizer and not lora_layers

    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    
    train_dataset = get_dataset(args.task, f'datasets/{args.task}', transform=None, split='train', tokenizer=tokenizer, hard_neg=args.hard_neg, neg_img=args.neg_img, mixed_neg=args.mixed_neg)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )


    val_dataset = get_dataset('mscoco_val', f'datasets/{args.task}', transform=None, split='val', neg_img=False, hard_neg=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    val_dataset2 = get_dataset('mscoco_val', f'datasets/{args.task}', transform=None, split='val', neg_img=True, hard_neg=False)
    val_dataloader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=8, shuffle=False, num_workers=0)
    # val_dataset2 = get_dataset('flickr30k_text', f'datasets/flickr30k_text', transform=None, split='val')
    # val_dataloader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=8, shuffle=False, num_workers=0)

    # val_dataset3 = get_dataset('vg_attribution', f'datasets/vg_attribution', transform=None)
    # val_dataloader3 = torch.utils.data.DataLoader(val_dataset3, batch_size=8, shuffle=False, num_workers=0)

    # val_dataset3 = get_dataset('coco_order', f'datasets/coco_order', transform=None)
    # val_dataloader3 = torch.utils.data.DataLoader(val_dataset3, batch_size=8, shuffle=False, num_workers=0)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    val_sentences = json.load(open('datasets/flickr30k/val_top10_RN50x64.json')).keys()
    val_sentences = list(val_sentences)[:10]
    args.validation_prompt = val_sentences

    args.validation_prompt = ["Two young guys with shaggy hair look at their hands while hanging out in the yard.", "Two young, White males are outside near many bushes.", "Two men in green shirts are standing in a yard.", "A man in a blue shirt standing in a garden.", "Two friends enjoy time spent together."]
    args.validation_prompt += ["A man sits in a chair while holding a large stuffed animal of a lion.", "A man is sitting on a chair holding a large stuffed animal.", "A man completes the finishing touches on a stuffed lion.", "A man holds a large stuffed lion toy.", "A man is smiling at a stuffed lion"]

    best_r1 = 0.0

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            
            if args.mixed_neg:
                neg_img_ = False
                hard_neg_ = False
                rand_neg_ = False
                if 0.5 < torch.rand(1):
                    neg_img_ = True
                else:
                    if 0.5 < torch.rand(1):
                        hard_neg_ = True
                    else:
                        rand_neg_ = True


            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space

                img, texts, _ = batch
                # idx = torch.randint(1, texts.shape[1], (1,))[0]
                if neg_img_:
                    img_neg = img[1][1]
                    txt_neg = None
                elif hard_neg_:
                    txt_neg = texts[:,1,:]
                    img_neg = None
                else:
                    txt_neg = texts[:,2,:]
                    img_neg = None
                
                img = img[1][0]
                text = texts[:,0,:]
    

                latents = vae.encode(img.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(text)[0]
                if txt_neg is not None:
                    encoder_hidden_states_neg = text_encoder(txt_neg)[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                if img_neg is not None:
                    latents_neg = vae.encode(img_neg.to(dtype=weight_dtype)).latent_dist.sample()
                    latents_neg = latents_neg * vae.config.scaling_factor

                    # Sample noise that we'll add to the negative latents
                    # noise_neg = torch.randn_like(latents_neg)

                    # Add noise to the negative latents according to the noise magnitude at each timestep
                    noisy_latents_neg = noise_scheduler.add_noise(latents_neg, noise, timesteps)

                    # Predict the noise residual for the negative latents
                    model_pred_neg = unet(noisy_latents_neg, timesteps, encoder_hidden_states).sample

                if txt_neg is not None:
                    model_pred_neg = unet(noisy_latents, timesteps, encoder_hidden_states_neg).sample
                if txt_neg is not None or img_neg is not None:
                    if args.relativistic:
                        diff_neg = F.mse_loss(model_pred_neg.float(), model_pred.float(), reduction="mean")
                    else:
                        diff_neg = F.mse_loss(model_pred_neg.float(), target.float(), reduction="mean")
                    if not args.unhinged:
                        loss_neg = torch.clamp(diff_neg, max=args.neg_loss_factor*loss.item())
                    else:
                        loss_neg = diff_neg
                    loss_neg = -loss_neg
                    # diff = torch.sigmoid(diff)
                    # diff_neg = torch.sigmoid(diff_neg)
                        
                    # loss = F.binary_cross_entropy(diff, torch.zeros_like(diff))
                    # loss_neg = F.binary_cross_entropy(diff_neg, torch.ones_like(diff_neg))

                    wandb.log({"loss_neg": loss_neg.item()})
                    loss = loss_neg + loss

                wandb.log({"loss": loss.item()})

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # wandb.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                accelerator.wait_for_everyone()

                # if global_step % args.checkpointing_steps == 250:
                if False:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        # logger.info(f"Saved state to {save_path}")
                        
                        ############ QUANTITAIVE EVALUATION #############
                        pipeline_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline_img2img = pipeline_img2img.to(accelerator.device)
                        pipeline_img2img.set_progress_bar_config(disable=True)


                        r1s = []
                        r5s = []
                        max_more_than_onces = 0
                        args.task = 'flickr30k_text'
                        for k, batch in tqdm(enumerate(val_dataloader2), total=len(val_dataloader2)):
                            if k % 15 != 0:
                                continue
                            # measure time for the following line
                            scores = score_batch(k, args, batch, pipeline_img2img)
                            score = evaluate_scores(args, scores, batch)

                            r1,r5, max_more_than_once = evaluate_scores(args, scores, batch)
                            r1s += r1
                            r5s += r5
                            max_more_than_onces += max_more_than_once
                            r1 = sum(r1s) / len(r1s)
                            r5 = sum(r5s) / len(r5s)
                            print(f'R@1: {r1}')
                            print(f'R@5: {r5}')
                            print(f'Max more than once: {max_more_than_onces}')
                            with open(f'{save_path}/results.txt', 'w') as f:
                                f.write(f'R@1: {r1}\n')
                                f.write(f'R@5: {r5}\n')
                                f.write(f'Max more than once: {max_more_than_onces}\n')
                                f.write(f"Sample size {len(r1s)}\n")
                        wandb.log({'R@1': r1, 'R@5': r5, 'Max more than once': max_more_than_onces})

                        # if r1 > best_r1:
                        # save model state to output_dir
                        

                        del pipeline_img2img
                        torch.cuda.empty_cache()
                        ############ QUANTITAIVE EVALUATION #############

                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}."
                        )
                        # create pipeline
                        pipeline = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        for seed in range(3):
                            generator = torch.Generator(device=accelerator.device).manual_seed(seed)

                            for kk, prompt in enumerate(args.validation_prompt):
                                pil_image = pipeline(prompt, num_inference_steps=30, generator=generator).images[0]
                                # save pil img to output folder with prompt written on image at top
                                image = np.array(pil_image)
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                image = cv2.putText(
                                    image, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                                )
                                img_path = os.path.join(save_path, f"validation_{kk}_{seed}.png")
                                cv2.imwrite(img_path, image)
                            
                        del pipeline
                        torch.cuda.empty_cache()

                elif (global_step % args.checkpointing_steps == 0 or global_step in [10, 20, 50, 100, 150, 200, 300, 400, 600   ]) and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    logger.info(f"Saving model checkpoint to {save_path}")
                    accelerator.save_state(save_path)
                    ############ QUANTITAIVE EVALUATION #############
                    pipeline_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline_img2img = pipeline_img2img.to(accelerator.device)
                    pipeline_img2img.set_progress_bar_config(disable=True)


                    metrics = []
                    max_more_than_onces = 0
                    for k, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                        if k % 15 != 0:
                            continue
                        # measure time for the following line
                        scores = score_batch(k, args, batch, pipeline_img2img)

                        args.task = 'mscoco_val'
                        acc, max_more_than_once = evaluate_scores(args, scores, batch)
                        metrics += acc
                        acc = sum(metrics) / len(metrics)
                        max_more_than_onces += max_more_than_once
                        print(f'MSCOCO Val Accuracy Txt: {acc}')
                        print(f'Max more than once: {max_more_than_onces}')
                        with open(f'{save_path}/results_txt.txt', 'w') as f:
                            f.write(f'MSCOCO Val Accuracy Txt: {acc}\n')
                            f.write(f'Max more than once: {max_more_than_onces}\n')
                            f.write(f"Sample size {len(metrics)}\n")
                    wandb.log({'MSCOCO Val Accuracy Txt': acc, 'Max more than once': max_more_than_onces})
                    txt_acc = acc

                    metrics = []
                    max_more_than_onces = 0
                    for k, batch in tqdm(enumerate(val_dataloader2), total=len(val_dataloader2)):
                        if k % 15 != 0:
                            continue
                        # measure time for the following line
                        scores = score_batch(k, args, batch, pipeline_img2img, img_retrieval=True)

                        args.task = 'mscoco_val'
                        acc, max_more_than_once = evaluate_scores(args, scores, batch)
                        metrics += acc
                        acc = sum(metrics) / len(metrics)
                        max_more_than_onces += max_more_than_once
                        print(f'MSCOCO Val Accuracy Img: {acc}')
                        print(f'Max more than once: {max_more_than_onces}')
                        with open(f'{save_path}/results_txt.txt', 'w') as f:
                            f.write(f'MSCOCO Val Accuracy Img: {acc}\n')
                            f.write(f'Max more than once: {max_more_than_onces}\n')
                            f.write(f"Sample size {len(metrics)}\n")
                    wandb.log({'MSCOCO Val Accuracy Img': acc, 'Max more than once': max_more_than_onces})
                    img_acc = acc
                    wandb.log({'Overall Val Accuracy': (txt_acc + img_acc) / 2})

                    del pipeline_img2img
                    torch.cuda.empty_cache()
            
                # elif False global_step % args.checkpointing_steps == 450 and accelerator.is_main_process:
                elif False:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    ############ QUANTITAIVE EVALUATION #############
                    pipeline_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline_img2img = pipeline_img2img.to(accelerator.device)
                    pipeline_img2img.set_progress_bar_config(disable=True)


                    metrics = []
                    max_more_than_onces = 0
                    for k, batch in tqdm(enumerate(val_dataloader3), total=len(val_dataloader3)):
                        if k % 60 != 0:
                            continue
                        # measure time for the following line
                        scores = score_batch(k, args, batch, pipeline_img2img)

                        args.task = 'vg_attribution'
                        acc, max_more_than_once = evaluate_scores(args, scores, batch)
                        metrics += acc
                        acc = sum(metrics) / len(metrics)
                        max_more_than_onces += max_more_than_once
                        print(f'VG Attribution Accuracy: {acc}')
                        print(f'Max more than once: {max_more_than_onces}')
                        with open(f'{save_path}/results.txt', 'w') as f:
                            f.write(f'VG Attribution Accuracy: {acc}\n')
                            f.write(f'Max more than once: {max_more_than_onces}\n')
                            f.write(f"Sample size {len(metrics)}\n")
                    wandb.log({'VG Attribution Accuracy': acc, 'Max more than once': max_more_than_onces})

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()