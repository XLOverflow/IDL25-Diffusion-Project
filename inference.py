import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(embed_dim=args.unet_ch, n_classes=args.num_classes, cond_drop_rate=0.0)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        shceduler_class = DDIMScheduler
    else:
        shceduler_class = DDPMScheduler
    # TOOD: scheduler
    scheduler = shceduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    ).to(device)

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline = DDPMPipeline(unet, scheduler, vae=vae, class_embedder=class_embedder)

    
    logger.info("***** Running Infrence *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    to_tensor = transforms.ToTensor()
    unet.eval()
    if scheduler:
        scheduler.eval()
    if class_embedder:
        class_embedder.eval()
    if vae:
        vae.eval()

    with torch.no_grad():
        if args.use_cfg:
            # generate 50 images per class
            for i in tqdm(range(args.num_classes)):
                logger.info(f"Generating 50 images for class {i}")
                batch_size = 50
                classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
                gen_images = pipeline(
                    batch_size=batch_size,
                    num_inference_steps=args.num_inference_steps,
                    classes=classes,
                    guidance_scale=args.cfg_guidance_scale,
                    generator=generator,
                    device=device,
                )
                batch_tensor = torch.stack([to_tensor(img) for img in gen_images])
                all_images.append(batch_tensor)
        else:
            # generate 5000 images
            target_samples = 5000
            batch_size = args.batch_size
            generated = 0
            progress = tqdm(total=target_samples)
            while generated < target_samples:
                current_batch = min(batch_size, target_samples - generated)
                gen_images = pipeline(
                    batch_size=current_batch,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    device=device,
                )
                batch_tensor = torch.stack([to_tensor(img) for img in gen_images])
                all_images.append(batch_tensor)
                generated += current_batch
                progress.update(current_batch)
            progress.close()
    
    # TODO: load validation images as reference batch
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.ImageFolder(args.data_dir, transform=eval_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    
    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics 
    
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    # TODO: compute FID and IS
    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception = InceptionScore(normalize=True).to(device)
    
    for images, _ in val_loader:
        fid.update(images.to(device), real=True)
    
    for batch in all_images:
        batch = batch.to(device)
        fid.update(batch, real=False)
        inception.update(batch)
    
    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    is_mean = is_mean.item()
    is_std = is_std.item()
    
    logger.info(f"FID: {fid_score:.4f}")
    logger.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    if wandb.run is not None:
        wandb.log({"FID": fid_score, "InceptionScore": is_mean, "InceptionScoreStd": is_std})
    
        
    


if __name__ == '__main__':
    main()
