#!/usr/bin/env python3
"""
LoRA Training Script for Character Consistency
Uses diffusers library for SDXL LoRA training
"""

import os
import json
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import accelerate

class CharacterDataset(Dataset):
    """Dataset for character LoRA training"""
    
    def __init__(self, data_dir: str, resolution: int = 1024):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.captions_dir = self.data_dir / "captions"
        
        # Find all images
        self.images = list(self.images_dir.glob("*.png"))
        self.images.extend(self.images_dir.glob("*.jpg"))
        
        # Load captions
        self.captions = []
        for img_path in self.images:
            cap_path = self.captions_dir / f"{img_path.stem}.txt"
            if cap_path.exists():
                with open(cap_path, 'r') as f:
                    self.captions.append(f.read().strip())
            else:
                self.captions.append("a portrait of a character")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.transform(image)
        caption = self.captions[idx]
        
        return {"pixel_values": image, "caption": caption}


class LoRATrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.setup_model()
        self.setup_dataset()
        self.setup_training()
    
    def setup_model(self):
        """Initialize model components"""
        print("Loading model components...")
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config["vae"] if "vae" in self.config else self.config["base_model"],
            subfolder="vae",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load tokenizer and text encoders
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config["base_model"],
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config["base_model"],
            subfolder="text_encoder",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load UNet
        self.unet = self.load_unet()
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config["base_model"],
            subfolder="scheduler"
        )
    
    def load_unet(self):
        """Load UNet with LoRA preparation"""
        from diffusers import UNet2DConditionModel
        
        unet = UNet2DConditionModel.from_pretrained(
            self.config["base_model"],
            subfolder="unet",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Freeze base model
        unet.requires_grad_(False)
        
        # Add LoRA layers
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=self.config.get("network_dim", 32),
            lora_alpha=self.config.get("network_alpha", 16),
            target_modules=["to_q", "to_v", "to_k", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
        )
        
        self.unet = get_peft_model(unet, lora_config)
        self.unet.print_trainable_parameters()
        
        return self.unet
    
    def setup_dataset(self):
        """Setup dataset and dataloader"""
        dataset_path = Path(self.config["dataset_path"])
        
        self.train_dataset = CharacterDataset(
            dataset_path,
            resolution=int(self.config["resolution"].split(",")[0])
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Dataset loaded: {len(self.train_dataset)} images")
    
    def setup_training(self):
        """Setup optimizer and scheduler"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config["learning_rate"]
        )
        
        # Scheduler
        self.lr_scheduler = get_scheduler(
            self.config["lr_scheduler"],
            optimizer=self.optimizer,
            num_warmup_steps=self.config["lr_warmup_steps"],
            num_training_steps=self.config["max_train_steps"]
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
    
    def encode_prompts(self, captions):
        """Encode text prompts"""
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(tokens.input_ids)[0]
        
        return encoder_hidden_states
    
    def train_step(self, batch):
        """Single training step"""
        # Move to device
        images = batch["pixel_values"].to(self.device, dtype=torch.float16)
        captions = batch["caption"]
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode prompts
        encoder_hidden_states = self.encode_prompts(captions)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states
            ).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()
        
        return loss.item()
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting LoRA Training")
        print("="*50 + "\n")
        
        self.unet.train()
        
        global_step = 0
        losses = []
        
        progress_bar = tqdm(range(self.config["max_train_steps"]), desc="Training")
        
        while global_step < self.config["max_train_steps"]:
            for batch in self.train_dataloader:
                if global_step >= self.config["max_train_steps"]:
                    break
                
                loss = self.train_step(batch)
                losses.append(loss)
                
                progress_bar.set_postfix({"loss": loss})
                progress_bar.update(1)
                
                global_step += 1
                
                # Save checkpoint
                if global_step % self.config["save_every_n_steps"] == 0:
                    self.save_checkpoint(global_step)
        
        progress_bar.close()
        
        # Save final model
        self.save_checkpoint("final")
        
        # Training summary
        print("\n" + "="*50)
        print("Training Complete")
        print("="*50)
        print(f"Total steps: {global_step}")
        print(f"Final loss: {losses[-1]:.6f}" if losses else "N/A")
        print(f"Average loss: {sum(losses)/len(losses):.6f}" if losses else "N/A")
        print(f"Model saved to: {self.config['output_path']}")
    
    def save_checkpoint(self, step):
        """Save LoRA checkpoint"""
        output_dir = Path(self.config["output_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        checkpoint_path = output_dir / f"{self.config['model_name']}_step_{step}.safetensors"
        
        # Extract only LoRA parameters
        lora_state_dict = {
            k: v for k, v in self.unet.state_dict().items()
            if "lora" in k
        }
        
        torch.save(lora_state_dict, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for character consistency")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    
    args = parser.parse_args()
    
    trainer = LoRATrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
