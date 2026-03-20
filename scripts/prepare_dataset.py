#!/usr/bin/env python3
"""
Dataset Preparation Script for Character LoRA Training
Handles cropping, captioning, and validation of training images
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict
import shutil
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

class DatasetPreparer:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "captions").mkdir(exist_ok=True)
        
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.webp'}
        self.target_size = (1024, 1024)
        
    def find_images(self) -> List[Path]:
        """Find all images in input directory"""
        images = []
        for ext in self.supported_formats:
            images.extend(self.input_dir.glob(f"*{ext}"))
            images.extend(self.input_dir.glob(f"*{ext.upper()}"))
        return images
    
    def validate_image(self, img_path: Path) -> Dict:
        """Validate image quality and features"""
        try:
            img = Image.open(img_path)
            
            # Check minimum size
            if img.size[0] < 512 or img.size[1] < 512:
                return {"valid": False, "reason": f"Image too small: {img.size}"}
            
            # Check aspect ratio (should be close to square for portraits)
            aspect = img.size[0] / img.size[1]
            if aspect < 0.5 or aspect > 2.0:
                return {"valid": False, "reason": f"Extreme aspect ratio: {aspect:.2f}"}
            
            # Check file size
            file_size = img_path.stat().st_size
            if file_size < 10 * 1024:  # 10KB
                return {"valid": False, "reason": f"File too small: {file_size/1024:.1f}KB"}
            
            return {
                "valid": True,
                "size": img.size,
                "format": img.format,
                "mode": img.mode
            }
            
        except Exception as e:
            return {"valid": False, "reason": str(e)}
    
    def preprocess_image(self, img_path: Path) -> Image.Image:
        """Preprocess image for training"""
        img = Image.open(img_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Center crop to square
        width, height = img.size
        if width != height:
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            img = img.crop((left, top, right, bottom))
        
        # Resize to target size
        img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return img
    
    def generate_caption(self, img_path: Path, img: Image.Image) -> str:
        """Generate caption for the image"""
        # Basic caption structure
        base_caption = "a portrait of [character_name]"
        
        # In production, you'd use a vision-language model here
        # For now, return a placeholder that needs manual review
        
        return base_caption
    
    def save_image_and_caption(self, img: Image.Image, caption: str, index: int):
        """Save processed image and its caption"""
        # Save image
        img_filename = f"char_{index:04d}.png"
        img_path = self.output_dir / "images" / img_filename
        img.save(img_path, "PNG")
        
        # Save caption
        cap_filename = f"char_{index:04d}.txt"
        cap_path = self.output_dir / "captions" / cap_filename
        with open(cap_path, 'w') as f:
            f.write(caption)
        
        return img_path, cap_path
    
    def prepare_metadata(self, images_data: List[Dict]):
        """Create metadata file for dataset"""
        metadata = {
            "dataset_name": self.output_dir.name,
            "total_images": len(images_data),
            "target_size": self.target_size,
            "images": images_data
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
    
    def run(self, auto_caption: bool = False):
        """Run dataset preparation pipeline"""
        print("\n" + "="*50)
        print("Dataset Preparation Pipeline")
        print("="*50 + "\n")
        
        # Find images
        print("Step 1: Finding images...")
        images = self.find_images()
        print(f"Found {len(images)} images")
        
        if not images:
            print("❌ No images found!")
            return
        
        # Validate and process images
        print("\nStep 2: Validating images...")
        valid_images = []
        images_data = []
        
        for img_path in tqdm(images, desc="Processing"):
            # Validate
            validation = self.validate_image(img_path)
            if not validation["valid"]:
                print(f"⚠️  Skipping {img_path.name}: {validation['reason']}")
                continue
            
            valid_images.append(img_path)
            
            # Preprocess
            processed_img = self.preprocess_image(img_path)
            
            # Generate caption
            caption = self.generate_caption(img_path, processed_img)
            
            # Save
            index = len(valid_images)
            img_out, cap_out = self.save_image_and_caption(processed_img, caption, index)
            
            # Store metadata
            images_data.append({
                "original": str(img_path),
                "processed": str(img_out),
                "caption_file": str(cap_out),
                "caption": caption,
                "validation": validation
            })
        
        print(f"\n✅ Valid images: {len(valid_images)}/{len(images)}")
        
        # Create metadata
        print("\nStep 3: Creating metadata...")
        self.prepare_metadata(images_data)
        
        # Generate report
        report_path = self.output_dir / "preparation_report.txt"
        with open(report_path, 'w') as f:
            f.write("Dataset Preparation Report\n")
            f.write("="*40 + "\n")
            f.write(f"Total images found: {len(images)}\n")
            f.write(f"Valid images: {len(valid_images)}\n")
            f.write(f"Skipped images: {len(images) - len(valid_images)}\n")
            f.write("\nSkipped images:\n")
            for img_path in images:
                if img_path not in valid_images:
                    val = self.validate_image(img_path)
                    if not val["valid"]:
                        f.write(f"  - {img_path.name}: {val.get('reason', 'Unknown')}\n")
        
        print(f"\n✅ Report saved to: {report_path}")
        
        # Summary
        print("\n" + "="*50)
        print("Summary")
        print("="*50)
        print(f"Processed images: {len(valid_images)}")
        print(f"Output directory: {self.output_dir}")
        print("\nNext steps:")
        print("1. Review generated captions in dataset/captions/")
        print("2. Update captions with proper character name and details")
        print("3. Verify image quality and remove any poor images")
        print("4. Run LoRA training with: python scripts/train_lora.py --config configs/lora_config.json")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA training")
    parser.add_argument("--input", required=True, help="Input directory with source images")
    parser.add_argument("--output", default="dataset/train", help="Output directory for processed dataset")
    parser.add_argument("--auto-caption", action="store_true", help="Generate auto-captions (requires BLIP model)")
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.input, args.output)
    preparer.run(auto_caption=args.auto_caption)


if __name__ == "__main__":
    main()
