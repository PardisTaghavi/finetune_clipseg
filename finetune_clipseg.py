import os
import glob
import cv2
import numpy as np
import torch
import random
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoProcessor, 
    CLIPSegForImageSegmentation,
    get_cosine_schedule_with_warmup
)

# Define the Cityscapes stuff classes and their corresponding colors
STUFF_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", 
    "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky"
]

NUM_TO_COLOR = {
    0: [128, 64, 128],   # road
    1: [244, 35, 232],   # sidewalk
    2: [70, 70, 70],     # building
    3: [102, 102, 156],  # wall
    4: [190, 153, 153],  # fence
    5: [153, 153, 153],  # pole
    6: [250, 170, 30],   # traffic light
    7: [220, 220, 0],    # traffic sign
    8: [107, 142, 35],   # vegetation
    9: [152, 251, 152],  # terrain
    10: [70, 130, 180]   # sky
}

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='val', crop_size=(970, 2048), resize_size=(1024, 2048)):
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.resize_size = resize_size
        
        # Define Cityscapes IDs to training IDs mapping
        self.id_to_trainid = {
            7: 0,      # road
            8: 1,      # sidewalk
            11: 2,     # building
            12: 3,     # wall
            13: 4,     # fence
            17: 5,     # pole
            19: 6,     # traffic light
            20: 7,     # traffic sign
            21: 8,     # vegetation
            22: 9,     # terrain
            23: 10,    # sky
        }
        
        # Set up file paths
        self.image_dir = os.path.join(self.root_dir, 'leftImg8bit', self.split)
        self.label_dir = os.path.join(self.root_dir, 'gtFine', self.split)
        
        # Get image and label files
        self.image_files = glob.glob(os.path.join(self.image_dir, '*/*_leftImg8bit.png'))
        self.image_files.sort()
        
        # Generate label paths by replacing filename suffix and directory name
        self.label_files = [
            f.replace("_leftImg8bit.png", "_gtFine_labelIds.png").replace("leftImg8bit", "gtFine")
            for f in self.image_files
        ]
        self.label_files.sort()
        
        print(f"Found {len(self.image_files)} images in {self.split} split")

    def __len__(self):
        return len(self.image_files)

    def convert_labels(self, label):
        """Convert Cityscapes IDs to training IDs."""
        label_copy = np.ones_like(label) * 255  # Initialize with ignore label
        
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
            
        # Set instance classes to ignore label (we only care about stuff classes)
        for i in range(24, 34):  # Instance class IDs
            label_copy[label == i] = 255
            
        return label_copy

    def __getitem__(self, idx):
        # Load image using PIL
        image_path = self.image_files[idx]
        image = Image.open(image_path)
        
        # Crop and resize image
        image = image.crop((0, 0, self.crop_size[1], self.crop_size[0]))
        image = image.resize((self.resize_size[1], self.resize_size[0]))
        # Convert PIL image to numpy array
        image = np.array(image)
        
        # Load label using cv2
        label_path = self.label_files[idx]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # Crop and resize label
        label = label[:self.crop_size[0], :self.crop_size[1]]
        label = cv2.resize(label, (self.resize_size[1], self.resize_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convert labels to training IDs
        label = self.convert_labels(label)
        
        # Create binary masks for each class
        binary_masks = []
        for i in range(len(STUFF_CLASSES)):
            mask = np.zeros_like(label, dtype=np.float32)
            mask[label == i] = 1.0
            binary_masks.append(mask)
        binary_masks = np.stack(binary_masks, axis=0)
        
        return {
            "image": image,  # Now a numpy array
            "label": label,
            "binary_masks": binary_masks,
            "file_name": os.path.basename(image_path)
        }
def train_model(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_name)
    
    # Move model to device
    model.to(device)
    
    # Create dataset and dataloader
    train_dataset = CityscapesDataset(
        root_dir=args.data_dir,
        split="train",
        crop_size=(970, 2048),
        resize_size=(1024, 2048)
    )
    
    val_dataset = CityscapesDataset(
        root_dir=args.data_dir,
        split="val",
        crop_size=(970, 2048),
        resize_size=(1024, 2048)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    num_training_steps = args.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"]
            binary_masks = batch["binary_masks"]
            
            # Process images for each class prompt
            total_loss = 0
            
            for class_idx, class_name in enumerate(STUFF_CLASSES):
                # Get the binary mask for current class
                class_masks = binary_masks[:, class_idx].to(device)
                
                # Process image with text prompt
                inputs = processor(
                    text=[class_name] * len(images),
                    images=images,
                    padding="max_length",
                    return_tensors="pt"
                ).to(device)
                
                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Resize logits to match the mask size
                resized_logits = torch.nn.functional.interpolate(
                    logits.unsqueeze(1),
                    size=(1024, 2048),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)
                
                # Binary cross entropy loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    resized_logits, 
                    class_masks
                )
                
                total_loss += loss
                
                # Backward pass and optimization step for each class
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
            
            # Update progress bar
            avg_loss = total_loss / len(STUFF_CLASSES)
            train_loss += avg_loss.item()
            progress_bar.set_postfix({"loss": avg_loss.item()})
            
            # Break early for debugging
            if args.debug and batch_idx >= 5:
                break
        
        # Calculate average training loss
        avg_train_loss = train_loss / min(len(train_loader), 6 if args.debug else float('inf'))
        
        # Validation
        # Validation
        model.eval()
        val_loss = 0.0
        
        # Initialize accumulators for mIoU computation
        intersection_total = [0.0 for _ in range(len(STUFF_CLASSES))]
        union_total = [0.0 for _ in range(len(STUFF_CLASSES))]
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                images = batch["image"]
                binary_masks = batch["binary_masks"]
                
                total_loss = 0
                
                for class_idx, class_name in enumerate(STUFF_CLASSES):
                    # Get the binary mask for current class
                    class_masks = binary_masks[:, class_idx].to(device)
                    
                    # Process image with text prompt
                    inputs = processor(
                        text=[class_name] * len(images),
                        images=images,
                        padding="max_length",
                        return_tensors="pt"
                    ).to(device)
                    
                    # Forward pass
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Resize logits to match the mask size
                    resized_logits = torch.nn.functional.interpolate(
                        logits.unsqueeze(1),
                        size=(1024, 2048),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)
                    
                    # Binary cross entropy loss
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        resized_logits, 
                        class_masks
                    )
                    
                    total_loss += loss
                    
                    # Compute predictions with sigmoid thresholding (default threshold: 0.5)
                    preds = torch.sigmoid(resized_logits) > 0.5
                    gt = class_masks > 0.5  # ground truth is already binary, but ensure boolean
                    
                    # Compute intersection and union for IoU
                    intersection = (preds & gt).sum().item()
                    union = (preds | gt).sum().item()
                    intersection_total[class_idx] += intersection
                    union_total[class_idx] += union
                
                avg_loss = total_loss / len(STUFF_CLASSES)
                val_loss += avg_loss.item()
                
                # Break early for debugging if needed
                if args.debug and batch_idx >= 5:
                    break
        
        # Calculate average validation loss
        avg_val_loss = val_loss / min(len(val_loader), 6 if args.debug else float('inf'))
        print(f"Epoch {epoch+1}/{args.num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Compute per-class IoU and mIoU
        iou_per_class = []
        for i in range(len(STUFF_CLASSES)):
            if union_total[i] > 0:
                iou = intersection_total[i] / union_total[i]
            else:
                iou = 0.0
            iou_per_class.append(iou)
            print(f"Class '{STUFF_CLASSES[i]}' IoU: {iou:.4f}")
        
        miou = sum(iou_per_class) / len(iou_per_class)
        print(f"Validation mIoU: {miou:.4f}")
        
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving best model with val loss: {best_val_loss:.4f}")
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save model state dict
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"clipseg_finetuned_semantic_cityscapes_{args.learning_rate:.0e}.pth"))
    
    # Save final model
    print("Saving final model")
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"clipseg_finetuned_semantic_cityscapes_final_{args.learning_rate:.0e}.pth"))
    
    return model, processor

def visualize_predictions(model, processor, dataset, output_dir, class_thresholds=None, num_samples=5, device="cuda"):
    """Visualize model predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Default thresholds if not provided
    if class_thresholds is None:
        class_thresholds = {i: 0.3 for i in range(len(STUFF_CLASSES))}
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    model.eval()
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image = sample["image"]
            true_label = sample["label"]
            file_name = sample["file_name"]
            
            # Initialize stuff mask and assigned pixels
            stuff_mask = torch.zeros(len(STUFF_CLASSES), 1024, 2048)
            assigned_pixels = torch.zeros(1024, 2048, dtype=torch.bool)
            
            # Process each class
            for i, class_name in enumerate(STUFF_CLASSES):
                inputs = processor(text=class_name, images=image, padding=True, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Resize logits to match the image size
                logits = logits.unsqueeze(0)
                logits = torch.nn.functional.interpolate(
                    logits, 
                    size=(1024, 2048), 
                    mode="bilinear", 
                    align_corners=False
                )
                logits = logits.squeeze(0)
                logits = torch.sigmoid(logits)
                
                # Create mask
                mask = logits > class_thresholds[i]
                
                # Avoid assigning already assigned pixels
                mask = mask & ~assigned_pixels
                
                # Update stuff mask and assigned pixels
                stuff_mask[i] = mask
                assigned_pixels = assigned_pixels | mask
            
            # Create color mask
            color_mask = colorize_mask(stuff_mask)
            
            # Convert true label to color mask
            true_color_mask = np.zeros((true_label.shape[0], true_label.shape[1], 3), dtype=np.uint8)
            for i in range(len(STUFF_CLASSES)):
                true_color_mask[true_label == i] = NUM_TO_COLOR[i]
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            # Ground truth label
            axes[1].imshow(true_color_mask)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            
            # Predicted label
            axes[2].imshow(color_mask)
            axes[2].set_title("Prediction")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"vis_{file_name}"))
            plt.close()

def colorize_mask(mask):
    """Convert a tensor of class masks to a color image."""
    mask = mask.detach().cpu().numpy()
    color_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        color = NUM_TO_COLOR[i]
        color_mask[mask[i] == 1] = color
    return color_mask

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIPSeg on Cityscapes for stuff segmentation")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Cityscapes dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--model_name", type=str, default="CIDAS/clipseg-rd64-refined", help="Model name")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer batches)")
    
    args = parser.parse_args()
    
    # Train model
    model, processor = train_model(args)
    
    # Visualize predictions if requested
    if args.visualize:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_dataset = CityscapesDataset(
            root_dir=args.data_dir,
            split="val",
            crop_size=(970, 2048),
            resize_size=(1024, 2048)
        )
        
        # Define class-specific thresholds
        class_thresholds = {
            0: 0.3,  # road
            1: 0.3,  # sidewalk
            2: 0.3,  # building
            3: 0.3,  # wall
            4: 0.3,  # fence
            5: 0.3,  # pole
            6: 0.3,  # traffic light
            7: 0.3,  # traffic sign
            8: 0.3,  # vegetation
            9: 0.3,  # terrain
            10: 0.3  # sky
        }
        
        visualize_predictions(
            model=model,
            processor=processor,
            dataset=val_dataset,
            output_dir=os.path.join(args.output_dir, "visualizations"),
            class_thresholds=class_thresholds,
            device=device
        )

if __name__ == "__main__":
    main()
