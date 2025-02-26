import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from torch.nn import functional as F

# Define the Cityscapes stuff classes
STUFF_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", 
    "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky"
]

# Define the mapping between our training IDs and Cityscapes IDs
TRAINID_TO_ID = {
    0: 7,       # road
    1: 8,       # sidewalk
    2: 11,      # building
    3: 12,      # wall
    4: 13,      # fence
    5: 17,      # pole
    6: 19,      # traffic light
    7: 20,      # traffic sign
    8: 21,      # vegetation
    9: 22,      # terrain
    10: 23,     # sky
}

# Define thresholds for each class
CLASS_THRESHOLDS = {
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

def create_pseudo_labels(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_name)
    
    # Load fine-tuned weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get all leftImg8bit images paths
    image_files = []
    for split in ["train", "val", "test"]:
        split_images = glob.glob(os.path.join(args.data_dir, "leftImg8bit", split, "*", "*_leftImg8bit.png"))
        image_files.extend(split_images)
    
    image_files.sort()
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Creating pseudo labels"):
        # Extract split and city information from path
        path_parts = image_path.split(os.sep)
        split = path_parts[-3]
        city = path_parts[-2]
        file_name = path_parts[-1]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(args.data_dir, "gtClip", split, city)
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output path (with similar naming convention to gtFine)
        output_name = file_name.replace("_leftImg8bit.png", "_gtClip_labelIds.png")
        output_path = os.path.join(output_dir, output_name)
        
        # Skip if output already exists
        if os.path.exists(output_path) and not args.overwrite:
            continue
            
        # Load image
        image = Image.open(image_path)
        
        # Resize image to the expected size for processing
        image_resized = image.resize((2048, 1024))
        
        # Initialize stuff mask and assigned pixels
        stuff_mask = torch.zeros((len(STUFF_CLASSES), 1024, 2048), device=device)
        assigned_pixels = torch.zeros((1024, 2048), dtype=torch.bool, device=device)
        
        # Get prediction for each class
        with torch.no_grad():
            for i, class_name in enumerate(STUFF_CLASSES):
                inputs = processor(
                    text=class_name, 
                    images=np.array(image_resized), 
                    padding=True, 
                    return_tensors="pt"
                ).to(device)
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Resize logits to match the image size
                logits = logits.unsqueeze(0)
                logits = F.interpolate(
                    logits, 
                    size=(1024, 2048), 
                    mode="bilinear", 
                    align_corners=False
                )
                logits = logits.squeeze(0)
                probs = torch.sigmoid(logits)
                
                # Create mask using class-specific threshold
                mask = probs > CLASS_THRESHOLDS[i]
                
                # Avoid assigning already assigned pixels
                mask = mask & ~assigned_pixels
                
                # Update stuff mask and assigned pixels
                stuff_mask[i] = mask.float()
                assigned_pixels = assigned_pixels | mask
        
        # Convert one-hot representation to label map
        pseudo_label = torch.zeros((1024, 2048), dtype=torch.uint8, device=device)
        
        # Compute IoU for this image (if we have ground truth)
        gt_path = image_path.replace("leftImg8bit", "gtFine").replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        
        if os.path.exists(gt_path):
            # Load ground truth
            gt_label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_label = cv2.resize(gt_label, (2048, 1024), interpolation=cv2.INTER_NEAREST)
            
            # Convert Cityscapes IDs to training IDs for IoU computation
            id_to_trainid = {v: k for k, v in TRAINID_TO_ID.items()}
            gt_converted = np.ones_like(gt_label) * 255  # Initialize with ignore label
            
            for k, v in id_to_trainid.items():
                gt_converted[gt_label == k] = v
            
            # Set instance classes to ignore label (we only care about stuff classes)
            for i in range(24, 34):  # Instance class IDs
                gt_converted[gt_label == i] = 255
                
            # Compute IoU for each class
            iou_per_class = []
            for i in range(len(STUFF_CLASSES)):
                pred = stuff_mask[i].cpu().numpy()
                gt = (gt_converted == i).astype(np.float32)
                
                intersection = np.logical_and(pred > 0, gt > 0).sum()
                union = np.logical_or(pred > 0, gt > 0).sum()
                
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 0.0
                
                iou_per_class.append(iou)
            
            miou = np.mean(iou_per_class)
            print(f"Image {file_name} - mIoU: {miou:.4f}")
            
            # Skip saving if mIoU is below threshold
            if miou < args.miou_threshold:
                print(f"Skipping image {file_name} due to low mIoU: {miou:.4f}")
                continue
        
        # If no ground truth exists or mIoU is good enough, create the pseudo-label
        for i in range(len(STUFF_CLASSES)):
            # Assign pixels to the current class if they have the highest probability
            pseudo_label[stuff_mask[i] > 0] = i
        
        # Convert training IDs back to original Cityscapes IDs for saving
        pseudo_label_np = pseudo_label.cpu().numpy()
        output_label = np.ones_like(pseudo_label_np) * 255  # Initialize with unlabeled
        
        for train_id, id in TRAINID_TO_ID.items():
            output_label[pseudo_label_np == train_id] = id
        
        # Save pseudo label
        cv2.imwrite(output_path, output_label)

def main():
    parser = argparse.ArgumentParser(description="Create pseudo labels using fine-tuned CLIPSeg model")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Cityscapes dataset")
    parser.add_argument("--model_name", type=str, default="CIDAS/clipseg-rd64-refined", help="Base model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model weights")
    parser.add_argument("--miou_threshold", type=float, default=0.3, help="mIoU threshold for saving pseudo labels")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing pseudo labels")
    
    args = parser.parse_args()
    
    create_pseudo_labels(args)

if __name__ == "__main__":
    main()
