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
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import defaultdict
import json

# SAM2 imports
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Define the Cityscapes stuff classes
STUFF_CLASSES = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 
    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
    9: 'terrain', 10: 'sky'
}

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

# Reverse mapping
ID_TO_TRAINID = {v: k for k, v in TRAINID_TO_ID.items()}

# Color mapping for visualization
NUM_TO_COLOR = {
    0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70],  
    3: [102, 102, 156], 4: [190, 153, 153], 5: [153, 153, 153],
    6: [250, 170, 30], 7: [220, 220, 0], 8: [107, 142, 35], 
    9: [152, 251, 152], 10: [70, 130, 180]
}

# Define threshold configurations
CLIPSEG_THRESHOLDS = {
    0: 0.7,  # road
    1: 0.7,  # sidewalk
    2: 0.6,  # building
    3: 0.5,  # wall
    4: 0.3,  # fence
    5: 0.3,  # pole
    6: 0.3,  # traffic light
    7: 0.3,  # traffic sign
    8: 0.5,  # vegetation
    9: 0.3,  # terrain
    10: 0.6  # sky
}

def initialize_sam2(checkpoint_path, config_path, config_name, device):
    """Initialize SAM2 model"""
    # Reset Hydra's global state to avoid conflicts
    GlobalHydra.instance().clear()
    
    with initialize(version_base=None, config_path=config_path):
        sam2_model = build_sam2(config_name, checkpoint_path, device=device)
    
    return SAM2ImagePredictor(sam2_model)

def get_clipseg_masks(model, processor, image, device):
    """Get masks from CLIPSeg for all classes"""
    stuff_mask = torch.zeros((len(STUFF_CLASSES), image.shape[0], image.shape[1]), device=device)
    assigned_pixels = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.bool, device=device)
    
    for i in range(len(STUFF_CLASSES)):
        prompt = f"{STUFF_CLASSES[i]}"
        
        inputs = processor(text=prompt, images=image, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        logits = logits.unsqueeze(0)
        logits = F.interpolate(logits, size=(image.shape[0], image.shape[1]), mode="bilinear", align_corners=False)
        logits = logits.squeeze(0)
        probs = torch.sigmoid(logits)
        
        mask = probs > CLIPSEG_THRESHOLDS[i]
        mask = mask & ~assigned_pixels
        
        stuff_mask[i] = mask.float()
        assigned_pixels = assigned_pixels | mask
        
    return stuff_mask, assigned_pixels

def sample_points(mask, num_samples=10):
    """Sample points from mask for SAM2 prompting"""
    idx_points = torch.nonzero(mask, as_tuple=False)
    
    if idx_points.size(0) >= num_samples:
        selected_indices = np.random.choice(idx_points.size(0), num_samples, replace=False)
        selected_points = idx_points[selected_indices]
    else:
        # If fewer than num_samples valid points, pad with zeros
        selected_points = torch.zeros(num_samples, 2, dtype=torch.int64)
        selected_points[:idx_points.size(0)] = idx_points
    
    return selected_points

def colorize_mask(mask):
    """Convert mask to RGB visualization"""
    mask = mask.detach().cpu().numpy()
    color_mask = np.zeros((mask.shape[1], mask.shape[2], 3))
    
    for i in range(mask.shape[0]):
        color = NUM_TO_COLOR[i]
        color_mask[mask[i] == 1] = color
    
    return color_mask.astype(np.uint8)

def load_ground_truth(gt_path, size=(2048, 1024)):
    """Load and convert ground truth to the proper format"""
    gt_label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_label = cv2.resize(gt_label, size, interpolation=cv2.INTER_NEAREST)
    
    # Convert Cityscapes IDs to training IDs
    gt_converted = np.ones_like(gt_label) * 255  # Initialize with ignore label
    
    for k, v in ID_TO_TRAINID.items():
        gt_converted[gt_label == k] = v
    
    # Set instance classes to ignore label (we only care about stuff classes)
    for i in range(24, 34):  # Instance class IDs
        gt_converted[gt_label == i] = 255
        
    return gt_converted

def compute_iou(pred, gt):
    """Compute IoU between prediction and ground truth"""
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    
    if union > 0:
        return intersection / union
    else:
        return 0.0

def create_confusion_matrix(pred_label, gt_label, num_classes=11):
    """Create confusion matrix for calculating mIoU"""
    mask = (gt_label >= 0) & (gt_label < num_classes)
    confusion = np.bincount(
        num_classes * gt_label[mask].astype(int) + pred_label[mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return confusion

def evaluate_cityscapes_val(args):
    # Set device and random seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Using device: {device}")
    print(f"Evaluating with fusion mode: {args.fusion_mode}")
    
    # Load CLIPSeg model and processor
    processor = AutoProcessor.from_pretrained(args.clipseg_model_name)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(args.clipseg_model_name)
    
    # Load fine-tuned weights
    clipseg_model.load_state_dict(torch.load(args.clipseg_model_path, map_location=device))
    clipseg_model.to(device)
    clipseg_model.eval()
    
    # Initialize SAM2
    sam2_predictor = initialize_sam2(
        args.sam2_checkpoint,
        args.sam2_config_dir,
        args.sam2_config_name,
        device
    )
    
    # Get validation set images
    val_image_files = glob.glob(os.path.join(args.data_dir, "leftImg8bit", "val", "*", "*_leftImg8bit.png"))
    val_image_files.sort()
    print(f"Found {len(val_image_files)} validation images")
    
    # Prepare results storage
    confusion_matrix = np.zeros((len(STUFF_CLASSES), len(STUFF_CLASSES)), dtype=np.int64)
    city_results = defaultdict(list)
    image_results = {}
    
    # Process all validation images
    for image_path in tqdm(val_image_files, desc="Evaluating validation set"):
        # Extract city information from path
        path_parts = image_path.split(os.sep)
        city = path_parts[-2]
        file_name = path_parts[-1]
        image_id = file_name.replace("_leftImg8bit.png", "")


        #leftImg8bit: /media/avalocal/T7/pardis/pardis/perception_system/datasets/cityscapes_complete/leftImg8bit/val/lindau/lindau_000000_000019_leftImg8bit.png
        #gtFine:      /media/avalocal/T7/pardis/pardis/perception_system/datasets/cityscapes_complete/gtFine/val/lindau/lindau_000000_000019_gtFine_labelIds.png
        
        # Load ground truth
        # gt_path = image_path.replace("leftImg8bit", "gtFine").replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        gt_path = image_path.replace("_leftImg8bit.png", "_gtFine_labelIds.png").replace("leftImg8bit", "gtFine")
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {image_path}")
            print(f"Warning gt_path: {gt_path}")
            continue
        
        gt_label = load_ground_truth(gt_path)
        
        # Load and preprocess image
        pil_image = Image.open(image_path)
        pil_image = pil_image.crop((0, 0, 2048, 970))
        pil_image = pil_image.resize((2048, 1024))
        np_image = np.array(pil_image)
        
        # Step 1: Generate CLIPSeg masks and sample points
        clipseg_mask, assigned_pixels = get_clipseg_masks(
            clipseg_model, processor, np_image, device
        )
        
        # Sample points for each class from CLIPSeg masks
        num_samples = args.num_samples
        points = torch.zeros(len(STUFF_CLASSES), num_samples, 2)
        
        for i in range(len(STUFF_CLASSES)):
            selected_points = sample_points(clipseg_mask[i], num_samples=num_samples)
            points[i] = selected_points
        
        # Step 2: Use SAM2 to refine masks using the sampled points
        sam2_predictor.set_image(np_image)
        combined_mask = torch.zeros((len(STUFF_CLASSES), 1024, 2048), device=device)
        
        for i in range(len(STUFF_CLASSES)):
            # Convert points format: swap x, y to y, x for SAM2
            coords = points[i].clone()
            sam_points = coords[:, [1, 0]].cpu().numpy()  # Swap coordinates for SAM2
            
            # Filter out zero points (which were padding)
            valid_points = np.all(sam_points != 0, axis=1)
            if not np.any(valid_points):
                continue  # Skip if no valid points
                
            valid_sam_points = sam_points[valid_points]
            
            if len(valid_sam_points) > 0:
                # Get SAM2 prediction
                masks, scores, _ = sam2_predictor.predict(
                    point_coords=valid_sam_points,
                    point_labels=np.ones(len(valid_sam_points), dtype=np.int64),
                    multimask_output=False,
                )
                
                # Process SAM2 mask
                if masks.shape[0] > 0:  # Check if any masks were returned
                    sam_mask = torch.from_numpy(masks).to(device)
                    
                    # Apply the selected fusion mode
                    if args.fusion_mode == "union":
                        fused_mask = (sam_mask.squeeze(0) | clipseg_mask[i].bool()).float()
                    elif args.fusion_mode == "intersection":
                        fused_mask = (sam_mask.squeeze(0) & clipseg_mask[i].bool()).float()
                    elif args.fusion_mode == "weighted":
                        fused_mask = (args.sam_weight * sam_mask.squeeze(0).float() + 
                                     (1 - args.sam_weight) * clipseg_mask[i]).float()
                        fused_mask = (fused_mask > 0.5).float()  # Threshold
                    elif args.fusion_mode == "clipseg_only":
                        fused_mask = clipseg_mask[i].float()
                    else:  # sam_only
                        fused_mask = sam_mask.squeeze(0).float()
                    
                    combined_mask[i] = fused_mask
        
        # Create pseudo-label from combined mask
        pseudo_label = torch.zeros((1024, 2048), dtype=torch.uint8, device=device)
        for i in range(len(STUFF_CLASSES)):
            pseudo_label[combined_mask[i] > 0] = i
        
        # Convert to numpy for evaluation
        pseudo_label_np = pseudo_label.cpu().numpy()
        
        # Compute IoU for each class
        class_ious = []
        for i in range(len(STUFF_CLASSES)):
            pred = combined_mask[i].cpu().numpy()
            gt = (gt_label == i).astype(np.float32)
            
            iou = compute_iou(pred, gt)
            class_ious.append(iou)
            
            # Store city-wise results
            city_results[city].append(iou)
        
        # Update confusion matrix
        confusion_matrix += create_confusion_matrix(pseudo_label_np, gt_label, len(STUFF_CLASSES))
        
        # Store per-image results
        image_results[image_id] = {
            "city": city,
            "class_ious": {STUFF_CLASSES[i]: float(class_ious[i]) for i in range(len(STUFF_CLASSES))},
            "mean_iou": float(np.mean(class_ious))
        }
        
        # Save visualization if requested
        if args.save_vis:
            # Create visualization directory
            vis_dir = os.path.join(args.output_dir, "visualizations", city)
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_name = file_name.replace("_leftImg8bit.png", "_vis.png")
            vis_path = os.path.join(vis_dir, vis_name)
            
            # Generate colorized mask
            color_mask = colorize_mask(combined_mask)
            
            # Create figure with image, prediction, and ground truth
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            axs[0].imshow(np_image)
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            # Prediction
            axs[1].imshow(color_mask)
            axs[1].set_title(f'Prediction (mIoU: {np.mean(class_ious):.4f})')
            axs[1].axis('off')
            
            # Ground truth visualization
            gt_colored = np.zeros((gt_label.shape[0], gt_label.shape[1], 3), dtype=np.uint8)
            for i in range(len(STUFF_CLASSES)):
                gt_colored[gt_label == i] = NUM_TO_COLOR[i]
            
            axs[2].imshow(gt_colored)
            axs[2].set_title('Ground Truth')
            axs[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # Calculate per-class IoU from confusion matrix
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - intersection
    
    # Handle division by zero
    class_iou = np.zeros_like(intersection, dtype=np.float32)
    valid = union > 0
    class_iou[valid] = intersection[valid] / union[valid]
    
    # Calculate mIoU
    mean_iou = np.mean(class_iou)
    
    # Calculate city-wise mIoU
    city_miou = {city: np.mean(ious) for city, ious in city_results.items()}
    
    # Print results
    headers = ["Class", "IoU"]
    table_data = [[STUFF_CLASSES[i], f"{class_iou[i]:.4f}"] for i in range(len(STUFF_CLASSES))]
    table_data.append(["Mean", f"{mean_iou:.4f}"])
    
    print("\n===== Class-wise IoU Results =====")
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    
    print("\n===== City-wise mIoU Results =====")
    city_table = [[city, f"{miou:.4f}"] for city, miou in city_miou.items()]
    city_table.append(["Overall", f"{mean_iou:.4f}"])
    print(tabulate(city_table, headers=["City", "mIoU"], tablefmt="pretty"))
    
    # Save detailed results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save class IoU results
        class_results = {
            "class_iou": {STUFF_CLASSES[i]: float(class_iou[i]) for i in range(len(STUFF_CLASSES))},
            "mean_iou": float(mean_iou)
        }
        
        with open(os.path.join(args.output_dir, "class_results.json"), 'w') as f:
            json.dump(class_results, f, indent=2)
        
        # Save city-wise results
        with open(os.path.join(args.output_dir, "city_results.json"), 'w') as f:
            json.dump({
                "city_miou": {k: float(v) for k, v in city_miou.items()},
                "overall_miou": float(mean_iou)
            }, f, indent=2)
        
        # Save per-image results
        with open(os.path.join(args.output_dir, "image_results.json"), 'w') as f:
            json.dump(image_results, f, indent=2)
            
        # Create comparison plots
        if args.compare_modes and len(args.compare_modes) > 0:
            # Create the path to store comparison results
            comparison_dir = os.path.join(args.output_dir, "comparisons")
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Load other results if they exist
            comparison_data = {args.fusion_mode: class_results}
            for mode in args.compare_modes:
                mode_file = os.path.join(args.output_dir.replace(args.fusion_mode, mode), "class_results.json")
                if os.path.exists(mode_file):
                    with open(mode_file, 'r') as f:
                        comparison_data[mode] = json.load(f)
            
            # Create class-wise comparison bar chart
            if len(comparison_data) > 1:
                # Set up class-wise comparison plot
                plt.figure(figsize=(14, 8))
                
                # Get all classes
                classes = list(STUFF_CLASSES.values())
                x = np.arange(len(classes))
                width = 0.8 / len(comparison_data)
                
                # Plot each mode
                for i, (mode, data) in enumerate(comparison_data.items()):
                    class_ious = [data["class_iou"][cls] for cls in classes]
                    plt.bar(x + i*width - 0.4 + width/2, class_ious, width, label=mode)
                
                plt.xlabel('Classes')
                plt.ylabel('IoU')
                plt.title('Class-wise IoU Comparison Between Fusion Modes')
                plt.xticks(x, classes, rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(comparison_dir, "class_comparison.png"), dpi=150)
                plt.close()
                
                # Create mIoU comparison bar chart
                plt.figure(figsize=(10, 6))
                modes = list(comparison_data.keys())
                mious = [data["mean_iou"] for data in comparison_data.values()]
                
                plt.bar(modes, mious)
                plt.xlabel('Fusion Modes')
                plt.ylabel('mIoU')
                plt.title('Mean IoU Comparison Between Fusion Modes')
                plt.ylim(0, 1.0)
                
                # Add value labels on top of bars
                for i, v in enumerate(mious):
                    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(comparison_dir, "miou_comparison.png"), dpi=150)
                plt.close()
    
    return mean_iou, class_iou

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM2+CLIPSeg on Cityscapes validation set")
    
    # Dataset and output paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Cityscapes dataset")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_vis", action="store_true", help="Save visualizations of the results")
    
    # CLIPSeg model arguments
    parser.add_argument("--clipseg_model_name", type=str, default="CIDAS/clipseg-rd64-refined", 
                       help="Base CLIPSeg model name")
    parser.add_argument("--clipseg_model_path", type=str, required=True, 
                       help="Path to fine-tuned CLIPSeg model weights")
    
    # SAM2 model arguments
    parser.add_argument("--sam2_checkpoint", type=str, required=True,
                       help="Path to SAM2 checkpoint file")
    parser.add_argument("--sam2_config_dir", type=str, required=True,
                       help="Path to SAM2 config directory")
    parser.add_argument("--sam2_config_name", type=str, default="sam2.1_hiera_l.yaml",
                       help="Name of SAM2 config file")
    
    # Processing parameters
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of points to sample for each class")
    parser.add_argument("--fusion_mode", type=str, default="weighted", 
                       choices=["union", "intersection", "weighted", "sam_only", "clipseg_only"],
                       help="Mode for fusing CLIPSeg and SAM2 masks")
    parser.add_argument("--sam_weight", type=float, default=0.7,
                       help="Weight for SAM2 mask when using weighted fusion mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Comparison
    parser.add_argument("--compare_modes", nargs="+", default=[],
                       help="Compare with results from other fusion modes")
    
    args = parser.parse_args()
    
    # Update output dir with fusion mode
    args.output_dir = os.path.join(args.output_dir, args.fusion_mode)
    
    # Run evaluation
    miou, class_iou = evaluate_cityscapes_val(args)
    
    print(f"\nEvaluation complete! Overall mIoU: {miou:.4f}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
