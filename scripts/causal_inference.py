import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import yaml
import sys
import os
from pathlib import Path
import random

# Add project root to path
sys.path.append(os.getcwd())

from data.dataset import RFMiDDataset, get_val_transforms
from models.detector import DiseaseDetector

# -----------------------------------------------------------------------------
# 1. Occlusion Sensitivity Implementation
# -----------------------------------------------------------------------------
class OcclusionSensitivity:
    def __init__(self, model, device, window_size=40, stride=10):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.stride = stride

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.to(self.device)
        
        # Get original probability
        with torch.no_grad():
            logits = self.model(x)
            orig_prob = torch.softmax(logits, dim=1)[0, 1].item()
        
        b, c, h, w = x.shape
        heatmap = np.zeros((h, w))
        
        # Slide window
        # We batch this to make it faster
        batch_images = []
        coords = []
        
        for y in range(0, h - self.window_size + 1, self.stride):
            for x_pos in range(0, w - self.window_size + 1, self.stride):
                img_occ = x.clone()
                # Occlude with mean value (0 because normalized)
                img_occ[:, :, y:y+self.window_size, x_pos:x_pos+self.window_size] = 0
                batch_images.append(img_occ)
                coords.append((y, x_pos))
                
                if len(batch_images) >= 32: # Batch size
                    batch_tensor = torch.cat(batch_images, dim=0)
                    with torch.no_grad():
                        logits_batch = self.model(batch_tensor)
                        probs_batch = torch.softmax(logits_batch, dim=1)[:, 1].cpu().numpy()
                    
                    # Calculate drop
                    for i, (cy, cx) in enumerate(coords):
                        drop = orig_prob - probs_batch[i]
                        heatmap[cy:cy+self.window_size, cx:cx+self.window_size] += drop
                    
                    batch_images = []
                    coords = []
        
        # Process remaining
        if batch_images:
            batch_tensor = torch.cat(batch_images, dim=0)
            with torch.no_grad():
                logits_batch = self.model(batch_tensor)
                probs_batch = torch.softmax(logits_batch, dim=1)[:, 1].cpu().numpy()
            for i, (cy, cx) in enumerate(coords):
                drop = orig_prob - probs_batch[i]
                heatmap[cy:cy+self.window_size, cx:cx+self.window_size] += drop

        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-7)
        
        return heatmap, logits

# -----------------------------------------------------------------------------
# 2. Causal Analyzer
# -----------------------------------------------------------------------------
class CausalAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Use Occlusion Sensitivity instead of Grad-CAM
        self.sensitivity = OcclusionSensitivity(model, device)
        
    def analyze_iterative(self, image_tensor, original_image_np, max_steps=10, cure_threshold=0.5):
        """
        Perform Iterative Causal Erasure.
        Removes lesions one by one until the model predicts 'Healthy'.
        """
        current_img_np = original_image_np.copy()
        current_tensor = image_tensor.clone()
        
        history = []
        
        # Initial Prediction
        with torch.no_grad():
            logits = self.model(current_tensor)
            current_prob = torch.softmax(logits, dim=1)[0, 1].item()
            
        history.append({
            'step': 0,
            'image': current_img_np.copy(),
            'prob': current_prob,
            'heatmap': None
        })
        
        if current_prob < cure_threshold:
            return history

        for step in range(1, max_steps + 1):
            # 1. Get Sensitivity Map for current state
            heatmap, _ = self.sensitivity(current_tensor, class_idx=1)
            
            # 2. Threshold to find top region (Top 10% - More aggressive but precise)
            # We pick the most salient feature to remove first
            threshold = np.percentile(heatmap, 90) 
            roi_mask = (heatmap > threshold).astype(np.float32)
            
            # Dilate to cover the lesion (Medium size)
            kernel = np.ones((13,13), np.uint8)
            roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
            
            # 3. Heal (Inpaint)
            img_uint8 = (current_img_np * 255).astype(np.uint8)
            mask_uint8 = (roi_mask * 255).astype(np.uint8)
            healed_uint8 = cv2.inpaint(img_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
            healed_img_np = healed_uint8.astype(np.float32) / 255.0
            
            # 4. Predict
            healed_tensor = self._to_tensor(healed_img_np)
            with torch.no_grad():
                logits = self.model(healed_tensor)
                new_prob = torch.softmax(logits, dim=1)[0, 1].item()
            
            # Record
            history.append({
                'step': step,
                'image': healed_img_np.copy(),
                'prob': new_prob,
                'heatmap': heatmap,
                'mask': roi_mask
            })
            
            # Update current
            current_img_np = healed_img_np
            current_tensor = healed_tensor
            
            # Stop if cured
            if new_prob < cure_threshold:
                break
                
        return history

    def _to_tensor(self, img_np):
        # Convert numpy [224,224,3] -> tensor [1,3,224,224]
        # Assuming img_np is float32 [0,1]
        t = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        # Normalize (using ImageNet stats as per training)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
        return t.unsqueeze(0).to(self.device)

# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Iterative Causal Inference')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--detector_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    # Load Model
    print("Loading Detector...")
    model = DiseaseDetector(backbone_name="efficientnet_b0", pretrained=False)
    ckpt = torch.load(args.detector_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.to(device)
    model.eval()
    
    # Load Data
    print("Loading Test Data...")
    test_csv = Path(config['paths']['test_labels'])
    test_img_dir = Path(config['paths']['test_images'])
    
    # We need a transform that DOESN'T normalize yet, so we can visualize
    # But for the model we need normalization.
    # We'll handle normalization manually in the analyzer.
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    raw_transform = A.Compose([
        A.Resize(224, 224),
        A.ToFloat(max_value=255.0), # Scale to [0,1]
        ToTensorV2()
    ])
    
    dataset = RFMiDDataset(
        csv_path=test_csv,
        image_dir=test_img_dir,
        mode="multilabel",
        transform=raw_transform,
        disease_columns=config['dataset']['disease_columns']
    )
    
    # Initialize Analyzer
    analyzer = CausalAnalyzer(model, device)
    
    # Create Output Dir
    output_dir = Path("outputs/causal_analysis_iterative")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select Random High-Risk Samples (No need to filter for single disease anymore!)
    print("Selecting High-Risk Samples (Prob > 0.9)...")
    candidates = []
    
    # Quick scan
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for i in indices:
        if len(candidates) >= args.num_samples:
            break
            
        # Check Ground Truth (Any disease)
        _, labels = dataset[i]
        if torch.sum(labels) == 0: continue
            
        # Check Prediction
        img_tensor, _ = dataset[i]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        norm_tensor = analyzer._to_tensor(img_np)
        
        with torch.no_grad():
            logits = model(norm_tensor)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            
        if prob > 0.90:
            candidates.append(i)
            
    print(f"Running Iterative Analysis on {len(candidates)} images...")
    
    for idx in candidates:
        image_id = dataset.df.iloc[idx][config['dataset']['image_id_column']]
        print(f"Analyzing {image_id}...")
        
        img_tensor, _ = dataset[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        norm_tensor = analyzer._to_tensor(img_np)
        
        history = analyzer.analyze_iterative(norm_tensor, img_np)
        
        # Visualize Sequence
        n_steps = len(history)
        fig, axes = plt.subplots(2, n_steps, figsize=(4 * n_steps, 8))
        
        # Handle case where n_steps=1 (already healthy/failed)
        if n_steps == 1:
            axes = np.array([[axes[0]], [axes[1]]]) # Reshape to 2x1
        
        for i, step_data in enumerate(history):
            # Row 1: Image
            ax_img = axes[0, i] if n_steps > 1 else axes[0]
            ax_img.imshow(step_data['image'])
            status = "HEALTHY" if step_data['prob'] < 0.5 else "DISEASE"
            ax_img.set_title(f"Step {step_data['step']}\nRisk: {step_data['prob']:.1%}\n{status}")
            ax_img.axis('off')
            
            # Row 2: Heatmap (for previous step that led to this)
            ax_map = axes[1, i] if n_steps > 1 else axes[1]
            if step_data['heatmap'] is not None:
                heatmap = cv2.applyColorMap(np.uint8(255 * step_data['heatmap']), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                # Overlay on previous image (which is history[i-1]['image'])
                prev_img = history[i-1]['image']
                overlay = heatmap * 0.4 + prev_img * 0.6
                ax_map.imshow(overlay)
                ax_map.set_title(f"Targeting Region {step_data['step']}")
            else:
                ax_map.text(0.5, 0.5, "Start", ha='center', va='center')
                ax_map.set_title("Initial State")
            ax_map.axis('off')
            
        plt.suptitle(f"Iterative Causal Erasure - {image_id}", fontsize=16)
        plt.tight_layout()
        save_path = output_dir / f"iterative_causal_{image_id}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved -> {save_path}")

if __name__ == '__main__':
    main()
