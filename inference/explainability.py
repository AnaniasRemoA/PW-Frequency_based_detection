import torch
import torch.nn.functional as F
import numpy as np
import cv2

class DeepfakeGradCAM:
    def __init__(self, model, target_layer_name="model.resnet.s5"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        
        # Find the target layer and register hooks
        target_layer = self._get_target_layer(target_layer_name)
        if target_layer is None:
            raise ValueError(f"Could not find target layer: {target_layer_name}")
            
        target_layer.register_forward_hook(self._save_activation)
        
        # Disable gradients for model_ft to prevent custom autograd bug
        for param in self.model.model_ft.parameters():
            param.requires_grad = False
            
    def _get_target_layer(self, name):
        for n, m in self.model.named_modules():
            if n == name:
                return m
        return None
        
    def _save_activation(self, module, input, output):
        if isinstance(output, (list, tuple)):
            self.activations = output[0]
        else:
            self.activations = output
            
        if hasattr(self.activations, "requires_grad") and self.activations.requires_grad:
            self.activations.register_hook(self._save_tensor_gradient)

    def _save_tensor_gradient(self, grad):
        self.gradients = grad

    def generate_heatmap(self, img_tensor, ft_tensor):
        """
        Generates a Grad-CAM heatmap for the given inputs.
        Args:
            img_tensor: [B, C, T, H, W]
            ft_tensor: [B, C, H, W] (frequency tensor)
        Returns:
            heatmap (numpy array of shape (H, W)): Values in [0, 1]
        """
        with torch.enable_grad():
            self.model.eval()
            self.model.zero_grad()
            
            # Ensure img_tensor requires grad for backprop
            img_tensor = img_tensor.clone().detach().requires_grad_(True)
            ft_tensor = ft_tensor.clone().detach().requires_grad_(False)
                
            # Forward pass
            output = self.model(img_tensor, ft_tensor)
            
            # Backward pass
            output.sum().backward()
                
            # self.gradients: [B, C, T, H', W']
            # self.activations: [B, C, T, H', W']
            if self.gradients is not None:
                weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
                cam = (weights * self.activations).sum(dim=1, keepdim=True)
            else:
                cam = torch.mean(self.activations.abs(), dim=1, keepdim=True)
            
            # Apply ReLU
            cam = F.relu(cam)
        
        # Mean across temporal dimension to get a single spatial heatmap
        cam = cam.mean(dim=2)  # shape: [B, 1, H', W']
        
        # Resize to match input spatial size
        B, C, T, H, W = img_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the heatmap onto the image.
    Args:
        img: RGB image numpy array (H, W, 3) in [0, 255]
        heatmap: numpy array (H, W) in [0, 1]
    Returns:
        overlayed_img: RGB image numpy array
    """
    # Convert heatmap to color map
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed_img
