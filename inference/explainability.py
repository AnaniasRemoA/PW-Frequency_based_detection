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
        target_layer.register_full_backward_hook(self._save_gradient)
        
        # Disable gradients for model_ft to prevent custom autograd bug
        for param in self.model.model_ft.parameters():
            param.requires_grad = False
            
    def _get_target_layer(self, name):
        for n, m in self.model.named_modules():
            if n == name:
                return m
        return None
        
    def _save_activation(self, module, input, output):
        if isinstance(output, list) or isinstance(output, tuple):
            self.activations = output[0]
        else:
            self.activations = output
            
    def _save_gradient(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple) or isinstance(grad_output, list):
            self.gradients = grad_output[0]
        else:
            self.gradients = grad_output

    def generate_heatmap(self, img_tensor, ft_tensor):
        """
        Generates a Grad-CAM heatmap for the given inputs.
        Args:
            img_tensor: [B, C, T, H, W]
            ft_tensor: [B, C, H, W] (frequency tensor)
        Returns:
            heatmap (numpy array of shape (H, W)): Values in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Ensure img_tensor requires grad for backprop
        if not img_tensor.requires_grad:
            img_tensor.requires_grad_(True)
            
        # Forward pass
        output = self.model(img_tensor, ft_tensor)
        
        # Backward pass
        if output.dim() == 2 and output.shape[1] == 1:
            output.backward(torch.ones_like(output))
        else:
            output.backward(torch.ones_like(output))
            
        # self.gradients: [B, C, T, H', W']
        # self.activations: [B, C, T, H', W']
        
        # Global average pooling over spatial and temporal dimensions
        # Pool across T, H', W'
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        
        # Weight the activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
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
