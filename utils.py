import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Register hooks on the target layer
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for the input tensor and target class.
        Args:
            input_tensor: input batch tensor, shape (1, C, H, W)
            class_idx: class index for which to generate heatmap. If None, uses predicted class.
        Returns:
            heatmap: numpy array of shape (H, W), values normalized 0-1
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # Forward pass

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward(retain_graph=True)  # Backward pass

        gradients = self.gradients  # Gradients of target layer (shape: [batch, channels, h, w])
        activations = self.activations  # Activations of target layer (same shape)

        # Global average pooling of gradients over spatial dimensions
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # shape: [channels]

        # Weight the activations by corresponding gradients
        weighted_activations = activations[0] * pooled_gradients[:, None, None]

        # Sum over channels to get the raw heatmap
        heatmap = weighted_activations.sum(dim=0).cpu().numpy()

        # Apply ReLU
        heatmap = np.maximum(heatmap, 0)

        # Normalize heatmap to [0, 1]
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        # Resize heatmap to input size (usually 224x224)
        import cv2
        heatmap = cv2.resize(heatmap, (input_tensor.size(3), input_tensor.size(2)))

        return heatmap

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
