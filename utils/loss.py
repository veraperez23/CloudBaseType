import torch
import torch.nn as nn
import torch.nn.functional as F


# class CustomCrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ce = nn.CrossEntropyLoss()
#         # Create a class index tensor like [0, 1, 2, ..., num_classes-1]
#         self.class_indices = torch.arange(9).float().view(1, -1)  # shape (1, num_classes)

#     def forward(self, logits, labels):
#         # Softmax probabilities (differentiable)
#         probs = F.softmax(logits, dim=1)  # shape: (batch_size, num_classes)

#         # Expand class indices to match batch
#         if probs.device != self.class_indices.device:
#             self.class_indices = self.class_indices.to(probs.device)

#         # Compute expected class index for each prediction (differentiable approximation of argmax)
#         expected_class = (probs * self.class_indices).sum(dim=1)  # shape: (batch_size,)

#         # Compute squared distance to true label (converted to float)
#         distance = (expected_class - labels.float()) ** 2
#         mean_distance = torch.sqrt(distance.mean())

#         # Standard CE loss
#         ce_loss = self.ce(logits, labels)

#         # Combine both losses
#         total_loss = ce_loss * mean_distance
#         return total_loss
    