from transformers import AutoModel
from torch import nn
import torch

class Dinov2(nn.Module):
    def __init__(
            self, 
            model_name='facebook/dinov2-base',
        ):
        super(Dinov2, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, x, return_pooler_output=True):
        x = {
            "pixel_values": x,
        }
        if return_pooler_output:
            return self.model(**x).pooler_output
        else:
            return self.model(**x).last_hidden_state