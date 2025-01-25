from transformers import AutoModel
from torch import nn
import torch

class Mertv1(nn.Module):
    def __init__(
            self, 
            model_name='m-a-p/MERT-v1-95M',
        ):
        super(Mertv1, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    @torch.no_grad()
    def forward(self, x, return_pooler_output=True,):
        x = {
            "input_values": x,
        }

        output = self.model(**x)

        # take the hidden states from the last layer
        last_layer_hidden_states = output.last_hidden_state

        # for utterance level classification tasks, you can simply reduce the representation in time
        time_reduced_hidden_states = last_layer_hidden_states.mean(-2)

        return time_reduced_hidden_states