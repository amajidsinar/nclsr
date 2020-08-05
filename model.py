import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.register_buffer('hidden', Variable(torch.zeros(1, self.hidden_size)))

    def forward(self, name_tensors):
        outputs = []
        for name_tensor in name_tensors:
            hidden = self.hidden
            # import pdb; pdb.set_trace()
            assert torch.all(torch.eq(hidden, torch.zeros(1, self.hidden_size)))
            for char_tensor in name_tensor:
                for char_tensor in name_tensor:
                    combined = torch.cat((char_tensor, hidden), 1)
                    hidden = self.i2h(combined)
                    output = self.i2o(combined)
            outputs.append(output)
        outputs = torch.cat(outputs)
        return outputs

    