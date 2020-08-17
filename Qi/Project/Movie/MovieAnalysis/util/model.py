from torch import nn

class TxtModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TxtModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        output = self.classifier(x.double())
        return output.squeeze(1)