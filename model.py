import torch
import torch.nn as nn
import torch.nn.functional as F


class BilBlock(nn.Module):

    def __init__(self, num_features, hidden_size=16):
        super(BilBlock, self).__init__()

        self.double_net = nn.Bilinear(num_features, num_features, hidden_size)

        self.mlp = nn.Sequential(
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_features),
            nn.Tanh(),
            nn.BatchNorm1d(num_features),
        )

    def forward(self, a_tensor, b_tensor):
        duo = self.double_net(a_tensor, b_tensor)
        return self.mlp(duo)


class MBREncoder(nn.Module):

    def __init__(self, num_features, hidden_size=16):
        super(MBREncoder, self).__init__()

        self.double_net = BilBlock(num_features, hidden_size)

        self.single_net = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_features),
            nn.Tanh(),
            nn.BatchNorm1d(num_features),
        )

        self.concat_net = nn.Sequential(
            nn.Linear(num_features * 3, num_features * 3),
            nn.Tanh(),
            nn.BatchNorm1d(num_features * 3),
            nn.Linear(num_features * 3, num_features),
            nn.Tanh(),
            nn.BatchNorm1d(num_features),
        )

    def forward(self, a_tensor, b_tensor):
        duo = self.double_net(a_tensor, b_tensor)
        a_output = self.single_net(a_tensor)
        b_output = self.single_net(b_tensor)

        concat = torch.cat((duo, a_output, b_output), dim=1)

        return self.concat_net(concat)


class IEJModel(nn.Module):
    def __init__(self, num_dimensions, num_features=4, hidden_size=16, num_layers=2):
        super(IEJModel, self).__init__()
        self.encoder = MBREncoder(num_features, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(num_dimensions * num_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        for _ in range(num_layers):
            self.decoder.append(nn.Linear(hidden_size, hidden_size))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.BatchNorm1d(hidden_size))

        self.decoder.append(nn.Linear(hidden_size, num_dimensions))
        self.decoder.append(nn.Sigmoid())

    def forward(self, a_tensor, b_tensor):
        # Input: (batch_size, num_dimensions, num_features)
        a_slices = torch.unbind(a_tensor, dim=1)
        b_slices = torch.unbind(b_tensor, dim=1)

        x = torch.cat([self.encoder(a, b) for a, b in zip(a_slices, b_slices)], dim=1)
        x = self.decoder(x)
        x = F.normalize(x, dim=1, p=2)

        return x


class SimpleIEJ(nn.Module):
    def __init__(self, num_dimensions, num_features=4, hidden_size=16, num_layers=2):
        super(SimpleIEJ, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(2 * num_dimensions * num_features, hidden_size),
            nn.Tanh(),
        )
        for _ in range(num_layers):
            self.model.append(nn.Linear(hidden_size, hidden_size))
            self.model.append(nn.Tanh())
        self.model.append(nn.Linear(hidden_size, num_dimensions))
        self.model.append(nn.Sigmoid())

    def forward(self, a_tensor, b_tensor):
        a_tensor = self.flatten(a_tensor)
        b_tensor = self.flatten(b_tensor)

        x = torch.cat((a_tensor, b_tensor), dim=1)

        x = self.model(x)
        x = F.normalize(x, dim=1, p=2)

        return x
