import torch
import torch.nn as nn
import torchvision.models as models


class CNNImageEncoder(nn.Module):
    def __init__(self, out_dim, freeze_backbone=True):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)



class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class DualImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder16 = CNNImageEncoder(out_dim=32)
        self.encoder18 = CNNImageEncoder(out_dim=8)

    def forward(self, img16, img18):
        emb16 = self.encoder16(img16)
        emb18 = self.encoder18(img18)
        return torch.cat([emb16, emb18], dim=1)  # 40-dim



class LateFusionModel(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        self.image_encoder = DualImageEncoder()
        self.tabular_encoder = TabularMLP(tabular_dim)

        self.regressor = nn.Sequential(
            nn.Linear(64 + 40, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img16, img18, tabular):
        img_emb = self.image_encoder(img16, img18)
        tab_emb = self.tabular_encoder(tabular)
        fused = torch.cat([tab_emb, img_emb], dim=1)
        return self.regressor(fused).squeeze(1)



class DualZoomResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = DualImageEncoder()

        self.head = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, img16, img18):
        emb = self.image_encoder(img16, img18)
        return self.head(emb).squeeze(1)


class MultiScaleResidualFusion(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, xgb_pred, r16, r18):
        return xgb_pred + r16 + self.alpha * r18
