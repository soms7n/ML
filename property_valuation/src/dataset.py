import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class PropertyDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_root,
        split,
        mode,                  # "tabular", "fusion", "residual"
        xgb_pred_col=None      # required for residual mode
    ):
        super().__init__()

        if mode not in ["tabular", "fusion", "residual"]:
            raise ValueError(f"Invalid mode: {mode}")

        if mode == "residual" and xgb_pred_col is None:
            raise ValueError("xgb_pred_col must be provided for residual mode")

        self.df = pd.read_csv(csv_path)

        self.df["id"] = (
            self.df["id"]
            .astype(str)
            .str.replace(".0", "", regex=False)
        )

        self.image_root = image_root
        self.split = split
        self.mode = mode
        self.xgb_pred_col = xgb_pred_col

        self.transform = IMAGE_TRANSFORM

        self.tabular_cols = [
            c for c in self.df.columns
            if c not in ["id", "log_price", "lat", "long"]
        ]


    def _load_image(self, zoom, pid):
        folder = "test" if self.split == "test" else "train"
        img_path = os.path.join(
            self.image_root,
            zoom,
            folder,
            f"{pid}.png"
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["id"]

        tabular_np = row[self.tabular_cols].astype("float32").values
        tabular = torch.from_numpy(tabular_np)
        y = None
        if "log_price" in self.df.columns:
            y = torch.tensor(row["log_price"], dtype=torch.float32)
        img16 = self._load_image("zoom16", pid)
        img18 = self._load_image("zoom18", pid)
        if self.mode == "tabular":
            return tabular, y
        if self.mode == "fusion":
            return img16, img18, tabular, y
        if self.mode == "residual":
            if "log_price" in self.df.columns:
                residual = row["log_price"] - row[self.xgb_pred_col]
                residual = torch.tensor(residual, dtype=torch.float32)
            else:
                residual = torch.tensor(0.0, dtype=torch.float32)

            return img16, img18, residual
        raise RuntimeError("Unreachable dataset state")


