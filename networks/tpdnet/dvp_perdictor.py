import torch 
from .segformer import SegFormer

class DVPPredictorV2(torch.nn.Module):
    def __init__(self, C_out=3) -> None:
        super().__init__()
        self.backbone = SegFormer(pretrianed=True, 
                                  num_labels=C_out)

        for p in self.backbone.model.segformer.encoder.parameters():
            p.requires_grad = False
        
    def forward(self, x: torch.Tensor):
        B, T, C, W, H = x.shape
        x = x.reshape(B*T, C, W, H)
        features = list(self.backbone.forward_features(x))
        features_high_level = features[-1]
        _, C2, W2, H2 = features_high_level.shape
        features_high_level = features_high_level.reshape(B, T, C2, W2, H2)

        for i in range(len(features)):
            f = features[i]
            BT, C, W1, H1 = f.shape
            f = f.reshape(B, T, C, W1, H1)
            ## average temporal pooling 
            f = f.mean(dim=1)
            features[i] = f 
        
        x = self.backbone.decode_head(features)
        return x, features_high_level
    