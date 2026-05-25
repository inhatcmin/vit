import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

# =========================
# 1. 설정
# =========================
DATA_DIR = "./grid"

TRAIN_DIR = os.path.join(DATA_DIR, "train/good")
TEST_DIR = os.path.join(DATA_DIR, "test")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

IMG_SIZE = 224
MAX_TRAIN_IMAGES = None   # 빠르게 테스트하려면 50 등으로 설정


# =========================
# 2. Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# =========================
# 3. ResNet Patch Feature Extractor
# =========================
class ResNetPatchExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

        self.features = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3
        )

    def forward(self, x):
        feat = self.features(x)
        # [B, C, H, W] -> [B, H*W, C]
        feat = feat.permute(0, 2, 3, 1)
        feat = feat.reshape(feat.size(0), -1, feat.size(-1))
        return feat


# =========================
# 4. ViT Patch Feature Extractor
# =========================
class ViTPatchExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights)

    def forward(self, x):
        n = x.shape[0]

        # image -> patch embedding
        x = self.model._process_input(x)

        # class token 추가
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # transformer encoder
        x = self.model.encoder(x)

        # class token 제거, patch token만 사용
        patch_tokens = x[:, 1:, :]   # [B, 196, 768]
        return patch_tokens


# =========================
# 5. Feature 추출
# =========================
def extract_patch_features(model, img_path):
    model.eval()
    x = load_image(img_path)

    with torch.no_grad():
        feat = model(x)

    feat = feat.squeeze(0).cpu().numpy()
    return feat


# =========================
# 6. Memory Bank 생성
# =========================
def build_memory_bank(model, train_dir):
    features = []
    files = sorted(os.listdir(train_dir))

    if MAX_TRAIN_IMAGES is not None:
        files = files[:MAX_TRAIN_IMAGES]

    for fname in tqdm(files, desc="Build memory bank"):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(train_dir, fname)
            feat = extract_patch_features(model, img_path)
            features.append(feat)

    memory_bank = np.concatenate(features, axis=0)
    print("memory bank shape:", memory_bank.shape)
    return memory_bank


# =========================
# 7. PatchCore Test
# =========================
def evaluate_patchcore(model_name, model):
    print("\n==============================")
    print("Model:", model_name)
    print("==============================")

    model = model.to(device)
    model.eval()

    memory_bank = build_memory_bank(model, TRAIN_DIR)

    # nearest neighbor 준비
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(memory_bank)

    y_true = []
    y_score = []
    results = []

    for defect_type in sorted(os.listdir(TEST_DIR)):
        defect_dir = os.path.join(TEST_DIR, defect_type)

        if not os.path.isdir(defect_dir):
            continue

        for fname in tqdm(sorted(os.listdir(defect_dir)), desc=f"Test {defect_type}"):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(defect_dir, fname)
            patch_feat = extract_patch_features(model, img_path)

            distances, _ = nn.kneighbors(patch_feat)

            # PatchCore image-level anomaly score
            score = distances.max()

            label = 0 if defect_type == "good" else 1

            y_true.append(label)
            y_score.append(score)

            results.append({
                "model": model_name,
                "file": fname,
                "defect_type": defect_type,
                "label": label,
                "score": score
            })

    auc = roc_auc_score(y_true, y_score)
    print("AUROC:", auc)

    return auc, results


# =========================
# 8. 실행
# =========================
resnet_model = ResNetPatchExtractor()
vit_model = ViTPatchExtractor()

resnet_auc, resnet_results = evaluate_patchcore("PatchCore_ResNet18", resnet_model)
vit_auc, vit_results = evaluate_patchcore("PatchCore_ViT_B16", vit_model)

# =========================
# 9. 결과 저장
# =========================
all_results = resnet_results + vit_results
df = pd.DataFrame(all_results)
df.to_csv("patchcore_resnet_vs_vit_results.csv", index=False)

print("\n===== Final Result =====")
print("PatchCore + ResNet18 AUROC:", resnet_auc)
print("PatchCore + ViT-B/16 AUROC:", vit_auc)

print("\n결과 파일 저장 완료: patchcore_resnet_vs_vit_results.csv")
