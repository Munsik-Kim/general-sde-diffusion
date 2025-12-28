import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

from src.sde.vp_sde import VPSDE
from src.sde.ve_sde import VESDE
from src.models.score_mlp import ScoreMLP
from src.models.score_unet import ScoreUNet

@torch.no_grad()
def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎨 Sampling {args.dataset} ({args.sde_type.upper()})...")
    
    # 모델 로드
    if args.model_type == 'mlp':
        model = ScoreMLP(hidden_dim=256).to(device)
        data_shape = (2000, 2)
    else:
        model = ScoreUNet().to(device)
        data_shape = (64, 1, 28, 28)

    ckpt_path = f"{args.out_dir}/ckpt_{args.dataset}_{args.sde_type}_{args.model_type}.pth"
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: {ckpt_path} not found.")
        return

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # SDE 로드
    if args.sde_type == 'vp':
        sde = VPSDE() # VP-SDE의 beta_max 등을 train과 동일하게 맞춰주는 것이 좋습니다.
    else:
        sde = VESDE()

    # Sampling Loop
    x = sde.prior_sampling(data_shape).to(device)
    n_steps = 1000
    time_steps = torch.linspace(sde.T, 1e-3, n_steps).to(device)
    dt = time_steps[0] - time_steps[1]

    # ★★★ 수정된 루프 시작 ★★★
    # enumerate를 사용하여 현재 몇 번째 스텝인지 파악합니다.
    for i, t in enumerate(time_steps):
        batch_t = torch.ones(data_shape[0], 1).to(device) * t
        
        # 1. SDE 정보 가져오기
        drift, diffusion = sde.sde(x, batch_t)
        _, std = sde.marginal_prob(x, batch_t)
        
        # 2. 모델 예측
        noise_pred = model(x, batch_t)
        
        # [안전장치 1] std가 너무 작으면(0에 가까우면) 나눗셈 폭발 방지를 위해 클리핑
        # 1e-4보다 작은 std는 1e-4로 고정하여 계산 안정성 확보
        std = torch.maximum(std, torch.tensor(1e-4).to(device))
        
        # Score 변환 (Score = -Noise / std)
        score = -noise_pred / std
        
        # 3. Reverse SDE Update
        reverse_drift = drift - (diffusion ** 2) * score
        
        # [안전장치 2] 마지막 단계(t가 거의 0일 때)에서는 노이즈(z)를 더하지 않음
        # 다 그려진 그림에 노이즈를 뿌리는 현상 방지
        if i < len(time_steps) - 1:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
            
        x = x - reverse_drift * dt + diffusion * torch.sqrt(dt) * z
        
        # [안전장치 3] 값이 너무 튀지 않게 강제로 -1 ~ 1 사이로 누름 (Clamping)
        # 이미지 생성 시 픽셀 값이 발산하는 것을 막아줌
        if args.dataset == 'mnist':
            x = torch.clamp(x, -1.0, 1.0)
    # ★★★ 수정된 루프 끝 ★★★

    # 결과 시각화
    plt.figure(figsize=(8, 8))
    if args.dataset == 'swiss_roll':
        data_np = x.cpu().numpy()
        plt.scatter(data_np[:, 0], data_np[:, 1], s=1, c='teal' if args.sde_type=='ve' else 'orange')
        plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5)
        plt.title(f"Generated Swiss Roll ({args.sde_type.upper()}-SDE)")
    else:
        # Denormalize (-1~1 -> 0~1)
        x = (x + 1) / 2.0
        x = x.clamp(0, 1)
        grid = torchvision.utils.make_grid(x, nrow=8)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"Generated MNIST ({args.sde_type.upper()}-SDE)")

    save_path = f"{args.out_dir}/result_{args.dataset}_{args.sde_type}.png"
    plt.savefig(save_path)
    print(f"🎉 Result saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["swiss_roll", "mnist"])
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "mlp", "unet"])
    parser.add_argument("--sde_type", type=str, default="vp", choices=["vp", "ve"])
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    if args.model_type == "auto":
        args.model_type = "unet" if args.dataset == "mnist" else "mlp"
        
    sample(args)