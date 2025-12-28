import torch
import torch.optim as optim
import argparse
import os
import csv
import time
from tqdm import tqdm

from src.sde.vp_sde import VPSDE
from src.sde.ve_sde import VESDE
from src.datasets import get_dataset
from src.models.score_mlp import ScoreMLP
from src.models.score_unet import ScoreUNet

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training {args.dataset} | Model: {args.model_type} | SDE: {args.sde_type.upper()} (Target: Noise)")
    
    # 모델 선택
    if args.model_type == 'mlp':
        model = ScoreMLP(hidden_dim=256).to(device)
    else:
        model = ScoreUNet().to(device)

    # SDE 선택
    if args.sde_type == 'vp':
        sde = VPSDE(beta_min=0.1, beta_max=20.0)
    elif args.sde_type == 've':
        sde = VESDE(sigma_min=0.01, sigma_max=50.0)
    else:
        raise ValueError("Unknown SDE type")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataloader = get_dataset(args.dataset, batch_size=args.batch_size)
    
    os.makedirs(args.out_dir, exist_ok=True)

    # 로그 파일 설정
    log_path = f"{args.out_dir}/train_log_{args.dataset}_{args.sde_type}.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg_Loss', 'Time_Sec'])
    print(f"📄 Logging to {log_path}")

    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0
        num_items = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        
        for x_0 in pbar:
            if isinstance(x_0, list): x_0 = x_0[0]
            x_0 = x_0.to(device)
            
            # (1) t 샘플링
            t = torch.rand(x_0.shape[0], 1).to(device) * (sde.T - 1e-3) + 1e-3
            
            # (2) Perturbation
            mean, std = sde.marginal_prob(x_0, t)
            z = torch.randn_like(x_0) # 정답 노이즈
            x_t = mean + std * z
            
            # (3) 모델 예측: ★★★ 이제 Noise를 예측합니다 ★★★
            noise_pred = model(x_t, t)
            
            # (4) Loss 계산: ★★★ 단순 MSE (Simple Diffusion) ★★★
            # 예측 노이즈와 실제 노이즈(z)의 차이만 줄이면 됨
            sum_dims = list(range(1, x_0.dim()))
            loss = torch.mean(torch.sum((noise_pred - z)**2, dim=sum_dims))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_items += 1
            pbar.set_postfix(loss=loss.item())

        # 로그 저장
        avg_loss = total_loss / num_items
        elapsed = time.time() - start_time
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{avg_loss:.4f}", f"{elapsed:.1f}"])
            
    # 모델 저장
    save_name = f"{args.out_dir}/ckpt_{args.dataset}_{args.sde_type}_{args.model_type}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"✅ Training Finished! Saved to {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="swiss_roll", choices=["swiss_roll", "mnist"])
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "unet"])
    parser.add_argument("--sde_type", type=str, default="vp", choices=["vp", "ve"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="outputs")
    
    args = parser.parse_args()
    
    if args.dataset == 'mnist' and args.model_type == 'mlp':
        args.model_type = 'unet'
        args.batch_size = 128
        
    train(args)