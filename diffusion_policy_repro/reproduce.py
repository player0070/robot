"""
PushT Diffusion Policy 재현 스크립트

사용법:
  python reproduce.py --mode lowdim              # state-based 학습
  python reproduce.py --mode image               # image-based 학습
  python reproduce.py --mode lowdim --eval       # 학습 완료 후 평가
  python reproduce.py --mode lowdim --epochs 500 # 빠른 테스트
"""

import sys
import argparse
import pathlib

# line-buffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_config(mode: str, epochs: int, device: str, seed: int) -> DictConfig:
    config_map = {
        "lowdim": "train_diffusion_unet_lowdim_workspace",
        "image":  "train_diffusion_unet_hybrid_workspace",
    }
    assert mode in config_map, f"mode는 lowdim 또는 image 여야 함"

    config_name = config_map[mode]
    config_path = str(pathlib.Path(__file__).parent / "diffusion_policy" / "config")

    output_dir = str(pathlib.Path(f"data/outputs/reproduce_pusht_{mode}").resolve())

    with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=[
                f"training.device={device}",
                f"training.seed={seed}",
                f"training.num_epochs={epochs}",
                f"training.rollout_every={epochs+1}",  # 학습 중 rollout 비활성화
                f"training.checkpoint_every=100",
                f"logging.mode=offline",
            ]
        )
    OmegaConf.resolve(cfg)
    return cfg, output_dir


def train(mode: str, epochs: int, device: str, seed: int):
    cfg, output_dir = get_config(mode, epochs, device, seed)

    print("=" * 50)
    print(f"  Task   : PushT {mode}")
    print(f"  Epochs : {epochs}")
    print(f"  Device : {device}")
    print(f"  Seed   : {seed}")
    print(f"  Output : {output_dir}")
    print("=" * 50)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.run()


def evaluate(mode: str, device: str):
    import torch
    import dill
    import json
    import wandb
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    run_dir = pathlib.Path(f"data/outputs/reproduce_pusht_{mode}")
    ckpt_dir = run_dir / "checkpoints"

    # latest.ckpt 우선, 없으면 epoch 번호 기준 가장 마지막 체크포인트
    latest = ckpt_dir / "latest.ckpt"
    if latest.exists():
        ckpt_path = latest
    else:
        ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"),
                       key=lambda p: int(p.stem.split("=")[1].split("-")[0]))
        assert ckpts, f"체크포인트 없음: {ckpt_dir}"
        ckpt_path = ckpts[-1]
    output_dir = run_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Eval] 체크포인트: {ckpt_path}")

    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))
    log = env_runner.run(policy)

    # 결과 출력
    print("\n[결과]")
    for k, v in log.items():
        if not isinstance(v, wandb.sdk.data_types.video.Video):
            print(f"  {k}: {v}")

    with open(output_dir / "eval_log.json", "w") as f:
        json.dump(
            {k: v._path if isinstance(v, wandb.sdk.data_types.video.Video) else v
             for k, v in log.items()},
            f, indent=2
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   default="lowdim", choices=["lowdim", "image"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3000,
                        help="학습 epoch 수 (빠른 테스트: 500, 논문 재현: 5000)")
    parser.add_argument("--eval",   action="store_true", help="학습 없이 평가만 실행")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.mode, args.device)
    else:
        train(args.mode, args.epochs, args.device, args.seed)