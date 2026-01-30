import argparse
from config.config import get_config
from step1_prepare import run_prepare
from step2_run import run_inference
from step3_eval import run_eval


def main():
    parser = argparse.ArgumentParser(description="MASR Framework")
    parser.add_argument('--dataset', type=str, required=True, choices=['wn18rr', 'fb15k237'])
    parser.add_argument('--mode', type=str, default='openai_suite', choices=['openai_suite', 'qwen_suite'])
    parser.add_argument('--step', type=int, default=2, choices=[1, 2, 3], help="1:Prepare, 2:Inference, 3:Eval")

    # 消融实验开关参数
    parser.add_argument('--ablation', type=str, default='full_model',
                        choices=['full_model', 'no_visual', 'no_paths', 'no_neighbors', 'no_cot'])

    args = parser.parse_args()

    # 处理消融覆盖逻辑
    overrides = None
    if args.ablation == 'no_visual':
        overrides = {"use_visual": False}
    elif args.ablation == 'no_paths':
        overrides = {"use_paths": False}
    elif args.ablation == 'no_neighbors':
        overrides = {"use_neighbors": False}
    elif args.ablation == 'no_cot':
        overrides = {"use_cot": False}

    # 获取配置
    cfg = get_config(args.dataset, args.mode, ablation_task=args.ablation, overrides=overrides)

    if args.step == 1:
        run_prepare(args.dataset, args.mode)
    elif args.step == 2:
        run_inference(cfg)
    elif args.step == 3:
        run_eval(cfg)


if __name__ == "__main__":
    main()