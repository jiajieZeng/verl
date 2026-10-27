"""Opt-in Add OPD entry. The original train_add4.py remains unchanged."""
import os


def main(expected_student_type=None, expected_loss_type=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--opd-config', required=True)
    args, remaining = parser.parse_known_args()
    from omegaconf import OmegaConf
    profile = OmegaConf.load(args.opd_config)
    if expected_student_type is not None and profile.student_model_type != expected_student_type:
        raise ValueError('This entry requires its specified student architecture')
    if expected_loss_type is not None and profile.loss_type != expected_loss_type:
        raise ValueError('This entry requires full-vocabulary reverse KL')
    sys.argv = [sys.argv[0], *remaining]
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    import verl.utils.fsdp_utils as fsdp_utils
    import verl.trainer.sft_trainer as sft
    from train_add4 import independent_validation_batch
    from add4_qwen_models import root_owned_qwen_head
    from add4_opd_trainer import install_opd
    fsdp_utils._select_fsdp2_wrap_targets = root_owned_qwen_head(fsdp_utils._select_fsdp2_wrap_targets)
    sft.SFTTrainer._build_dataloader = independent_validation_batch(sft.SFTTrainer._build_dataloader)
    install_opd(sft, args.opd_config)
    sft.main()


if __name__ == '__main__':
    main()
