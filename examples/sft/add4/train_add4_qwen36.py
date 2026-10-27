"""Qwen3.6 MoE SFT entry, sharing Add data/validation without editing dense SFT."""
import os


def install_moe_sft(sft):
    from add4_qwen_models import validate_student_model
    base = sft.SFTTrainer

    class Qwen36SFTTrainer(base):
        def _build_ckpt_handler(self):
            super()._build_ckpt_handler()
            if self.config.trainer.save_freq == -1:
                # The shared trainer also saves unconditionally on its final step.
                self.ckpt_handler.save_checkpoint = self._skip_checkpoint
            elif os.environ.get('ADD4_KEEP_EPOCH_CHECKPOINTS') == '1':
                from add4_checkpoint_retention import install_epoch_retention
                required = {'model', 'optimizer', 'extra'}
                if not required.issubset(set(self.config.checkpoint.save_contents)):
                    raise ValueError('Epoch retention requires full model/optimizer/extra checkpoints')
                install_epoch_retention(self.ckpt_handler, self.steps_per_epoch,
                                        int(os.environ['ADD4_KEEP_LATEST_CHECKPOINTS']))

        @staticmethod
        def _skip_checkpoint(*args, **kwargs):
            return None

        def _build_config(self):
            super()._build_config()
            validate_student_model(self.model_config.hf_config, 'qwen3_5_moe')
            if self.config.engine.strategy != 'fsdp2':
                raise ValueError('The Qwen3.6 Add SFT entry requires FSDP2')
            if self.config.model.lora_rank or self.config.model.lora_adapter_path:
                raise ValueError('This Qwen3.6 SFT entry is for full-parameter training')

    sft.SFTTrainer = Qwen36SFTTrainer


def main():
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    import verl.utils.fsdp_utils as fsdp_utils
    import verl.trainer.sft_trainer as sft
    if layers := os.environ.get('ADD4_CPU_OFFLOAD_LAYERS'):
        from add4_partial_offload import install_partial_offload
        install_partial_offload([int(index) for index in layers.split(',')])
    from add4_qwen_models import root_owned_qwen_head
    from train_add4 import independent_validation_batch, validation_before_training
    fsdp_utils._select_fsdp2_wrap_targets = root_owned_qwen_head(fsdp_utils._select_fsdp2_wrap_targets, 'qwen3_5_moe')
    sft.SFTTrainer._build_dataloader = independent_validation_batch(sft.SFTTrainer._build_dataloader)
    sft.SFTTrainer.fit = validation_before_training(sft.SFTTrainer.fit)
    install_moe_sft(sft)
    sft.main()


if __name__ == '__main__':
    main()
