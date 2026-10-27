"""Opt-in OPD adapter for the existing VERL SFT engine and checkpoint lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import re
import uuid

from add4_opd_client import SGLangClient
from add4_opd_core import prompt_from_sft_ids, restrict_fused_head, validate_vocabulary
from add4_opd_preflight import check_services, service_identity
from add4_qwen_models import SUPPORTED_STUDENT_TYPES, validate_student_model
from add4_opd_monitor import phase, observed


def validate_config(config):
    opd = config.opd
    if config.engine.strategy != 'fsdp2' or config.engine.ulysses_sequence_parallel_size != 1:
        raise ValueError('Add OPD currently requires FSDP2 with sequence parallel size 1')
    if config.data.pad_mode != 'no_padding' or not config.model.use_remove_padding:
        raise ValueError('Add OPD requires the existing packed no_padding SFT path')
    if not config.model.use_fused_kernels or config.model.fused_kernel_options.impl_backend != 'torch':
        raise ValueError('Add OPD requires the torch fused-head backend for valid-vocabulary normalization')
    if config.model.lora_rank or config.model.lora_adapter_path:
        raise ValueError('This Add OPD adapter supports full-parameter training')
    if opd.student_model_type not in SUPPORTED_STUDENT_TYPES or opd.export_shard_bytes <= 0:
        raise ValueError('Unsupported student architecture or invalid export shard budget')
    if opd.student_url.rstrip('/') == opd.teacher_url.rstrip('/'):
        raise ValueError('Student rollout and frozen teacher must be different services')
    if not opd.get('teacher_weight_version'):
        raise ValueError('Pin the frozen teacher_weight_version before starting OPD')
    if opd.temperature <= 0 or opd.max_new_tokens <= 0 or opd.max_concurrency <= 0:
        raise ValueError('Invalid OPD sampling configuration')
    if opd.loss_type == 'full_vocab_reverse_kl':
        if opd.tail_bucket:
            raise ValueError('Full-vocabulary reverse KL does not use a tail bucket')
        if opd.teacher_top_k is not None or not 0 < opd.teacher_score_chunk_size <= 32:
            raise ValueError('Full-vocabulary OPD requires teacher_top_k=null and score chunks of 1..32')
        if config.trainer.nnodes != 1 or opd.max_concurrency > 4:
            raise ValueError('Full-vocabulary score files require single-node training and at most 4 requests')
        if not Path(opd.full_score_dir).resolve().is_relative_to('/data/zengjiajie'):
            raise ValueError('Full score cache must live under /data/zengjiajie')
    elif opd.loss_type != 'teacher_topk_reverse_kl' or not 0 < opd.teacher_top_k <= opd.valid_vocab_size:
        raise ValueError('Expected top-k or full-vocabulary reverse KL')
    elif not opd.tail_bucket:
        raise ValueError('Top-k reverse KL requires tail_bucket=true; the partial sum can leak probability mass')
    if opd.topk_renormalize or opd.distillation_temperature != 1.0:
        raise ValueError('OPD uses global-V raw probabilities without subset softmax, at temperature 1')
    if opd.entropy_coef < 0 or opd.head_chunk_size <= 0 or config.engine.get('pad_to_length', False):
        raise ValueError('Invalid entropy/head chunk configuration or unsupported static padding')
    if opd.context_length != config.data.max_length:
        raise ValueError('OPD and dataset context limits must match')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', opd.run_id):
        raise ValueError('Set a unique single-component OPD run_id')
    for name in ('sync_dir', 'audit_dir'):
        path = Path(opd[name]).resolve()
        if not path.is_relative_to('/data/zengjiajie'):
            raise ValueError(f'opd.{name} must be under /data/zengjiajie')


def opd_or_sft_loss(config, model_output, data, dp_group=None):
    from verl.workers.utils.losses import sft_loss
    if 'opd_teacher_ids' not in data and 'opd_full_rows' not in data:
        # Existing fixed-label validation retains its original SFT objective.
        return sft_loss(config, model_output, data, dp_group)
    import torch
    mask = torch.roll(data['loss_mask'].values(), -1).bool()
    values = model_output['opd_losses'].values()[mask]
    if not values.numel() or not torch.isfinite(values).all() or data['batch_num_tokens'] <= 0:
        raise ValueError('Empty or nonfinite OPD loss')
    scale = data['dp_size'] / data['batch_num_tokens']
    loss = values.sum() * scale
    metrics = {f'opd/{name.removeprefix("opd_")}': (tensor.values()[mask].detach().sum() * scale).item()
               for name, tensor in model_output.items() if name != 'opd_losses'}
    if 'opd_teacher_ids' in data:
        from add4_opd_topk import TEACHER_TAIL_MASS_FLOOR
        teacher = data['opd_teacher_logp'].values()[data['loss_mask'].values().bool()].detach().double()
        if teacher.shape[-1] < (config.opd.valid_vocab_size if config is not None else float('inf')):
            tail = -teacher.logsumexp(-1).expm1()
            metrics['opd/teacher_tail_floor_rate'] = ((tail < TEACHER_TAIL_MASS_FLOOR).sum() * scale).item()
            metrics['opd/teacher_tail_mass'] = (tail.clamp_min(TEACHER_TAIL_MASS_FLOOR).sum() * scale).item()
            if 'opd_student_mass' in model_output:
                metrics['opd/student_tail_mass'] = ((1 - model_output['opd_student_mass'].values()[mask]).sum() * scale).item()
    return loss, metrics


class OPDBatchAdapter:
    def __init__(self, trainer):
        import torch.distributed as dist
        from omegaconf import OmegaConf
        self.trainer = trainer
        self.config = OmegaConf.to_container(trainer.config.opd, resolve=True)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.step = trainer.resume_global_step
        self.student = SGLangClient(self.config['student_url'], self.config['http_timeout'])
        self.teacher = SGLangClient(self.config['teacher_url'], self.config['http_timeout'])
        for role, client in [('student', self.student), ('teacher', self.teacher)]:
            client.monitor, client.role = getattr(trainer, '_opd_monitor', None), role
        # Serialize in the rollout environment; the SFT image need not install SGLang.
        self.processor = Path(self.config['student_logit_processor_file']).read_text().strip()
        serialized = json.loads(self.processor)
        if set(serialized) != {'callable'} or not bytes.fromhex(serialized['callable']):
            raise ValueError('Invalid student logit processor artifact')
        self.attempt = self.rank_zero(lambda: uuid.uuid4().hex)
        def acquire_lock():
            import fcntl
            root = Path(self.config['audit_dir']) / self.config['run_id']
            root.mkdir(parents=True, exist_ok=True)
            self.run_lock = (root / 'run.lock').open('a')
            fcntl.flock(self.run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.rank_zero(acquire_lock)
        tokenizer = trainer.model_config.tokenizer
        validate_vocabulary(tokenizer, self.config['valid_vocab_size'], self.config['output_vocab_size'])
        self.eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if not 0 <= self.eos < self.config['valid_vocab_size']:
            raise ValueError('Missing Add EOS token')
        def preflight():
            report = check_services(self.student, self.teacher, tokenizer, self.config)
            folder = Path(self.config['audit_dir']) / self.config['run_id'] / self.attempt
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'preflight.json').write_text(json.dumps(report, indent=2))
            return report['identities']
        identities = self.rank_zero(preflight)
        self.teacher_identity, self.student_identity = identities['teacher'], identities['student']

    def verify_services(self):
        """Recheck roles before a student update; never POST a weight mutation to teacher."""
        def verify():
            if service_identity(self.teacher) != self.teacher_identity:
                raise ValueError('Frozen teacher identity changed during training')
            student = service_identity(self.student)
            if student != self.student_identity:
                raise ValueError('Student endpoint changed role; refusing weight update')
        with phase(self, 'service_probe'):
            self.rank_zero(verify)
        monitor = getattr(getattr(self, 'trainer', None), '_opd_monitor', None)
        if monitor:
            monitor.count('services/identity_checks_passed')

    def rank_zero(self, fn):
        """Propagate an HTTP/IO failure before other ranks enter another collective."""
        import torch.distributed as dist
        result = [None]
        if self.rank == 0:
            try:
                result[0] = {'ok': True, 'value': fn()}
            except Exception as error:
                result[0] = {'ok': False, 'error': type(error).__name__}
        dist.broadcast_object_list(result, src=0)
        if not result[0]['ok']:
            raise RuntimeError(f"OPD rank-zero operation failed ({result[0]['error']}); no optimizer update")
        return result[0]['value']

    def sync_weights(self):
        """Stream a bounded, two-slot HF export; no growing per-step snapshots."""
        from add4_opd_export import StudentShardWriter
        trainer = self.trainer
        folder = Path(self.config['sync_dir']) / self.config['run_id'] / f'slot-{self.step % 2}'
        self.rank_zero(lambda: folder.mkdir(parents=True, exist_ok=True))
        writer = StudentShardWriter(folder, self.config['export_shard_bytes']) if self.rank == 0 else None
        error = None
        parameters, lora = trainer.engine.get_per_tensor_param()
        if lora is not None:
            raise ValueError('Unexpected LoRA weights in full-parameter OPD export')
        for name, tensor in parameters:
            # All ranks consume every FSDP all-gather, even after rank-zero IO fails.
            if self.rank == 0 and error is None:
                try:
                    writer.add(name, tensor)
                except Exception as exc:
                    error = type(exc).__name__
        def finish():
            if error:
                raise RuntimeError(f'OPD weight export failed: {error}')
            writer.flush()
            trainer.model_config.hf_config.save_pretrained(folder)
            trainer.model_config.tokenizer.save_pretrained(folder)
            index_path = folder / 'model.safetensors.index.json'
            temporary = index_path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'metadata': {'total_size': writer.total_bytes}, 'weight_map': writer.mapping}))
            temporary.replace(index_path)
            version = f"{self.config['run_id']}:{self.step}"
            monitor = getattr(trainer, '_opd_monitor', None)
            if monitor:
                monitor.count('weights/export_bytes', writer.total_bytes)
            with phase(self, 'student_reload'):
                result = self.student.request('/update_weights_from_disk', {
                    'model_path': str(folder), 'load_format': 'safetensors',
                    'flush_cache': True, 'weight_version': version})
            if result.get('success') is not True:
                raise RuntimeError('Student weight update failed')
            return version
        version = self.rank_zero(finish)
        self.student_identity = {'model_type': self.config['student_model_type'], 'served_model_name': str(folder),
                                 'model_path': str(folder), 'weight_version': version}
        return version

    def prepare(self, data, sequence_lengths):
        import torch
        import torch.distributed as dist
        import verl.utils.tensordict_utils as tu
        from tensordict import NonTensorData
        trainer = self.trainer
        self.step += 1
        self.verify_services()
        with phase(self, 'weight_sync'):
            version = self.sync_weights()
        try:
            prompts = [prompt_from_sft_ids(ids.tolist(), mask.tolist())
                       for ids, mask in zip(data['input_ids'].unbind(), data['loss_mask'].unbind(), strict=True)]
            local = {'ok': True, 'prompts': prompts}
        except Exception as error:
            local = {'ok': False, 'error': type(error).__name__}
        gathered = [None] * self.world
        dist.all_gather_object(gathered, local)
        if not all(item['ok'] for item in gathered):
            raise ValueError('At least one rank has an invalid Add prompt span; no optimizer update')
        all_prompts = [item['prompts'] for item in gathered]

        def generate_all():
            flat = [prompt for group in all_prompts for prompt in group]
            def one(pair):
                index, prompt = pair
                response = observed(self, 'rollout_service', self.student.sample, prompt, self.config, self.eos, self.processor,
                    seed=self.config['seed'] + self.step * len(flat) + index, expected_weight_version=version)
                teacher = observed(self, 'teacher_service', self.teacher.score_topk, prompt, response, self.config['valid_vocab_size'],
                                                 self.config['output_vocab_size'], self.config['teacher_top_k'],
                                                 expected_weight_version=self.teacher_identity['weight_version'])
                return {'prompt_ids': prompt, 'response_ids': response,
                        'teacher_ids': teacher['ids'], 'teacher_logp': teacher['logprobs']}
            with ThreadPoolExecutor(max_workers=self.config['max_concurrency']) as pool:
                records = list(pool.map(one, enumerate(flat)))
            audit = Path(self.config['audit_dir']) / self.config['run_id'] / self.attempt
            audit.mkdir(parents=True, exist_ok=True)
            with (audit / f'step-{self.step:08d}.json').open('x') as stream:
                json.dump({'step': self.step, 'student_weight_version': version,
                           'temperature': self.config['temperature'], 'teacher_temperature': 1.0,
                           'distillation_temperature': 1.0, 'loss_type': self.config['loss_type'],
                           'teacher_top_k': self.config['teacher_top_k'], 'topk_renormalize': False,
                           'tail_bucket': self.config['tail_bucket'], 'entropy_coef': self.config['entropy_coef'],
                           'valid_vocab_size': self.config['valid_vocab_size'], 'samples': records}, stream)
            return records
        with phase(self, 'generate_score_wall'):
            all_records = self.rank_zero(generate_all)
        monitor = getattr(trainer, '_opd_monitor', None)
        if monitor:
            monitor.records(all_records)
        self.verify_services()
        offset = sum(len(group) for group in all_prompts[:self.rank])
        records = all_records[offset:offset + len(prompts)]
        position_axes = data['position_ids'].shape[1] if data['position_ids'].dim() == 3 else None
        ids, masks, teachers, teacher_ids, positions = [], [], [], [], []
        for record in records:
            prompt, response = record['prompt_ids'], record['response_ids']
            length = len(prompt) + len(response)
            ids.append(torch.tensor(prompt + response, dtype=torch.long))
            masks.append(torch.tensor([0] * len(prompt) + [1] * len(response), dtype=torch.long))
            teachers.append(torch.cat([torch.zeros(len(prompt), self.config['teacher_top_k']),
                                       torch.tensor(record['teacher_logp'], dtype=torch.float32)]))
            teacher_ids.append(torch.cat([torch.zeros(len(prompt), self.config['teacher_top_k'], dtype=torch.long),
                                          torch.tensor(record['teacher_ids'], dtype=torch.long)]))
            position = torch.arange(length, dtype=torch.long)
            positions.append(position.unsqueeze(0).repeat(position_axes, 1) if position_axes else position)
        for name, values in [('input_ids', ids), ('loss_mask', masks), ('position_ids', positions),
                             ('opd_teacher_logp', teachers), ('opd_teacher_ids', teacher_ids)]:
            data[name] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
        tu.assign_non_tensor(data, temperature=1.0, compute_loss=True, return_model_output=False,
                            global_token_num=NonTensorData(sequence_lengths(trainer, data)))
        return data


def install_opd(sft, opd_config_path):
    """Called only by train_add4_opd.py; never imported by train_add4.py."""
    base = sft.SFTTrainer

    class Add4OPDTrainer(base):
        def _build_dataloader(self):
            from add4_opd_monitor_runtime import build_loaders
            build_loaders(self, super()._build_dataloader)

        def fit(self):
            from add4_opd_monitor_runtime import run_monitored
            return run_monitored(self, sft, super().fit)

        def _build_config(self):
            from omegaconf import OmegaConf, open_dict
            with open_dict(self.config):
                self.config.opd = OmegaConf.load(opd_config_path)
            validate_config(self.config)
            super()._build_config()
            validate_student_model(self.model_config.hf_config, self.config.opd.student_model_type)

        def _build_engine(self):
            from add4_opd_engine import install_topk_engine, wrap_training_metrics
            from verl.utils.experimental.torch_functional import FusedLinearForPPO
            if self.config.opd.loss_type == 'full_vocab_reverse_kl':
                from add4_opd_full_engine import install_full_engine
                install_full_engine(self.config.opd)
            else:
                install_topk_engine(self.config.opd)
            FusedLinearForPPO.forward = restrict_fused_head(FusedLinearForPPO.forward,
                self.config.opd.valid_vocab_size, self.config.opd.output_vocab_size)
            super()._build_engine()
            self.training_client.set_loss_fn(partial(opd_or_sft_loss, config=self.config))
            self.training_client._postprocess_output = wrap_training_metrics(
                self.training_client._postprocess_output, self.engine.get_data_parallel_size)

        def _get_batch_seqlens(self, data):
            from add4_opd_monitor_runtime import prepare_monitored
            return prepare_monitored(self, lambda: self._prepare_opd(data))

        def _prepare_opd(self, data):
            if not hasattr(self, '_opd_adapter'):
                if self.config.opd.loss_type == 'full_vocab_reverse_kl':
                    from add4_opd_full_trainer import FullOPDBatchAdapter
                    self._opd_adapter = FullOPDBatchAdapter(self)
                else:
                    self._opd_adapter = OPDBatchAdapter(self)
            self._opd_adapter.prepare(data, base._get_batch_seqlens)
            return base._get_batch_seqlens(self, data)

    sft.SFTTrainer = Add4OPDTrainer
