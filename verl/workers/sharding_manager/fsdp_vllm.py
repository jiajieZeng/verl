# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
from typing import Union
import os
import time
from collections import OrderedDict
from typing import List

import torch
from peft import PeftModel
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from vllm import AsyncLLMEngine

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class FSDPVLLMShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = 'dummy_hf',
        layered_summon: bool = True
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer intialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine

        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        else:
            # vLLM > v0.6.3
            try:
                self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None
            except:
                self.model_runner = self.inference_engine.engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None
            
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = 'dummy' not in load_format
        if is_version_ge(pkg='vllm', minver='0.7.3'):
            VLLMHijack.hijack()
        
        # Multi-LoRA support: mapping from adapter_name to lora_int_id
        # This allows different agents to use different LoRA adapters during inference
        self.adapter_lora_ids = {}  # e.g., {"agent_lora_0": 1001, "agent_lora_1": 1002}

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        def __collect_lora_params(adapter_name: str = None)->OrderedDict:
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            
            Args:
                adapter_name: If specified, collect params for this specific adapter.
                              Otherwise collect params for the currently active adapter.
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            peft_model = self.module._fsdp_wrapped_module
            
            # If adapter_name specified, switch to it first
            original_adapter = None
            if adapter_name is not None and hasattr(peft_model, 'active_adapter'):
                original_adapter = peft_model.active_adapter
                if hasattr(peft_model, 'set_adapter'):
                    peft_model.set_adapter(adapter_name)
            
            try:
                if fsdp_version(self.module) > 0:
                    if self.layered_summon:
                        if not self.base_sync_done:
                            raise ValueError("To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let rollout.load_format=safetensors")
                        lora_params = layered_summon_lora_params(self.module, adapter_name=adapter_name)
                    else:
                        with FSDP.summon_full_params(self.module, writeback=False):
                            if self.base_sync_done:
                                lora_params = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
                                lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu() 
                                            for name, param in lora_params.items()}
                            else:
                                model = peft_model.base_model.model
                                orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device else 'cuda'
                                model = model.to('cpu')
                                for name, param in model.state_dict().items():
                                    if any(x in name for x in ['_flat_param', 'lora_']):
                                        continue
                                    name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                                    lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu()
                                model = model.to(orig_dev)
                        torch.cuda.empty_cache()
                else:
                    if self.base_sync_done:
                        lora_params = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
                    else:
                        model = peft_model.base_model.model
                        orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device else 'cuda'
                        model = model.to('cpu')
                        for name, param in model.state_dict().items():
                            if any(x in name for x in ['_flat_param', 'lora_']):
                                continue
                            name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                            lora_params[name] = param.detach().cpu()
                        model = model.to(orig_dev)
            finally:
                # Restore original adapter if we changed it
                if original_adapter is not None and hasattr(peft_model, 'set_adapter'):
                    peft_model.set_adapter(original_adapter)
            
            return lora_params

        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        get_torch_device().empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        peft_config = None
        peft_configs_dict = None  # For multi-LoRA mode
        params = None
        multi_lora_params = None  # For multi-LoRA mode: {adapter_name: params}
        
        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            peft_model = self.module._fsdp_wrapped_module
            all_adapter_names = list(peft_model.peft_config.keys())
            
            # Check for multi-LoRA mode (more than one adapter, excluding 'default')
            # Multi-LoRA adapters are named like 'agent_lora_0', 'agent_lora_1', etc.
            multi_lora_adapters = [name for name in all_adapter_names if name.startswith('agent_lora_')]
            
            if len(multi_lora_adapters) >= 1:
                # Multi-LoRA mode (includes single agent_lora_* adapter):
                # collect params for each adapter separately
                logger.info(f"[Multi-LoRA] Detected {len(multi_lora_adapters)} LoRA adapters: {multi_lora_adapters}")
                multi_lora_params = {}
                peft_configs_dict = {}
                for adapter_name in multi_lora_adapters:
                    adapter_params = __collect_lora_params(adapter_name=adapter_name)
                    multi_lora_params[adapter_name] = adapter_params
                    peft_configs_dict[adapter_name] = peft_model.peft_config.get(adapter_name)
                    logger.info(f"[Multi-LoRA] Collected {len(adapter_params)} params for adapter '{adapter_name}'")
            elif 'default' in all_adapter_names:
                # Single LoRA mode with 'default' adapter
                peft_config = peft_model.peft_config.get('default')
                params = __collect_lora_params(adapter_name='default')
                logger.info(f"[Single-LoRA] Using 'default' adapter")
            else:
                # Fallback: use whatever adapter is active
                active_adapter = getattr(peft_model, 'active_adapter', all_adapter_names[0] if all_adapter_names else None)
                if active_adapter and active_adapter in all_adapter_names:
                    peft_config = peft_model.peft_config.get(active_adapter)
                    params = __collect_lora_params(adapter_name=active_adapter)
                    logger.info(f"[Single-LoRA] Using active adapter: '{active_adapter}'")
                else:
                    # No valid adapter found, collect all params
                    logger.warning(f"[LoRA] No valid adapter found in {all_adapter_names}, collecting all params")
                    params = __collect_lora_params()
        else:
            params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            if params is not None:
                self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            if params is not None:
                del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            if multi_lora_params is not None:
                # Multi-LoRA mode: load each adapter separately
                self.update_multi_lora_params(multi_lora_params, peft_configs_dict)
                del multi_lora_params
            else:
                # Single LoRA or full model mode
                self.update_params(params, peft_config=peft_config)
                del params
            
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            get_torch_device().empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        model = self.model_runner.model
        if peft_config:
            if self.base_sync_done:
                lora_int_id=int(time.time_ns() % 0x7FFFFFFF)
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                # Handle different inference engine types:
                # - LLM (standard mode): has llm_engine attribute
                # - WorkerWrapperBase (SPMD mode): has worker attribute
                if hasattr(self.inference_engine, 'llm_engine'):
                    self.inference_engine.llm_engine.add_lora(lora_reqest)
                elif hasattr(self.inference_engine, 'worker'):
                    self.inference_engine.worker.add_lora(lora_reqest)
                else:
                    raise AttributeError(
                        f"Cannot find add_lora method. inference_engine type: {type(self.inference_engine)}. "
                        "Expected LLM (with llm_engine) or WorkerWrapperBase (with worker)."
                    )
                # Store the lora_int_id for later use
                self.adapter_lora_ids['default'] = lora_int_id
                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}, lora_int_id: {lora_int_id}")
                return
            else:
                def replace_lora_wrapper(k):
                    stacked_params = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
                        return k.replace(".weight", ".base_layer.weight")
                    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
                        return k.replace(".bias", ".base_layer.bias")
                    return k
                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        # Load weights for non-LoRA mode or when base model not yet synced
        patch_vllm_moe_model_weight_loader(model)
        device = get_torch_device().current_device()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in updated_params.items()))

        self.base_sync_done = True
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")
    
    def update_multi_lora_params(self, multi_lora_params: dict, peft_configs_dict: dict):
        """
        Load multiple LoRA adapters into vLLM for multi-agent training.
        
        Args:
            multi_lora_params: Dict mapping adapter_name to params, e.g., 
                              {"agent_lora_0": {...params...}, "agent_lora_1": {...params...}}
            peft_configs_dict: Dict mapping adapter_name to peft_config
        """
        if not self.base_sync_done:
            raise ValueError("Multi-LoRA mode requires base model to be preloaded in vLLM. "
                           "Please set rollout.load_format=safetensors")
        
        base_lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
        
        for idx, (adapter_name, params) in enumerate(multi_lora_params.items()):
            peft_config = peft_configs_dict.get(adapter_name)
            if peft_config is None:
                logger.warning(f"[Multi-LoRA] No peft_config found for adapter '{adapter_name}', skipping")
                continue
            
            # Generate unique lora_int_id for each adapter
            # Use base_id + idx to ensure uniqueness
            lora_int_id = base_lora_int_id + idx
            
            lora_request = TensorLoRARequest(
                lora_name=adapter_name,  # Use adapter_name as the LoRA name
                lora_int_id=lora_int_id,
                lora_path=f"multi_lora_{adapter_name}",
                peft_config=asdict(peft_config),
                lora_tensors=params,
            )
            
            # Add LoRA to vLLM engine
            if hasattr(self.inference_engine, 'llm_engine'):
                self.inference_engine.llm_engine.add_lora(lora_request)
            elif hasattr(self.inference_engine, 'worker'):
                self.inference_engine.worker.add_lora(lora_request)
            else:
                raise AttributeError(
                    f"Cannot find add_lora method. inference_engine type: {type(self.inference_engine)}. "
                    "Expected LLM (with llm_engine) or WorkerWrapperBase (with worker)."
                )
            
            # Store the mapping for later use during inference
            self.adapter_lora_ids[adapter_name] = lora_int_id
            logger.info(f"[Multi-LoRA] Loaded adapter '{adapter_name}' with lora_int_id={lora_int_id}, params_count={len(params)}")
        
        logger.info(f"[Multi-LoRA] Loaded {len(multi_lora_params)} LoRA adapters: {list(self.adapter_lora_ids.keys())}")
