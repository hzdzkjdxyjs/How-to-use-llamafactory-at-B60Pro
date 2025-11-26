# 怎么用xpu的后端通信库进行通信
 - DeepSpeed在底层上来说是进程之间进行通信，说起来也很搞笑啊，我在测试DeepSpeed进行多卡训练的时候，根本没装任何和通信有关的库
 - 所以现在我们希望训练的时候能从底层调用xpu的优化后的通信库来训练

---

## 验证您的系统是否可以支持多卡通信
 - gloo通信，（创建文件 执行脚本）

```bash
import torch.distributed as dist

def main():
    dist.init_process_group("gloo")
    print("Backend =", dist.get_backend())
    print("Rank =", dist.get_rank())

if __name__ == "__main__":
    main()
````

<img width="1037" height="202" alt="image" src="https://github.com/user-attachments/assets/28fd10fc-0e52-4b3b-8889-df8081bc76eb" />

 - oneccl通信，（创建文件 执行脚本）

```bash
import oneccl_bindings_for_pytorch 
import torch.distributed as dist

def main():
    dist.init_process_group("ccl")
    print("Backend =", dist.get_backend(), "Rank =", dist.get_rank())

if __name__ == "__main__":
    main()
````

<img width="1241" height="406" alt="image" src="https://github.com/user-attachments/assets/ff3a9ce4-6c9c-4f5a-bf2b-f17deea172e7" />


## 设置环境变量开始多机训练
```python
export ZE_AFFINITY_MASK=0.1       # 两张 GPU
export CCL_WORKER_COUNT=2
export CCL_LOG_LEVEL=info
export FI_PROVIDER=tcp
export CCL_ZE_IPC_EXCHANGE=sockets
llamafactory-cli train examples/train_lora/qwen3-0.6B_lora_sft.yaml
````

  - 我很奇怪的就是训练过的耗时和用DeepSpeed差不多
<details><summary> 点击展开训练过程 </summary>

```
[INFO|2025-11-26 12:43:38] llamafactory.launcher:143 >> Initializing 2 distributed tasks at: 127.0.0.1:36775
W1126 12:43:39.684000 4016406 site-packages/torch/distributed/run.py:774] 
W1126 12:43:39.684000 4016406 site-packages/torch/distributed/run.py:774] *****************************************
W1126 12:43:39.684000 4016406 site-packages/torch/distributed/run.py:774] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1126 12:43:39.684000 4016406 site-packages/torch/distributed/run.py:774] *****************************************
/root/anaconda3/envs/b60/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/root/anaconda3/envs/b60/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
[W1126 12:43:44.675193702 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
[W1126 12:43:44.675193617 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
My guessed rank = 1
My guessed rank = 0
[INFO|2025-11-26 12:43:45] llamafactory.hparams.parser:143 >> Set `ddp_find_unused_parameters` to False in DDP training since LoRA is enabled.
[INFO|2025-11-26 12:43:45] llamafactory.hparams.parser:468 >> Process rank: 0, world size: 2, device: xpu:0, distributed training: True, compute dtype: torch.bfloat16
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,157 >> loading file vocab.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,157 >> loading file merges.txt
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,157 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,157 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,157 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,157 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,157 >> loading file chat_template.jinja
[INFO|2025-11-26 12:43:45] llamafactory.hparams.parser:468 >> Process rank: 1, world size: 2, device: xpu:1, distributed training: True, compute dtype: torch.bfloat16
[INFO|tokenization_utils_base.py:2364] 2025-11-26 12:43:45,432 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|configuration_utils.py:763] 2025-11-26 12:43:45,432 >> loading configuration file /root/models/Qwen3-0.6B/config.json
[INFO|configuration_utils.py:839] 2025-11-26 12:43:45,434 >> Model config Qwen3Config {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,434 >> loading file vocab.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,434 >> loading file merges.txt
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,434 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,434 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,434 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,434 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 12:43:45,434 >> loading file chat_template.jinja
[INFO|tokenization_utils_base.py:2364] 2025-11-26 12:43:45,729 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|2025-11-26 12:43:45] llamafactory.data.loader:143 >> Loading dataset identity.json...
/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
2025:11:26-12:43:45:(4016677) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
2025:11:26-12:43:45:(4016677) |CCL_INFO| process launcher: hydra, local_proc_idx: -1, local_proc_count: -1
2025:11:26-12:43:45:(4016677) |CCL_INFO| initializing level-zero api
2025:11:26-12:43:45:(4016677) |CCL_INFO| initializing level-zero
2025:11:26-12:43:45:(4016677) |CCL_INFO| Total hardware threads: 1280
2025:11:26-12:43:45:(4016677) |CCL_INFO| auto tune with port counts enabled
2025:11:26-12:43:45:(4016677) |CCL_INFO| ze fabric ports: 0 were able to be detected
2025:11:26-12:43:45:(4016677) |CCL_INFO| initialized level-zero
2025:11:26-12:43:45:(4016677) |CCL_INFO| could not initialize umf api
2025:11:26-12:43:45:(4016677) |CCL_INFO| OS info: { Linux b60 6.14.0-1006-intel #6-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  1 00:03:01 UTC 2025 x86_64 }
Converting format of dataset (num_proc=16): 182 examples [00:00, 218.98 examples/s]                                                               
[INFO|2025-11-26 12:43:47] llamafactory.data.loader:143 >> Loading dataset alpaca_en_demo.json...
Converting format of dataset (num_proc=16): 1998 examples [00:00, 2413.49 examples/s]                                                             
/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
2025:11:26-12:43:47:(4016676) |CCL_WARN| value of CCL_LOG_LEVEL changed to be info (default:warn)
2025:11:26-12:43:47:(4016676) |CCL_WARN| value of CCL_WORKER_COUNT changed to be 4 (default:1)
2025:11:26-12:43:47:(4016676) |CCL_WARN| value of CCL_ATL_TRANSPORT changed to be ofi (default:mpi)
2025:11:26-12:43:47:(4016676) |CCL_WARN| value of CCL_ZE_IPC_EXCHANGE changed to be sockets (default:pidfd)
2025:11:26-12:43:47:(4016676) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
2025:11:26-12:43:47:(4016676) |CCL_INFO| process launcher: hydra, local_proc_idx: -1, local_proc_count: -1
2025:11:26-12:43:47:(4016676) |CCL_INFO| initializing level-zero api
2025:11:26-12:43:47:(4016676) |CCL_INFO| initializing level-zero
2025:11:26-12:43:47:(4016676) |CCL_INFO| Total hardware threads: 1280
2025:11:26-12:43:47:(4016676) |CCL_INFO| auto tune with port counts enabled
2025:11:26-12:43:47:(4016676) |CCL_INFO| ze fabric ports: 0 were able to be detected
2025:11:26-12:43:47:(4016676) |CCL_INFO| initialized level-zero
2025:11:26-12:43:47:(4016676) |CCL_INFO| could not initialize umf api
2025:11:26-12:43:47:(4016676) |CCL_INFO| OS info: { Linux b60 6.14.0-1006-intel #6-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  1 00:03:01 UTC 2025 x86_64 }
2025:11:26-12:43:48:(4016677) |CCL_INFO| fi_version: 1.10
2025:11:26-12:43:48:(4016676) |CCL_INFO| fi_version: 1.10
2025:11:26-12:43:48:(4016677) |CCL_INFO| coord: global [ idx 1, cnt 2 ], local [ idx 1, cnt 2 ]
2025:11:26-12:43:48:(4016676) |CCL_INFO| libfabric version: 1.21.0-impi
2025:11:26-12:43:48:(4016676) |CCL_INFO| coord: global [ idx 0, cnt 2 ], local [ idx 0, cnt 2 ]
2025:11:26-12:43:48:(4016676) |CCL_INFO| found 1 nic(s) according to all filters
2025:11:26-12:43:48:(4016676) |CCL_INFO| provider: tcp
2025:11:26-12:43:48:(4016676) |CCL_INFO|   nic: { name tcp:br-1894cb0b4cfd, state unknown, speed 1.25 GB/s }
2025:11:26-12:43:48:(4016676) |CCL_INFO|   mr_mode: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO|   threading: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO|   tx_ctx_cnt: 256
2025:11:26-12:43:48:(4016676) |CCL_INFO|   max_ep_tx_ctx: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO|   max_msg_size: 18446744073709551615
2025:11:26-12:43:48:(4016677) |CCL_INFO| found 1 nic(s) according to all filters
2025:11:26-12:43:48:(4016676) |CCL_INFO| tcp tag_bits: 64, max_tag: 18446744073709551615, mem_tag_format: 12297829382473034410
2025:11:26-12:43:48:(4016676) |CCL_INFO| ep_idx: 0, active_prov_idxs: 0 
2025:11:26-12:43:48:(4016676) |CCL_INFO| ep_idx: 1, active_prov_idxs: 0 
2025:11:26-12:43:48:(4016676) |CCL_INFO| ep_idx: 2, active_prov_idxs: 0 
2025:11:26-12:43:48:(4016676) |CCL_INFO| ep_idx: 3, active_prov_idxs: 0 
2025:11:26-12:43:48:(4016676) |CCL_INFO| atl-ofi:
{
  prov_count: 1
  nw_prov_count: 1
  nw_prov_first_idx: 0
  mnic_type: none
  mnic_include_names: <empty>
  mnic_exclude_names: <empty>
  mnic_count: 1
  mnic_offset: none
  max_retry_count: 10000
  progress_mode: 1
  hmem: 0
}
2025:11:26-12:43:48:(4016676) |CCL_INFO| atl attrs:
{
  in: { shm: 0, hmem: 0, sync_coll: 0, extra_ep: 0, ep_count: 4, mnic_type: none, mnic_count: 4, mnic_offset: none }
  out: { shm: 0, hmem: 0, mnic_type: none, mnic_count: 1, tag_bits: 64, max_tag: 18446744073709551615 }
}
2025:11:26-12:43:48:(4016677) |CCL_INFO| tcp tag_bits: 64, max_tag: 18446744073709551615, mem_tag_format: 12297829382473034410
2025:11:26-12:43:48:(4016677) |CCL_INFO| start workers for local process [1:2]
2025:11:26-12:43:48:(4016677) |CCL_INFO| local_proc_idx: 1, local_proc_count: 2 are set by ATL transport2025:11:26-12:43:48:(4016676) |CCL_INFO| start workers for local process [0:2]

2025:11:26-12:43:48:(4016676) |CCL_INFO| local_proc_idx: 0, local_proc_count: 2 are set by ATL transport
2025:11:26-12:43:48:(4016676) |CCL_INFO| library version: Gold-2021.15.2 2025-05-12T 08:06:42Z (HEAD/ae8376ff)
2025:11:26-12:43:48:(4016677) |CCL_INFO| local process [1:2]: worker: 0, cpu: 59, numa: 12025:11:26-12:43:48:(4016676) |CCL_INFO| specification version: 1.0

2025:11:26-12:43:48:(4016677) |CCL_INFO| local process [1:2]: worker: 1, cpu: 58, numa: 12025:11:26-12:43:48:(4016676) |CCL_INFO| compute backend: DPCPP

2025:11:26-12:43:48:(4016677) |CCL_INFO| local process [1:2]: worker: 2, cpu: 57, numa: 12025:11:26-12:43:48:(4016676) |CCL_INFO| build mode: release

2025:11:26-12:43:48:(4016677) |CCL_INFO| local process [1:2]: worker: 3, cpu: 56, numa: 12025:11:26-12:43:48:(4016676) |CCL_INFO| C compiler: IntelLLVM 2025.1.0

2025:11:26-12:43:48:(4016676) |CCL_INFO| C++ compiler: IntelLLVM 2025.1.0
2025:11:26-12:43:48:(4016676) |CCL_INFO| hwloc initialized: 1
{
  membind_thread_supported: 1
  numa: {os_idx: 0, memory: 128615 MB, cores: 16, cpus: 32, membind: 1}
  numa: {os_idx: 1, memory: 128964 MB, cores: 16, cpus: 32, membind: 1}
}
2025:11:26-12:43:48:(4016676) |CCL_INFO| local process [0:2]: worker: 0, cpu: 63, numa: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| local process [0:2]: worker: 1, cpu: 62, numa: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| local process [0:2]: worker: 2, cpu: 61, numa: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| local process [0:2]: worker: 3, cpu: 60, numa: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_WORKER_COUNT: 4
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_WORKER_OFFLOAD: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_WORKER_WAIT: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_LOG_LEVEL: info
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ABORT_ON_THROW: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_QUEUE_DUMP: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SCHED_DUMP: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SCHED_PROFILE: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ENTRY_MAX_UPDATE_TIME_SEC: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_FRAMEWORK: none
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ATL_TRANSPORT: ofi
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KVS_MODE: pmi
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KVS_CONNECTION_TIMEOUT: 120
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KVS_MPI_ALLGATHER: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KVS_USE_MPI_RANKS: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ATL_SHM: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ATL_RMA: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ATL_HMEM: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ATL_SEND_PROXY: none
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ATL_CACHE: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_MNIC: none
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_MNIC_NAME: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_MNIC_COUNT: 4
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_MNIC_OFFSET: none
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALGO_FALLBACK: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHER: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHERV: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLREDUCE: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLTOALL: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLTOALLV: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_BARRIER: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_BCAST: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_BROADCAST: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_RECV: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_SCATTER: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SEND: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHER_SCALEOUT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHERV_SCALEOUT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLREDUCE_SCALEOUT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLTOALL_SCALEOUT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLTOALLV_SCALEOUT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_SCALEOUT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_SCATTER_SCALEOUT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_UNORDERED_COLL: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_FUSION: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_FUSION_BYTES_THRESHOLD: 16384
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_FUSION_COUNT_THRESHOLD: 256
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_FUSION_CHECK_URGENT: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_FUSION_CYCLE_MS: 0.2
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_PRIORITY: none
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SPIN_COUNT: 1000
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_YIELD: pause
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_MAX_SHORT_SIZE: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_BCAST_PART_COUNT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_CACHE_KEY: match_id
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_CACHE_FLUSH: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_BUFFER_CACHE: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_STRICT_ORDER: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_STAGING_BUFFER: regular
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_OP_SYNC: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_OFI_ENABLE_HOSTNAME_SHARING: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_OFI_INIT_ENABLE_HOSTNAME_SHARING: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_CHUNK_COUNT: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_MIN_CHUNK_SIZE: 65536
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_RS_CHUNK_COUNT: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_RS_MIN_CHUNK_SIZE: 65536
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHERV_TOPO_LARGE_SCALE: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHERV_TOPO_READ: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLTOALLV_TOPO_READ: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_SCATTER_TOPO_READ: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_SCATTER_FALLBACK_ALGO: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHERV_MONOLITHIC_KERNEL: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLTOALLV_MONOLITHIC_KERNEL: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLTOALLV_MONOLITHIC_READ_KERNEL: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLGATHERV_PIPE_CHUNK_COUNT: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLREDUCE_PIPE_CHUNK_COUNT: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_SCATTER_PIPE_CHUNK_COUNT: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_REDUCE_PIPE_CHUNK_COUNT: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLREDUCE_TMP_BUF: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD: 524288
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLREDUCE_MEDIUM_THRESHOLD: 16777216
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLREDUCE_SCALEOUT_THRESHOLD: 4294967296
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLREDUCE_SCALEOUT: auto
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_REDUCE_SCATTER_TMP_BUF: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_REDUCE_SCATTER_SMALL_THRESHOLD: 2097152
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_REDUCE_SCATTER_MEDIUM_THRESHOLD: 67108864
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_REDUCE_SCATTER_SCALEOUT_THRESHOLD: 4294967296
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_REDUCE_SCATTER_SCALEOUT: auto
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLGATHERV_TMP_BUF: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD: 131072
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLGATHERV_MEDIUM_THRESHOLD: 2097152
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD: 1048576
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ENABLE_SYCL_KERNELS: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_CCL_BARRIER: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_KERNEL_SYNC: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_SINGLE_NODE_ALGORITHM: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_AUTO_USE_TMP_BUF: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_COPY_ENGINE: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_KERNEL_COPY: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ESIMD: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_FULL_VECTOR: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_TMP_BUF_SIZE: 402653184
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_SCALEOUT_HOST_BUF_SIZE: 1073741824
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_SCALEOUT_DEVICE_BUF_SIZE: 1073741824
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_KERNELS_LINE_SIZE: 128
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_SCALEOUT_BUF_ALLOC_MODE: hwloc
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_MAX_PIPELINE_CHUNK_SIZE: 33554432
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_PIPELINE_CHUNK_SIZE: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_ENABLE_DIRECT_GPU_RDMA: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_SUB_COMMUICATOR: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLREDUCE_NREDUCE_BUFFERING: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_DTREE_PARTITION_COUNT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLREDUCE_2D_CHUNK_COUNT: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLREDUCE_2D_MIN_CHUNK_SIZE: 65536
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLREDUCE_2D_SWITCH_DIMS: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_CHECK_INPLACE_ALIASING: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ALLTOALL_SCATTER_MAX_OPS: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_BACKEND: native
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_LOCAL_RANK: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_LOCAL_SIZE: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_PROCESS_LAUNCHER: hydra
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_MPI_LIBRARY_PATH: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_OFI_LIBRARY_PATH: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_UMF_ENABLE: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_UMF_LIBRARY_PATH: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_TOPO_ALGO: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_TOPO_COLOR: ze
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_TOPO_P2P_ACCESS: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_TOPO_WA_FABRIC_VERTEX_CONNECTION_CHECK: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_USE_MPI_BCAST_WA: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_USE_ROOT_PRINT_WA: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_PATH: /root/anaconda3/envs/b60/lib/ccl/kernels/
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_DEBUG: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_GROUP_SIZE: 32
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_GROUP_COUNT: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_MEM_ALIGN: 128
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_SYNC: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_1S_LEAD: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_1S_USE_COPY_OPS: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_1S_IPC_WA: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_KERNEL_CLOSE_FD_WA: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_SYCL_OUTPUT_EVENT: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_BARRIER_SYNC: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_DEPS_SYNC: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_USE_HMEM: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_BARRIER: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_BIDIR_ALGO: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_CACHE: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_DEVICE_CACHE_EVICT_SMALLEST: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_DEVICE_CACHE_UPPER_LIMIT: 838860800
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_DEVICE_CACHE_NUM_BLOCKS_IN_CHUNK: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_DEVICE_CACHE_POLICY: chunk
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_PTR_REGISTER_THRESHOLD: 4294967296
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_CACHE_OPEN_IPC_HANDLES: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD: 1000
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD: 1000
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_CACHE_GET_IPC_HANDLES: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_SINGLE_LIST: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_DISABLE_FAMILY_CHECK: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_DISABLE_PORT_CHECK: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_ENABLE_OVERSUBSCRIPTION_FALLBACK: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_ENABLE_OVERSUBSCRIPTION_THROW: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_SERIALIZE: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_COPY_ENGINE: link
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_H2D_COPY_ENGINE: none
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_D2D_COPY_ENGINE: none
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_MAX_COMPUTE_QUEUES: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_MAX_COPY_QUEUES: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_ENABLE_CCS_FALLBACK_FOR_COPY: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_LIST_DUMP: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_QUEUE_INDEX_OFFSET: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_CLOSE_IPC_WA: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_LIBRARY_PATH: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_ENABLE: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_FINI_WA: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_MULTI_WORKERS: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_AUTO_TUNE_PORTS: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_IPC_EXCHANGE: sockets
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_DRM_BDF_SUPPORT: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_PT2PT_READ: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ZE_TYPE2_TUNE_PORTS: undetected
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_DRMFD_DEV_RENDER_DIR_PATH: /dev/dri/by-path/
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_DRMFD_DEV_RENDER_SUFFIX: -render
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_IPC_ALLGATHERV_WA: 1
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_PMIX_LIBRARY_PATH: <not specified>
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ITT_LEVEL: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_DEBUG_TIMESTAMPS_LEVEL: 0
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_BF16: avx512bf
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_FP16: avx512fp16
2025:11:26-12:43:48:(4016676) |CCL_INFO| CCL_ROOT: /root/anaconda3/envs/b60
2025:11:26-12:43:48:(4016676) |CCL_INFO| I_MPI_ROOT: /root/anaconda3/envs/b60
2025:11:26-12:43:48:(4016676) |CCL_INFO| FI_PROVIDER_PATH: /root/anaconda3/envs/b60/lib:/usr/lib/x86_64-linux-gnu/libfabric
2025:11:26-12:43:48:(4016676) |CCL_INFO| FI_PROVIDER: tcp
2025:11:26-12:43:48:(4016676) |CCL_WARN| device_family is unknown, topology discovery could be incorrect, it might result in suboptimal performance
2025:11:26-12:43:48:(4016677) |CCL_WARN| device_family is unknown, topology discovery could be incorrect, it might result in suboptimal performance
2025:11:26-12:43:48:(4016676) |CCL_INFO| no ports detected2025:11:26-12:43:48:(4016677) |CCL_INFO| no ports detected

2025:11:26-12:43:48:(4016676) |CCL_INFO| level zero driver version: 17008127
2025:11:26-12:43:48:(4016677) |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices2025:11:26-12:43:48:(4016676) |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices

2025:11:26-12:43:48:(4016677) |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices2025:11:26-12:43:48:(4016676) |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices

2025:11:26-12:43:48:(4016676) |CCL_INFO| all vertices connected: is_matrix_connected: 0
 fabric connectivity matrix: 
              [R]0x4728bf98              [R]0x480b1b68
[R]0x4728bf98              X              0
[R]0x480b1b68              0              X

2025:11:26-12:43:48:(4016676) |CCL_INFO| ze_rank_info_vec: 
{
  comm_size: 2
    host: { idx: 0, name: b60 }
      rank: { idx: 0, device_uuid: { 134, 128, 17, 226, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0 }, subdev_count: 0, subdev_id: na }
      rank: { idx: 1, device_uuid: { 134, 128, 17, 226, 0, 0, 0, 0, 155, 0, 0, 0, 0, 0, 0, 0 }, subdev_count: 0, subdev_id: na }
}
2025:11:26-12:43:48:(4016676) |CCL_INFO| host: 0, plane 0 contains ranks: 0 
2025:11:26-12:43:48:(4016676) |CCL_INFO| host: 0, plane 1 contains ranks: 1 
2025:11:26-12:43:48:(4016676) |CCL_INFO| rank_info_vec: 
{
  comm_size: 2
    host: { idx: 0, name: b60 }
      rank: { idx: 0, local_proc_idx: 0, uuid: 04016676-57518397-1764132228718450 }
      rank: { idx: 1, local_proc_idx: 1, uuid: 04016677-19674206-1764132228718456 }
}
2025:11:26-12:43:48:(4016676) |CCL_INFO| topo_manager:
{
  comm_size: 2
  single_node: 1
  single_card: 1
  host_rank_counts: 2 
  intra_card_colors: 0 0 
  inter_card_colors: 0 1 
  p2p_access: 1
}
2025:11:26-12:43:48:(4016676) |CCL_INFO| stream: { type: gpu, in_order: 1, device: Intel(R) Graphics [0xe211], device_family: unknown }
2025:11:26-12:43:48:(4016677) |CCL_INFO| stream: { type: gpu, in_order: 1, device: Intel(R) Graphics [0xe211], device_family: unknown }
Running tokenizer on dataset (num_proc=16): 2180 examples [00:02, 441.26 examples/s]                                                              
training example:
input_ids:
[151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198, 9707, 0, 358, 1079, 5867, 606, 38154, 458, 15235, 17847, 7881, 553, 5867, 3094, 3417, 13, 2585, 646, 358, 7789, 498, 3351, 30, 151645, 198]
inputs:
<|im_start|>user
hi<|im_end|>
<|im_start|>assistant
Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?<|im_end|>

label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, 9707, 0, 358, 1079, 5867, 606, 38154, 458, 15235, 17847, 7881, 553, 5867, 3094, 3417, 13, 2585, 646, 358, 7789, 498, 3351, 30, 151645, 198]
labels:
Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?<|im_end|>

[INFO|configuration_utils.py:763] 2025-11-26 12:43:51,357 >> loading configuration file /root/models/Qwen3-0.6B/config.json
[INFO|configuration_utils.py:839] 2025-11-26 12:43:51,358 >> Model config Qwen3Config {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|2025-11-26 12:43:51] llamafactory.model.model_utils.kv_cache:143 >> KV cache is disabled during training.
[WARNING|_logger.py:93] 2025-11-26 12:43:51,485 >> `torch_dtype` is deprecated! Use `dtype` instead!
[INFO|modeling_utils.py:1169] 2025-11-26 12:43:51,486 >> loading weights file /root/models/Qwen3-0.6B/model.safetensors
[INFO|modeling_utils.py:2341] 2025-11-26 12:43:51,486 >> Instantiating Qwen3ForCausalLM model under default dtype torch.bfloat16.
[INFO|configuration_utils.py:986] 2025-11-26 12:43:51,489 >> Generate config GenerationConfig {
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "use_cache": false
}

`torch_dtype` is deprecated! Use `dtype` instead!
[INFO|configuration_utils.py:939] 2025-11-26 12:43:52,686 >> loading configuration file /root/models/Qwen3-0.6B/generation_config.json
[INFO|configuration_utils.py:986] 2025-11-26 12:43:52,686 >> Generate config GenerationConfig {
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95
}

[INFO|dynamic_module_utils.py:423] 2025-11-26 12:43:52,687 >> Could not locate the custom_generate/generate.py inside /root/models/Qwen3-0.6B.
[INFO|2025-11-26 12:43:52] llamafactory.model.model_utils.checkpointing:143 >> Gradient checkpointing enabled.
[INFO|2025-11-26 12:43:52] llamafactory.model.model_utils.attention:143 >> Using torch SDPA for faster training and inference.
[INFO|2025-11-26 12:43:52] llamafactory.model.adapter:143 >> Upcasting trainable params to float32.
[INFO|2025-11-26 12:43:52] llamafactory.model.adapter:143 >> Fine-tuning method: LoRA
[INFO|2025-11-26 12:43:52] llamafactory.model.model_utils.misc:143 >> Found linear modules: q_proj,up_proj,gate_proj,down_proj,v_proj,o_proj,k_proj
[INFO|2025-11-26 12:43:52] llamafactory.model.loader:143 >> trainable params: 5,046,272 || all params: 601,096,192 || trainable%: 0.8395
[WARNING|trainer.py:906] 2025-11-26 12:43:53,007 >> The model is already on multiple devices. Skipping the move to device specified in `args`.
[INFO|trainer.py:749] 2025-11-26 12:43:53,010 >> Using auto half precision backend
[WARNING|2025-11-26 12:43:53] llamafactory.train.callbacks:154 >> Previous trainer log in this folder will be deleted.
[WARNING|trainer.py:982] 2025-11-26 12:43:53,011 >> The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
The model is already on multiple devices. Skipping the move to device specified in `args`.
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
[INFO|trainer.py:2519] 2025-11-26 12:43:54,296 >> ***** Running training *****
[INFO|trainer.py:2520] 2025-11-26 12:43:54,296 >>   Num examples = 1,090
[INFO|trainer.py:2521] 2025-11-26 12:43:54,296 >>   Num Epochs = 3
[INFO|trainer.py:2522] 2025-11-26 12:43:54,296 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2525] 2025-11-26 12:43:54,296 >>   Total train batch size (w. parallel, distributed & accumulation) = 16
[INFO|trainer.py:2526] 2025-11-26 12:43:54,296 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2527] 2025-11-26 12:43:54,296 >>   Total optimization steps = 207
[INFO|trainer.py:2528] 2025-11-26 12:43:54,299 >>   Number of trainable parameters = 5,046,272
  0%|                                                                                                                     | 0/207 [00:00<?, ?it/s]2025:11:26-12:43:55:(4016676) |CCL_INFO| stream: { type: gpu, in_order: 1, device: Intel(R) Graphics [0xe211], device_family: unknown }
2025:11:26-12:43:55:(4016677) |CCL_INFO| stream: { type: gpu, in_order: 1, device: Intel(R) Graphics [0xe211], device_family: unknown }
{'loss': 1.745, 'grad_norm': 1.0916047096252441, 'learning_rate': 4.2857142857142856e-05, 'epoch': 0.15}                                          
{'loss': 1.6616, 'grad_norm': 0.9074978232383728, 'learning_rate': 9.047619047619048e-05, 'epoch': 0.29}                                          
{'loss': 1.5027, 'grad_norm': 0.7538744807243347, 'learning_rate': 9.954424340791196e-05, 'epoch': 0.44}                                          
{'loss': 1.3501, 'grad_norm': 0.6855493783950806, 'learning_rate': 9.770696282000244e-05, 'epoch': 0.59}                                          
{'loss': 1.3937, 'grad_norm': 0.5986014008522034, 'learning_rate': 9.451192254041758e-05, 'epoch': 0.73}                                          
{'loss': 1.4793, 'grad_norm': 0.8447734117507935, 'learning_rate': 9.005005472346924e-05, 'epoch': 0.88}                                          
{'loss': 1.3035, 'grad_norm': 0.7054071426391602, 'learning_rate': 8.444834595378434e-05, 'epoch': 1.01}                                          
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 207/207 [09:16<00:00,  2.69s/it]
[INFO|trainer.py:4309] 2025-11-26 12:53:10,334 >> Saving model checkpoint to saves/Kllama_Qwen3-0.6B
[INFO|configuration_utils.py:763] 2025-11-26 12:53:10,347 >> loading configuration file /root/models/Qwen3-0.6B/config.json
[INFO|configuration_utils.py:839] 2025-11-26 12:53:10,348 >> Model config Qwen3Config {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2421] 2025-11-26 12:53:10,477 >> chat template saved in saves/Kllama_Qwen3-0.6B/chat_template.jinja
[INFO|tokenization_utils_base.py:2590] 2025-11-26 12:53:10,478 >> tokenizer config file saved in saves/Kllama_Qwen3-0.6B/tokenizer_config.json
[INFO|tokenization_utils_base.py:2599] 2025-11-26 12:53:10,478 >> Special tokens file saved in saves/Kllama_Qwen3-0.6B/special_tokens_map.json
***** train metrics *****
  epoch                    =        3.0
  total_flos               =  1351978GF
  train_loss               =     1.3686
  train_runtime            = 0:09:16.03
  train_samples_per_second =      5.881
  train_steps_per_second   =      0.372
Figure saved at: saves/Kllama_Qwen3-0.6B/training_loss.png
[WARNING|2025-11-26 12:53:10] llamafactory.extras.ploting:148 >> No metric eval_loss to plot.
[WARNING|2025-11-26 12:53:10] llamafactory.extras.ploting:148 >> No metric eval_accuracy to plot.
[INFO|modelcard.py:456] 2025-11-26 12:53:10,747 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
2025:11:26-12:53:10:(4016676) |CCL_INFO| finalizing atl-ofi
2025:11:26-12:53:10:(4016676) |CCL_INFO| finalized atl-ofi
2025:11:26-12:53:11:(4016677) |CCL_INFO| finalizing level-zero
2025:11:26-12:53:11:(4016677) |CCL_INFO| finalized level-zero
2025:11:26-12:53:11:(4016676) |CCL_INFO| finalizing level-zero
2025:11:26-12:53:11:(4016676) |CCL_INFO| finalized level-zero                                                                 
 ```
</details

 # 但是如果使用MPI通信框架则
```bash
export CCL_ATL_TRANSPORT=mpi
export CCL_ZE_IPC_EXCHANGE=sockets
export FI_PROVIDER=tcp
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
llamafactory-cli train examples/train_lora/qwen3-0.6B_lora_sft.yaml
````
 - 但是似乎不支持训练
<details><summary> 点击展开训练过程 </summary>

```
[INFO|2025-11-26 13:08:41] llamafactory.launcher:143 >> Initializing 2 distributed tasks at: 127.0.0.1:51481
[INFO|2025-11-26 13:08:42] llamafactory.launcher:143 >> Initializing 2 distributed tasks at: 127.0.0.1:45265
W1126 13:08:43.134000 140616 site-packages/torch/distributed/run.py:774] 
W1126 13:08:43.134000 140616 site-packages/torch/distributed/run.py:774] *****************************************
W1126 13:08:43.134000 140616 site-packages/torch/distributed/run.py:774] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1126 13:08:43.134000 140616 site-packages/torch/distributed/run.py:774] *****************************************
W1126 13:08:43.307000 140655 site-packages/torch/distributed/run.py:774] 
W1126 13:08:43.307000 140655 site-packages/torch/distributed/run.py:774] *****************************************
W1126 13:08:43.307000 140655 site-packages/torch/distributed/run.py:774] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1126 13:08:43.307000 140655 site-packages/torch/distributed/run.py:774] *****************************************
/root/anaconda3/envs/b60/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/root/anaconda3/envs/b60/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
/root/anaconda3/envs/b60/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
[W1126 13:08:47.819292275 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
/root/anaconda3/envs/b60/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
My guessed rank = 1
[W1126 13:08:47.103702133 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
My guessed rank = 0
[W1126 13:08:47.228211471 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
[W1126 13:08:47.228211394 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
My guessed rank = 0
My guessed rank = 1
[INFO|2025-11-26 13:08:48] llamafactory.hparams.parser:468 >> Process rank: 1, world size: 2, device: xpu:1, distributed training: True, compute dtype: torch.bfloat16
[INFO|2025-11-26 13:08:48] llamafactory.hparams.parser:143 >> Set `ddp_find_unused_parameters` to False in DDP training since LoRA is enabled.
[INFO|2025-11-26 13:08:48] llamafactory.hparams.parser:468 >> Process rank: 0, world size: 2, device: xpu:0, distributed training: True, compute dtype: torch.bfloat16
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,581 >> loading file vocab.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,581 >> loading file merges.txt
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,581 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,581 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,581 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,581 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,581 >> loading file chat_template.jinja
[INFO|2025-11-26 13:08:48] llamafactory.hparams.parser:143 >> Set `ddp_find_unused_parameters` to False in DDP training since LoRA is enabled.
[INFO|2025-11-26 13:08:48] llamafactory.hparams.parser:468 >> Process rank: 0, world size: 2, device: xpu:0, distributed training: True, compute dtype: torch.bfloat16
[INFO|2025-11-26 13:08:48] llamafactory.hparams.parser:468 >> Process rank: 1, world size: 2, device: xpu:1, distributed training: True, compute dtype: torch.bfloat16
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,723 >> loading file vocab.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,723 >> loading file merges.txt
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,723 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,723 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,723 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,723 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,723 >> loading file chat_template.jinja
[INFO|tokenization_utils_base.py:2364] 2025-11-26 13:08:48,854 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|configuration_utils.py:763] 2025-11-26 13:08:48,855 >> loading configuration file /root/models/Qwen3-0.6B/config.json
[INFO|configuration_utils.py:839] 2025-11-26 13:08:48,856 >> Model config Qwen3Config {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,857 >> loading file vocab.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,857 >> loading file merges.txt
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,857 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,857 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,857 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,857 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:48,857 >> loading file chat_template.jinja
/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
2025:11:26-13:08:48:(140879) |CCL_WARN| value of CCL_LOG_LEVEL changed to be info (default:warn)
2025:11:26-13:08:48:(140879) |CCL_WARN| value of CCL_WORKER_COUNT changed to be 4 (default:1)
2025:11:26-13:08:48:(140879) |CCL_WARN| value of CCL_ZE_IPC_EXCHANGE changed to be sockets (default:pidfd)
2025:11:26-13:08:48:(140879) |CCL_INFO| process launcher: hydra, local_proc_idx: 0, local_proc_count: 2
2025:11:26-13:08:48:(140879) |CCL_INFO| initializing level-zero api
2025:11:26-13:08:48:(140879) |CCL_INFO| initializing level-zero
2025:11:26-13:08:48:(140879) |CCL_INFO| Total hardware threads: 1280
2025:11:26-13:08:48:(140879) |CCL_INFO| auto tune with port counts enabled
2025:11:26-13:08:48:(140879) |CCL_INFO| ze fabric ports: 0 were able to be detected
2025:11:26-13:08:48:(140879) |CCL_INFO| initialized level-zero
2025:11:26-13:08:48:(140879) |CCL_INFO| could not initialize umf api
2025:11:26-13:08:48:(140879) |CCL_INFO| OS info: { Linux b60 6.14.0-1006-intel #6-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  1 00:03:01 UTC 2025 x86_64 }
[INFO|tokenization_utils_base.py:2364] 2025-11-26 13:08:48,998 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|configuration_utils.py:763] 2025-11-26 13:08:48,998 >> loading configuration file /root/models/Qwen3-0.6B/config.json
[INFO|configuration_utils.py:839] 2025-11-26 13:08:49,000 >> Model config Qwen3Config {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:49,000 >> loading file vocab.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:49,000 >> loading file merges.txt
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:49,000 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:49,000 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:49,001 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:49,001 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2093] 2025-11-26 13:08:49,001 >> loading file chat_template.jinja
[INFO|tokenization_utils_base.py:2364] 2025-11-26 13:08:49,136 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|2025-11-26 13:08:49] llamafactory.data.loader:143 >> Loading dataset identity.json...
[INFO|tokenization_utils_base.py:2364] 2025-11-26 13:08:49,296 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|2025-11-26 13:08:49] llamafactory.data.loader:143 >> Loading dataset identity.json...
/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
2025:11:26-13:08:49:(140884) |CCL_INFO| process launcher: hydra, local_proc_idx: 1, local_proc_count: 2
2025:11:26-13:08:49:(140884) |CCL_INFO| initializing level-zero api
2025:11:26-13:08:49:(140884) |CCL_INFO| initializing level-zero
2025:11:26-13:08:49:(140884) |CCL_INFO| Total hardware threads: 1280
2025:11:26-13:08:49:(140884) |CCL_INFO| auto tune with port counts enabled
2025:11:26-13:08:49:(140884) |CCL_INFO| ze fabric ports: 0 were able to be detected
2025:11:26-13:08:49:(140884) |CCL_INFO| initialized level-zero
2025:11:26-13:08:49:(140884) |CCL_INFO| could not initialize umf api
2025:11:26-13:08:49:(140884) |CCL_INFO| OS info: { Linux b60 6.14.0-1006-intel #6-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  1 00:03:01 UTC 2025 x86_64 }
Converting format of dataset (num_proc=16): 182 examples [00:00, 246.15 examples/s]       
[INFO|2025-11-26 13:08:51] llamafactory.data.loader:143 >> Loading dataset alpaca_en_demo.json...
Converting format of dataset (num_proc=16): 1998 examples [00:00, 2470.57 examples/s]       
/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
2025:11:26-13:08:51:(140878) |CCL_WARN| value of CCL_LOG_LEVEL changed to be info (default:warn)
2025:11:26-13:08:51:(140878) |CCL_WARN| value of CCL_WORKER_COUNT changed to be 4 (default:1)
2025:11:26-13:08:51:(140878) |CCL_WARN| value of CCL_ZE_IPC_EXCHANGE changed to be sockets (default:pidfd)
2025:11:26-13:08:51:(140878) |CCL_INFO| process launcher: hydra, local_proc_idx: 0, local_proc_count: 2
2025:11:26-13:08:51:(140878) |CCL_INFO| initializing level-zero api
2025:11:26-13:08:51:(140878) |CCL_INFO| initializing level-zero
2025:11:26-13:08:51:(140878) |CCL_INFO| Total hardware threads: 1280
2025:11:26-13:08:51:(140878) |CCL_INFO| auto tune with port counts enabled
2025:11:26-13:08:51:(140878) |CCL_INFO| ze fabric ports: 0 were able to be detected
2025:11:26-13:08:51:(140878) |CCL_INFO| initialized level-zero
2025:11:26-13:08:51:(140878) |CCL_INFO| could not initialize umf api
2025:11:26-13:08:51:(140878) |CCL_INFO| OS info: { Linux b60 6.14.0-1006-intel #6-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  1 00:03:01 UTC 2025 x86_64 }
[cli_0]: write_line error; fd=9 buf=:cmd=init pmi_version=1 pmi_subversion=1
:
system msg for write_line failure : Broken pipe
[cli_0]: Unable to write to PMI_fd
[cli_0]: write_line error; fd=9 buf=:cmd=get_appnum
:
system msg for write_line failure : Broken pipe
Abort(1090831) on node 0 (rank 0 in comm 0): Fatal error in PMPI_Init_thread: Unknown error class, error stack:
MPIR_Init_thread(196): 
MPID_Init(1612)......: 
MPIR_pmi_init(142)...: PMI_Get_appnum returned -1
[cli_0]: write_line error; fd=9 buf=:cmd=abort exitcode=1090831
:
system msg for write_line failure : Broken pipe
[cli_0]: write_line error; fd=9 buf=:cmd=init pmi_version=1 pmi_subversion=1
:
system msg for write_line failure : Invalid argument
[cli_0]: Unable to write to PMI_fd
[cli_0]: write_line error; fd=9 buf=:cmd=get_appnum
:
system msg for write_line failure : Invalid argument
Abort(1090831) on node 0 (rank 0 in comm 0): Fatal error in PMPI_Init_thread: Unknown error class, error stack:
MPIR_Init_thread(196): 
MPID_Init(1612)......: 
MPIR_pmi_init(142)...: PMI_Get_appnum returned -1
[cli_0]: write_line error; fd=9 buf=:cmd=abort exitcode=1090831
:
system msg for write_line failure : Invalid argument
Converting format of dataset (num_proc=16): 100%|██████████| 91/91 [00:00<?, ? examples/s]E1126 13:08:52.560000 140616 site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: -11) local_rank: 0 (pid: 140878) of binary: /root/anaconda3/envs/b60/bin/python3.10
Traceback (most recent call last):
  File "/root/anaconda3/envs/b60/bin/torchrun", line 7, in <module>
    sys.exit(main())
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 357, in wrapper
    return f(*args, **kwargs)
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/run.py", line 901, in main
    run(args)
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/run.py", line 892, in run
    elastic_launch(
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 143, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 277, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
========================================================
/root/LLaMA-Factory/src/llamafactory/launcher.py FAILED
--------------------------------------------------------
Failures:
[1]:
  time      : 2025-11-26_13:08:52
  host      : b60
  rank      : 1 (local_rank: 1)
  exitcode  : -11 (pid: 140879)
  error_file: <N/A>
  traceback : Signal 11 (SIGSEGV) received by PID 140879
--------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-11-26_13:08:52
  host      : b60
  rank      : 0 (local_rank: 0)
  exitcode  : -11 (pid: 140878)
  error_file: <N/A>
  traceback : Signal 11 (SIGSEGV) received by PID 140878
========================================================
Converting format of dataset (num_proc=16): 182 examples [00:00, 213.69 examples/s]       
Traceback (most recent call last):
  File "/root/anaconda3/envs/b60/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/root/anaconda3/envs/b60/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/root/LLaMA-Factory/src/llamafactory/cli.py", line 31, in <module>
    main()
  File "/root/LLaMA-Factory/src/llamafactory/cli.py", line 24, in main
    launcher.launch()
  File "/root/LLaMA-Factory/src/llamafactory/launcher.py", line 115, in launch
    process = subprocess.run(
  File "/root/anaconda3/envs/b60/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['torchrun', '--nnodes', '1', '--node_rank', '0', '--nproc_per_node', '2', '--master_addr', '127.0.0.1', '--master_port', '51481', '/root/LLaMA-Factory/src/llamafactory/launcher.py', 'examples/train_lora/qwen3-0.6B_lora_sft.yaml']' returned non-zero exit status 1.
[INFO|2025-11-26 13:08:52] llamafactory.data.loader:143 >> Loading dataset alpaca_en_demo.json...
Converting format of dataset (num_proc=16): 1998 examples [00:00, 2354.95 examples/s]       
/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
2025:11:26-13:08:54:(140883) |CCL_INFO| process launcher: hydra, local_proc_idx: 1, local_proc_count: 2
2025:11:26-13:08:54:(140883) |CCL_INFO| initializing level-zero api
2025:11:26-13:08:54:(140883) |CCL_INFO| initializing level-zero
2025:11:26-13:08:54:(140883) |CCL_INFO| Total hardware threads: 1280
2025:11:26-13:08:54:(140883) |CCL_INFO| auto tune with port counts enabled
2025:11:26-13:08:54:(140883) |CCL_INFO| ze fabric ports: 0 were able to be detected
2025:11:26-13:08:54:(140883) |CCL_INFO| initialized level-zero
2025:11:26-13:08:54:(140883) |CCL_INFO| could not initialize umf api
2025:11:26-13:08:54:(140883) |CCL_INFO| OS info: { Linux b60 6.14.0-1006-intel #6-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  1 00:03:01 UTC 2025 x86_64 }
[cli_1]: write_line error; fd=10 buf=:cmd=init pmi_version=1 pmi_subversion=1
:
system msg for write_line failure : Broken pipe
[cli_1]: Unable to write to PMI_fd
[cli_1]: write_line error; fd=10 buf=:cmd=get_appnum
:
system msg for write_line failure : Broken pipe
Abort(1090831) on node 0 (rank 0 in comm 0): Fatal error in PMPI_Init_thread: Unknown error class, error stack:
MPIR_Init_thread(196): 
MPID_Init(1612)......: 
MPIR_pmi_init(142)...: PMI_Get_appnum returned -1
[cli_1]: write_line error; fd=10 buf=:cmd=abort exitcode=1090831
:
system msg for write_line failure : Broken pipe
[cli_1]: readline failed
[cli_1]: write_line error; fd=10 buf=:cmd=get_maxes
:
system msg for write_line failure : Broken pipe
[cli_1]: write_line error; fd=10 buf=:cmd=get_appnum
:
system msg for write_line failure : Broken pipe
Abort(1090831) on node 0 (rank 0 in comm 0): Fatal error in PMPI_Init_thread: Unknown error class, error stack:
MPIR_Init_thread(196): 
MPID_Init(1612)......: 
MPIR_pmi_init(142)...: PMI_Get_appnum returned -1
[cli_1]: write_line error; fd=10 buf=:cmd=abort exitcode=1090831
:
system msg for write_line failure : Broken pipe
W1126 13:08:55.240000 140655 site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 140884 closing signal SIGTERM
E1126 13:08:55.454000 140655 site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: -11) local_rank: 0 (pid: 140883) of binary: /root/anaconda3/envs/b60/bin/python3.10
Traceback (most recent call last):
  File "/root/anaconda3/envs/b60/bin/torchrun", line 7, in <module>
    sys.exit(main())
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 357, in wrapper
    return f(*args, **kwargs)
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/run.py", line 901, in main
    run(args)
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/run.py", line 892, in run
    elastic_launch(
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 143, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/root/anaconda3/envs/b60/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 277, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
========================================================
/root/LLaMA-Factory/src/llamafactory/launcher.py FAILED
--------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
--------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-11-26_13:08:55
  host      : b60
  rank      : 0 (local_rank: 0)
  exitcode  : -11 (pid: 140883)
  error_file: <N/A>
  traceback : Signal 11 (SIGSEGV) received by PID 140883
========================================================
Traceback (most recent call last):
  File "/root/anaconda3/envs/b60/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/root/anaconda3/envs/b60/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/root/LLaMA-Factory/src/llamafactory/cli.py", line 31, in <module>
    main()
  File "/root/LLaMA-Factory/src/llamafactory/cli.py", line 24, in main
    launcher.launch()
  File "/root/LLaMA-Factory/src/llamafactory/launcher.py", line 115, in launch
    process = subprocess.run(
  File "/root/anaconda3/envs/b60/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['torchrun', '--nnodes', '1', '--node_rank', '0', '--nproc_per_node', '2', '--master_addr', '127.0.0.1', '--master_port', '45265', '/root/LLaMA-Factory/src/llamafactory/launcher.py', 'examples/train_lora/qwen3-0.6B_lora_sft.yaml']' returned non-zero exit status 1.
 ```
</details
