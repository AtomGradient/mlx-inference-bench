# Prompt Lookup Speculative Decoding 实现文档

## 概述

本文档描述了为 mlx-vlm 框架实现的 Prompt Lookup Speculative Decoding（N-gram 投机解码）。该实现针对 Qwen3.5-9B 模型在 Apple M2 Ultra 上的推理进行了优化。

**最终结论**: 由于 Qwen3.5 的混合注意力架构（75% GatedDeltaNet linear attention），speculative decoding 在该模型上**无法实现加速**，最多持平（1.00x）。但实现本身架构正确，对纯 Transformer 模型有潜在价值。

## 目录

1. [算法原理](#算法原理)
2. [架构挑战：Qwen3.5 混合注意力](#架构挑战)
3. [核心实现](#核心实现)
4. [迭代优化历程](#迭代优化历程)
5. [性能剖析](#性能剖析)
6. [Benchmark 结果](#benchmark-结果)
7. [结论与未来方向](#结论与未来方向)

---

## 算法原理

Prompt Lookup Speculative Decoding 利用 N-gram 匹配从已有 token（prompt + 已生成）中预测接下来的 token，无需额外的 draft model。

### 流程

```
1. 标准生成一个 token
2. 用最后 N 个 token 在历史中查找 N-gram 匹配
3. 如果匹配，取后续 K 个 token 作为 draft
4. 批量前向传播验证所有 draft token
5. 接受匹配的 token，拒绝不匹配的
6. 从第一个拒绝位置的 bonus token 继续
```

### 优势

- **无需 draft model**: 零额外内存开销
- **适合总结/复述任务**: prompt 与输出有高 N-gram 重叠
- **批量前向**: 多个 token 共享一次权重读取

---

## 架构挑战

### Qwen3.5 混合注意力架构

Qwen3.5-9B 使用 32 层，其中：
- **24 层 (75%)**: GatedDeltaNet (linear attention)
- **8 层 (25%)**: 标准 Full Attention (head_dim=256)

### GatedDeltaNet 的限制

```
state_t = gate * state_{t-1} + key * delta_t
```

递推关系**不可并行化**。即使输入 5 个 token 的 batch，GatedDeltaNet 层仍然必须逐个 token 顺序处理。这意味着 batch 前向传播无法获得理论上的线性加速：

| Batch Size | 理论加速 | 实测加速 | 原因 |
|-----------|---------|---------|------|
| 2 | 2.00x | 1.76x | GatedDeltaNet 顺序处理 |
| 3 | 3.00x | 2.24x | |
| 5 | 5.00x | 2.57x | |
| 8 | 8.00x | 3.20x | |

### ArraysCache 不支持 Trim

标准 speculative decoding 通过 trim KV cache 回滚被拒绝的 token。但 GatedDeltaNet 使用 ArraysCache（存储 conv_state 和 recurrent_state），**不支持 trim 操作**。

**解决方案**: 零开销 checkpoint/restore 机制。

---

## 核心实现

### 文件结构

```
mlx-vlm-optimized/
├── speculative_generate.py   # 主要实现：投机解码生成器
├── cache_utils.py            # cache checkpoint/restore 工具
├── benchmark.py              # 标准 vs 投机 Benchmark
└── profile_forward.py        # 前向传播性能剖析
```

### 1. 零开销 Cache Checkpoint (`cache_utils.py`)

**关键发现**: GatedDeltaNet 在每次前向传播时创建**新的** array 对象（通过 `mx.concatenate` 和 `gated_delta_update`），旧的 Python 引用保持有效。因此 checkpoint 只需保存 Python 引用，无需深拷贝。

```python
def save_cache_checkpoint(cache):
    """保存 checkpoint — 几乎零开销 (0.01ms)"""
    checkpoint = []
    for c in cache:
        if isinstance(c, ArraysCache):
            # 仅保存 Python 引用列表！旧数组不会被原地修改
            checkpoint.append(("arrays", list(c.cache)))
        elif isinstance(c, KVCache):
            # 仅保存 offset 整数
            checkpoint.append(("kv", c.offset))
    return checkpoint

def restore_cache_checkpoint(cache, checkpoint):
    """恢复 checkpoint — 几乎零开销 (0.04ms)"""
    for c, cp in zip(cache, checkpoint):
        if cp[0] == "arrays":
            c.cache = cp[1]      # 恢复引用
        elif cp[0] == "kv":
            c.offset = cp[1]     # 恢复 offset
```

实测开销：checkpoint=0.01ms, restore=0.04ms。

### 2. N-gram Lookup (`NgramLookup` class)

```python
class NgramLookup:
    def __init__(self, ngram_size=3, max_draft=5):
        self._table = defaultdict(list)  # N-gram → [positions]

    def build(self, tokens):    # 从 prompt 构建初始表
    def update(self, tokens, start_pos):  # 增量更新
    def lookup(self, context, all_tokens): # 查找 draft tokens
```

- 使用 hash map 存储 N-gram 到位置的映射
- 支持增量更新（每生成一个 token 调用一次）
- 返回最长匹配的后续 token 序列

### 3. 生成循环 (V5 设计)

```
Standard async pipeline (零开销):
┌──────────────┐
│ GPU: compute  │  ← 异步启动下一个 token 的计算
│ next_y       │
├──────────────┤
│ CPU: yield    │  ← 输出当前 token
│ current_y    │
├──────────────┤
│ CPU: N-gram   │  ← 查找 draft (GPU 仍在工作)
│ lookup       │
└──────────────┘

Speculative path (on N-gram match):
┌──────────────┐
│ eval(next_y)  │  ← 等待已经在计算的 next_y
├──────────────┤
│ draft[0]      │  ← 免费检查：next_y == draft[0]?
│ match?        │
├──────────────┤
│ checkpoint    │  ← 0.01ms
├──────────────┤
│ batch verify  │  ← 33ms (5 tokens)
│ [remaining]   │
├──────────────┤
│ batch sample  │  ← 比较所有位置
├──────────────┤
│ restore +     │  ← 0.04ms + 18ms (部分接受时)
│ reprocess     │
└──────────────┘
```

核心设计原则：
- **标准路径零开销**: GPU 已在异步计算 next_y，N-gram lookup 在 CPU 上并行
- **draft[0] 免费检查**: next_y 已经计算好，只需比较值
- **批量采样**: 单次 `mx.eval(matches, draft_sampled)` 替代逐个评估

---

## 迭代优化历程

### V1: 基础实现
- 同步 pipeline：GPU 空闲等待 CPU yield
- **结果**: 0.88x（12% 开销来自 pipeline 断裂）

### V2: Eval timing 修复
- 移动 eval 到 yield 前
- **结果**: 仍有 ~5% 开销

### V3: 标准 async pipeline
- 完全匹配 `generate_step` 的异步结构
- GPU 在 yield 之前就开始计算下一个 token
- **结果**: 1.00x（零开销！）

### V4: 顺序验证（去除 batch）
- 假设：checkpoint/restore 是瓶颈
- 每个 draft token 独立用 _step_single 验证
- **结果**: 0.93x（更慢！每次 _step_single 都读 10GB 权重）

### V5: 批量验证 + 零开销 checkpoint + 批量采样（最终方案）
- 发现 ArraysCache 引用保存安全 → 零开销 checkpoint
- 单次 batch 前向传播验证所有 draft
- 批量采样：一次性比较所有位置
- **结果**: short=1.00x, medium=0.99x, long=0.99x, summary=0.95-1.00x

### V6: Draft[0]-only（实验）
- 只接受第一个 draft match，不做 batch verify
- **结果**: summary=0.96x（需要更多 forward passes，不如 V5）

---

## 性能剖析

### 前向传播耗时 (`profile_forward.py`)

测试环境: Qwen3.5-9B-8bit, Apple M2 Ultra, ~20 warmup tokens

| 操作 | 耗时 (ms) | 备注 |
|------|----------|------|
| 单 token forward | 17.08 | 基准 |
| Batch(2) forward | 19.40 | 1.76x vs 2×single |
| Batch(3) forward | 22.84 | 2.24x vs 3×single |
| Batch(5) forward | 33.28 | 2.57x vs 5×single |
| Batch(8) forward | 42.71 | 3.20x vs 8×single |
| Checkpoint (save) | 0.01 | Python 引用保存 |
| Restore | 0.04 | Python 引用恢复 |
| Reprocess (2 tok) | ~18 | restore 后 cache 更新 |

### 投机步骤成本分析

**Batch(5), accept 2 (典型场景)**:
```
Checkpoint:     0.01ms
Batch forward: 33.28ms  (验证 5 个 draft)
Sampling:       0.52ms  (批量比较)
Restore:        0.04ms
Reprocess:     18.42ms  (重放 accepted 前缀)
────────────────────────
Total:         52.27ms → 3 tokens (2 accepted + 1 bonus) = 17.4ms/tok

Standard:      17.08ms/tok (async pipeline 有效约 15.5ms/tok)
```

结论：**投机解码在 Qwen3.5 上 per-token 成本与标准路径几乎相同**。

### 瓶颈分析

1. **GatedDeltaNet 顺序递推**: 75% 的层在 batch 中仍按顺序处理
   - Batch(5) 只获得 2.57x 加速（vs 理论 5x）
   - 这是**架构级限制**，无法通过软件优化解决

2. **Reprocess 开销**: 部分接受时需要重放已接受 token 更新 cache
   - 约 18ms，几乎等于一次标准前向传播

3. **Pipeline 中断**: batch verify 引入同步点
   - 标准路径通过 `mx.async_eval` 隐藏约 1.5ms 延迟
   - 投机路径需要 `mx.eval` 等待结果

---

## Benchmark 结果

### V5 Final (ngram=3, draft=5)

| Prompt | Std (t/s) | Spec (t/s) | Speedup | Accept% | Tok/Step |
|--------|-----------|------------|---------|---------|----------|
| short | 67.9 | 67.8 | 1.00x | 0% | 1.04 |
| medium | 65.3 | 64.9 | 0.99x | 0% | 0.97 |
| long | 65.0 | 64.6 | 0.99x | 50% | 1.01 |
| summary | 64.5 | 61.0 | 0.95x | 52% | 1.29 |

### 最优参数 (ngram=2, draft=3)

| Prompt | Std (t/s) | Spec (t/s) | Speedup | Accept% |
|--------|-----------|------------|---------|---------|
| summary | 64.1 | 64.4 | 1.00x | 63.5% |

更小的 draft 数量减少了 batch verify 和 reprocess 开销。

### 关键观察

1. **无匹配时零开销**: short/medium prompt 几乎没有 N-gram 匹配，speculative 路径不触发，速度完全一致
2. **高匹配反而更慢**: summary prompt 有 52-63% acceptance，但 batch verify + reprocess 开销抵消了收益
3. **内存零增长**: Peak memory 完全一致（零开销 checkpoint 不分配新内存）

---

## 结论与未来方向

### 核心结论

**Speculative decoding 在 Qwen3.5 上无法实现加速**，根本原因是 GatedDeltaNet 的顺序递推关系限制了 batch 前向传播的加速比。

| 因素 | 纯 Transformer | Qwen3.5 (混合) |
|------|---------------|----------------|
| Batch(5) 加速比 | ~5x | 2.57x |
| 单次投机步骤收益 | 显著 | 微弱 |
| Reprocess 相对开销 | 低 | 高 (与 batch verify 相当) |
| 总体预期加速 | 1.3-2x | 0.95-1.00x |

### 可复用成果

1. **零开销 checkpoint/restore**: ArraysCache 引用保存机制，适用于任何使用 GatedDeltaNet 的模型
2. **N-gram lookup 实现**: 高效的增量 N-gram 表，可复用于其他投机解码场景
3. **async pipeline 设计模式**: 保持 GPU 持续工作的生成循环结构

### 未来方向

1. **在纯 Transformer 模型上测试**: 如 Llama-3, Gemma 等，batch(5) 应接近 5x 加速
2. **Draft model 方案**: 使用小型 draft model（如 0.5B）替代 N-gram lookup，可能获得更高 acceptance rate
3. **Prefill 优化**: MLX 在 prompt processing 上比 llama.cpp 慢 1.9-3.6x，这是更大的优化空间
4. **混合递推-注意力架构优化**: 探索将 GatedDeltaNet 的递推状态在 batch 中预计算的可能性

---

## 运行说明

```bash
# 激活环境
source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate
cd /Users/alex/Documents/Codes/RefSources/mlx-vlm-optimized

# 运行 benchmark
python benchmark.py --prompt all --max-tokens 256 --runs 3

# 指定参数
python benchmark.py --prompt summary --ngram-size 2 --max-draft 3

# 性能剖析
python profile_forward.py
```
