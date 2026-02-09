# GNNWR Changelog

## [Unreleased] — 2026-02

### Bug Fixes

- **修复 DataParallel 双重包装**（models.py）：模型通过 `nn.DataParallel` 包装后保存到磁盘，重新加载时 `torch.load` 返回的已经是 DataParallel 对象，代码会再次包装导致模型结构损坏、推理结果错误。在 `run()`、`load_model()`、`reg_result()` 三处加入 `isinstance` 检查。
- **修复除零错误**（datasets.py）：`minmax_scaler` 和 `standard_scaler` 在特征列所有值相同时（max == min 或 std == 0）触发除零。添加零值分母保护，常量列归一化后设为 0。
- **修复奇异矩阵崩溃**（utils.py）：DIAGNOSIS 中 X'X 矩阵奇异或病态时 `torch.linalg.inv` / `torch.inverse` 崩溃。改用条件数检测 + 伪逆（`torch.linalg.pinv`）兜底。
- **修复 rescale() 逻辑错误**（datasets.py）：仅传入 x 或 y 之一时，因另一方 scale_info 为 None 导致误抛异常。改为独立检查 x 和 y。
- **修复可变默认参数**（networks.py）：`SWNN`、`STPNN`、`STNN_SPNN` 的 `activate_func` 使用可变对象 `nn.PReLU()` / `nn.ReLU()` 作为默认参数，多个实例共享同一激活函数。改为 `None` + 运行时创建。
- **修复重复目录创建**（models.py）：`GNNWR` 和 `GTNNWR` 构造函数中 `os.makedirs` 被调用 4 次（含重复），替换为 `os.makedirs(..., exist_ok=True)`。
- **修复变量名拼写错误**（models.py）：`weigth_decay` → `weight_decay`。

### Code Quality

- **替换废弃 API**（models.py）：将 9 处 `.data` 属性替换为 `.detach()`，遵循 PyTorch 官方推荐。
- **消除裸 except 子句**（models.py）：`except:` 改为 `except (ZeroDivisionError, RuntimeError, FloatingPointError) as e:`，避免吞掉 `KeyboardInterrupt`。
- **修复 `__str__` 方法**（models.py）：原实现通过 `print()` 输出并返回空字符串，改为返回格式化字符串。
- **修复 LinearNetwork `__str__`**（networks.py）：处理 `drop_out=0` 时 `nn.Identity` 没有 `.p` 属性的情况。

### Performance

- **ManhattanDistance 优化**（datasets.py）：将 numpy 广播运算替换为 `scipy.spatial.distance.cdist`。

  | 数据规模 | 优化前 | 优化后 | 加速比 | 内存节省 |
  |---------|-------|-------|-------|---------|
  | 1K | 22.5ms | 2.0ms | 11.0x | 25% |
  | 5K | 502ms | 79.8ms | 6.3x | 25% |
  | 10K | 2.01s | 308ms | 6.5x | 25% |
  | 30K | 15.2s | 2.29s | 6.6x | 25% |

- **BasicDistance 多后端支持**（datasets.py）：新增 `backend`（'scipy'/'torch'/'auto'）和 `device`（'cpu'/'cuda'）参数，支持 GPU 加速距离计算。默认行为不变。

  | 数据规模 | scipy | torch CPU | torch GPU | GPU 加速比 |
  |---------|-------|-----------|-----------|-----------|
  | 1K | 2.5ms | 1.5ms | 0.21ms | 12x |
  | 5K | 87.8ms | 16.7ms | 0.48ms | 184x |
  | 10K | 347ms | 49.5ms | 1.30ms | 268x |
  | 50K | 9.03s | 476ms | 25.4ms | 355x |

- **DataLoader 张量缓存**（datasets.py）：`baseDataset.__getitem__` 使用 `torch.from_numpy` 懒加载缓存，避免每次调用 `torch.tensor()` 重复内存分配。`_invalidate_tensor_cache()` 在 `scale()` 等数据变更后自动清除缓存。

  | 数据规模 | 优化前 (10K 次访问) | 优化后 | 加速比 |
  |---------|-------------------|-------|-------|
  | 1K | 178ms | 56ms | 3.2x |
  | 10K | 203ms | 56ms | 3.6x |
  | 50K | 382ms | 55ms | 6.9x |

### New Features

- **KNN 稀疏距离矩阵**（datasets.py, models.py）：新增 `KNNDistance()` 函数，只存储 K 个最近邻的距离，大幅降低内存占用。`init_dataset` 新增 `knn_k`、`reference_size` 参数。

  | 数据规模 | 完整矩阵内存 | KNN k=100 内存 | 内存节省 |
  |---------|------------|--------------|---------|
  | 10K | 305MB | 7.6MB | 97.5% |
  | 50K | 7.45GB | 38MB | 99.5% |
  | 100K | 29.8GB | 76MB | 99.8% |

- **DistanceProjection 投影层**（networks.py, models.py）：在 SWNN 前端加入可学习线性投影，将高维距离向量压缩到固定 `embed_dim`，解耦 knn_k 与网络结构。GNNWR / GTNNWR 新增 `embed_dim` 参数。

  在 100K 模拟数据上的消融实验结果（NVIDIA A800-SXM4-80GB）：

  | knn_k | embed_dim | 参数量 | Test R² | vs 无投影 |
  |------:|----------:|-------:|--------:|:---------:|
  | 1000 | none | 301K | 0.7306 | — |
  | 1000 | 256 | 367K | 0.7379 | +0.0073 |
  | 2000 | none | 557K | 0.7694 | — |
  | 2000 | 256 | 623K | **0.7924** | **+0.0230** |

  结论：knn_k >= 1000 时，embed_dim=256 在参数量相近的条件下提升 R²，投影层起到 bottleneck 正则化效果。

- **DIAGNOSIS lite 模式**（utils.py, models.py）：大数据集（n > 10000）自动跳过 Hat 矩阵计算（O(n²) 内存），R² 和 RMSE 仍可正常使用，AIC/AICc/F-test 仅在非 lite 模式下可用。
