# Sternberg工作记忆任务 - 快速入门

## 新增文件

我已经为你创建了以下文件来实现Sternberg工作记忆任务：

1. **`examples/models/sternberg_1area.py`** - 单区域RNN模型（推荐先用这个测试）
2. **`examples/models/sternberg_3areas.py`** - 三区域RNN模型（Sensory → Memory → Decision）
3. **`examples/test_sternberg.py`** - 测试脚本，可视化trial生成
4. **`STERNBERG_TASK_GUIDE.md`** - 详细的实现指南和说明文档

## 快速开始

### 步骤1: 测试任务设置

首先运行测试脚本，确保任务定义正确：

```bash
cd /Users/yifei/Desktop/UCLA/multi-area-cleaned

# 测试单区域模型
python examples/test_sternberg.py sternberg_1area

# 测试三区域模型
python examples/test_sternberg.py sternberg_3areas
```

这会：
- 生成几个示例trials（match, non-match, catch）
- 显示输入、输出、掩码的可视化
- 显示连接矩阵的结构
- 打印详细的trial信息

### 步骤2: 训练网络

```bash
# 训练单区域模型（快速，推荐先用这个）
python examples/do.py examples/models/sternberg_1area.py train

# 训练三区域模型
python examples/do.py examples/models/sternberg_3areas.py train
```

训练过程会：
- 自动保存到 `saved_rnns_server_apr/data/` 目录
- 定期输出性能指标
- 达到目标性能（75%）后自动停止

### 步骤3: 分析结果

训练完成后，你可以使用现有的分析工具：

```python
import cPickle as pickle
from pycog import RNN

# 加载训练好的模型
with open('saved_rnns_server_apr/data/sternberg_1area.pkl', 'rb') as f:
    data = pickle.load(f)

rnn = RNN('saved_rnns_server_apr/data/sternberg_1area.pkl', verbose=True)

# 运行网络并分析活动
# ... 使用sims/下的分析脚本
```

## Sternberg任务说明

### 任务结构

```
时间线:
[Fixation] → [Item1][Item2][Item3]... → [Delay] → [Probe] → [Response]
   200ms      300ms each (2-6 items)    1-2s      500ms      500ms
```

### 输入编码

- **输入通道**: 9个（8个items + 1个probe信号）
  - 通道0-7: 代表8个可能的items（one-hot编码）
  - 通道8: probe信号（只在probe期激活）

### 输出

- **输出通道**: 2个
  - 通道0: Non-match
  - 通道1: Match

### 条件

- **Set sizes**: [2, 4, 6] - 记忆集合大小
- **Match types**: Match vs. Non-match
- **总条件数**: 3 × 2 = 6

## 自定义参数

你可以编辑模型文件来调整参数：

### 调整任务难度

```python
# 在 sternberg_1area.py 或 sternberg_3areas.py 中修改:

# 改变记忆集合大小
set_sizes = [2, 3, 4, 5, 6]  # 增加更多难度

# 改变item数量
n_items = 10  # 更多可能的items

# 改变延迟时长
delay_duration = [2000, 4000]  # 更长的延迟期
```

### 调整网络结构

```python
# 改变神经元数量
N = 500  # 更大的网络

# 改变连接概率
ff_prop = 0.2  # 更强的前馈连接
fb_prop = 0.1  # 更强的反馈连接
```

### 调整训练参数

```python
# 改变学习率
learning_rate = 1e-4  # 更慢/更快的学习

# 改变性能目标
TARGET_PERFORMANCE = 85  # 更高的目标准确率
```

## 多区域结构说明

在三区域模型中：

1. **Sensory Area** (前1/3神经元)
   - 接收所有输入
   - 负责编码items
   - 向Memory区域发送信息

2. **Memory Area** (中间1/3神经元)
   - 维持工作记忆
   - 强循环连接（持续活动）
   - 向Decision区域发送信息

3. **Decision Area** (后1/3神经元)
   - 比较probe与记忆
   - 生成match/non-match决策
   - 连接到输出层

### 连接模式

```
输入 → Sensory Area ⇄ Memory Area ⇄ Decision Area → 输出
         (ff: 10%)      (ff: 10%)
         (fb: 5%)       (fb: 5%)
```

## 预期结果

### 训练时间

- 单区域 (300 units): ~30-60分钟（取决于硬件）
- 三区域 (300 units): ~45-90分钟

### 性能指标

- **目标**: >75% 准确率
- **典型结果**: 80-90% 准确率
- **Set size effect**: 更大的set size通常准确率略低（符合行为学规律）

### 网络活动特征

你应该能观察到：
1. **编码期**: 顺序的item表征
2. **延迟期**: 持续的神经活动（维持记忆）
3. **Probe期**: 比较过程的激活
4. **响应期**: 明确的match/non-match决策

## 故障排除

### 问题1: 准确率停在50%

**原因**: 网络没有学习，只是随机猜测

**解决**:
- 检查输出和mask是否正确
- 降低学习率
- 增加训练时间
- 检查正则化参数

### 问题2: 训练很慢

**解决**:
- 减少神经元数量 (N=200)
- 减少set sizes (只用[2,4])
- 使用GPU训练
- 减少validation频率

### 问题3: 测试脚本报错

**常见错误**:
```python
# 如果看到 "No module named 'pycog'"
# 确保已经运行:
add2virtualenv /Users/yifei/Desktop/UCLA/multi-area-cleaned
add2virtualenv /Users/yifei/Desktop/UCLA/multi-area-cleaned/pycog
```

### 问题4: 内存不足

**解决**:
- 减少batch size
- 减少神经元数量
- 减少trial长度（缩短延迟期）

## 进阶分析

### 分析区域间信息流

```python
# 使用 dPCA 分析不同区域的编码
# 参考 sims/null_potent_dpca.py
```

### 分析工作记忆容量

```python
# 绘制准确率 vs. set size
# 分析反应时间随load增加
```

### 分析持续活动

```python
# 提取延迟期的神经活动
# 分析维持记忆的机制
```

## 下一步

1. ✅ 运行测试脚本验证设置
2. ✅ 训练基础模型（sternberg_1area）
3. ✅ 训练多区域模型（sternberg_3areas）
4. 🔄 使用现有分析工具分析结果
5. 🔄 尝试不同的网络配置和任务参数
6. 🔄 与其他任务（如原始的decision-making任务）比较

## 参考资料

- **详细文档**: `STERNBERG_TASK_GUIDE.md`
- **原始论文**: Sternberg, S. (1966). High-speed scanning in human memory. Science.
- **原始仓库示例**: `examples/models/2020-04-10_cb_simple_*.py`

## 需要帮助？

如果遇到问题：
1. 查看 `STERNBERG_TASK_GUIDE.md` 获取详细说明
2. 检查测试脚本的输出
3. 对比原始模型文件 (`2020-04-10_cb_simple_1area.py`)
4. 检查 `pycog/tasktools.py` 中的工具函数

祝实验顺利！ 🧠


