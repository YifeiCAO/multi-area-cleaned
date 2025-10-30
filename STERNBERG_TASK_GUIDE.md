# 将Multi-Area RNN改造为Sternberg工作记忆任务指南

## 概述

本指南说明如何将这个multi-area RNN仓库改造成实现Sternberg工作记忆任务。我已经创建了两个示例文件作为参考：
- `examples/models/sternberg_1area.py` - 单区域版本（用于快速测试）
- `examples/models/sternberg_3areas.py` - 三区域版本（Sensory → Memory → Decision）

## Sternberg工作记忆任务简介

Sternberg任务是一个经典的工作记忆任务，包含以下阶段：

1. **注视期 (Fixation)**: 初始准备阶段
2. **编码期 (Encoding)**: 顺序呈现多个items（如数字、字母等）
3. **延迟期 (Delay)**: 保持items在工作记忆中，无刺激呈现
4. **探测期 (Probe)**: 呈现一个探测item
5. **响应期 (Response)**: 判断探测item是否在记忆集合中（match/non-match）

任务的关键挑战：
- 需要编码多个items
- 需要在延迟期维持这些信息
- 需要将探测item与记忆集合进行比较

## 改造步骤

### 1. 定义任务参数

```python
# Sternberg任务特定参数
set_sizes = [2, 4, 6]  # 记忆集合的大小（需要记住的item数量）
n_items = 8  # 可能的item总数（如8个不同的数字）
nconditions = len(set_sizes) * 2  # 条件数 = set_size * (match/non-match)

# 时间参数（单位：毫秒）
fixation_duration = 200  # 注视期时长
item_duration = 300  # 每个item的呈现时长
delay_duration = [1000, 2000]  # 延迟期时长范围（随机）
probe_duration = 500  # 探测item呈现时长
response_duration = 500  # 响应时长
```

### 2. 定义网络结构

#### 输入层 (Nin)
```python
Nin = n_items + 1  # n_items个输入通道（每个item一个）+ 1个探测信号通道
```

- 前`n_items`个通道：每个通道代表一个可能的item
- 最后1个通道：标记探测期（帮助网络区分编码期和探测期）

#### 隐藏层 (N)
对于多区域版本，将神经元分成多个功能区域：

```python
N = 300  # 总神经元数

# 三区域结构示例
EXC_SENSORY = EXC[:Nexc // 3]      # 感觉区（编码）
EXC_MEMORY = EXC[Nexc // 3:2*Nexc // 3]  # 记忆区（维持）
EXC_DECISION = EXC[2*Nexc // 3:]    # 决策区（比较和输出）
```

#### 输出层 (Nout)
```python
Nout = 2  # 两个输出：[non-match, match]
```

### 3. 设计连接矩阵

#### 输入连接 (Cin)
```python
# 只有感觉区接收输入
Cin = np.zeros((N, Nin))
Cin[EXC_SENSORY + INH_SENSORY, :] = 1
```

#### 循环连接 (Crec)
对于多区域模型，建立层级结构：
- Sensory → Memory（前馈连接 + 稀疏反馈）
- Memory → Decision（前馈连接 + 稀疏反馈）
- 每个区域内部全连接

```python
ff_prop = 0.1   # 前馈连接概率
fb_prop = 0.05  # 反馈连接概率

# 例：Sensory → Memory连接
for i in EXC_SENSORY:
    Crec[EXC_MEMORY, i] = 1 * (rng.uniform(size=len(EXC_MEMORY)) < ff_prop)
    Crec[i, EXC_MEMORY] = 1 * (rng.uniform(size=len(EXC_MEMORY)) < fb_prop)
```

#### 输出连接 (Cout)
```python
# 只从决策区读取输出
Cout = np.zeros((Nout, N))
Cout[:, EXC_DECISION] = 1
```

### 4. 实现generate_trial函数

这是最关键的部分，定义每个trial的输入和期望输出。

```python
def generate_trial(rng, dt, params):
    # 1. 确定trial类型和参数
    if params['name'] in ['gradient', 'test']:
        set_size = rng.choice(set_sizes)
        is_match = rng.choice([0, 1])
    
    # 2. 计算各epoch的时间
    encoding_duration = set_size * item_duration
    delay_dur = rng.uniform(delay_duration[0], delay_duration[1])
    # ... 计算其他时间点
    
    # 3. 生成记忆集合和探测item
    memory_set = rng.choice(n_items, size=set_size, replace=False)
    if is_match:
        probe_item = rng.choice(memory_set)
    else:
        probe_item = rng.choice([i for i in range(n_items) if i not in memory_set])
    
    # 4. 构建输入矩阵 X
    X = np.zeros((len(t), Nin))
    # 顺序呈现记忆集合中的items
    for i, item in enumerate(memory_set):
        item_start = encoding_start + i * item_duration
        item_end = item_start + item_duration
        item_idx = get_idx(t, (item_start, item_end))
        X[item_idx, item] = 1.0
    
    # 呈现探测item
    X[probe_idx, probe_item] = 1.0
    X[probe_idx, -1] = 1.0  # 探测信号
    
    # 5. 构建目标输出矩阵 Y 和掩码矩阵 M
    Y = np.zeros((len(t), Nout))
    M = np.zeros_like(Y)
    
    if is_match:
        Y[response_idx, 1] = 1.0  # match
    else:
        Y[response_idx, 0] = 1.0  # non-match
    
    M[all_relevant_idx, :] = 1  # 指定哪些时间点的输出需要计入损失
    
    return trial
```

### 5. 实现性能评估函数

```python
def performance_sternberg(trials, z):
    """
    评估网络在Sternberg任务上的准确率
    """
    dt = trials[0]['info']['dt']
    ends = [len(trial['t']) - 1 for trial in trials]
    
    # 提取响应期末的网络输出
    choices = [np.argmax(z[end - buffer, i]) for i, end in enumerate(ends)]
    
    # 计算准确率
    correct = []
    for choice, trial in zip(choices, trials):
        if not trial['info']['catch']:
            correct.append(choice == trial['info']['is_match'])
    
    return 100 * sum(correct) / len(correct)
```

### 6. 设置训练参数

```python
# 学习率和梯度裁剪
learning_rate = 5e-5
max_gradient_norm = 0.1

# 正则化
lambda2_in = 1
lambda2_rec = 1
lambda2_out = 1

# 噪声
var_in = 0.10**2
var_rec = 0.05**2

# 性能目标
TARGET_PERFORMANCE = 75

def terminate(pcorrect_history):
    return np.mean(pcorrect_history[-1:]) > TARGET_PERFORMANCE
```

## 训练网络

### 基本训练命令

```bash
# 训练单区域模型（快速测试）
python examples/do.py examples/models/sternberg_1area.py train

# 训练三区域模型
python examples/do.py examples/models/sternberg_3areas.py train
```

### 指定保存路径

```bash
python examples/do.py examples/models/sternberg_1area.py train \
    --savefile saved_rnns_server_apr/data/sternberg_1area.pkl
```

## 关键设计考虑

### 1. Multi-Area的优势

对于Sternberg任务，multi-area结构可以：
- **Sensory区域**：专门负责编码输入items
- **Memory区域**：维持工作记忆（强循环连接，持续活动）
- **Decision区域**：比较探测item与记忆内容

这种功能分离有助于：
- 理解不同脑区在工作记忆中的作用
- 研究区域间的信息流动
- 提高网络的可解释性

### 2. 任务难度调节

可以通过以下方式调节难度：
- **Set size**: 更大的记忆集合更难
- **延迟时长**: 更长的延迟更难维持
- **Item相似性**: 可以使用更相似的items
- **干扰**: 在延迟期加入干扰刺激

### 3. 分析方向

训练完成后可以分析：
- 不同区域的神经活动模式
- 工作记忆容量（通过set size）
- 延迟期的持续活动
- 区域间的信息传递（用dPCA等方法）
- Load-dependent effects（反应时间随set size增加）

## 可能的变体

### 1. 四区域模型

```python
# Sensory → Encoding → Memory → Decision
EXC_SENSORY = EXC[:Nexc // 4]
EXC_ENCODING = EXC[Nexc // 4:Nexc // 2]
EXC_MEMORY = EXC[Nexc // 2:3*Nexc // 4]
EXC_DECISION = EXC[3*Nexc // 4:]
```

### 2. 不同的输入表示

```python
# 使用分布式表示而非one-hot
X[item_idx, item] = 1.0
X[item_idx, (item+1) % n_items] = 0.5  # 相邻items有部分激活
```

### 3. 不同的读出方式

```python
# 使用单输出（连续值）而非两个输出
Nout = 1
Y[response_idx, 0] = 1.0 if is_match else -1.0
```

## 调试建议

1. **先训练简单版本**：
   - 从单区域、小set size (如2)开始
   - 使用固定的延迟时长
   - 确保能达到高准确率

2. **检查trial生成**：
   ```python
   # 生成几个试验并可视化
   model = Model('examples/models/sternberg_1area.py')
   trial = model.m.generate_trial(np.random.RandomState(0), 10, {'name': 'test'})
   print(trial['info'])
   import matplotlib.pyplot as plt
   plt.imshow(trial['inputs'].T, aspect='auto')
   plt.show()
   ```

3. **监控训练过程**：
   - 观察损失是否下降
   - 检查不同条件下的准确率
   - 可视化网络活动

4. **常见问题**：
   - 如果准确率在50%：网络可能没有学到任务规则
   - 如果在编码期就有输出：增大mask的权重
   - 如果延迟期活动消失：增加循环连接强度或减少噪声

## 进一步扩展

1. **添加位置信息**：编码items的呈现顺序
2. **Serial vs. Parallel搜索**：研究搜索策略
3. **分心任务**：在延迟期加入其他任务
4. **多模态**：同时使用视觉和听觉刺激
5. **层级结构**：实现更复杂的类别记忆

## 参考文件

- `examples/models/sternberg_1area.py` - 单区域实现
- `examples/models/sternberg_3areas.py` - 三区域实现
- `pycog/tasktools.py` - 任务工具函数
- `pycog/model.py` - 模型基类
- `examples/models/2020-04-10_cb_simple_3areas.py` - 原始三区域示例

## 总结

改造这个仓库实现Sternberg任务的核心步骤：

1. ✅ 创建新的模型文件（如`sternberg_1area.py`）
2. ✅ 定义任务参数（set sizes, n_items, timing）
3. ✅ 设置网络结构（Nin, N, Nout, 区域划分）
4. ✅ 构建连接矩阵（Cin, Crec, Cout）
5. ✅ 实现`generate_trial`函数（生成输入和目标输出）
6. ✅ 实现`performance`函数（评估准确率）
7. ✅ 训练网络并分析结果

祝训练顺利！如有问题，请查看示例文件或联系原作者。


