# Sternberg工作记忆任务实现 📝

本文档概述了将multi-area RNN改造为Sternberg工作记忆任务的所有新增文件。

---

## 📦 新增文件清单

### 1️⃣ 模型定义文件

| 文件 | 说明 | 推荐指数 |
|------|------|---------|
| `examples/models/sternberg_1area.py` | 单区域RNN模型 | ⭐⭐⭐⭐⭐ |
| `examples/models/sternberg_3areas.py` | 三区域RNN模型 (Sensory→Memory→Decision) | ⭐⭐⭐⭐ |

### 2️⃣ 测试和可视化工具

| 文件 | 说明 | 用途 |
|------|------|------|
| `examples/test_sternberg.py` | 完整测试脚本 | 验证任务设置，可视化多个trials |
| `examples/visualize_single_trial.py` | 单trial可视化 | 快速查看一个trial的详细结构 |

### 3️⃣ 文档

| 文件 | 说明 | 适合对象 |
|------|------|---------|
| `STERNBERG_QUICKSTART.md` | 快速入门指南 | 想快速开始的用户 |
| `STERNBERG_TASK_GUIDE.md` | 详细实现指南 | 想深入理解的用户 |
| `CHANGES_SUMMARY.md` | 改动总结 | 想了解所有改动的用户 |
| `README_STERNBERG.md` (本文件) | 文件清单 | 所有用户 |

---

## 🚀 三步快速开始

### Step 1: 可视化任务 (1分钟)

```bash
cd /Users/yifei/Desktop/UCLA/multi-area-cleaned

# 快速可视化一个trial
python examples/visualize_single_trial.py

# 或者完整测试
python examples/test_sternberg.py sternberg_1area
```

### Step 2: 训练模型 (30-60分钟)

```bash
# 单区域模型（推荐先试这个）
python examples/do.py examples/models/sternberg_1area.py train

# 三区域模型
python examples/do.py examples/models/sternberg_3areas.py train
```

### Step 3: 分析结果

使用`sims/`目录下的现有分析工具分析训练好的网络。

---

## 📊 任务说明

### Sternberg工作记忆任务

**任务流程**:
```
Fixation → Item1, Item2, ... → Delay → Probe → Response
 (200ms)    (300ms × N items)   (1-2s)  (500ms)  (500ms)
```

**任务目标**: 
判断探测item (probe) 是否在之前呈现的记忆集合中。

**网络输入**: 
- 8个通道代表8个可能的items
- 1个通道表示probe信号

**网络输出**: 
- 2个通道: [Non-match, Match]

**条件**: 
- 记忆集合大小: 2, 4, 6 items
- 类型: Match vs. Non-match
- 总计: 6个条件

---

## 🏗️ 网络结构

### 单区域模型 (`sternberg_1area.py`)

```
Input (9) → RNN (300 neurons, fully connected) → Output (2)
```

**特点**:
- 简单直接
- 训练快速
- 适合快速测试

### 三区域模型 (`sternberg_3areas.py`)

```
Input (9) → Sensory (100) → Memory (100) → Decision (100) → Output (2)
              ↓              ↓               ↓
           Encoding      Maintenance     Comparison
```

**特点**:
- 功能分离
- 层级结构
- 适合深入分析

**连接模式**:
- 前馈连接: 10% 概率
- 反馈连接: 5% 概率
- 区域内全连接

---

## 📈 预期结果

### 性能指标
- **目标准确率**: >75%
- **典型表现**: 80-90%
- **训练时间**: 30-90分钟

### 网络活动特征
1. **编码期**: 顺序的item表征
2. **延迟期**: 持续活动（记忆维持）
3. **Probe期**: 比较过程激活
4. **响应期**: 明确的match/non-match决策

### Set Size效应
更大的记忆集合 → 略低的准确率（符合人类行为）

---

## 🔧 自定义和调整

### 调整任务难度

编辑模型文件 (`sternberg_1area.py` 或 `sternberg_3areas.py`):

```python
# 简化任务（更快训练）
set_sizes = [2, 4]
delay_duration = [1000, 1000]
N = 200

# 增加难度
set_sizes = [2, 4, 6, 8]
n_items = 10
delay_duration = [2000, 5000]
```

### 调整网络结构

```python
# 改变神经元数量
N = 500

# 改变连接强度
ff_prop = 0.2  # 前馈连接概率
fb_prop = 0.1  # 反馈连接概率
```

### 调整训练参数

```python
# 学习率
learning_rate = 1e-4

# 性能目标
TARGET_PERFORMANCE = 85
```

---

## 🐛 故障排除

### 问题: "No module named pycog"

**解决**:
```bash
add2virtualenv /Users/yifei/Desktop/UCLA/multi-area-cleaned
add2virtualenv /Users/yifei/Desktop/UCLA/multi-area-cleaned/pycog
```

### 问题: 准确率停在50%

**可能原因**: 网络没有学习，只是随机猜测

**解决方法**:
1. 检查输出和mask设置
2. 降低学习率
3. 增加训练时间
4. 减小正则化强度

### 问题: 训练很慢

**解决方法**:
1. 减少神经元数量 (`N = 200`)
2. 减少条件数 (`set_sizes = [2, 4]`)
3. 使用GPU (如果可用)
4. 缩短延迟时长

### 问题: 内存不足

**解决方法**:
1. 减少batch size
2. 减少神经元数量
3. 缩短trial长度

---

## 📚 详细文档链接

| 问题 | 查看文档 |
|------|---------|
| 如何快速开始？ | `STERNBERG_QUICKSTART.md` |
| 如何理解实现细节？ | `STERNBERG_TASK_GUIDE.md` |
| 做了哪些改动？ | `CHANGES_SUMMARY.md` |
| 有哪些文件？ | `README_STERNBERG.md` (本文件) |

---

## 🔍 文件用途速查

### 想快速看看任务长什么样？
→ 运行 `python examples/visualize_single_trial.py`

### 想测试任务设置是否正确？
→ 运行 `python examples/test_sternberg.py sternberg_1area`

### 想开始训练？
→ 运行 `python examples/do.py examples/models/sternberg_1area.py train`

### 想理解代码如何实现？
→ 阅读 `STERNBERG_TASK_GUIDE.md`

### 想了解参数如何调整？
→ 阅读 `STERNBERG_QUICKSTART.md`

---

## ✅ 验证清单

在开始研究之前，请确认：

- [ ] 已安装所有依赖 (`pip install -r requirements.txt`)
- [ ] pycog已添加到Python路径
- [ ] 可视化脚本能正常运行
- [ ] 理解Sternberg任务的基本流程
- [ ] 知道如何调整参数

---

## 🎯 建议的学习路径

1. **Day 1**: 运行可视化脚本，理解任务结构
2. **Day 2**: 训练单区域模型，观察学习曲线
3. **Day 3**: 训练三区域模型，比较性能
4. **Day 4+**: 深入分析，研究区域间信息流动

---

## 📞 需要帮助？

1. 先查看相应的文档（见上方"详细文档链接"）
2. 检查测试脚本的输出
3. 参考原始模型文件 (`examples/models/2020-04-10_cb_simple_*.py`)
4. 查看 `pycog/tasktools.py` 中的工具函数

---

## 🌟 研究方向建议

### 基础分析
- [ ] 训练曲线和收敛性
- [ ] 不同条件下的准确率
- [ ] Set size对性能的影响

### 进阶分析
- [ ] 网络活动的PCA/dPCA分析
- [ ] 延迟期持续活动的机制
- [ ] 区域间的信息流动

### 深入研究
- [ ] 工作记忆容量的神经机制
- [ ] 与其他任务的比较
- [ ] 多区域的功能分工

---

## 📝 总结

**已创建**: 
- ✅ 2个可用的模型文件
- ✅ 2个测试/可视化脚本  
- ✅ 4个详细文档

**可以做**:
- ✅ 立即开始训练
- ✅ 自定义任务参数
- ✅ 使用现有分析工具

**下一步**:
1. 运行 `python examples/visualize_single_trial.py`
2. 训练模型
3. 分析结果

---

**创建日期**: 2025-10-21  
**状态**: ✅ 就绪可用  
**版本**: 1.0

祝研究顺利！🧠✨



