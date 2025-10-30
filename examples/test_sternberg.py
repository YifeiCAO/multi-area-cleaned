#!/usr/bin/env python
"""
测试Sternberg任务的trial生成和可视化

使用方法:
    python examples/test_sternberg.py sternberg_1area
    python examples/test_sternberg.py sternberg_3areas
"""
from __future__ import division
from __future__ import print_function

import sys
import os
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from pycog import Model

def visualize_trial(trial, title="Sternberg Trial"):
    """
    可视化一个Sternberg trial的输入、输出和掩码
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # 输入
    ax = axes[0]
    im = ax.imshow(trial['inputs'].T, aspect='auto', cmap='Blues', interpolation='nearest')
    ax.set_ylabel('Input Channel')
    ax.set_title(f'{title} - Inputs')
    plt.colorbar(im, ax=ax)
    
    # 添加epoch标记
    if 'epochs' in trial:
        epochs = trial['epochs']
        dt = trial['info']['dt']
        y_max = trial['inputs'].shape[1]
        
        for name, val in epochs.items():
            if name == 'T':
                continue
            start, end = val
            start_idx = int(start / dt)
            end_idx = int(end / dt)
            ax.axvline(start_idx, color='red', linestyle='--', alpha=0.5)
            ax.text(start_idx, y_max * 0.95, name, rotation=90, 
                       verticalalignment='top', fontsize=8)
    
    # 目标输出
    if 'outputs' in trial:
        ax = axes[1]
        im = ax.imshow(trial['outputs'].T, aspect='auto', cmap='RdYlGn', 
                      interpolation='nearest', vmin=0, vmax=1)
        ax.set_ylabel('Output Channel')
        ax.set_title('Target Outputs')
        plt.colorbar(im, ax=ax)
    
    # 掩码
    if 'mask' in trial:
        ax = axes[2]
        im = ax.imshow(trial['mask'].T, aspect='auto', cmap='Greys', 
                      interpolation='nearest', vmin=0, vmax=1)
        ax.set_ylabel('Output Channel')
        ax.set_xlabel('Time Step')
        ax.set_title('Mask')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


def print_trial_info(trial):
    """
    打印trial的详细信息
    """
    print("\n" + "="*60)
    print("Trial Information")
    print("="*60)
    
    info = trial['info']
    
    if info['catch']:
        print("Trial Type: CATCH TRIAL")
    else:
        print("Trial Type: NORMAL TRIAL")
        print(f"Set Size: {info['set_size']}")
        print(f"Memory Set: {info['memory_set']}")
        print(f"Probe Item: {info['probe_item']}")
        print(f"Is Match: {'YES' if info['is_match'] else 'NO'}")
        print(f"Correct Choice: {info['choice']}")
        print(f"Delay Duration: {info['delay_duration']} ms")
    
    print(f"\nTiming (ms):")
    for name, val in sorted(info['epochs'].items()):
        if name == 'T':
            continue
        start, end = val
        print(f"  {name:15s}: {start:6d} - {end:6d} ({end-start:4d} ms)")
    print(f"  {'Total':15s}: {info['epochs']['T']} ms")
    
    print(f"\nInput shape: {trial['inputs'].shape}")
    if 'outputs' in trial:
        print(f"Output shape: {trial['outputs'].shape}")
    
    print("="*60 + "\n")


def test_model(model_name):
    """
    测试指定的模型
    """
    # 加载模型
    modelfile = f'examples/models/{model_name}.py'
    
    if not os.path.exists(modelfile):
        print(f"Error: Model file {modelfile} not found!")
        return
    
    print(f"Loading model: {modelfile}")
    model = Model(modelfile)
    
    # 测试参数
    print(f"\nModel Parameters:")
    print(f"  Nin  = {model.m.Nin}")
    print(f"  N    = {model.m.N}")
    print(f"  Nout = {model.m.Nout}")
    print(f"  dt   = {model.m.dt} ms")
    
    if hasattr(model.m, 'set_sizes'):
        print(f"  Set Sizes = {model.m.set_sizes}")
        print(f"  N Items   = {model.m.n_items}")
    
    # 生成几个示例trials
    rng = np.random.RandomState(42)
    
    # Normal trial - match
    print("\n" + "#"*60)
    print("# Test 1: Normal Trial (Match)")
    print("#"*60)
    params = {
        'name': 'test',
        'target_output': True,
        'set_size': model.m.set_sizes[0],
        'is_match': 1
    }
    trial1 = model.m.generate_trial(rng, model.m.dt, params)
    print_trial_info(trial1)
    fig1 = visualize_trial(trial1, "Test 1: Match Trial")
    
    # Normal trial - non-match
    print("\n" + "#"*60)
    print("# Test 2: Normal Trial (Non-Match)")
    print("#"*60)
    params = {
        'name': 'test',
        'target_output': True,
        'set_size': model.m.set_sizes[-1],  # largest set size
        'is_match': 0
    }
    trial2 = model.m.generate_trial(rng, model.m.dt, params)
    print_trial_info(trial2)
    fig2 = visualize_trial(trial2, "Test 2: Non-Match Trial")
    
    # Catch trial
    print("\n" + "#"*60)
    print("# Test 3: Catch Trial")
    print("#"*60)
    params = {
        'name': 'test',
        'target_output': True,
        'catch': True
    }
    trial3 = model.m.generate_trial(rng, model.m.dt, params)
    print_trial_info(trial3)
    fig3 = visualize_trial(trial3, "Test 3: Catch Trial")
    
    # 检查连接矩阵
    print("\n" + "#"*60)
    print("# Connectivity Matrices")
    print("#"*60)
    print(f"Cin shape:  {model.m.Cin.shape}")
    print(f"Crec shape: {model.m.Crec.shape}")
    print(f"Cout shape: {model.m.Cout.shape}")
    
    print(f"\nCin statistics:")
    print(f"  Non-zero entries: {np.sum(model.m.Cin > 0)}/{model.m.Cin.size}")
    print(f"  Mean: {np.mean(model.m.Cin):.4f}")
    
    print(f"\nCrec statistics:")
    print(f"  Non-zero entries: {np.sum(model.m.Crec > 0)}/{model.m.Crec.size}")
    print(f"  Mean: {np.mean(model.m.Crec):.4f}")
    print(f"  Max: {np.max(model.m.Crec):.4f}")
    
    print(f"\nCout statistics:")
    print(f"  Non-zero entries: {np.sum(model.m.Cout > 0)}/{model.m.Cout.size}")
    print(f"  Mean: {np.mean(model.m.Cout):.4f}")
    
    # 可视化连接矩阵
    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Cin
    im = axes[0].imshow(model.m.Cin, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[0].set_title('Input Connectivity (Cin)')
    axes[0].set_xlabel('Input')
    axes[0].set_ylabel('Neuron')
    plt.colorbar(im, ax=axes[0])
    
    # Crec
    im = axes[1].imshow(model.m.Crec, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[1].set_title('Recurrent Connectivity (Crec)')
    axes[1].set_xlabel('From Neuron')
    axes[1].set_ylabel('To Neuron')
    plt.colorbar(im, ax=axes[1])
    
    # Add area boundaries if multi-area
    if hasattr(model.m, 'EXC_SENSORY') and hasattr(model.m, 'EXC_MEMORY'):
        n_sensory = len(model.m.EXC_SENSORY) + len(model.m.INH_SENSORY)
        axes[1].axhline(n_sensory, color='yellow', linestyle='--', linewidth=2)
        axes[1].axvline(n_sensory, color='yellow', linestyle='--', linewidth=2)
        
        if hasattr(model.m, 'EXC_DECISION'):
            n_memory = len(model.m.EXC_MEMORY) + len(model.m.INH_MEMORY)
            axes[1].axhline(n_sensory + n_memory, color='yellow', linestyle='--', linewidth=2)
            axes[1].axvline(n_sensory + n_memory, color='yellow', linestyle='--', linewidth=2)
    
    # Cout
    im = axes[2].imshow(model.m.Cout, aspect='auto', cmap='Greens', interpolation='nearest')
    axes[2].set_title('Output Connectivity (Cout)')
    axes[2].set_xlabel('Neuron')
    axes[2].set_ylabel('Output')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    print("\n" + "="*60)
    print("All tests completed! Close the plot windows to exit.")
    print("="*60)
    
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_sternberg.py <model_name>")
        print("Example: python test_sternberg.py sternberg_1area")
        sys.exit(1)
    
    model_name = sys.argv[1]
    test_model(model_name)


