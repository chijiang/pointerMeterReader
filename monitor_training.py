#!/usr/bin/env python3
"""
训练进度监控脚本
"""

import os
import time
import subprocess
from pathlib import Path


def check_training_status():
    """检查训练状态"""
    
    # 检查训练进程
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        training_processes = []
        for line in lines:
            if 'train_segmentation.py' in line and 'grep' not in line:
                training_processes.append(line.strip())
        
        if training_processes:
            print("🟢 训练正在运行中:")
            for proc in training_processes:
                parts = proc.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                print(f"   PID: {pid}, CPU: {cpu}%, 内存: {mem}%")
        else:
            print("🔴 没有发现训练进程")
            return False
            
    except Exception as e:
        print(f"检查进程失败: {e}")
        return False
    
    # 检查日志文件
    log_file = Path("outputs/segmentation/logs/training.log")
    if log_file.exists():
        stat = log_file.stat()
        size = stat.st_size
        mtime = time.ctime(stat.st_mtime)
        print(f"\n📝 训练日志:")
        print(f"   文件大小: {size} 字节")
        print(f"   最后修改: {mtime}")
        
        # 读取最后几行
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    print("   最新日志:")
                    for line in lines[-5:]:  # 显示最后5行
                        print(f"     {line.strip()}")
        except Exception as e:
            print(f"   读取日志失败: {e}")
    else:
        print("🔴 训练日志文件不存在")
    
    # 检查checkpoint文件
    checkpoint_dir = Path("outputs/segmentation/checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            print(f"\n💾 已保存的checkpoints: {len(checkpoints)} 个")
            # 显示最新的checkpoint
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            mtime = time.ctime(latest.stat().st_mtime)
            print(f"   最新: {latest.name} ({mtime})")
        else:
            print("\n💾 尚未保存checkpoint")
    
    # 检查TensorBoard事件文件
    tb_dir = Path("outputs/segmentation/logs")
    if tb_dir.exists():
        tb_files = list(tb_dir.glob("events.out.tfevents.*"))
        if tb_files:
            latest_tb = max(tb_files, key=lambda p: p.stat().st_mtime)
            size = latest_tb.stat().st_size
            mtime = time.ctime(latest_tb.stat().st_mtime)
            print(f"\n📊 TensorBoard事件文件:")
            print(f"   最新: {latest_tb.name}")
            print(f"   大小: {size} 字节, 最后修改: {mtime}")
    
    return True


def main():
    print("🔍 分割训练进度监控")
    print("=" * 50)
    
    if check_training_status():
        print("\n✅ 训练系统运行正常")
        print("\n💡 提示:")
        print("   - 训练进程正在后台运行")
        print("   - 可以使用 tensorboard --logdir outputs/segmentation/logs 查看详细进度")
        print("   - 训练日志保存在 outputs/segmentation/logs/training.log")
        print("   - 按 Ctrl+C 退出监控（不会影响训练）")
        
        try:
            print("\n⏰ 持续监控中（每30秒刷新一次）...")
            while True:
                time.sleep(30)
                print("\n" + "=" * 50)
                print(f"刷新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)
                if not check_training_status():
                    print("❌ 训练进程已停止")
                    break
        except KeyboardInterrupt:
            print("\n\n👋 退出监控")
    else:
        print("\n❌ 训练未在运行")


if __name__ == "__main__":
    main() 