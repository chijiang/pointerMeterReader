#!/usr/bin/env python3
"""
液晶数字表检测演示脚本

这个脚本展示了完整的训练和推理流程：
1. 验证数据集
2. 训练模型（小规模演示）
3. 推理测试
4. 结果展示
"""

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

def get_project_root():
    """获取项目根目录"""
    current_dir = Path.cwd()
    if current_dir.name == "demo":
        return current_dir.parent.parent.parent
    elif current_dir.name == "digital_meter_detection":
        return current_dir.parent.parent
    elif current_dir.name == "scripts":
        return current_dir.parent
    else:
        return current_dir

def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def print_step(step: str, description: str):
    """打印步骤"""
    print(f"\n📋 步骤 {step}: {description}")
    print("-" * 50)

def run_command(command: str, description: str = "") -> bool:
    """运行命令并显示结果"""
    if description:
        print(f"🔄 {description}")
    
    print(f"💻 执行命令: {command}")
    
    result = os.system(command)
    
    if result == 0:
        print("✅ 命令执行成功")
        return True
    else:
        print(f"❌ 命令执行失败，退出代码: {result}")
        return False

def check_requirements():
    """检查必要的依赖"""
    print_step("0", "检查环境依赖")
    
    required_packages = ['ultralytics', 'opencv-python', 'matplotlib', 'numpy', 'pyyaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 未安装")
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖已安装")
    return True

def validate_dataset():
    """验证数据集"""
    print_step("1", "验证数据集")
    
    project_root = get_project_root()
    dataset_path = project_root / "data" / "digital_meters"
    validation_script = project_root / "scripts" / "digital_meter_detection" / "validation" / "validate_digital_meter_dataset.py"
    
    if not dataset_path.exists():
        print(f"❌ 数据集目录不存在: {dataset_path}")
        return False
    
    return run_command(
        f"cd {project_root} && python {validation_script} --dataset data/digital_meters",
        "验证液晶数字表数据集"
    )

def train_demo_model():
    """训练演示模型（小规模）"""
    print_step("2", "训练演示模型")
    
    project_root = get_project_root()
    training_script = project_root / "scripts" / "digital_meter_detection" / "training" / "train_digital_meter_yolo.py"
    
    # 创建演示配置文件（训练轮数较少，用于快速演示）
    demo_config_path = project_root / "config" / "digital_meter_demo_config.yaml"
    original_config_path = project_root / "config" / "digital_meter_yolo_config.yaml"
    
    # 复制原配置文件并修改训练轮数
    if original_config_path.exists():
        shutil.copy(original_config_path, demo_config_path)
        
        # 修改配置以缩短演示时间
        with open(demo_config_path, 'r') as f:
            content = f.read()
        
        # 减少训练轮数用于演示
        content = content.replace("epochs: 200", "epochs: 20")
        content = content.replace("save_period: 20", "save_period: 5")
        content = content.replace("patience: 50", "patience: 10")
        content = content.replace("experiment_name: 'digital_meter_detection'", 
                                "experiment_name: 'digital_meter_demo'")
        
        with open(demo_config_path, 'w') as f:
            f.write(content)
        
        print(f"📝 创建演示配置文件: {demo_config_path}")
        print("⚡ 演示模式：训练轮数设置为20轮（正式训练建议200轮）")
    
    return run_command(
        f"cd {project_root} && python {training_script} --config config/digital_meter_demo_config.yaml",
        "开始训练演示模型（20轮）"
    )

def find_best_model():
    """查找最新训练的最佳模型"""
    project_root = get_project_root()
    runs_dir = project_root / "runs" / "detect"
    if not runs_dir.exists():
        return None
    
    # 查找最新的训练结果目录
    demo_dirs = list(runs_dir.glob("digital_meter_demo_*"))
    if not demo_dirs:
        return None
    
    # 按修改时间排序，取最新的
    latest_dir = max(demo_dirs, key=lambda p: p.stat().st_mtime)
    best_model = latest_dir / "weights" / "best.pt"
    
    if best_model.exists():
        return str(best_model)
    
    return None

def test_inference():
    """测试推理功能"""
    print_step("3", "测试模型推理")
    
    project_root = get_project_root()
    inference_script = project_root / "scripts" / "digital_meter_detection" / "inference" / "digital_meter_inference.py"
    
    # 查找最佳模型
    best_model = find_best_model()
    if not best_model:
        print("❌ 未找到训练好的模型")
        return False
    
    print(f"🎯 使用模型: {best_model}")
    
    # 选择一张测试图像
    test_images = list((project_root / "data" / "digital_meters" / "images").glob("*.jpg"))
    if not test_images:
        print("❌ 未找到测试图像")
        return False
    
    test_image = test_images[0]  # 选择第一张图像
    print(f"🖼️  测试图像: {test_image}")
    
    # 创建演示输出目录
    demo_output = f"outputs/demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 计算相对于项目根目录的路径
    model_rel_path = Path(best_model).relative_to(project_root)
    image_rel_path = test_image.relative_to(project_root)
    
    return run_command(
        f"cd {project_root} && python {inference_script} --model {model_rel_path} --input {image_rel_path} --output {demo_output}",
        "执行液晶表检测推理"
    )

def show_results():
    """展示结果"""
    print_step("4", "展示训练结果")
    
    project_root = get_project_root()
    
    # 显示训练结果目录
    runs_dir = project_root / "runs" / "detect"
    if runs_dir.exists():
        demo_dirs = list(runs_dir.glob("digital_meter_demo_*"))
        if demo_dirs:
            latest_dir = max(demo_dirs, key=lambda p: p.stat().st_mtime)
            print(f"📂 训练结果目录: {latest_dir}")
            
            # 列出主要文件
            important_files = [
                "weights/best.pt",
                "weights/last.pt", 
                "results.png",
                "confusion_matrix.png",
                "training_summary.md"
            ]
            
            print("📄 重要文件:")
            for file_path in important_files:
                full_path = latest_dir / file_path
                if full_path.exists():
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ❌ {file_path} (不存在)")
    
    # 显示推理结果目录
    outputs_dir = project_root / "outputs"
    if outputs_dir.exists():
        demo_outputs = list(outputs_dir.glob("demo_results_*"))
        if demo_outputs:
            latest_output = max(demo_outputs, key=lambda p: p.stat().st_mtime)
            print(f"\n📂 推理结果目录: {latest_output}")
            
            # 检查ROI文件
            roi_dir = latest_output / "rois"
            if roi_dir.exists():
                roi_files = list(roi_dir.glob("*.jpg"))
                print(f"🎯 提取的ROI区域: {len(roi_files)} 个")
            
            # 检查检测结果
            result_files = list(latest_output.glob("*_detection_*.jpg"))
            if result_files:
                print(f"🖼️  检测结果可视化: {len(result_files)} 个")

def cleanup_demo_files():
    """清理演示文件"""
    print_step("5", "清理演示文件（可选）")
    
    project_root = get_project_root()
    
    response = input("是否要清理演示生成的文件？(y/N): ").strip().lower()
    
    if response == 'y':
        # 删除演示配置文件
        demo_config = project_root / "config" / "digital_meter_demo_config.yaml"
        if demo_config.exists():
            demo_config.unlink()
            print(f"🗑️  删除: {demo_config}")
        
        # 可选：删除演示训练结果（用户确认）
        runs_dir = project_root / "runs" / "detect"
        if runs_dir.exists():
            demo_dirs = list(runs_dir.glob("digital_meter_demo_*"))
            if demo_dirs:
                response2 = input(f"发现 {len(demo_dirs)} 个演示训练目录，是否删除？(y/N): ").strip().lower()
                if response2 == 'y':
                    for demo_dir in demo_dirs:
                        shutil.rmtree(demo_dir)
                        print(f"🗑️  删除: {demo_dir}")
        
        print("✅ 清理完成")
    else:
        print("📁 保留演示文件")

def main():
    """主演示流程"""
    print_header("液晶数字表检测系统演示")
    
    print("🎪 这个演示将展示完整的液晶表检测流程:")
    print("   1. 验证数据集格式和完整性")
    print("   2. 训练YOLO v10模型（演示版：20轮）")
    print("   3. 使用训练好的模型进行推理")
    print("   4. 展示检测结果和ROI提取")
    print()
    print("⏱️  预计耗时: 10-30分钟（取决于硬件性能）")
    print("🖥️  推荐使用GPU或Apple Silicon芯片加速")
    
    response = input("\n是否继续演示？(Y/n): ").strip().lower()
    if response == 'n':
        print("👋 演示取消")
        return
    
    start_time = time.time()
    
    try:
        # 步骤0: 检查环境
        if not check_requirements():
            print("❌ 环境检查失败，请安装必要的依赖包")
            return
        
        # 步骤1: 验证数据集
        if not validate_dataset():
            print("❌ 数据集验证失败")
            return
        
        # 步骤2: 训练模型
        print(f"\n⚠️  开始训练演示模型...")
        print("📝 注意：这是演示版本，只训练20轮")
        print("🚀 正式使用时建议训练200轮以获得更好的性能")
        
        train_start = time.time()
        if not train_demo_model():
            print("❌ 模型训练失败")
            return
        train_time = time.time() - train_start
        print(f"⏱️  训练耗时: {train_time/60:.1f} 分钟")
        
        # 步骤3: 推理测试
        if not test_inference():
            print("❌ 推理测试失败")
            return
        
        # 步骤4: 展示结果
        show_results()
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        print_header("演示完成")
        print(f"🎉 恭喜！液晶数字表检测系统演示成功完成")
        print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
        print()
        print("📋 接下来您可以:")
        print("   1. 查看训练结果和模型性能")
        print("   2. 使用更多图像测试推理功能")
        print("   3. 调整配置参数重新训练")
        print("   4. 集成到完整的液晶表识别pipeline中")
        print()
        print("📖 更多信息请参考: DIGITAL_METER_DETECTION_README.md")
        
        # 步骤5: 清理（可选）
        cleanup_demo_files()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 