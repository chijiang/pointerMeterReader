#!/usr/bin/env python3
"""
分割模型端到端训练流水线
从COCO数据集到ONNX模型的完整流程

使用方法:
python scripts/segmentation_pipeline.py \
    --coco-json data/coco_data/result_coco.json \
    --coco-images data/coco_data/images \
    --config config/segmentation_config.yaml \
    --output-onnx models/segmentation/segmentation_model.onnx

或使用默认参数:
python scripts/segmentation_pipeline.py
"""

import os
import sys
import argparse
import yaml
import logging
import shutil
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# 导入数据转换模块
from tools.data_preparation.convert_coco_to_segmentation import convert_coco_to_segmentation


class SegmentationPipeline:
    """分割模型训练流水线"""
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置日志
        self._setup_logging()
        
        # 加载配置
        self.config = self._load_config()
        
        # 设置临时目录
        self.temp_dir = Path(args.temp_dir) / f'pipeline_{self.timestamp}'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("=" * 80)
        logging.info("分割模型端到端训练流水线")
        logging.info("=" * 80)
        logging.info(f"COCO JSON: {args.coco_json}")
        logging.info(f"COCO Images: {args.coco_images}")
        logging.info(f"配置文件: {args.config}")
        logging.info(f"输出ONNX: {args.output_onnx}")
        logging.info(f"临时目录: {self.temp_dir}")
        logging.info("=" * 80)
    
    def _setup_logging(self):
        """设置日志"""
        log_level = logging.INFO if not self.args.debug else logging.DEBUG
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
            ]
        )
    
    def _load_config(self):
        """加载配置文件"""
        logging.info(f"加载配置文件: {self.args.config}")
        
        with open(self.args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def step1_convert_data(self):
        """步骤1: 转换COCO数据为分割格式"""
        logging.info("\n" + "=" * 80)
        logging.info("步骤 1/4: 转换COCO数据为分割格式")
        logging.info("=" * 80)
        
        # 检查输入文件
        if not os.path.exists(self.args.coco_json):
            raise FileNotFoundError(f"COCO JSON文件不存在: {self.args.coco_json}")
        
        if not os.path.exists(self.args.coco_images):
            raise FileNotFoundError(f"COCO images目录不存在: {self.args.coco_images}")
        
        # 设置输出目录
        if self.args.segmentation_dir:
            seg_output_dir = Path(self.args.segmentation_dir)
        else:
            seg_output_dir = self.temp_dir / 'segmentation_data'
        
        seg_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换数据
        logging.info(f"开始转换数据...")
        convert_coco_to_segmentation(
            coco_json_path=self.args.coco_json,
            coco_images_dir=self.args.coco_images,
            output_dir=seg_output_dir,
            train_ratio=self.args.train_ratio
        )
        
        # 更新配置中的数据路径
        self.config['data']['root_dir'] = str(seg_output_dir)
        # 转换脚本生成的是 SegmentationClass，不是 SegmentationClass_unified
        self.config['data']['mask_dir'] = 'SegmentationClass'
        # 更新 split_dir 路径
        self.config['data']['split_dir'] = 'ImageSets/Segmentation'
        self.segmentation_dir = seg_output_dir
        
        logging.info(f"✓ 数据转换完成，输出到: {seg_output_dir}")
        logging.info(f"  - 图像目录: {self.config['data']['image_dir']}")
        logging.info(f"  - 掩码目录: {self.config['data']['mask_dir']}")
        logging.info(f"  - 划分目录: {self.config['data']['split_dir']}")
        
        return seg_output_dir
    
    def step2_update_config(self):
        """步骤2: 更新训练配置"""
        logging.info("\n" + "=" * 80)
        logging.info("步骤 2/4: 更新训练配置")
        logging.info("=" * 80)
        
        # 创建临时配置文件
        temp_config_path = self.temp_dir / 'training_config.yaml'
        
        # 更新配置
        if self.args.epochs:
            self.config['training']['epochs'] = self.args.epochs
        
        if self.args.batch_size:
            self.config['data']['batch_size'] = self.args.batch_size
        
        if self.args.learning_rate:
            self.config['training']['learning_rate'] = float(self.args.learning_rate)
        
        # 更新保存路径
        output_base = self.temp_dir / 'training_output'
        self.config['save']['checkpoint_dir'] = str(output_base / 'checkpoints')
        self.config['save']['log_dir'] = str(output_base / 'logs')
        self.config['save']['best_model_path'] = str(output_base / 'best_model.pth')
        
        self.config['visualization']['prediction_dir'] = str(output_base / 'predictions')
        
        # 保存临时配置
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
        
        self.temp_config_path = temp_config_path
        
        logging.info(f"✓ 配置更新完成")
        logging.info(f"  - Epochs: {self.config['training']['epochs']}")
        logging.info(f"  - Batch size: {self.config['data']['batch_size']}")
        logging.info(f"  - Learning rate: {self.config['training']['learning_rate']}")
        logging.info(f"  - 配置文件: {temp_config_path}")
        
        return temp_config_path
    
    def step3_train_model(self):
        """步骤3: 训练模型"""
        logging.info("\n" + "=" * 80)
        logging.info("步骤 3/4: 训练分割模型")
        logging.info("=" * 80)
        
        # 动态导入训练模块（避免在不训练时加载所有依赖）
        from scripts.train_segmentation import SegmentationTrainer
        
        # 创建训练器
        trainer = SegmentationTrainer(self.config)
        
        # 开始训练
        logging.info("开始训练...")
        trainer.train()
        
        # 保存训练信息
        training_info = {
            'timestamp': self.timestamp,
            'config': self.config,
            'best_miou': float(trainer.best_miou),
            'best_model_path': self.config['save']['best_model_path']
        }
        
        info_path = self.temp_dir / 'training_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        self.best_model_path = self.config['save']['best_model_path']
        self.best_miou = trainer.best_miou
        
        logging.info(f"✓ 训练完成")
        logging.info(f"  - 最佳mIoU: {self.best_miou:.4f}")
        logging.info(f"  - 最佳模型: {self.best_model_path}")
        
        return self.best_model_path
    
    def step4_export_onnx(self):
        """步骤4: 导出ONNX模型"""
        logging.info("\n" + "=" * 80)
        logging.info("步骤 4/4: 导出ONNX模型")
        logging.info("=" * 80)
        
        # 检查最佳模型是否存在
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(f"找不到训练好的模型: {self.best_model_path}")
        
        # 创建模型
        import torchvision.models.segmentation as segmentation
        
        model_config = self.config['model']
        num_classes = model_config['num_classes']
        
        if model_config['architecture'] == 'deeplabv3_resnet50':
            model = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
        elif model_config['architecture'] == 'deeplabv3_resnet101':
            model = segmentation.deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
        elif model_config['architecture'] == 'deeplabv3_mobilenet_v3_large':
            model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"不支持的模型架构: {model_config['architecture']}")
        
        # 获取设备
        if self.config['device'] == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
        elif self.config['device'].startswith('cuda') and torch.cuda.is_available():
            device = torch.device(self.config['device'])
        else:
            device = torch.device('cpu')
        
        model = model.to(device)
        
        # 加载权重
        logging.info(f"加载模型权重: {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path, map_location=device, weights_only=False)
        # 使用 strict=False 以忽略 aux_classifier（辅助分类器在推理时不需要）
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        # 创建示例输入
        input_size = self.config['data']['augmentation'].get('random_crop', [224, 224])
        example_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # 导出ONNX
        output_onnx = Path(self.args.output_onnx)
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        
        onnx_config = self.config.get('export', {}).get('onnx', {})
        
        logging.info(f"导出ONNX模型到: {output_onnx}")
        
        try:
            torch.onnx.export(
                model,
                example_input,
                str(output_onnx),
                export_params=True,
                opset_version=onnx_config.get('opset_version', 11),
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=onnx_config.get('dynamic_axes', {
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                })
            )
            logging.info(f"✓ ONNX模型导出成功: {output_onnx}")
            
            # 验证ONNX模型
            if self.args.verify_onnx:
                self._verify_onnx(output_onnx)
            
        except Exception as e:
            logging.error(f"✗ ONNX导出失败: {e}")
            raise
        
        return output_onnx
    
    def _verify_onnx(self, onnx_path):
        """验证ONNX模型"""
        logging.info("验证ONNX模型...")
        
        try:
            import onnx
            import onnxruntime as ort
            
            # 检查ONNX模型
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logging.info("  ✓ ONNX模型结构验证通过")
            
            # 测试推理
            ort_session = ort.InferenceSession(str(onnx_path))
            input_name = ort_session.get_inputs()[0].name
            
            input_size = self.config['data']['augmentation'].get('random_crop', [224, 224])
            test_input = torch.randn(1, 3, input_size[0], input_size[1]).numpy()
            
            outputs = ort_session.run(None, {input_name: test_input})
            logging.info(f"  ✓ ONNX推理测试通过，输出形状: {outputs[0].shape}")
            
        except ImportError as e:
            logging.warning(f"  ! 无法验证ONNX模型（缺少依赖）: {e}")
        except Exception as e:
            logging.error(f"  ✗ ONNX验证失败: {e}")
    
    def cleanup(self):
        """清理临时文件"""
        if self.args.keep_temp:
            logging.info(f"\n保留临时文件: {self.temp_dir}")
        else:
            if self.args.segmentation_dir:
                # 如果用户指定了分割数据目录，不删除它
                logging.info(f"保留分割数据: {self.segmentation_dir}")
            
            # 可以选择删除其他临时文件
            logging.info(f"临时文件保存在: {self.temp_dir}")
    
    def run(self):
        """运行完整流水线"""
        try:
            start_time = datetime.now()
            
            # 步骤1: 转换数据
            if not self.args.skip_conversion:
                self.step1_convert_data()
            else:
                logging.info("\n跳过数据转换步骤")
                if not self.args.segmentation_dir:
                    raise ValueError("跳过数据转换时必须指定 --segmentation-dir")
                self.segmentation_dir = Path(self.args.segmentation_dir)
                self.config['data']['root_dir'] = str(self.segmentation_dir)
                
                # 检测并使用正确的 mask 目录
                if (self.segmentation_dir / 'SegmentationClass_unified').exists():
                    self.config['data']['mask_dir'] = 'SegmentationClass_unified'
                    logging.info(f"  使用mask目录: SegmentationClass_unified")
                elif (self.segmentation_dir / 'SegmentationClass').exists():
                    self.config['data']['mask_dir'] = 'SegmentationClass'
                    logging.info(f"  使用mask目录: SegmentationClass")
                
                # 检测 split_dir
                if (self.segmentation_dir / 'ImageSets' / 'Segmentation').exists():
                    self.config['data']['split_dir'] = 'ImageSets/Segmentation'
                elif (self.segmentation_dir / 'ImageSets').exists():
                    self.config['data']['split_dir'] = 'ImageSets'
            
            # 步骤2: 更新配置
            self.step2_update_config()
            
            # 步骤3: 训练模型
            if not self.args.skip_training:
                self.step3_train_model()
            else:
                logging.info("\n跳过训练步骤")
                if not self.args.model_checkpoint:
                    raise ValueError("跳过训练时必须指定 --model-checkpoint")
                self.best_model_path = self.args.model_checkpoint
                self.config['save']['best_model_path'] = self.best_model_path
            
            # 步骤4: 导出ONNX
            self.step4_export_onnx()
            
            # 清理
            self.cleanup()
            
            # 总结
            end_time = datetime.now()
            duration = end_time - start_time
            
            logging.info("\n" + "=" * 80)
            logging.info("流水线执行完成!")
            logging.info("=" * 80)
            logging.info(f"总耗时: {duration}")
            logging.info(f"输出ONNX模型: {self.args.output_onnx}")
            if hasattr(self, 'best_miou'):
                logging.info(f"最佳mIoU: {self.best_miou:.4f}")
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error(f"\n流水线执行失败: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='分割模型端到端训练流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流水线
  python scripts/segmentation_pipeline.py \\
      --coco-json data/coco_data/result_coco.json \\
      --coco-images data/coco_data/images \\
      --config config/segmentation_config.yaml \\
      --output-onnx models/segmentation/model.onnx
  
  # 使用默认参数
  python scripts/segmentation_pipeline.py
  
  # 跳过数据转换（使用已有分割数据）
  python scripts/segmentation_pipeline.py \\
      --skip-conversion \\
      --segmentation-dir data/segmentation_new
  
  # 只导出ONNX（跳过训练）
  python scripts/segmentation_pipeline.py \\
      --skip-conversion \\
      --skip-training \\
      --model-checkpoint outputs/segmentation/best_model.pth \\
      --output-onnx models/segmentation/model.onnx
        """
    )
    
    # 输入数据
    parser.add_argument(
        '--coco-json',
        type=str,
        default='data/coco_data/result_coco.json',
        help='COCO格式的标注文件路径'
    )
    parser.add_argument(
        '--coco-images',
        type=str,
        default='data/coco_data/images',
        help='COCO图像目录路径'
    )
    
    # 配置
    parser.add_argument(
        '--config',
        type=str,
        default='config/segmentation_config.yaml',
        help='训练配置文件路径'
    )
    
    # 输出
    parser.add_argument(
        '--output-onnx',
        type=str,
        default='models/segmentation/segmentation_model.onnx',
        help='输出ONNX模型路径'
    )
    
    # 数据转换参数
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例（默认: 0.8）'
    )
    parser.add_argument(
        '--segmentation-dir',
        type=str,
        default=None,
        help='分割数据输出目录（默认: 临时目录）'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数（覆盖配置文件）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小（覆盖配置文件）'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='学习率（覆盖配置文件）'
    )
    
    # 流程控制
    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='跳过数据转换步骤'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='跳过训练步骤'
    )
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default=None,
        help='已训练模型的checkpoint路径（跳过训练时使用）'
    )
    
    # 其他选项
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='temp/pipeline',
        help='临时文件目录'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='保留临时文件'
    )
    parser.add_argument(
        '--verify-onnx',
        action='store_true',
        default=True,
        help='验证导出的ONNX模型'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    args = parser.parse_args()
    
    # 创建并运行流水线
    pipeline = SegmentationPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

