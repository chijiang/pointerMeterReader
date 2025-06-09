#!/usr/bin/env python3
"""
液晶数字表检测推理脚本

此脚本参考指针表训练脚本架构，提供完整的液晶数字表检测推理功能。
包含检测、过滤、可视化、分析等完整功能。

功能特性：
1. 智能模型加载和设备选择
2. 单张图像和批量推理
3. 智能结果过滤和后处理
4. 丰富的可视化功能
5. 详细的分析报告生成
6. ROI提取和保存

作者: chijiang
日期: 2025-06-09
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 导入ultralytics库
try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    print("❌ 错误：请安装ultralytics库")
    print("运行: pip install ultralytics")
    sys.exit(1)


class DigitalMeterDetector:
    """液晶数字表检测器"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "auto",
    ):
        """
        初始化检测器

        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            device: 设备类型
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._get_device(device)

        # 获取项目根目录
        self.project_root = self._get_project_root()

        # 解析模型路径
        if not self.model_path.is_absolute():
            self.model_path = self.project_root / self.model_path

        # 加载模型
        self.model = self._load_model()

        # 设置输出目录
        self.setup_output_dirs()

        # 过滤参数
        self.filter_config = {
            "min_area": 1600,  # 最小面积（像素）
            "max_area": 360000,  # 最大面积（像素）
            "min_aspect_ratio": 1.2,  # 最小宽高比
            "max_aspect_ratio": 6.0,  # 最大宽高比
        }

        print(f"🚀 液晶数字表检测器已初始化")
        print(f"📁 模型路径: {self.model_path}")
        print(f"🎯 设备: {self.device}")
        print(f"📊 置信度阈值: {self.conf_threshold}")

    def _get_project_root(self) -> Path:
        """智能获取项目根目录"""
        current_dir = Path.cwd()
        if current_dir.name == "inference":
            return current_dir.parent.parent.parent
        elif current_dir.name == "digital_meter_detection":
            return current_dir.parent.parent
        elif current_dir.name == "scripts":
            return current_dir.parent
        else:
            return current_dir

    def _get_device(self, device: str) -> str:
        """智能检测最佳设备"""
        if device != "auto":
            return device

        # 自动检测设备
        import torch

        if torch.cuda.is_available():
            device = "0"
            print(f"🔥 使用CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("🍎 使用Apple MPS加速")
        else:
            device = "cpu"
            print("💻 使用CPU")

        return device

    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        try:
            print(f"📦 加载模型: {self.model_path}")

            if not self.model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

            model = YOLO(str(self.model_path))

            # 设置设备
            model.to(self.device)

            print(f"✅ 模型加载成功")
            return model

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

    def setup_output_dirs(self):
        """设置输出目录"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 输出目录结构
        self.output_root = self.project_root / "outputs" / "digital_meter_inference"
        self.result_dir = self.output_root / f"inference_{self.timestamp}"
        self.viz_dir = self.result_dir / "visualizations"
        self.roi_dir = self.result_dir / "rois"
        self.analysis_dir = self.result_dir / "analysis"

        # 创建目录
        for dir_path in [
            self.result_dir,
            self.viz_dir,
            self.roi_dir,
            self.analysis_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"📁 输出目录结构:")
        print(f"  - 结果: {self.result_dir}")
        print(f"  - 可视化: {self.viz_dir}")
        print(f"  - ROI: {self.roi_dir}")
        print(f"  - 分析: {self.analysis_dir}")

    def detect_single_image(self, image_path: Union[str, Path]) -> Dict:
        """
        检测单张图像

        Args:
            image_path: 图像路径

        Returns:
            检测结果字典
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        print(f"🔍 检测图像: {image_path.name}")

        # 执行检测
        results = self.model(
            str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # 解析结果
        detections = self._parse_detections(results[0], image_path)

        # 过滤结果
        filtered_detections = self._filter_detections(detections)

        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "raw_detections": detections,
            "filtered_detections": filtered_detections,
            "detection_count": len(filtered_detections),
            "timestamp": datetime.now().isoformat(),
        }

    def _parse_detections(self, result, image_path: Path) -> List[Dict]:
        """解析检测结果"""
        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            # 获取图像尺寸
            image = cv2.imread(str(image_path))
            img_height, img_width = image.shape[:2]

            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 0

                detection = {
                    "id": i,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "class_name": "digital_meter",
                    "width": float(width),
                    "height": float(height),
                    "area": float(area),
                    "aspect_ratio": float(aspect_ratio),
                    "center_x": float((x1 + x2) / 2),
                    "center_y": float((y1 + y2) / 2),
                    "relative_center_x": float((x1 + x2) / 2 / img_width),
                    "relative_center_y": float((y1 + y2) / 2 / img_height),
                    "relative_width": float(width / img_width),
                    "relative_height": float(height / img_height),
                }

                detections.append(detection)

        return detections

    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """过滤检测结果"""
        filtered = []

        for detection in detections:
            # 面积过滤
            if detection["area"] < self.filter_config["min_area"]:
                continue
            if detection["area"] > self.filter_config["max_area"]:
                continue

            # 宽高比过滤
            if detection["aspect_ratio"] < self.filter_config["min_aspect_ratio"]:
                continue
            if detection["aspect_ratio"] > self.filter_config["max_aspect_ratio"]:
                continue

            filtered.append(detection)

        return filtered

    def extract_roi(
        self, image_path: Union[str, Path], detection: Dict, padding: int = 20
    ) -> np.ndarray:
        """
        提取ROI区域

        Args:
            image_path: 图像路径
            detection: 检测结果
            padding: 边界填充

        Returns:
            ROI图像
        """
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        # 获取边界框
        x1, y1, x2, y2 = detection["bbox"]

        # 添加填充
        x1 = max(0, int(x1 - padding))
        y1 = max(0, int(y1 - padding))
        x2 = min(width, int(x2 + padding))
        y2 = min(height, int(y2 + padding))

        # 提取ROI
        roi = image[y1:y2, x1:x2]

        return roi

    def visualize_detections(
        self, image_path: Union[str, Path], detections: List[Dict], save: bool = True
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image_path: 图像路径
            detections: 检测结果列表
            save: 是否保存结果

        Returns:
            可视化图像
        """
        # 加载图像
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建副本用于绘制
        viz_image = image_rgb.copy()

        # 绘制检测框
        for detection in detections:
            x1, y1, x2, y2 = [int(x) for x in detection["bbox"]]
            conf = detection["confidence"]

            # 绘制边界框
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 绘制标签
            label = f"digital_meter {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # 绘制标签背景
            cv2.rectangle(
                viz_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (255, 0, 0),
                -1,
            )

            # 绘制标签文字
            cv2.putText(
                viz_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # 保存可视化结果
        if save:
            image_name = Path(image_path).stem
            viz_file = self.viz_dir / f"{image_name}_detections.jpg"
            viz_bgr = cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(viz_file), viz_bgr)
            print(f"💾 可视化结果保存: {viz_file}")

        return viz_image

    def create_detection_summary(self, all_results: List[Dict]) -> Dict:
        """创建检测总结"""
        print("\n📊 生成检测总结...")

        total_images = len(all_results)
        total_detections = sum(len(r["filtered_detections"]) for r in all_results)
        images_with_detections = sum(
            1 for r in all_results if len(r["filtered_detections"]) > 0
        )

        # 置信度统计
        all_confidences = []
        all_areas = []
        all_aspect_ratios = []

        for result in all_results:
            for detection in result["filtered_detections"]:
                all_confidences.append(detection["confidence"])
                all_areas.append(detection["area"])
                all_aspect_ratios.append(detection["aspect_ratio"])

        summary = {
            "statistics": {
                "total_images": total_images,
                "images_with_detections": images_with_detections,
                "detection_rate": (
                    images_with_detections / total_images if total_images > 0 else 0
                ),
                "total_detections": total_detections,
                "avg_detections_per_image": (
                    total_detections / total_images if total_images > 0 else 0
                ),
            },
            "confidence_stats": {
                "mean": float(np.mean(all_confidences)) if all_confidences else 0,
                "std": float(np.std(all_confidences)) if all_confidences else 0,
                "min": float(np.min(all_confidences)) if all_confidences else 0,
                "max": float(np.max(all_confidences)) if all_confidences else 0,
            },
            "area_stats": {
                "mean": float(np.mean(all_areas)) if all_areas else 0,
                "std": float(np.std(all_areas)) if all_areas else 0,
                "min": float(np.min(all_areas)) if all_areas else 0,
                "max": float(np.max(all_areas)) if all_areas else 0,
            },
            "aspect_ratio_stats": {
                "mean": float(np.mean(all_aspect_ratios)) if all_aspect_ratios else 0,
                "std": float(np.std(all_aspect_ratios)) if all_aspect_ratios else 0,
                "min": float(np.min(all_aspect_ratios)) if all_aspect_ratios else 0,
                "max": float(np.max(all_aspect_ratios)) if all_aspect_ratios else 0,
            },
            "filter_config": self.filter_config,
            "model_config": {
                "model_path": str(self.model_path),
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "device": self.device,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def plot_detection_analysis(self, all_results: List[Dict]):
        """绘制检测分析图表"""
        print("\n📈 生成分析图表...")

        # 收集数据
        confidences = []
        areas = []
        aspect_ratios = []
        detection_counts = []

        for result in all_results:
            detection_counts.append(len(result["filtered_detections"]))
            for detection in result["filtered_detections"]:
                confidences.append(detection["confidence"])
                areas.append(detection["area"])
                aspect_ratios.append(detection["aspect_ratio"])

        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("液晶数字表检测分析", fontsize=16, fontweight="bold")

        # 置信度分布
        if confidences:
            axes[0, 0].hist(
                confidences, bins=20, alpha=0.7, color="blue", edgecolor="black"
            )
            axes[0, 0].axvline(
                np.mean(confidences),
                color="red",
                linestyle="--",
                label=f"均值: {np.mean(confidences):.3f}",
            )
            axes[0, 0].set_title("置信度分布")
            axes[0, 0].set_xlabel("置信度")
            axes[0, 0].set_ylabel("频次")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 面积分布
        if areas:
            axes[0, 1].hist(areas, bins=20, alpha=0.7, color="green", edgecolor="black")
            axes[0, 1].axvline(
                np.mean(areas),
                color="red",
                linestyle="--",
                label=f"均值: {np.mean(areas):.0f}",
            )
            axes[0, 1].set_title("检测框面积分布")
            axes[0, 1].set_xlabel("面积 (像素²)")
            axes[0, 1].set_ylabel("频次")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 宽高比分布
        if aspect_ratios:
            axes[0, 2].hist(
                aspect_ratios, bins=20, alpha=0.7, color="orange", edgecolor="black"
            )
            axes[0, 2].axvline(
                np.mean(aspect_ratios),
                color="red",
                linestyle="--",
                label=f"均值: {np.mean(aspect_ratios):.2f}",
            )
            axes[0, 2].set_title("宽高比分布")
            axes[0, 2].set_xlabel("宽高比")
            axes[0, 2].set_ylabel("频次")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 每图像检测数量分布
        axes[1, 0].hist(
            detection_counts,
            bins=max(1, max(detection_counts) if detection_counts else 1),
            alpha=0.7,
            color="purple",
            edgecolor="black",
        )
        axes[1, 0].axvline(
            np.mean(detection_counts),
            color="red",
            linestyle="--",
            label=f"均值: {np.mean(detection_counts):.2f}",
        )
        axes[1, 0].set_title("每图像检测数量分布")
        axes[1, 0].set_xlabel("检测数量")
        axes[1, 0].set_ylabel("图像数量")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 置信度vs面积散点图
        if confidences and areas:
            scatter = axes[1, 1].scatter(
                areas,
                confidences,
                alpha=0.6,
                c=confidences,
                cmap="viridis",
                edgecolors="black",
                linewidth=0.5,
            )
            axes[1, 1].set_title("置信度 vs 面积")
            axes[1, 1].set_xlabel("面积 (像素²)")
            axes[1, 1].set_ylabel("置信度")
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label="置信度")

        # 宽高比vs置信度散点图
        if aspect_ratios and confidences:
            scatter2 = axes[1, 2].scatter(
                aspect_ratios,
                confidences,
                alpha=0.6,
                c=areas,
                cmap="plasma",
                edgecolors="black",
                linewidth=0.5,
            )
            axes[1, 2].set_title("宽高比 vs 置信度")
            axes[1, 2].set_xlabel("宽高比")
            axes[1, 2].set_ylabel("置信度")
            axes[1, 2].grid(True, alpha=0.3)
            if areas:
                plt.colorbar(scatter2, ax=axes[1, 2], label="面积")

        plt.tight_layout()

        # 保存图表
        analysis_file = self.analysis_dir / "detection_analysis.png"
        plt.savefig(analysis_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ 分析图表保存: {analysis_file}")

    def create_detection_gallery(self, all_results: List[Dict], max_images: int = 16):
        """创建检测结果画廊"""
        print(f"\n🖼️  创建检测结果画廊 (最多{max_images}张)...")

        # 选择有检测结果的图像
        results_with_detections = [
            r for r in all_results if len(r["filtered_detections"]) > 0
        ]

        if not results_with_detections:
            print("⚠️  没有检测到液晶数字表，跳过画廊创建")
            return

        # 按检测数量排序，选择最好的结果
        results_with_detections.sort(
            key=lambda x: len(x["filtered_detections"]), reverse=True
        )
        selected_results = results_with_detections[:max_images]

        # 计算网格大小
        n_images = len(selected_results)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols

        # 创建画廊
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle("液晶数字表检测结果画廊", fontsize=16, fontweight="bold")

        for idx, result in enumerate(selected_results):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 加载和可视化图像
            image_path = result["image_path"]
            detections = result["filtered_detections"]

            # 加载图像
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 绘制检测框
            for detection in detections:
                x1, y1, x2, y2 = [int(x) for x in detection["bbox"]]
                conf = detection["confidence"]

                # 绘制边界框
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # 绘制标签
                label = f"{conf:.2f}"
                cv2.putText(
                    image_rgb,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

            ax.imshow(image_rgb)
            ax.set_title(
                f"{Path(image_path).name}\n检测数: {len(detections)}", fontsize=10
            )
            ax.axis("off")

        # 隐藏多余的子图
        for idx in range(len(selected_results), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        # 保存画廊
        gallery_file = self.viz_dir / "detection_gallery.png"
        plt.savefig(gallery_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ 检测画廊保存: {gallery_file}")

    def process_batch(
        self, input_path: Union[str, Path], save_rois: bool = True
    ) -> List[Dict]:
        """
        批量处理图像

        Args:
            input_path: 输入路径（图像文件或目录）
            save_rois: 是否保存ROI区域

        Returns:
            检测结果列表
        """
        input_path = Path(input_path)

        # 获取图像文件列表
        if input_path.is_file():
            image_files = [input_path]
        elif input_path.is_dir():
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
                image_files.extend(input_path.glob(ext))
                image_files.extend(input_path.glob(ext.upper()))
        else:
            raise ValueError(f"输入路径无效: {input_path}")

        if not image_files:
            raise ValueError(f"未找到图像文件: {input_path}")

        print(f"🔍 开始批量检测，共 {len(image_files)} 张图像")

        all_results = []

        # 处理每张图像
        for image_file in tqdm(image_files, desc="处理图像"):
            try:
                # 检测
                result = self.detect_single_image(image_file)
                all_results.append(result)

                # 可视化
                self.visualize_detections(image_file, result["filtered_detections"])

                # 保存ROI
                if save_rois and result["filtered_detections"]:
                    for i, detection in enumerate(result["filtered_detections"]):
                        roi = self.extract_roi(image_file, detection)
                        roi_filename = f"{image_file.stem}_roi_{i+1}.jpg"
                        roi_path = self.roi_dir / roi_filename
                        cv2.imwrite(str(roi_path), roi)

            except Exception as e:
                print(f"⚠️  处理 {image_file.name} 时出错: {e}")
                continue

        return all_results

    def save_results(self, all_results: List[Dict], summary: Dict):
        """保存检测结果"""
        print("\n💾 保存检测结果...")

        # 保存详细结果
        results_file = self.result_dir / "detection_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 保存总结
        summary_file = self.result_dir / "detection_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 生成Markdown报告
        self._generate_markdown_report(summary, all_results)

        print(f"✅ 结果保存完成:")
        print(f"  - 详细结果: {results_file}")
        print(f"  - 检测总结: {summary_file}")

    def _generate_markdown_report(self, summary: Dict, all_results: List[Dict]):
        """生成Markdown格式报告"""
        md_content = f"""# 液晶数字表检测报告

## 检测总结
- **总图像数**: {summary['statistics']['total_images']}
- **有检测结果的图像**: {summary['statistics']['images_with_detections']}
- **检测成功率**: {summary['statistics']['detection_rate']:.2%}
- **总检测数**: {summary['statistics']['total_detections']}
- **平均每图检测数**: {summary['statistics']['avg_detections_per_image']:.2f}

## 置信度统计
- **均值**: {summary['confidence_stats']['mean']:.3f}
- **标准差**: {summary['confidence_stats']['std']:.3f}
- **最小值**: {summary['confidence_stats']['min']:.3f}
- **最大值**: {summary['confidence_stats']['max']:.3f}

## 检测框面积统计
- **均值**: {summary['area_stats']['mean']:.0f} 像素²
- **标准差**: {summary['area_stats']['std']:.0f} 像素²
- **最小值**: {summary['area_stats']['min']:.0f} 像素²
- **最大值**: {summary['area_stats']['max']:.0f} 像素²

## 宽高比统计
- **均值**: {summary['aspect_ratio_stats']['mean']:.2f}
- **标准差**: {summary['aspect_ratio_stats']['std']:.2f}
- **最小值**: {summary['aspect_ratio_stats']['min']:.2f}
- **最大值**: {summary['aspect_ratio_stats']['max']:.2f}

## 模型配置
- **模型路径**: {summary['model_config']['model_path']}
- **置信度阈值**: {summary['model_config']['conf_threshold']}
- **IoU阈值**: {summary['model_config']['iou_threshold']}
- **设备**: {summary['model_config']['device']}

## 过滤配置
- **最小面积**: {summary['filter_config']['min_area']} 像素²
- **最大面积**: {summary['filter_config']['max_area']} 像素²
- **最小宽高比**: {summary['filter_config']['min_aspect_ratio']}
- **最大宽高比**: {summary['filter_config']['max_aspect_ratio']}

## 生成文件
- 检测结果可视化: `visualizations/`
- ROI区域图像: `rois/`
- 分析图表: `analysis/detection_analysis.png`
- 检测画廊: `visualizations/detection_gallery.png`

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        md_file = self.result_dir / "detection_report.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="液晶数字表检测推理脚本")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="训练好的YOLO模型路径（相对于项目根目录或绝对路径）",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="输入图像文件或目录路径"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)"
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU阈值 (默认: 0.5)")
    parser.add_argument(
        "--device", type=str, default="auto", help="设备类型 (auto/cpu/0/mps)"
    )
    parser.add_argument("--no-rois", action="store_true", help="不保存ROI区域")
    parser.add_argument("--output", type=str, help="自定义输出目录")

    args = parser.parse_args()

    # 处理模型路径（相对于项目根目录）
    model_path = Path(args.model)
    if not model_path.is_absolute():
        # 获取项目根目录
        current_dir = Path.cwd()
        if current_dir.name == "inference":
            project_root = current_dir.parent.parent.parent
        elif current_dir.name == "digital_meter_detection":
            project_root = current_dir.parent.parent
        elif current_dir.name == "scripts":
            project_root = current_dir.parent
        else:
            project_root = current_dir

        model_path = project_root / model_path

    # 检查模型文件
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        sys.exit(1)

    args.model = str(model_path)

    try:
        # 创建检测器
        detector = DigitalMeterDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        )

        # 如果指定了自定义输出目录
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            detector.result_dir = output_dir
            detector.viz_dir = output_dir / "visualizations"
            detector.roi_dir = output_dir / "rois"
            detector.analysis_dir = output_dir / "analysis"

            for dir_path in [detector.viz_dir, detector.roi_dir, detector.analysis_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

        # 批量处理
        results = detector.process_batch(args.input, save_rois=not args.no_rois)

        # 生成总结
        summary = detector.create_detection_summary(results)

        # 生成分析图表
        detector.plot_detection_analysis(results)

        # 创建检测画廊
        detector.create_detection_gallery(results)

        # 保存结果
        detector.save_results(results, summary)

        # 输出统计信息
        print(f"\n🎉 检测完成!")
        print(f"📊 处理图像: {summary['statistics']['total_images']} 张")
        print(f"🎯 检测成功: {summary['statistics']['images_with_detections']} 张")
        print(f"📈 检测成功率: {summary['statistics']['detection_rate']:.2%}")
        print(f"🔍 总检测数: {summary['statistics']['total_detections']} 个")
        print(f"📁 结果目录: {detector.result_dir}")

    except Exception as e:
        print(f"❌ 检测过程出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
