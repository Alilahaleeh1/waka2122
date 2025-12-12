# -*- coding: utf-8 -*-
"""
محمل الإعدادات - لتحميل وتعديل إعدادات YAML
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json


class ConfigLoader:
    """فئة لتحميل وإدارة إعدادات المشروع"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        تهيئة محمل الإعدادات
        
        Args:
            config_path: مسار ملف الإعدادات
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """تحميل الإعدادات من ملف YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError(f"ملف الإعدادات {self.config_path} فارغ أو غير صالح")
            
            return config
            
        except FileNotFoundError:
            print(f"⚠️  ملف الإعدادات {self.config_path} غير موجود، إنشاء إعدادات افتراضية...")
            return self._create_default_config()
        except yaml.YAMLError as e:
            raise ValueError(f"خطأ في تحليل ملف YAML: {e}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """إنشاء إعدادات افتراضية"""
        default_config = {
            "project": {
                "name": "DeepSeek Mini",
                "version": "1.0.0",
                "author": "Your Name",
                "description": "نموذج لغوي عصبي صغير"
            },
            "model": {
                "vocab_size": 50000,
                "d_model": 768,
                "n_heads": 12,
                "n_layers": 12,
                "max_seq_len": 2048,
                "dropout": 0.1,
                "ffn_dim": 3072,
                "use_bias": True
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 3e-4,
                "warmup_steps": 2000,
                "total_steps": 100000
            }
        }
        
        # حفظ الإعدادات الافتراضية
        self.save_config(default_config, self.config_path)
        return default_config
    
    def _validate_config(self) -> None:
        """التحقق من صحة الإعدادات"""
        required_sections = ["project", "model", "training"]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"قسم {section} مفقود في ملف الإعدادات")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        الحصول على قيمة إعداد
        
        Args:
            key: مفتاح الإعداد (يمكن أن يكون متداخلاً باستخدام النقاط)
            default: القيمة الافتراضية إذا لم يوجد المفتاح
        
        Returns:
            قيمة الإعداد
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        تعيين قيمة إعداد
        
        Args:
            key: مفتاح الإعداد (يمكن أن يكون متداخلاً باستخدام النقاط)
            value: القيمة الجديدة
        """
        keys = key.split('.')
        config_ref = self.config
        
        # التنقل إلى المكان الصحيح
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # تعيين القيمة
        config_ref[keys[-1]] = value
    
    def save_config(self, config: Optional[Dict[str, Any]] = None, 
                   path: Optional[str] = None) -> None:
        """
        حفظ الإعدادات إلى ملف
        
        Args:
            config: الإعدادات لحفظها (إذا كان None، يتم حفظ الإعدادات الحالية)
            path: مسار الحفظ (إذا كان None، يتم استخدام المسار الحالي)
        """
        if config is None:
            config = self.config
        
        if path is None:
            path = self.config_path
        
        # التأكد من وجود المجلد
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """الحصول على الإعدادات كقاموس"""
        return self.config.copy()
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """تحديث الإعدادات بقيم جديدة"""
        self._deep_update(self.config, new_config)
    
    def _deep_update(self, original: Dict[str, Any], 
                    new: Dict[str, Any]) -> None:
        """تحديث متداخل للقاموس"""
        for key, value in new.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def print_config(self) -> None:
        """طباعة الإعدادات بشكل منسق"""
        print("=" * 60)
        print("📋 إعدادات المشروع:")
        print("=" * 60)
        print(json.dumps(self.config, indent=2, ensure_ascii=False))
        print("=" * 60)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    دالة مساعدة لتحميل الإعدادات
    
    Args:
        config_path: مسار ملف الإعدادات
    
    Returns:
        الإعدادات المحملة
    """
    loader = ConfigLoader(config_path)
    return loader.to_dict()


if __name__ == "__main__":
    # اختبار المحمل
    config = load_config()
    print("✅ تم تحميل الإعدادات بنجاح")
    print(f"اسم المشروع: {config['project']['name']}")
    print(f"إصدار النموذج: {config['model']['n_layers']} طبقات")