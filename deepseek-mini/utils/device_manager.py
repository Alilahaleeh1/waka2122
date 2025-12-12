# -*- coding: utf-8 -*-
"""
مدير الجهاز - لإدارة الأجهزة (GPU/CPU) وتخصيص الذاكرة
"""

import torch
import gc
import psutil
import platform
from typing import Optional, Tuple, Dict, Any
import warnings


class DeviceManager:
    """مدير الأجهزة والذاكرة"""
    
    def __init__(self, auto_select: bool = True):
        """
        تهيئة مدير الجهاز
        
        Args:
            auto_select: تحديد الجهاز تلقائياً
        """
        self.device_info = self._get_device_info()
        self.selected_device = None
        
        if auto_select:
            self.selected_device = self.select_best_device()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """الحصول على معلومات الجهاز"""
        info = {
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "freq": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
                "memory": psutil.virtual_memory().total / (1024**3),  #GB
            },
            "cuda": {
                "available": torch.cuda.is_available(),
                "devices": [],
                "driver": None
            },
            "system": {
                "os": platform.system(),
                "version": platform.version(),
                "machine": platform.machine()
            }
        }
        
        # معلومات CUDA إذا كانت متاحة
        if info["cuda"]["available"]:
            info["cuda"]["driver"] = torch.version.cuda
            info["cuda"]["devices"] = []
            
            for i in range(torch.cuda.device_count()):
                device_props = {
                    "name": torch.cuda.get_device_name(i),
                    "memory": torch.cuda.get_device_properties(i).total_memory / (1024**3),  #GB
                    "capability": torch.cuda.get_device_capability(i),
                    "current_memory": torch.cuda.memory_allocated(i) / (1024**3)  #GB
                }
                info["cuda"]["devices"].append(device_props)
        
        return info
    
    def select_best_device(self, preference: str = "cuda") -> torch.device:
        """
        اختيار أفضل جهاز متاح
        
        Args:
            preference: التفضيل ("cuda", "mps", "cpu")
        
        Returns:
            جهاز PyTorch
        """
        # محاولة CUDA أولاً
        if preference == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"✅ تم اختيار GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ذاكرة GPU: {self.get_gpu_memory()[0]:.2f} GB")
        
        # محاولة MPS (لـ Apple Silicon)
        elif preference == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✅ تم اختيار Apple Silicon (MPS)")
        
        # استخدام CPU كخيار أخير
        else:
            device = torch.device("cpu")
            print(f"⚠️  استخدام CPU ({self.device_info['cpu']['cores']} نواة)")
        
        self.selected_device = device
        return device
    
    def get_device(self, device_str: str = "auto") -> torch.device:
        """
        الحصول على جهاز بناءً على السلسلة
        
        Args:
            device_str: "cuda", "cpu", "mps", أو "auto"
        
        Returns:
            جهاز PyTorch
        """
        if device_str == "auto":
            return self.select_best_device()
        
        elif device_str == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            else:
                warnings.warn("CUDA غير متاح، استخدام CPU بدلاً من ذلك")
                return torch.device("cpu")
        
        elif device_str == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                warnings.warn("MPS غير متاح، استخدام CPU بدلاً من ذلك")
                return torch.device("cpu")
        
        elif device_str == "cpu":
            return torch.device("cpu")
        
        else:
            raise ValueError(f"جهاز غير معروف: {device_str}")
    
    def get_gpu_memory(self, device_id: int = 0) -> Tuple[float, float]:
        """
        الحصول على ذاكرة GPU
        
        Args:
            device_id: معرف جهاز GPU
        
        Returns:
            ذاكرة مستخدمة، ذاكرة كلية (GB)
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0
        
        torch.cuda.synchronize(device_id)
        used = torch.cuda.memory_allocated(device_id) / (1024**3)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        
        return used, total
    
    def clear_memory(self) -> None:
        """مسح الذاكرة"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
    
    def monitor_memory(self) -> Dict[str, float]:
        """
        مراقبة استخدام الذاكرة
        
        Returns:
            قاموس بمعلومات الذاكرة
        """
        memory_info = {}
        
        # ذاكرة النظام
        sys_memory = psutil.virtual_memory()
        memory_info["system_used"] = sys_memory.used / (1024**3)
        memory_info["system_total"] = sys_memory.total / (1024**3)
        memory_info["system_percent"] = sys_memory.percent
        
        # ذاكرة GPU إذا كانت متاحة
        if torch.cuda.is_available():
            used, total = self.get_gpu_memory()
            memory_info["gpu_used"] = used
            memory_info["gpu_total"] = total
            memory_info["gpu_percent"] = (used / total) * 100 if total > 0 else 0
        
        return memory_info
    
    def print_device_info(self) -> None:
        """طباعة معلومات الجهاز"""
        print("=" * 60)
        print("🖥️  معلومات الجهاز:")
        print("=" * 60)
        
        print(f"نظام التشغيل: {self.device_info['system']['os']} {self.device_info['system']['version']}")
        print(f"المعالج: {self.device_info['cpu']['cores']} نواة، {self.device_info['cpu']['threads']} خيط")
        print(f"ذاكرة النظام: {self.device_info['cpu']['memory']:.2f} GB")
        
        if self.device_info["cuda"]["available"]:
            print(f"\n🎮 CUDA متاح: نعم")
            print(f"   إصدار السائق: {self.device_info['cuda']['driver']}")
            
            for i, device in enumerate(self.device_info["cuda"]["devices"]):
                print(f"\n   GPU {i}: {device['name']}")
                print(f"     الذاكرة: {device['memory']:.2f} GB")
                print(f"     الإمكانية: {device['capability'][0]}.{device['capability'][1]}")
        else:
            print("\n🎮 CUDA متاح: لا")
        
        print("=" * 60)
    
    def optimize_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        تحسين النموذج للاستدلال
        
        Args:
            model: النموذج المراد تحسينه
        
        Returns:
            النموذج المحسن
        """
        model.eval()
        
        if torch.cuda.is_available():
            # تحسين الذاكرة
            torch.cuda.empty_cache()
            
            # استخدام التوقعات
            with torch.no_grad():
                model = model.to(self.selected_device)
        
        return model
    
    def optimize_for_training(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        تحسين النموذج للتدريب
        
        Args:
            model: النموذج المراد تحسينه
        
        Returns:
            النموذج المحسن
        """
        model.train()
        
        if torch.cuda.is_available():
            # استخدام Automatic Mixed Precision (AMP)
            try:
                from torch.cuda.amp import autocast
                model.amp_enabled = True
            except ImportError:
                model.amp_enabled = False
        
        return model


def get_available_device(preference: str = "cuda") -> torch.device:
    """
    دالة مساعدة للحصول على الجهاز المتاح
    
    Args:
        preference: تفضيل الجهاز
    
    Returns:
        جهاز PyTorch
    """
    manager = DeviceManager(auto_select=False)
    return manager.get_device(preference)


if __name__ == "__main__":
    # اختبار مدير الجهاز
    manager = DeviceManager()
    manager.print_device_info()
    
    device = manager.select_best_device()
    print(f"\n✅ الجهاز المختار: {device}")
    
    memory_info = manager.monitor_memory()
    print(f"\n📊 استخدام الذاكرة: {memory_info}")