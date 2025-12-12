# -*- coding: utf-8 -*-
"""
نظام التسجيل (Logging) للمشروع
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """مصمم سجلات ملون"""
    
    # ألوان ANSI
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """تنسيق السجل مع الألوان"""
        log_message = super().format(record)
        
        if sys.stdout.isatty():  # فقط في الطرفية
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            return f"{color}{log_message}{self.COLORS['RESET']}"
        
        return log_message


class JSONFormatter(logging.Formatter):
    """مصمم سجلات بتنسيق JSON"""
    
    def format(self, record):
        """تنسيق السجل كـ JSON"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.threadName,
            'process': record.processName
        }
        
        # إضافة البيانات الإضافية
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        # إضافة الاستثناء إذا كان موجوداً
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class Logger:
    """مدير السجلات الرئيسي"""
    
    _instance = None
    
    def __new__(cls):
        """نمط Singleton"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """تهيئة المسجل"""
        if self._initialized:
            return
        
        # إنشاء المسجل
        self.logger = logging.getLogger('DeepSeekMini')
        self.logger.setLevel(logging.DEBUG)
        
        # مستوى التسجيل الحالي
        self.current_level = logging.INFO
        
        # المجلدات
        self.log_dir = Path.home() / '.deepseek_mini' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # المعالجات
        self.handlers = {}
        
        # التهيئة
        self._setup_default_handlers()
        self._initialized = True
        
        # تسجيل بدء التشغيل
        self.info("🚀 تم تهيئة نظام التسجيل")
    
    def _setup_default_handlers(self):
        """إعداد المعالجات الافتراضية"""
        # معالج وحدة التحكم
        self.add_console_handler()
        
        # معالج الملفات
        self.add_file_handler()
    
    def add_console_handler(self, level=logging.INFO):
        """إضافة معالج وحدة التحكم"""
        if 'console' in self.handlers:
            return self.handlers['console']
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # تنسيق ملون
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.handlers['console'] = handler
        
        return handler
    
    def add_file_handler(self, level=logging.DEBUG, max_bytes=10*1024*1024, backup_count=5):
        """إضافة معالج الملفات"""
        if 'file' in self.handlers:
            return self.handlers['file']
        
        # اسم الملف
        log_file = self.log_dir / f'deepseek_{datetime.now().strftime("%Y%m%d")}.log'
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setLevel(level)
        
        # تنسيق مفصل
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.handlers['file'] = handler
        
        return handler
    
    def add_json_handler(self, level=logging.INFO):
        """إضافة معالج JSON"""
        if 'json' in self.handlers:
            return self.handlers['json']
        
        json_file = self.log_dir / f'logs_{datetime.now().strftime("%Y%m%d")}.json'
        
        handler = logging.FileHandler(json_file, encoding='utf-8')
        handler.setLevel(level)
        
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.handlers['json'] = handler
        
        return handler
    
    def set_level(self, level):
        """تعيين مستوى التسجيل"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        self.current_level = level
        self.logger.setLevel(level)
        
        for handler in self.handlers.values():
            handler.setLevel(level)
        
        self.info(f"تم تعيين مستوى التسجيل إلى {logging.getLevelName(level)}")
    
    def get_level(self):
        """الحصول على مستوى التسجيل الحالي"""
        return self.current_level
    
    def debug(self, message, extra_data=None):
        """تسجيل رسالة تصحيح"""
        if extra_data:
            self.logger.debug(message, extra={'extra_data': extra_data})
        else:
            self.logger.debug(message)
    
    def info(self, message, extra_data=None):
        """تسجيل رسالة معلومات"""
        if extra_data:
            self.logger.info(message, extra={'extra_data': extra_data})
        else:
            self.logger.info(message)
    
    def warning(self, message, extra_data=None):
        """تسجيل رسالة تحذير"""
        if extra_data:
            self.logger.warning(message, extra={'extra_data': extra_data})
        else:
            self.logger.warning(message)
    
    def error(self, message, extra_data=None, exc_info=False):
        """تسجيل رسالة خطأ"""
        if extra_data:
            self.logger.error(message, extra={'extra_data': extra_data}, exc_info=exc_info)
        else:
            self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message, extra_data=None):
        """تسجيل رسالة حرجة"""
        if extra_data:
            self.logger.critical(message, extra={'extra_data': extra_data})
        else:
            self.logger.critical(message)
    
    def exception(self, message, extra_data=None):
        """تسجيل استثناء"""
        if extra_data:
            self.logger.exception(message, extra={'extra_data': extra_data})
        else:
            self.logger.exception(message)
    
    def log_performance(self, operation, duration, **kwargs):
        """تسجيل أداء العملية"""
        message = f"⏱️  {operation}: {duration:.3f} ثانية"
        
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message += f" [{extra_info}]"
        
        self.info(message)
    
    def log_memory_usage(self, context=""):
        """تسجيل استخدام الذاكرة"""
        import psutil
        import torch
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        message = f"💾 استخدام الذاكرة {context}:"
        message += f"\n  النظام: {memory_info.rss / 1024 / 1024:.1f} MB"
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            memory_cached = torch.cuda.memory_reserved() / 1024 / 1024
            message += f"\n  GPU مخصصة: {memory_allocated:.1f} MB"
            message += f"\n  GPU محجوزة: {memory_cached:.1f} MB"
        
        self.debug(message)
    
    def log_model_info(self, model):
        """تسجيل معلومات النموذج"""
        import torch
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"🧠 معلومات النموذج:")
        self.info(f"  المعلمات الكلية: {total_params:,}")
        self.info(f"  المعلمات القابلة للتدريب: {trainable_params:,}")
        self.info(f"  الطبقات: {len(list(model.children()))}")
        
        if hasattr(model, 'get_config'):
            config = model.get_config()
            for key, value in config.items():
                self.info(f"  {key}: {value}")
    
    def log_training_start(self, config):
        """تسجيل بدء التدريب"""
        self.info("🎯 بدء التدريب")
        self.info(f"  الإعدادات: {json.dumps(config, indent=2, default=str)}")
    
    def log_training_progress(self, epoch, step, loss, lr, **kwargs):
        """تسجيل تقدم التدريب"""
        message = f"📈 التدريب - الدورة {epoch}, الخطوة {step}:"
        message += f"\n  الخسارة: {loss:.4f}"
        message += f"\n  معدل التعلم: {lr:.2e}"
        
        if kwargs:
            for key, value in kwargs.items():
                message += f"\n  {key}: {value}"
        
        self.info(message)
    
    def log_generation(self, prompt, response, tokens_per_second=None):
        """تسجيل التوليد"""
        self.info("🤖 التوليد:")
        self.info(f"  المطالبة: {prompt[:100]}..." if len(prompt) > 100 else f"  المطالبة: {prompt}")
        self.info(f"  الرد: {response[:100]}..." if len(response) > 100 else f"  الرد: {response}")
        
        if tokens_per_second:
            self.info(f"  السرعة: {tokens_per_second:.1f} رمز/ثانية")
    
    def get_log_files(self):
        """الحصول على قائمة ملفات السجلات"""
        log_files = []
        
        if self.log_dir.exists():
            for file in self.log_dir.iterdir():
                if file.is_file() and file.suffix in ['.log', '.json']:
                    log_files.append(file)
        
        return sorted(log_files, reverse=True)  # الأحدث أولاً
    
    def clear_old_logs(self, days_to_keep=30):
        """مسح السجلات القديمة"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.get_log_files():
            file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            if file_date < cutoff_date:
                try:
                    log_file.unlink()
                    self.info(f"تم مسح السجل القديم: {log_file.name}")
                except Exception as e:
                    self.error(f"فشل مسح السجل {log_file.name}: {e}")
    
    def export_logs(self, output_path=None):
        """تصدير السجلات"""
        if output_path is None:
            output_path = self.log_dir / f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for log_file in self.get_log_files()[:10]:  # آخر 10 ملفات
                f.write(f"\n{'='*80}\n")
                f.write(f"ملف: {log_file.name}\n")
                f.write(f"{'='*80}\n\n")
                
                try:
                    with open(log_file, 'r', encoding='utf-8') as log_f:
                        f.write(log_f.read())
                except Exception as e:
                    f.write(f"خطأ في قراءة الملف: {e}\n")
        
        self.info(f"تم تصدير السجلات إلى: {output_path}")
        return output_path
    
    def cleanup(self):
        """تنظيف الموارد"""
        for handler in self.handlers.values():
            handler.close()
            self.logger.removeHandler(handler)
        
        self.handlers.clear()
        self.info("تم تنظيف نظام التسجيل")


# دوال مساعدة للاستخدام السهل
def get_logger():
    """الحصول على مثيل المسجل"""
    return Logger().logger


def setup_logging(level='INFO', log_dir=None):
    """إعداد نظام التسجيل"""
    logger = Logger()
    
    if log_dir:
        logger.log_dir = Path(log_dir)
        logger.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.set_level(level)
    return logger


def log_debug(message, **kwargs):
    """تسجيل تصحيح"""
    Logger().debug(message, kwargs if kwargs else None)


def log_info(message, **kwargs):
    """تسجيل معلومات"""
    Logger().info(message, kwargs if kwargs else None)


def log_warning(message, **kwargs):
    """تسجيل تحذير"""
    Logger().warning(message, kwargs if kwargs else None)


def log_error(message, **kwargs):
    """تسجيل خطأ"""
    Logger().error(message, kwargs if kwargs else None)


def log_critical(message, **kwargs):
    """تسجيل حرج"""
    Logger().critical(message, kwargs if kwargs else None)


def log_exception(message, **kwargs):
    """تسجيل استثناء"""
    Logger().exception(message, kwargs if kwargs else None)


def log_performance(operation, duration, **kwargs):
    """تسجيل أداء"""
    Logger().log_performance(operation, duration, **kwargs)


if __name__ == "__main__":
    # اختبار نظام التسجيل
    logger = setup_logging(level='DEBUG')
    
    # اختبار مستويات التسجيل المختلفة
    logger.debug("هذه رسالة تصحيح")
    logger.info("هذه رسالة معلومات")
    logger.warning("هذه رسالة تحذير")
    logger.error("هذه رسالة خطأ")
    
    # اختبار مع بيانات إضافية
    logger.info("تسجيل مع بيانات إضافية", extra_data={"user": "test", "action": "login"})
    
    # اختبار تسجيل الأداء
    import time
    start = time.time()
    time.sleep(0.1)
    logger.log_performance("عملية الاختبار", time.time() - start, iterations=100)
    
    # اختبار تسجيل الذاكرة
    logger.log_memory_usage("بعد الاختبار")
    
    # عرض ملفات السجلات
    log_files = logger.get_log_files()
    print(f"\n📁 ملفات السجلات ({len(log_files)}):")
    for file in log_files[:3]:
        print(f"  {file.name}")
    
    # تنظيف
    logger.cleanup()