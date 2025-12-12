# -*- coding: utf-8 -*-
"""
أدوات مساعدة عامة للمشروع
"""

import os
import sys
import json
import yaml
import math
import random
import string
import hashlib
import shutil
import zipfile
import tempfile
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import time
import traceback
import signal


def setup_project_dirs():
    """إعداد مجلدات المشروع"""
    dirs = [
        "data/raw",
        "data/processed",
        "checkpoints",
        "models",
        "logs",
        "exports",
        "tmp"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 تم إنشاء مجلد: {dir_path}")
    
    return dirs


def load_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Any:
    """تحميل ملف JSON"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ خطأ في تحميل JSON من {file_path}: {e}")
        return {}


def save_json(data: Any, file_path: Union[str, Path], 
              indent: int = 2, encoding: str = 'utf-8') -> bool:
    """حفظ بيانات إلى ملف JSON"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        print(f"✅ تم حفظ JSON إلى {file_path}")
        return True
    except Exception as e:
        print(f"❌ خطأ في حفظ JSON إلى {file_path}: {e}")
        return False


def load_yaml(file_path: Union[str, Path]) -> Dict:
    """تحميل ملف YAML"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ خطأ في تحميل YAML من {file_path}: {e}")
        return {}


def save_yaml(data: Dict, file_path: Union[str, Path]) -> bool:
    """حفظ بيانات إلى ملف YAML"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ تم حفظ YAML إلى {file_path}")
        return True
    except Exception as e:
        print(f"❌ خطأ في حفظ YAML إلى {file_path}: {e}")
        return False


def generate_id(length: int = 8, prefix: str = "") -> str:
    """إنشاء معرف فريد"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}{timestamp}_{random_str}"


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """تجزئة نص"""
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    return hash_func(text.encode()).hexdigest()


def format_bytes(size: float) -> str:
    """تنسيق حجم بالبايت"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_time(seconds: float) -> str:
    """تنسيق الوقت"""
    if seconds < 1:
        return f"{seconds*1000:.1f} مللي ثانية"
    elif seconds < 60:
        return f"{seconds:.1f} ثانية"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:.0f} دقيقة {seconds:.0f} ثانية"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f} ساعة {minutes:.0f} دقيقة"


def format_number(number: float) -> str:
    """تنسيق الأرقام"""
    if number >= 1_000_000_000:
        return f"{number/1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return str(number)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """قسمة آمنة (تجنب القسمة على صفر)"""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """تحديد قيمة بين الحد الأدنى والأقصى"""
    return max(min_val, min(value, max_val))


def linear_interpolate(start: float, end: float, t: float) -> float:
    """استكمال خطي"""
    t = clamp(t, 0.0, 1.0)
    return start + (end - start) * t


def exponential_decay(start: float, decay_rate: float, step: int) -> float:
    """اضمحلال أسي"""
    return start * (decay_rate ** step)


def cosine_decay(start: float, end: float, step: int, total_steps: int) -> float:
    """اضمحلال جيب التمام"""
    progress = min(step / total_steps, 1.0)
    decay = 0.5 * (1 + math.cos(math.pi * progress))
    return end + (start - end) * decay


def set_random_seed(seed: int = 42):
    """تعيين بذرة عشوائية"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # لنتائج حتمية
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ تم تعيين البذرة العشوائية إلى {seed}")


def count_parameters(model) -> Dict[str, int]:
    """عد معلمات النموذج"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params
    }


def model_size_mb(model) -> float:
    """حجم النموذج بالميجابايت"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def clean_text(text: str) -> str:
    """تنظيف النص"""
    import re
    
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text)
    
    # إزالة المسافات في البداية والنهاية
    text = text.strip()
    
    # إصلاح علامات الترقيم
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)
    
    # إصلاح الأقواس
    text = re.sub(r'\s+([{\[(\<])', r'\1', text)
    text = re.sub(r'([}\]\)\>])\s+', r'\1', text)
    
    # إصلاح التنصيص
    text = re.sub(r'\s+["\'`]', r'"', text)
    text = re.sub(r'["\'`]\s+', r'"', text)
    
    return text


def arabic_normalize(text: str) -> str:
    """تطبيع النص العربي"""
    import re
    
    # تحويل الألف المقصورة إلى ألف
    text = text.replace('ى', 'ا')
    
    # تحويل التاء المربوطة إلى هاء
    text = text.replace('ة', 'ه')
    
    # إزالة التشكيل
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # تحويل الهمزة في أماكنها المختلفة
    text = text.replace('أ', 'ا')
    text = text.replace('إ', 'ا')
    text = text.replace('آ', 'ا')
    
    # إزالة التكرار
    text = re.sub(r'(.)\1+', r'\1', text)
    
    return text


def english_normalize(text: str) -> str:
    """تطبيع النص الإنجليزي"""
    import re
    
    # تحويل إلى أحرف صغيرة
    text = text.lower()
    
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text)
    
    # إزالة علامات الترقيم غير الأساسية
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
    
    # إصلاخ الاختصارات الشائعة
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """اقتصاص النص"""
    if len(text) <= max_length:
        return text
    
    # محاولة الاقتصاص عند كلمة كاملة
    if max_length > len(suffix):
        truncated = text[:max_length - len(suffix)]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length // 2:
            truncated = truncated[:last_space]
        
        return truncated + suffix
    
    return text[:max_length]


def split_text_into_chunks(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """تقسيم النص إلى أجزاء"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # البحث عن مكان جيد للقطع (نهاية جملة أو فقرة)
        cut_point = text.rfind('. ', start, end)
        if cut_point == -1:
            cut_point = text.rfind(' ', start, end)
        
        if cut_point > start:
            end = cut_point + 1
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def calculate_similarity(text1: str, text2: str) -> float:
    """حساب التشابه بين نصين"""
    from difflib import SequenceMatcher
    
    # استخدام SequenceMatcher من difflib
    return SequenceMatcher(None, text1, text2).ratio()


def backup_file(file_path: Union[str, Path], 
                backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """نسخ احتياطي للملف"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"⚠️  الملف غير موجود: {file_path}")
        return None
    
    if backup_dir is None:
        backup_dir = file_path.parent / "backups"
    
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # اسم الملف النسخ الاحتياطي
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    try:
        shutil.copy2(file_path, backup_path)
        print(f"✅ تم النسخ الاحتياطي لـ {file_path.name} إلى {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ فشل النسخ الاحتياطي: {e}")
        return None


def zip_directory(directory: Union[str, Path], 
                  output_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """ضغط مجلد"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"⚠️  المجلد غير موجود: {directory}")
        return None
    
    if output_path is None:
        output_path = directory.parent / f"{directory.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    output_path = Path(output_path)
    
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in directory.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(directory)
                    zipf.write(file, arcname)
        
        print(f"✅ تم ضغط {directory} إلى {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ فشل الضغط: {e}")
        return None


def extract_zip(zip_path: Union[str, Path], 
                extract_to: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """استخراج ملف مضغوط"""
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        print(f"⚠️  الملف المضغوط غير موجود: {zip_path}")
        return None
    
    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem
    
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)
        
        print(f"✅ تم استخراج {zip_path} إلى {extract_to}")
        return extract_to
    except Exception as e:
        print(f"❌ فشل الاستخراج: {e}")
        return None


def retry(max_attempts: int = 3, delay: float = 1.0, 
          exceptions: Tuple = (Exception,)):
    """مكرر لمحاولة الدوال المتكررة"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        print(f"❌ فشلت جميع المحاولات ({max_attempts}) للدالة {func.__name__}: {e}")
                        raise
                    
                    print(f"⚠️  محاولة {attempt}/{max_attempts} فشلت للدالة {func.__name__}: {e}")
                    time.sleep(delay * attempt)  # زيادة التأخير مع كل محاولة
            
            raise Exception(f"فشلت جميع المحاولات للدالة {func.__name__}")
        return wrapper
    return decorator


def timer(func):
    """مؤقت لقياس زمن تنفيذ الدوال"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        print(f"⏱️  {func.__name__} استغرقت {duration:.3f} ثانية")
        
        return result
    return wrapper


def memoize(func):
    """تخزين نتائج الدوال"""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    return wrapper


def singleton(cls):
    """نمط Singleton للفئات"""
    instances = {}
    
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return wrapper


def format_exception(e: Exception) -> str:
    """تنسيق الاستثناء"""
    return f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                       suffix: str = '', length: int = 50, fill: str = '█'):
    """طباعة شريط تقدم"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    
    if iteration == total:
        print()


def print_table(data: List[List[Any]], headers: List[str] = None):
    """طباعة جدول"""
    if not data:
        return
    
    if headers is None:
        headers = [f"Column {i+1}" for i in range(len(data[0]))]
    
    # حساب عرض الأعمدة
    col_widths = []
    for i in range(len(headers)):
        max_len = max(
            len(str(headers[i])),
            max(len(str(row[i])) for row in data) if data else 0
        )
        col_widths.append(max_len + 2)  # إضافة مسافة
    
    # طباعة الرأس
    header_line = "┌" + "┬".join("─" * w for w in col_widths) + "┐"
    print(header_line)
    
    header_cells = []
    for i, header in enumerate(headers):
        header_cells.append(f" {header:<{col_widths[i]-1}}")
    print("│" + "│".join(header_cells) + "│")
    
    separator_line = "├" + "┼".join("─" * w for w in col_widths) + "┤"
    print(separator_line)
    
    # طباعة البيانات
    for row in data:
        row_cells = []
        for i, cell in enumerate(row):
            row_cells.append(f" {str(cell):<{col_widths[i]-1}}")
        print("│" + "│".join(row_cells) + "│")
    
    footer_line = "└" + "┴".join("─" * w for w in col_widths) + "┘"
    print(footer_line)


def get_system_info() -> Dict[str, Any]:
    """الحصول على معلومات النظام"""
    import platform
    import psutil
    import torch
    
    info = {
        "system": {
            "os": platform.system(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        },
        "hardware": {
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3)
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "devices": []
        }
    }
    
    if info["gpu"]["available"]:
        for i in range(torch.cuda.device_count()):
            device_info = {
                "name": torch.cuda.get_device_name(i),
                "memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                "capability": torch.cuda.get_device_capability(i)
            }
            info["gpu"]["devices"].append(device_info)
    
    return info


def check_disk_space(path: Union[str, Path] = ".") -> Dict[str, float]:
    """التحقق من مساحة القرص"""
    import shutil
    
    path = Path(path)
    total, used, free = shutil.disk_usage(path)
    
    return {
        "total_gb": total / (1024**3),
        "used_gb": used / (1024**3),
        "free_gb": free / (1024**3),
        "used_percent": (used / total) * 100
    }


def clean_temp_files(max_age_hours: int = 24):
    """تنظيف الملفات المؤقتة القديمة"""
    temp_dir = Path("tmp")
    
    if not temp_dir.exists():
        return
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for file in temp_dir.rglob("*"):
        if file.is_file():
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            
            if file_time < cutoff_time:
                try:
                    file.unlink()
                    print(f"🧹 تم تنظيف الملف المؤقت: {file.name}")
                except Exception as e:
                    print(f"⚠️  فشل تنظيف الملف {file.name}: {e}")


class GracefulExit:
    """خروج سلس من البرنامج"""
    
    def __init__(self):
        self.exit_requested = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """معالج الإشارات"""
        print(f"\n⚠️  تم استقبال إشارة الخروج ({signum})")
        self.exit_requested = True
    
    def should_exit(self) -> bool:
        """التحقق إذا طلب الخروج"""
        return self.exit_requested


if __name__ == "__main__":
    # اختبار الأدوات المساعدة
    print("🧪 اختبار الأدوات المساعدة...")
    
    # إنشاء معرف
    id1 = generate_id()
    id2 = generate_id(prefix="model_")
    print(f"المعرف 1: {id1}")
    print(f"المعرف 2: {id2}")
    
    # تجزئة نص
    hash1 = hash_string("Hello World")
    print(f"تجزئة 'Hello World': {hash1[:16]}...")
    
    # تنسيق
    print(f"تنسيق بايت: {format_bytes(123456789)}")
    print(f"تنسيق وقت: {format_time(3665.5)}")
    print(f"تنسيق رقم: {format_number(1234567)}")
    
    # اقتصاص نص
    long_text = "هذا نص طويل جداً يحتاج إلى اقتصاص ليصبح مناسباً للعرض"
    truncated = truncate_text(long_text, 20)
    print(f"اقتصاص نص: {truncated}")
    
    # تنظيف نص
    messy_text = "  هذا   نص   به   مسافات   زائدة  .وأيضا  علامات ترقيم  غير  صحيحة  !  "
    cleaned = clean_text(messy_text)
    print(f"تنظيف نص: {cleaned}")
    
    print("\n✅ تم اختبار الأدوات المساعدة بنجاح!")