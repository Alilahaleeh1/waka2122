# -*- coding: utf-8 -*-
"""
مجموعة البيانات - لتحميد ومعالجة بيانات النص
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import os
import json
from pathlib import Path


class TextDataset(Dataset):
    """مجموعة بيانات نصية للتدريب"""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 max_length: int = 2048,
                 stride: int = 512,
                 lazy_loading: bool = False):
        """
        تهيئة مجموعة البيانات
        
        Args:
            data_path: مسار ملف البيانات
            tokenizer: Tokenizer
            max_length: الحد الأقصى لطول التسلسل
            stride: الخطوة عند تقسيم النصوص الطويلة
            lazy_loading: التحميل الكسول للبيانات الكبيرة
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.lazy_loading = lazy_loading
        
        # تحميل البيانات
        self.data = self._load_data()
        
        # تقسيم النصوص الطويلة إلى كتل
        self.sequences = self._create_sequences()
        
        print(f"✅ تم تحميل مجموعة بيانات: {len(self.sequences)} تسلسل")
    
    def _load_data(self) -> List[str]:
        """تحميل البيانات من الملف"""
        print(f"📂 جاري تحميل البيانات من {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ملف البيانات غير موجود: {self.data_path}")
        
        # التحقق من نوع الملف
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        if file_ext == '.pt':
            # ملف PyTorch
            data = torch.load(self.data_path)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'texts' in data:
                return data['texts']
            else:
                raise ValueError(f"تنسيق ملف غير معروف: {self.data_path}")
        
        elif file_ext == '.json':
            # ملف JSON
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # البحث عن مفتاح يحتوي على النصوص
                for key in ['texts', 'data', 'content']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
            
            raise ValueError(f"تنسيق JSON غير معروف: {self.data_path}")
        
        elif file_ext == '.txt':
            # ملف نصي
            with open(self.data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # تنظيف الأسطر
            texts = [line.strip() for line in lines if line.strip()]
            return texts
        
        else:
            raise ValueError(f"امتداد ملف غير مدعوم: {file_ext}")
    
    def _create_sequences(self) -> List[Dict[str, torch.Tensor]]:
        """إنشاء تسلسلات للتدريب"""
        sequences = []
        
        print("🔡 جاري تقسيم النصوص إلى تسلسلات...")
        
        for text in self.data:
            # ترميز النص
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(token_ids) <= self.max_length:
                # إذا كان النص قصيراً، أضفه كما هو
                sequence = self._prepare_sequence(token_ids)
                sequences.append(sequence)
            else:
                # تقسيم النص الطويل مع التداخل
                for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                    chunk = token_ids[i:i + self.max_length]
                    sequence = self._prepare_sequence(chunk)
                    sequences.append(sequence)
        
        return sequences
    
    def _prepare_sequence(self, token_ids: List[int]) -> Dict[str, torch.Tensor]:
        """تحضير تسلسل للتدريب"""
        # المدخلات هي كل الرموز ماعدا الأخير
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        
        # التسميات هي كل الرموز ماعدا الأول (shifted right)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        
        # قناع الانتباه (كلها 1 لأن لا حشو هنا)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def __len__(self) -> int:
        """طول مجموعة البيانات"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """الحصول على عينة"""
        return self.sequences[idx]
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        دالة تجميع للـ DataLoader
        
        Args:
            batch: قائمة العينات
        
        Returns:
            دفعة مجمعة
        """
        # الحصول على الحد الأقصى للطول في الدفعة
        max_len = max(item["input_ids"].size(0) for item in batch)
        
        # حشو جميع Tensors لنفس الطول
        input_ids = []
        attention_masks = []
        labels = []
        
        for item in batch:
            seq_len = item["input_ids"].size(0)
            
            if seq_len < max_len:
                # الحشو
                pad_len = max_len - seq_len
                pad_tensor = torch.full((pad_len,), self.tokenizer.vocab[self.tokenizer.pad_token])
                
                input_ids.append(torch.cat([item["input_ids"], pad_tensor]))
                attention_masks.append(torch.cat([item["attention_mask"], torch.zeros(pad_len)]))
                labels.append(torch.cat([item["labels"], pad_tensor]))
            else:
                input_ids.append(item["input_ids"])
                attention_masks.append(item["attention_mask"])
                labels.append(item["labels"])
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels)
        }
    
    def get_dataloader(self, 
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0) -> DataLoader:
        """
        الحصول على DataLoader
        
        Args:
            batch_size: حجم الدفعة
            shuffle: خلط البيانات
            num_workers: عدد العاملين
        
        Returns:
            DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple['TextDataset', 'TextDataset', 'TextDataset']:
        """
        تقسيم مجموعة البيانات
        
        Args:
            train_ratio: نسبة التدريب
            val_ratio: نسبة التحقق
        
        Returns:
            مجموعات تدريب، تحقق، اختبار
        """
        from copy import deepcopy
        
        total_size = len(self)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # إنشاء مجموعات فرعية
        train_dataset = deepcopy(self)
        train_dataset.sequences = self.sequences[:train_size]
        
        val_dataset = deepcopy(self)
        val_dataset.sequences = self.sequences[train_size:train_size + val_size]
        
        test_dataset = deepcopy(self)
        test_dataset.sequences = self.sequences[train_size + val_size:]
        
        print(f"📊 تقسيم البيانات:")
        print(f"   التدريب: {len(train_dataset)} تسلسل")
        print(f"   التحقق: {len(val_dataset)} تسلسل")
        print(f"   الاختبار: {len(test_dataset)} تسلسل")
        
        return train_dataset, val_dataset, test_dataset
    
    def save(self, path: str) -> None:
        """
        حفظ مجموعة البيانات
        
        Args:
            path: مسار الحفظ
        """
        save_data = {
            'data': self.data,
            'config': {
                'max_length': self.max_length,
                'stride': self.stride
            }
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_data, path)
        print(f"✅ تم حفظ مجموعة البيانات في {path}")
    
    @classmethod
    def load(cls, path: str, tokenizer, **kwargs) -> 'TextDataset':
        """
        تحميل مجموعة البيانات
        
        Args:
            path: مسار التحميل
            tokenizer: Tokenizer
            **kwargs: معاملات إضافية
        
        Returns:
            مجموعة البيانات المحملة
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"ملف البيانات غير موجود: {path}")
        
        data = torch.load(path)
        
        dataset = cls(
            data_path=path,
            tokenizer=tokenizer,
            max_length=data.get('config', {}).get('max_length', 2048),
            stride=data.get('config', {}).get('stride', 512),
            **kwargs
        )
        
        dataset.data = data['data']
        dataset.sequences = dataset._create_sequences()
        
        return dataset


class StreamingTextDataset(Dataset):
    """مجموعة بيانات دفق للبيانات الكبيرة جداً"""
    
    def __init__(self, 
                 data_paths: List[str],
                 tokenizer,
                 max_length: int = 2048):
        """
        تهيئة مجموعة بيانات الدفق
        
        Args:
            data_paths: قائمة مسارات الملفات
            tokenizer: Tokenizer
            max_length: الحد الأقصى للطول
        """
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # فهرسة الملفات
        self.file_index = self._build_file_index()
        
        print(f"✅ تم تحميل مجموعة بيانات دفق: {len(self.file_index)} عينة")
    
    def _build_file_index(self) -> List[Tuple[str, int, int]]:
        """بناء فهرس الملفات"""
        file_index = []
        
        for file_path in self.data_paths:
            if not os.path.exists(file_path):
                print(f"⚠️  ملف غير موجود: {file_path}")
                continue
            
            # حساب عدد الأسطر في الملف
            with open(file_path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            
            # إضافة كل سطر إلى الفهرس
            for line_idx in range(num_lines):
                file_index.append((file_path, line_idx))
        
        return file_index
    
    def __len__(self) -> int:
        """طول مجموعة البيانات"""
        return len(self.file_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """الحصول على عينة"""
        file_path, line_idx = self.file_index[idx]
        
        # قراءة السطر المحدد
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    text = line.strip()
                    break
        
        # ترميز النص
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        # اقتصاص إذا كان طويلاً
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # تحضير التسلسل
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_sample_dataset(output_path: str = "./data/processed/sample.pt", 
                         num_samples: int = 1000) -> None:
    """
    إنشاء مجموعة بيانات عينة للاختبار
    
    Args:
        output_path: مسار الحفظ
        num_samples: عدد العينات
    """
    import random
    import string
    
    # نصوص عينة
    arabic_texts = [
        "الذكاء الاصطناعي هو محاكاة عمليات الذكاء البشري بواسطة الآلات.",
        "التعلم العميق هو جزء من التعلم الآلي الذي يستخدم الشبكات العصبية.",
        "اللغة العربية هي لغة غنية بمفرداتها وتراكيبها النحوية.",
        "النمذجة اللغوية الإحصائية تستخدم للتنبؤ بالكلمات التالية في النص.",
        "المعالجة الطبيعية للغة هي مجال يهتم بتفاعل الحاسوب مع اللغة البشرية."
    ]
    
    english_texts = [
        "Artificial intelligence is the simulation of human intelligence processes by machines.",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Natural Language Processing enables computers to understand human language.",
        "Transformers have revolutionized the field of language modeling.",
        "Attention mechanisms allow models to focus on relevant parts of the input."
    ]
    
    # إنشاء بيانات عشوائية
    samples = []
    
    for _ in range(num_samples):
        # اختيار لغة عشوائية
        if random.random() < 0.5:
            text = random.choice(arabic_texts)
        else:
            text = random.choice(english_texts)
        
        # إضافة بعض التنوع
        words = text.split()
        if len(words) > 3:
            # تغيير ترتيب بعض الكلمات
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            text = " ".join(words)
        
        samples.append(text)
    
    # حفظ البيانات
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"texts": samples}, output_path)
    
    print(f"✅ تم إنشاء مجموعة بيانات عينة: {len(samples)} عينة في {output_path}")


if __name__ == "__main__":
    # اختبار مجموعة البيانات
    from tokenizer import Tokenizer
    
    # إنشاء Tokenizer
    tokenizer_config = {
        "vocab_size": 5000,
        "special_tokens": {
            "bos": "<bos>",
            "eos": "<eos>",
            "pad": "<pad>",
            "unk": "<unk>"
        }
    }
    
    tokenizer = Tokenizer(tokenizer_config)
    
    # إنشاء بيانات عينة
    create_sample_dataset()
    
    # تحميل مجموعة البيانات
    dataset = TextDataset(
        data_path="./data/processed/sample.pt",
        tokenizer=tokenizer,
        max_length=128
    )
    
    # الحصول على DataLoader
    dataloader = dataset.get_dataloader(batch_size=4, shuffle=True)
    
    # اختبار دفعة واحدة
    for batch in dataloader:
        print(f"\nدفعة:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        
        # فك ترميز النص الأول
        first_text = tokenizer.decode(batch['input_ids'][0].tolist())
        print(f"  النص الأول: {first_text[:100]}...")
        
        break
    
    print(f"\n✅ تم اختبار مجموعة البيانات بنجاح!")