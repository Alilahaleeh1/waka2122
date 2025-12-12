# -*- coding: utf-8 -*-
"""
Tokenizer مخصص - مع دعم متعدد اللغات والعربية
"""

import json
import os
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union
import torch


class Tokenizer:
    """Tokenizer مخصص مع دعم BPE"""
    
    def __init__(self, config: Dict):
        """
        تهيئة Tokenizer
        
        Args:
            config: إعدادات Tokenizer
        """
        self.config = config
        self.vocab_size = config.get("vocab_size", 50000)
        self.special_tokens = config.get("special_tokens", {})
        
        # قوائم المفردات
        self.vocab = {}
        self.inverse_vocab = {}
        
        # رموز خاصة
        self.bos_token = self.special_tokens.get("bos", "<bos>")
        self.eos_token = self.special_tokens.get("eos", "<eos>")
        self.pad_token = self.special_tokens.get("pad", "<pad>")
        self.unk_token = self.special_tokens.get("unk", "<unk>")
        
        # إضافة الرموز الخاصة إلى المفردات
        self._init_special_tokens()
        
        # أنماط Tokenization
        self.pattern = self._build_pattern()
        
        # BPE merges
        self.merges = {}
        self.bpe_ranks = {}
        
        # إحصائيات
        self.stats = {
            "total_tokens": 0,
            "unique_tokens": 0,
            "vocab_loaded": False
        }
    
    def _init_special_tokens(self) -> None:
        """تهيئة الرموز الخاصة"""
        special_tokens_list = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token
        ]
        
        for i, token in enumerate(special_tokens_list):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        
        self.special_token_ids = set(self.vocab.values())
    
    def _build_pattern(self) -> re.Pattern:
        """بناء نمط Tokenization"""
        # نمط بسيط للغات متعددة (يشمل العربية)
        pattern_parts = [
            r"\w+",                     # كلمات
            r"[\u0600-\u06FF]+",        # حروف عربية
            r"\d+",                     # أرقام
            r"[^\w\s\u0600-\u06FF]",    # علامات ترقيم
            r"\s+",                     # مسافات
        ]
        
        return re.compile("|".join(pattern_parts))
    
    def train(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """
        تدريب Tokenizer على نص
        
        Args:
            texts: قائمة النصوص للتدريب
            save_path: مسار حفظ المفردات (اختياري)
        """
        print("🔤 بدء تدريب Tokenizer...")
        
        # تجميع جميع النصوص
        all_text = " ".join(texts)
        
        # Tokenization أولي
        tokens = self.pattern.findall(all_text)
        
        # حساب التكرارات
        word_counts = Counter(tokens)
        
        # بناء المفردات من أكثر الكلمات تكراراً
        most_common = word_counts.most_common(self.vocab_size - len(self.special_token_ids))
        
        # إضافة الكلمات إلى المفردات
        start_idx = len(self.special_token_ids)
        for i, (word, count) in enumerate(most_common):
            idx = start_idx + i
            self.vocab[word] = idx
            self.inverse_vocab[idx] = word
        
        # تدريب BPE (نموذج مبسط)
        self._train_bpe(word_counts)
        
        # تحديث الإحصائيات
        self.stats["total_tokens"] = sum(word_counts.values())
        self.stats["unique_tokens"] = len(self.vocab)
        self.stats["vocab_loaded"] = True
        
        print(f"✅ تم تدريب Tokenizer على {len(texts)} نص")
        print(f"   حجم المفردات: {len(self.vocab)}")
        print(f"   إجمالي الرموز: {self.stats['total_tokens']}")
        
        # حفظ المفردات إذا طُلب
        if save_path:
            self.save(save_path)
    
    def _train_bpe(self, word_counts: Counter, num_merges: int = 10000) -> None:
        """تدريب خوارزمية BPE مبسطة"""
        print("   تدريب BPE...")
        
        # تهيئة الرموز كحروف فردية
        vocab = set()
        for word in word_counts.keys():
            vocab.update(word)
        
        # إضافة الرموز إلى المفردات
        for char in vocab:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
        
        # عمليات الدمج (مبسطة)
        merges = []
        for i in range(min(num_merges, len(vocab) * 10)):
            # هنا يجب تنفيذ خوارزمية BPE كاملة
            # لكننا سنبقيها مبسطة لأجل المثال
            break
        
        self.merges = {i: merge for i, merge in enumerate(merges)}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        تحويل النص إلى رموز
        
        Args:
            text: النص المراد ترميزه
            add_special_tokens: إضافة رموز خاصة
        
        Returns:
            قائمة بالرموز
        """
        # Tokenization باستخدام النمط
        tokens = self.pattern.findall(text)
        
        # تحويل إلى رموز
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.vocab[self.bos_token])
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # تطبيق BPE إذا كان متاحاً
                bpe_tokens = self._apply_bpe(token)
                for bpe_token in bpe_tokens:
                    if bpe_token in self.vocab:
                        token_ids.append(self.vocab[bpe_token])
                    else:
                        token_ids.append(self.vocab[self.unk_token])
        
        if add_special_tokens:
            token_ids.append(self.vocab[self.eos_token])
        
        return token_ids
    
    def _apply_bpe(self, token: str) -> List[str]:
        """تطبيق BPE على رمز"""
        # تطبيق مبسط لـ BPE
        if not self.merges:
            return [token]
        
        # هنا يجب تطبيق عمليات الدمج بالتسلسل
        # لكننا سنرجع الرمز كما هو لأجل المثال
        return [token]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        تحويل الرموز إلى نص
        
        Args:
            token_ids: قائمة الرموز
            skip_special_tokens: تخطي الرموز الخاصة
        
        Returns:
            النص المفكوك
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                
                if skip_special_tokens and token in [self.bos_token, self.eos_token, self.pad_token]:
                    continue
                
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        # تجميع النص
        text = "".join(tokens)
        
        # إصلاح المسافات
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save(self, path: str) -> None:
        """
        حفظ Tokenizer
        
        Args:
            path: مسار الحفظ
        """
        save_data = {
            "vocab": self.vocab,
            "inverse_vocab": self.inverse_vocab,
            "config": self.config,
            "merges": self.merges,
            "stats": self.stats
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ تم حفظ Tokenizer في {path}")
    
    def load(self, path: str) -> None:
        """
        تحميل Tokenizer
        
        Args:
            path: مسار التحميل
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"ملف Tokenizer غير موجود: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        self.vocab = save_data["vocab"]
        self.inverse_vocab = save_data["inverse_vocab"]
        self.config = save_data["config"]
        self.merges = save_data["merges"]
        self.stats = save_data["stats"]
        
        # تحديث الرموز الخاصة
        self.bos_token = self.config["special_tokens"]["bos"]
        self.eos_token = self.config["special_tokens"]["eos"]
        self.pad_token = self.config["special_tokens"]["pad"]
        self.unk_token = self.config["special_tokens"]["unk"]
        
        print(f"✅ تم تحميل Tokenizer من {path}")
        print(f"   حجم المفردات: {len(self.vocab)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        تقسيم النص إلى رموز نصية
        
        Args:
            text: النص المراد تقسيمه
        
        Returns:
            قائمة الرموز النصية
        """
        return self.pattern.findall(text)
    
    def get_vocab_size(self) -> int:
        """الحصول على حجم المفردات"""
        return len(self.vocab)
    
    def pad_sequence(self, sequences: List[List[int]], 
                    max_len: Optional[int] = None,
                    padding_side: str = "right") -> torch.Tensor:
        """
        حشو تسلسل من الرموز
        
        Args:
            sequences: قائمة التسلسلات
            max_len: الحد الأقصى للطول (إذا None، الحد الأقصى للتسلسلات)
            padding_side: جهة الحشو ("right" أو "left")
        
        Returns:
            Tensor محشو
        """
        pad_token_id = self.vocab[self.pad_token]
        
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) > max_len:
                # اقتصاص إذا كان أطول
                padded_seq = seq[:max_len]
            else:
                padded_seq = seq.copy()
                
                # الحشو
                padding_length = max_len - len(seq)
                padding = [pad_token_id] * padding_length
                
                if padding_side == "right":
                    padded_seq = padded_seq + padding
                else:
                    padded_seq = padding + padded_seq
            
            padded_sequences.append(padded_seq)
        
        return torch.tensor(padded_sequences)
    
    def batch_encode(self, texts: List[str], 
                    max_length: Optional[int] = None,
                    truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        ترميز مجموعة من النصوص
        
        Args:
            texts: قائمة النصوص
            max_length: الحد الأقصى للطول
            truncation: اقتصاص النصوص الطويلة
        
        Returns:
            قاموس بـ Tensors
        """
        all_token_ids = []
        all_attention_masks = []
        
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=True)
            
            # اقتصاص إذا طلب
            if max_length and truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            all_token_ids.append(token_ids)
        
        # الحصول على الحد الأقصى للطول
        if max_length is None:
            max_length = max(len(ids) for ids in all_token_ids)
        
        # الحشو
        input_ids = self.pad_sequence(all_token_ids, max_len=max_length)
        
        # إنشاء أقنعة الانتباه
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.vocab[self.pad_token]] = 0
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def print_stats(self) -> None:
        """طباعة إحصائيات Tokenizer"""
        print("=" * 60)
        print("📊 إحصائيات Tokenizer:")
        print("=" * 60)
        print(f"حجم المفردات: {self.get_vocab_size()}")
        print(f"إجمالي الرموز المدربة: {self.stats.get('total_tokens', 0)}")
        print(f"الرموز الفريدة: {self.stats.get('unique_tokens', 0)}")
        print(f"الرموز الخاصة: {list(self.special_tokens.values())}")
        print("=" * 60)


class ArabicTokenizer(Tokenizer):
    """Tokenizer مخصص للغة العربية"""
    
    def __init__(self, config: Dict):
        """تهيئة Tokenizer العربي"""
        super().__init__(config)
        
        # نمط محسن للعربية
        self.pattern = self._build_arabic_pattern()
    
    def _build_arabic_pattern(self) -> re.Pattern:
        """بناء نمط خاص بالعربية"""
        # نمط شامل للعربية مع دعم التشكيل
        arabic_pattern = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
        
        # نمط للكلمات الإنجليزية (للمصطلحات)
        english_pattern = r"\b[A-Za-z]+\b"
        
        # نمط للأرقام
        digit_pattern = r"\d+"
        
        # نمط لعلامات الترقيم
        punctuation_pattern = r"[^\w\s\u0600-\u06FF]"
        
        # نمط للمسافات
        space_pattern = r"\s+"
        
        return re.compile("|".join([
            arabic_pattern,
            english_pattern,
            digit_pattern,
            punctuation_pattern,
            space_pattern
        ]))
    
    def normalize_arabic(self, text: str) -> str:
        """
        تطبيع النص العربي
        
        Args:
            text: النص العربي
        
        Returns:
            النص المطبيع
        """
        # إزالة التشكيل (اختياري)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        
        # تحويل الألف المقصورة إلى ألف
        text = text.replace('ى', 'ا')
        
        # تحويل التاء المربوطة إلى هاء
        text = text.replace('ة', 'ه')
        
        # إزالة التكرار
        text = re.sub(r'(.)\1+', r'\1', text)
        
        return text.strip()


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """
    دالة مساعدة لتحميل Tokenizer
    
    Args:
        tokenizer_path: مسار ملف Tokenizer
    
    Returns:
        Tokenizer محمل
    """
    # تحميل الإعدادات
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        config = json.load(f)["config"]
    
    # تحديد نوع Tokenizer
    tokenizer_type = config.get("type", "standard")
    
    if tokenizer_type == "arabic":
        tokenizer = ArabicTokenizer(config)
    else:
        tokenizer = Tokenizer(config)
    
    tokenizer.load(tokenizer_path)
    return tokenizer


if __name__ == "__main__":
    # اختبار Tokenizer
    config = {
        "vocab_size": 10000,
        "special_tokens": {
            "bos": "<bos>",
            "eos": "<eos>",
            "pad": "<pad>",
            "unk": "<unk>"
        }
    }
    
    tokenizer = Tokenizer(config)
    
    # نص اختبار
    test_texts = [
        "مرحبا بك في DeepSeek Mini!",
        "هذا نموذج لمعالجة اللغة العربية والإنجليزية.",
        "Hello world! كيف الحال؟"
    ]
    
    # تدريب Tokenizer
    tokenizer.train(test_texts)
    
    # اختبار الترميز والتفكيك
    for text in test_texts:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        
        print(f"\nالنص: {text}")
        print(f"الرموز: {token_ids}")
        print(f"المفكوك: {decoded}")
    
    tokenizer.print_stats()