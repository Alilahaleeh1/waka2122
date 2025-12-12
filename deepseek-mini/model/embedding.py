# -*- coding: utf-8 -*-
"""
التضمين - تضمين الرموز والمواضع للنموذج اللغوي
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    """تضمين الرموز"""
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        """
        تهيئة تضمين الرموز
        
        Args:
            vocab_size: حجم المفردات
            d_model: بعد التضمين
            padding_idx: فهرس الحشو
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            d_model, 
            padding_idx=padding_idx
        )
        self.d_model = d_model
        
        # تهيئة الأوزان
        self._init_weights()
    
    def _init_weights(self) -> None:
        """تهيئة الأوزان"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # تعيين وزن الحشو إلى صفر
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        تمرير للأمام
        
        Args:
            x: Tensor للرموز [batch_size, seq_len]
        
        Returns:
            Tensor للتضمينات [batch_size, seq_len, d_model]
        """
        # تضمين الرموز وتطبيع حسب sqrt(d_model)
        embeddings = self.embedding(x) * math.sqrt(self.d_model)
        return embeddings
    
    def get_embedding_weight(self) -> torch.Tensor:
        """الحصول على أوزان التضمين"""
        return self.embedding.weight
    
    def set_embedding_weight(self, weight: torch.Tensor) -> None:
        """تعيين أوزان التضمين"""
        self.embedding.weight.data.copy_(weight)


class PositionalEncoding(nn.Module):
    """ترميز المواضع الجيبية"""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, dropout: float = 0.1):
        """
        تهيئة ترميز المواضع
        
        Args:
            d_model: بعد النموذج
            max_seq_len: الحد الأقصى لطول التسلسل
            dropout: نسبة التسرب
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # إنشاء مصفوفة ترميز المواضع
        pe = torch.zeros(max_seq_len, d_model)
        
        # حساب المواضع
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # تطبيق الدوال المثلثية
        pe[:, 0::2] = torch.sin(position * div_term)  # الأبعاد الزوجية
        pe[:, 1::2] = torch.cos(position * div_term)  # الأبعاد الفردية
        
        # إضافة بعد الدفعة وتسجيل كمعامل ثابت
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        تمرير للأمام
        
        Args:
            x: Tensor للتضمينات [batch_size, seq_len, d_model]
        
        Returns:
            Tensor مع ترميز المواضع [batch_size, seq_len, d_model]
        """
        # إضافة ترميز المواضع للتضمينات
        x = x + self.pe[:, :x.size(1), :]
        
        # تطبيق التسرب
        return self.dropout(x)
    
    def get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        الحصول على ترميز المواضع لتسلسل معين
        
        Args:
            seq_len: طول التسلسل
        
        Returns:
            Tensor لترميز المواضع [1, seq_len, d_model]
        """
        return self.pe[:, :seq_len, :]


class RotaryPositionalEmbedding(nn.Module):
    """تضمين المواضع الدوارة (RoPE)"""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        """
        تهيئة RoPE
        
        Args:
            d_model: بعد النموذج (يجب أن يكون زوجياً)
            max_seq_len: الحد الأقصى لطول التسلسل
        """
        super().__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model يجب أن يكون زوجياً لـ RoPE، لكنه {d_model}")
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # إنشاء ثيتا للترميز الدوار
        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('theta', theta)
        
        # إنشاء ذاكرة تخزين مؤقت
        self._cache = {}
    
    def _compute_rotary_matrix(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """حساب المصفوفة الدوارة"""
        if seq_len in self._cache:
            return self._cache[seq_len]
        
        # إنشاء المواضع
        positions = torch.arange(seq_len).float()
        
        # حساب زوايا الدوران
        angles = positions.unsqueeze(1) * self.theta.unsqueeze(0)  # [seq_len, d_model/2]
        
        # حساب جيب التمام والجيب
        cos = torch.cos(angles)  # [seq_len, d_model/2]
        sin = torch.sin(angles)  # [seq_len, d_model/2]
        
        # توسيع للأبعاد الزوجية والفردية
        cos = self._repeat_interleave(cos)  # [seq_len, d_model]
        sin = self._repeat_interleave(sin)  # [seq_len, d_model]
        
        # تخزين في الذاكرة المؤقتة
        self._cache[seq_len] = (cos, sin)
        
        return cos, sin
    
    def _repeat_interleave(self, x: torch.Tensor) -> torch.Tensor:
        """تكرار وتداخل القيم للأبعاد الزوجية والفردية"""
        # x: [seq_len, d_model/2]
        return x.repeat_interleave(2, dim=1)  # [seq_len, d_model]
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        تطبيق RoPE على Tensor
        
        Args:
            x: Tensor للإدخال [batch_size, seq_len, d_model]
            start_pos: موضع البداية (للتوليد المتزايد)
        
        Returns:
            Tensor مع تطبيق RoPE
        """
        batch_size, seq_len, d_model = x.shape
        
        if d_model != self.d_model:
            raise ValueError(f"البعد غير متطابق: {d_model} != {self.d_model}")
        
        # الحصول على مصفوفة الدوران
        cos, sin = self._compute_rotary_matrix(start_pos + seq_len)
        
        # اقتصاص لحجم التسلسل
        cos = cos[start_pos:start_pos + seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
        sin = sin[start_pos:start_pos + seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
        
        # فصل الأبعاد الزوجية والفردية
        x_even = x[..., 0::2]  # [batch_size, seq_len, d_model/2]
        x_odd = x[..., 1::2]   # [batch_size, seq_len, d_model/2]
        
        # تطبيق الدوران
        x_even_rot = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
        x_odd_rot = x_odd * cos[..., 1::2] + x_even * sin[..., 1::2]
        
        # إعادة تجميع الأبعاد
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even_rot
        x_rotated[..., 1::2] = x_odd_rot
        
        return x_rotated
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, 
                            start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        تطبيق RoPE على استعلامات ومفاتيح الاهتمام
        
        Args:
            q: استعلامات الاهتمام
            k: مفاتيح الاهتمام
            start_pos: موضع البداية
        
        Returns:
            استعلامات ومفاتيح مع تطبيق RoPE
        """
        q_rotated = self.forward(q, start_pos)
        k_rotated = self.forward(k, start_pos)
        
        return q_rotated, k_rotated


class EmbeddingLayer(nn.Module):
    """طبقة التضمين الكاملة (رموز + مواضع)"""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 max_seq_len: int = 2048,
                 dropout: float = 0.1,
                 positional_encoding: str = "sinusoidal",
                 padding_idx: int = 0):
        """
        تهيئة طبقة التضمين
        
        Args:
            vocab_size: حجم المفردات
            d_model: بعد النموذج
            max_seq_len: الحد الأقصى لطول التسلسل
            dropout: نسبة التسرب
            positional_encoding: نوع ترميز المواضع
            padding_idx: فهرس الحشو
        """
        super().__init__()
        
        # تضمين الرموز
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        
        # ترميز المواضع
        self.positional_encoding_type = positional_encoding
        
        if positional_encoding == "sinusoidal":
            self.positional_encoding = PositionalEncoding(
                d_model, max_seq_len, dropout
            )
        elif positional_encoding == "rotary":
            self.positional_encoding = RotaryPositionalEmbedding(
                d_model, max_seq_len
            )
            # نحتاج إلى تسرب منفصل لـ RoPE
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"نوع ترميز مواضع غير معروف: {positional_encoding}")
        
        # تسرب إضافي
        self.dropout_layer = nn.Dropout(dropout)
        
        # معايير التطبيع (اختياري)
        self.norm = nn.LayerNorm(d_model)
        self.use_norm = True
        
        # التخزين
        self.d_model = d_model
        self.vocab_size = vocab_size
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        تمرير للأمام
        
        Args:
            x: Tensor للرموز [batch_size, seq_len]
            start_pos: موضع البداية (لـ RoPE)
        
        Returns:
            Tensor للتضمينات [batch_size, seq_len, d_model]
        """
        # تضمين الرموز
        token_embeddings = self.token_embedding(x)
        
        # تطبيق ترميز المواضع
        if self.positional_encoding_type == "sinusoidal":
            embeddings = self.positional_encoding(token_embeddings)
        elif self.positional_encoding_type == "rotary":
            # RoPE يطبق لاحقاً في الاهتمام
            embeddings = token_embeddings
            embeddings = self.dropout(embeddings)
        
        # تطبيق التسرب
        embeddings = self.dropout_layer(embeddings)
        
        # التطبيع (اختياري)
        if self.use_norm:
            embeddings = self.norm(embeddings)
        
        return embeddings
    
    def get_input_embeddings(self) -> nn.Module:
        """الحصول على طبقة تضمين المدخلات"""
        return self.token_embedding
    
    def set_input_embeddings(self, embedding: nn.Module) -> None:
        """تعيين طبقة تضمين المدخلات"""
        self.token_embedding = embedding
    
    def tie_weights(self, output_embedding: nn.Module) -> None:
        """ربط أوزان تضمين المدخلات والمخرجات"""
        if isinstance(output_embedding, nn.Linear):
            # ربط مع طبقة خطية
            output_embedding.weight = self.token_embedding.embedding.weight
        elif isinstance(output_embedding, nn.Embedding):
            # ربط مع تضمين آخر
            output_embedding.weight = self.token_embedding.embedding.weight
    
    def compute_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        حساب ترميز المواضع
        
        Args:
            seq_len: طول التسلسل
        
        Returns:
            Tensor لترميز المواضع
        """
        if self.positional_encoding_type == "sinusoidal":
            return self.positional_encoding.get_positional_encoding(seq_len)
        else:
            # لـ RoPE، نحسب المصفوفة الدوارة
            cos, sin = self.positional_encoding._compute_rotary_matrix(seq_len)
            return cos, sin


class AdaptiveEmbedding(nn.Module):
    """تضمين تكيفي للنماذج الكبيرة"""
    
    def __init__(self, vocab_size: int, d_model: int, 
                 cutoffs: list = [20000, 40000], 
                 div_val: float = 4.0,
                 padding_idx: int = 0):
        """
        تهيئة التضمين التكيفي
        
        Args:
            vocab_size: حجم المفردات
            d_model: بعد النموذج
            cutoffs: نقاط قطع لتجميع المفردات
            div_val: قيمة التقسيم لأبعاد المجموعات
            padding_idx: فهرس الحشو
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.cutoffs = cutoffs + [vocab_size]
        self.div_val = div_val
        self.padding_idx = padding_idx
        
        # إنشاء مجموعات التضمين
        self.embeddings = nn.ModuleList()
        self.embedding_dims = []
        
        prev_cutoff = 0
        for i, cutoff in enumerate(self.cutoffs):
            # حساب البعد لهذه المجموعة
            if i == 0:
                # المجموعة الأولى لها البعد الكامل
                dim = d_model
            else:
                # تقسيم البعد حسب div_val
                dim = d_model // (div_val ** i)
            
            # إنشاء تضمين للمجموعة
            embedding = nn.Embedding(
                cutoff - prev_cutoff,
                dim,
                padding_idx=padding_idx if i == 0 else None
            )
            
            self.embeddings.append(embedding)
            self.embedding_dims.append(dim)
            
            prev_cutoff = cutoff
        
        # طبقة إسقاط لمواءمة الأبعاد
        self.projection = nn.ModuleList()
        for i, dim in enumerate(self.embedding_dims):
            if dim != d_model:
                proj = nn.Linear(dim, d_model, bias=False)
                self.projection.append(proj)
            else:
                self.projection.append(nn.Identity())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        تمرير للأمام
        
        Args:
            x: Tensor للرموز [batch_size, seq_len]
        
        Returns:
            Tensor للتضمينات [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.shape
        
        # إنشاء Tensor للإخراج
        output = torch.zeros(batch_size, seq_len, self.d_model, 
                           device=x.device, dtype=torch.float32)
        
        # معالجة كل مجموعة
        for i, embedding in enumerate(self.embeddings):
            # إنشاء قناع للرموز في هذه المجموعة
            if i == 0:
                mask = (x < self.cutoffs[i])
            else:
                mask = (x >= self.cutoffs[i-1]) & (x < self.cutoffs[i])
            
            if mask.any():
                # تحويل الفهارس للمجموعة الحالية
                indices = x[mask]
                if i > 0:
                    indices = indices - self.cutoffs[i-1]
                
                # الحصول على التضمينات
                emb = embedding(indices)
                
                # الإسقاط إذا لزم الأمر
                emb = self.projection[i](emb)
                
                # وضع في المواضع الصحيحة
                output[mask] = emb
        
        return output


def test_embeddings():
    """اختبار وظائف التضمين"""
    print("🧪 اختبار طبقات التضمين...")
    
    # معلمات اختبار
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    d_model = 512
    
    # إنشاء بيانات اختبار
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # اختبار TokenEmbedding
    print("\n1. اختبار TokenEmbedding:")
    token_embedding = TokenEmbedding(vocab_size, d_model)
    token_embeds = token_embedding(x)
    print(f"   الشكل: {token_embeds.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار PositionalEncoding
    print("\n2. اختبار PositionalEncoding:")
    pos_encoding = PositionalEncoding(d_model, max_seq_len=2048)
    pos_embeds = pos_encoding(token_embeds)
    print(f"   الشكل: {pos_embeds.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار RotaryPositionalEmbedding
    print("\n3. اختبار RotaryPositionalEmbedding:")
    try:
        rope = RotaryPositionalEmbedding(d_model)
        rope_embeds = rope(token_embeds)
        print(f"   الشكل: {rope_embeds.shape}")
        print(f"   ✓ تم بنجاح")
    except Exception as e:
        print(f"   ✗ خطأ: {e}")
    
    # اختبار EmbeddingLayer الكاملة
    print("\n4. اختبار EmbeddingLayer:")
    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=2048,
        positional_encoding="sinusoidal"
    )
    full_embeds = embedding_layer(x)
    print(f"   الشكل: {full_embeds.shape}")
    print(f"   ✓ تم بنجاح")
    
    print("\n✅ تم اختبار جميع طبقات التضمين بنجاح!")


if __name__ == "__main__":
    test_embeddings()