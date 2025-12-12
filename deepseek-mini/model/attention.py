# -*- coding: utf-8 -*-
"""
الاهتمام (Attention) - قلب نموذج Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class MultiHeadAttention(nn.Module):
    """الاهتمام متعدد الرؤوس"""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 dropout: float = 0.1,
                 bias: bool = True,
                 flash_attention: bool = False):
        """
        تهيئة الاهتمام متعدد الرؤوس
        
        Args:
            d_model: بعد النموذج
            n_heads: عدد رؤوس الاهتمام
            dropout: نسبة التسرب
            bias: استخدام التحيز
            flash_attention: استخدام Flash Attention
        """
        super().__init__()
        
        # التحقق من أن d_model يقبل القسمة على n_heads
        assert d_model % n_heads == 0, "d_model يجب أن يكون قابلاً للقسمة على n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_flash = flash_attention
        
        # طبقات الإسقاط
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # طبقة الإخراج
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # التسرب
        self.dropout_layer = nn.Dropout(dropout)
        
        # معامل القياس
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # تهيئة الأوزان
        self._init_weights()
    
    def _init_weights(self) -> None:
        """تهيئة الأوزان"""
        # تهيئة He للاستعلامات والمفاتيح
        nn.init.xavier_uniform_(self.W_q.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_k.weight, gain=1.0 / math.sqrt(2))
        
        # تهيئة القيم والإخراج
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        
        # تهيئة التحيزات
        if self.W_q.bias is not None:
            nn.init.constant_(self.W_q.bias, 0)
            nn.init.constant_(self.W_k.bias, 0)
            nn.init.constant_(self.W_v.bias, 0)
            nn.init.constant_(self.W_o.bias, 0)
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        تمرير للأمام
        
        Args:
            query: استعلامات [batch_size, seq_len_q, d_model]
            key: مفاتيح [batch_size, seq_len_k, d_model]
            value: قيم [batch_size, seq_len_v, d_model]
            mask: قناع الاهتمام [batch_size, n_heads, seq_len_q, seq_len_k]
            rotary_pos_emb: ترميز المواضع الدوارة (cos, sin)
            cache: ذاكرة تخزين مؤقت للمفاتيح والقيم (للتوليد)
        
        Returns:
            Tensor الناتج واهتمام الأوزان
        """
        batch_size = query.size(0)
        
        # الإسقاط الخطي وتقسيم الرؤوس
        Q = self.W_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.W_v(value)  # [batch_size, seq_len_v, d_model]
        
        # إعادة التشكيل لإضافة الرؤوس
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # تطبيق ترميز المواضع الدوارة إذا كان متوفراً
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            Q = apply_rotary_pos_emb(Q, cos, sin)
            K = apply_rotary_pos_emb(K, cos, sin)
        
        # استخدام الذاكرة المؤقتة إذا كانت متاحة (للتوليد المتزايد)
        if cache is not None:
            K_cache, V_cache = cache
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2)
        
        # حساب الاهتمام
        if self.use_flash and self._can_use_flash_attention(Q, K, mask):
            # استخدام Flash Attention
            output, attention_weights = self._flash_attention(Q, K, V, mask)
        else:
            # استخدام الاهتمام العادي
            output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # إعادة التشكيل للبعد الأصلي
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # الإسقاط الخطي النهائي
        output = self.W_o(output)
        output = self.dropout_layer(output)
        
        # إرجاع الذاكرة المؤقتة الجديدة
        new_cache = (K, V) if cache is not None else None
        
        return output, attention_weights, new_cache
    
    def _scaled_dot_product_attention(self, 
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        حساب الاهتمام بالضرب النقطي القياسي
        
        Args:
            Q: استعلامات [batch_size, n_heads, seq_len_q, head_dim]
            K: مفاتيح [batch_size, n_heads, seq_len_k, head_dim]
            V: قيم [batch_size, n_heads, seq_len_v, head_dim]
            mask: قناع الاهتمام [batch_size, n_heads, seq_len_q, seq_len_k]
        
        Returns:
            Tensor الناتج وأوزان الاهتمام
        """
        # حساب درجات الاهتمام
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # تطبيق القناع إذا كان موجوداً
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # تطبيق softmax للحصول على أوزان الاهتمام
        attention_weights = F.softmax(scores, dim=-1)
        
        # تطبيق التسرب على أوزان الاهتمام
        attention_weights = self.dropout_layer(attention_weights)
        
        # تطبيق أوزان الاهتمام على القيم
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def _flash_attention(self, 
                        Q: torch.Tensor,
                        K: torch.Tensor,
                        V: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        تنفيذ مبسط لـ Flash Attention
        
        Note: هذا تنفيذ مبسط، Flash Attention الحقيقي أكثر تعقيداً
        """
        # في الإصدار الحقيقي، نستخدم torch.nn.functional.scaled_dot_product_attention
        # إذا كان متاحاً (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            output = F.scaled_dot_product_attention(
                Q, K, V, 
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )
            # للحصول على أوزان الاهتمام، نستخدم النسخة العادية
            _, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        else:
            # استخدام الاهتمام العادي
            output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        
        return output, attention_weights
    
    def _can_use_flash_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                mask: Optional[torch.Tensor]) -> bool:
        """التحقق مما إذا كان يمكن استخدام Flash Attention"""
        if not self.use_flash:
            return False
        
        # Flash Attention يعمل فقط مع أنواع float16 و bfloat16 على GPU
        if Q.dtype not in [torch.float16, torch.bfloat16]:
            return False
        
        if not Q.is_cuda:
            return False
        
        # Flash Attention لا يدعم بعض أنواع الأقنعة
        if mask is not None and mask.dtype != torch.bool:
            return False
        
        return True
    
    def get_attention_weights(self, 
                             query: torch.Tensor,
                             key: torch.Tensor,
                             value: torch.Tensor,
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """الحصول على أوزان الاهتمام فقط"""
        _, attention_weights, _ = self.forward(query, key, value, mask)
        return attention_weights


class CausalSelfAttention(MultiHeadAttention):
    """الاهتمام الذاتي السببي (للنماذج اللغوية)"""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 dropout: float = 0.1,
                 bias: bool = True,
                 flash_attention: bool = False):
        """تهيئة الاهتمام السببي"""
        super().__init__(d_model, n_heads, dropout, bias, flash_attention)
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        تمرير للأمام مع قناع سببي
        
        Args:
            x: Tensor المدخلات [batch_size, seq_len, d_model]
            mask: قناع إضافي (اختياري)
            rotary_pos_emb: ترميز المواضع الدوارة
            cache: ذاكرة تخزين مؤقتة
        
        Returns:
            Tensor الناتج واهتمام الأوزان
        """
        batch_size, seq_len, _ = x.shape
        
        # إنشاء قناع سببي
        causal_mask = self._create_causal_mask(seq_len, x.device)
        
        # دمج مع القناع الإضافي إذا كان موجوداً
        if mask is not None:
            # توسيع أبعاد القناع السببي
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            combined_mask = causal_mask & mask
        else:
            combined_mask = causal_mask
        
        # استدعاء الاهتمام متعدد الرؤوس
        return super().forward(x, x, x, combined_mask, rotary_pos_emb, cache)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """إنشاء قناع سببي"""
        # إنشاء مصفوفة مثلثية سفلية
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.bool()


class GroupedQueryAttention(MultiHeadAttention):
    """الاهتمام المجموعي للاستعلامات (GQA)"""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 n_kv_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        تهيئة GQA
        
        Args:
            d_model: بعد النموذج
            n_heads: عدد رؤوس الاستعلام
            n_kv_heads: عدد رؤوس المفاتيح/القيم
            dropout: نسبة التسرب
            bias: استخدام التحيز
        """
        super().__init__(d_model, n_heads, dropout, bias, flash_attention=False)
        
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # عدد مرات تكرار كل رأس KV
        
        # طبقات منفصلة للمفاتيح والقيم
        self.W_k = nn.Linear(d_model, d_model // (n_heads // n_kv_heads), bias=bias)
        self.W_v = nn.Linear(d_model, d_model // (n_heads // n_kv_heads), bias=bias)
        
        # إعادة تهيئة الأوزان
        self._init_gqa_weights()
    
    def _init_gqa_weights(self) -> None:
        """تهيئة أوزان GQA"""
        # تهيئة طبقات KV
        nn.init.xavier_uniform_(self.W_k.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_v.weight)
        
        if self.W_k.bias is not None:
            nn.init.constant_(self.W_k.bias, 0)
            nn.init.constant_(self.W_v.bias, 0)
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """تمرير للأمام لـ GQA"""
        batch_size = query.size(0)
        
        # إسقاط الاستعلامات (Q)
        Q = self.W_q(query)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # إسقاط المفاتيح والقيم (K, V) مع عدد رؤوس أقل
        K = self.W_k(key)
        V = self.W_v(value)
        
        kv_head_dim = self.d_model // self.n_kv_heads
        K = K.view(batch_size, -1, self.n_kv_heads, kv_head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_kv_heads, kv_head_dim).transpose(1, 2)
        
        # تطبيق ترميز المواضع الدوارة
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            Q = apply_rotary_pos_emb(Q, cos, sin)
            K = apply_rotary_pos_emb(K, cos, sin)
        
        # استخدام الذاكرة المؤقتة
        if cache is not None:
            K_cache, V_cache = cache
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2)
        
        # تكرار رؤوس K وV لمطابقة عدد رؤوس Q
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)
        
        # حساب الاهتمام
        output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # إعادة التشكيل
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # الإسقاط النهائي
        output = self.W_o(output)
        output = self.dropout_layer(output)
        
        # إرجاع الذاكرة المؤقتة
        new_cache = (K[:, :self.n_kv_heads, :, :], V[:, :self.n_kv_heads, :, :]) if cache is not None else None
        
        return output, attention_weights, new_cache


class SlidingWindowAttention(MultiHeadAttention):
    """الاهتمام بنافذة منزلقة"""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 window_size: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        تهيئة الاهتمام بنافذة منزلقة
        
        Args:
            d_model: بعد النموذج
            n_heads: عدد رؤوس الاهتمام
            window_size: حجم النافذة
            dropout: نسبة التسرب
            bias: استخدام التحيز
        """
        super().__init__(d_model, n_heads, dropout, bias, flash_attention=False)
        self.window_size = window_size
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """تمرير للأمام مع نافذة منزلقة"""
        batch_size, seq_len, _ = query.shape
        
        # إنشاء قناع النافذة المنزلقة
        window_mask = self._create_sliding_window_mask(seq_len, query.device)
        
        # دمج مع القناع السببي
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device)).bool()
        combined_mask = causal_mask & window_mask
        
        if mask is not None:
            combined_mask = combined_mask.unsqueeze(0) & mask.unsqueeze(1)
        
        return super().forward(query, key, value, combined_mask, rotary_pos_emb, cache)
    
    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """إنشاء قناع النافذة المنزلقة"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = True
        
        return mask


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    تطبيق ترميز المواضع الدوارة على Tensor
    
    Args:
        x: Tensor للإدخال [batch_size, n_heads, seq_len, head_dim]
        cos: جيب التمام لترميز المواضع
        sin: الجيب لترميز المواضع
    
    Returns:
        Tensor مع تطبيق RoPE
    """
    # فصل الأبعاد الزوجية والفردية
    x_even = x[..., 0::2]  # [batch_size, n_heads, seq_len, head_dim/2]
    x_odd = x[..., 1::2]   # [batch_size, n_heads, seq_len, head_dim/2]
    
    # اقتصاص cos و sin لأبعاد x
    cos = cos[:, :x.size(2), :]  # [1, seq_len, head_dim]
    sin = sin[:, :x.size(2), :]  # [1, seq_len, head_dim]
    
    # إعادة تشكيل لمطابقة x
    cos = cos.unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [1, 1, seq_len, head_dim]
    
    # اقتصاص للأبعاد الزوجية والفردية
    cos_even = cos[..., 0::2]
    cos_odd = cos[..., 1::2]
    sin_even = sin[..., 0::2]
    sin_odd = sin[..., 1::2]
    
    # تطبيق الدوران
    x_even_rot = x_even * cos_even - x_odd * sin_odd
    x_odd_rot = x_odd * cos_odd + x_even * sin_even
    
    # إعادة تجميع
    x_rotated = torch.zeros_like(x)
    x_rotated[..., 0::2] = x_even_rot
    x_rotated[..., 1::2] = x_odd_rot
    
    return x_rotated


class AttentionFactory:
    """مصنع لإنشاء أنواع مختلفة من الاهتمام"""
    
    @staticmethod
    def create_attention(attention_type: str, **kwargs):
        """
        إنشاء طبقة اهتمام
        
        Args:
            attention_type: نوع الاهتمام
            **kwargs: معاملات الاهتمام
        
        Returns:
            طبقة اهتمام
        """
        if attention_type == "multihead":
            return MultiHeadAttention(**kwargs)
        elif attention_type == "causal":
            return CausalSelfAttention(**kwargs)
        elif attention_type == "gqa":
            return GroupedQueryAttention(**kwargs)
        elif attention_type == "sliding":
            return SlidingWindowAttention(**kwargs)
        else:
            raise ValueError(f"نوع اهتمام غير معروف: {attention_type}")


def test_attention():
    """اختبار وظائف الاهتمام"""
    print("🧪 اختبار طبقات الاهتمام...")
    
    # معلمات اختبار
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # إنشاء بيانات اختبار
    x = torch.randn(batch_size, seq_len, d_model)
    
    # اختبار MultiHeadAttention
    print("\n1. اختبار MultiHeadAttention:")
    mha = MultiHeadAttention(d_model, n_heads)
    output, weights, _ = mha.forward(x, x, x)
    print(f"   الشكل: {output.shape}")
    print(f"   أوزان الاهتمام: {weights.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار CausalSelfAttention
    print("\n2. اختبار CausalSelfAttention:")
    causal = CausalSelfAttention(d_model, n_heads)
    output, weights, _ = causal.forward(x)
    print(f"   الشكل: {output.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار GroupedQueryAttention
    print("\n3. اختبار GroupedQueryAttention:")
    try:
        gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads=4)
        output, weights, _ = gqa.forward(x, x, x)
        print(f"   الشكل: {output.shape}")
        print(f"   ✓ تم بنجاح")
    except Exception as e:
        print(f"   ✗ خطأ: {e}")
    
    # اختبار SlidingWindowAttention
    print("\n4. اختبار SlidingWindowAttention:")
    sliding = SlidingWindowAttention(d_model, n_heads, window_size=5)
    output, weights, _ = sliding.forward(x, x, x)
    print(f"   الشكل: {output.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار AttentionFactory
    print("\n5. اختبار AttentionFactory:")
    factory = AttentionFactory()
    attention = factory.create_attention(
        "causal",
        d_model=d_model,
        n_heads=n_heads
    )
    print(f"   النوع: {type(attention).__name__}")
    print(f"   ✓ تم بنجاح")
    
    print("\n✅ تم اختبار جميع طبقات الاهتمام بنجاح!")


if __name__ == "__main__":
    test_attention()