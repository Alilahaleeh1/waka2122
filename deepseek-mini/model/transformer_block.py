# -*- coding: utf-8 -*-
"""
كتلة Transformer - اللبنة الأساسية للنماذج اللغوية
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .attention import CausalSelfAttention
from .embedding import RotaryPositionalEmbedding


class FeedForward(nn.Module):
    """شبكة التغذية الأمامية"""
    
    def __init__(self, 
                 d_model: int, 
                 ffn_dim: int, 
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 bias: bool = True):
        """
        تهيئة شبكة التغذية الأمامية
        
        Args:
            d_model: بعد النموذج
            ffn_dim: بعد الطبقة المخفية
            dropout: نسبة التسرب
            activation: دالة التنشيط
            bias: استخدام التحيز
        """
        super().__init__()
        
        # الطبقة الأولى
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=bias)
        
        # الطبقة الثانية
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=bias)
        
        # التسرب
        self.dropout = nn.Dropout(dropout)
        
        # التنشيط
        self.activation_fn = self._get_activation_fn(activation)
        
        # تهيئة الأوزان
        self._init_weights()
    
    def _get_activation_fn(self, activation: str):
        """الحصول على دالة التنشيط"""
        if activation == "gelu":
            return F.gelu
        elif activation == "relu":
            return F.relu
        elif activation == "silu" or activation == "swish":
            return F.silu
        elif activation == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"دالة تنشيط غير معروفة: {activation}")
    
    def _init_weights(self) -> None:
        """تهيئة الأوزان"""
        # تهيئة He للطبقة الأولى
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        
        # تهيئة الطبقة الثانية
        nn.init.xavier_uniform_(self.fc2.weight)
        
        # تهيئة التحيزات
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        تمرير للأمام
        
        Args:
            x: Tensor المدخلات [batch_size, seq_len, d_model]
        
        Returns:
            Tensor المخرجات [batch_size, seq_len, d_model]
        """
        # الطبقة الأولى مع التنشيط
        hidden = self.fc1(x)
        hidden = self.activation_fn(hidden)
        hidden = self.dropout(hidden)
        
        # الطبقة الثانية
        output = self.fc2(hidden)
        output = self.dropout(output)
        
        return output


class RMSNorm(nn.Module):
    """تطبيع RMS (بديل لـ LayerNorm)"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        تهيئة RMSNorm
        
        Args:
            dim: بعد التطبيع
            eps: قيمة epsilon للاستقرار العددي
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        تمرير للأمام
        
        Args:
            x: Tensor المدخلات
        
        Returns:
            Tensor المخرجات
        """
        # حساب RMS (جذر متوسط المربعات)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # التطبيع
        x_norm = x / rms
        
        # التوسيع
        return x_norm * self.weight


class TransformerBlock(nn.Module):
    """كتلة Transformer كاملة"""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 bias: bool = True,
                 use_rmsnorm: bool = False,
                 rotary_emb: bool = False,
                 max_seq_len: int = 2048):
        """
        تهيئة كتلة Transformer
        
        Args:
            d_model: بعد النموذج
            n_heads: عدد رؤوس الاهتمام
            ffn_dim: بعد شبكة التغذية الأمامية
            dropout: نسبة التسرب
            activation: دالة تنشيط الـ FFN
            bias: استخدام التحيز
            use_rmsnorm: استخدام RMSNorm بدلاً من LayerNorm
            rotary_emb: استخدام ترميز المواضع الدوارة
            max_seq_len: الحد الأقصى لطول التسلسل
        """
        super().__init__()
        
        # الاهتمام الذاتي السببي
        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias
        )
        
        # شبكة التغذية الأمامية
        self.ffn = FeedForward(
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation,
            bias=bias
        )
        
        # تطبيع ما قبل الاهتمام
        if use_rmsnorm:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        # التسرب الإضافي
        self.dropout = nn.Dropout(dropout)
        
        # ترميز المواضع الدوارة
        self.rotary_emb = None
        if rotary_emb:
            self.rotary_emb = RotaryPositionalEmbedding(
                d_model=d_model,
                max_seq_len=max_seq_len
            )
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                start_pos: int = 0) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        تمرير للأمام
        
        Args:
            x: Tensor المدخلات [batch_size, seq_len, d_model]
            mask: قناع الاهتمام
            cache: ذاكرة تخزين مؤقت للمفاتيح والقيم
            start_pos: موضع البداية (للتوليد المتزايد)
        
        Returns:
            Tensor المخرجات وذاكرة التخزين المؤقت الجديدة
        """
        # تحضير ترميز المواضع الدوارة
        rotary_pos_emb = None
        if self.rotary_emb is not None:
            # حساب مصفوفة الدوران
            cos, sin = self.rotary_emb._compute_rotary_matrix(start_pos + x.size(1))
            rotary_pos_emb = (cos, sin)
        
        # تطبيع ما قبل الاهتمام
        norm_x = self.norm1(x)
        
        # الاهتمام الذاتي السببي
        attn_output, attn_weights, new_cache = self.attention(
            norm_x, 
            mask=mask,
            rotary_pos_emb=rotary_pos_emb,
            cache=cache
        )
        
        # الاتصال المتبقي مع الاهتمام
        x = x + self.dropout(attn_output)
        
        # تطبيع ما قبل FFN
        norm_x = self.norm2(x)
        
        # شبكة التغذية الأمامية
        ffn_output = self.ffn(norm_x)
        
        # الاتصال المتبقي مع FFN
        x = x + self.dropout(ffn_output)
        
        return x, attn_weights, new_cache


class ParallelTransformerBlock(TransformerBlock):
    """كتلة Transformer متوازية (مثل في PaLM)"""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 bias: bool = True,
                 use_rmsnorm: bool = False,
                 rotary_emb: bool = False,
                 max_seq_len: int = 2048):
        """تهيئة الكتلة المتوازية"""
        super().__init__(d_model, n_heads, ffn_dim, dropout, activation, 
                        bias, use_rmsnorm, rotary_emb, max_seq_len)
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                start_pos: int = 0) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """تمرير للأمام متوازي"""
        # تطبيع المدخلات
        norm_x = self.norm1(x)
        
        # تحضير ترميز المواضع الدوارة
        rotary_pos_emb = None
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb._compute_rotary_matrix(start_pos + x.size(1))
            rotary_pos_emb = (cos, sin)
        
        # تشغيل الاهتمام وFFN بالتوازي
        attn_output, attn_weights, new_cache = self.attention(
            norm_x, 
            mask=mask,
            rotary_pos_emb=rotary_pos_emb,
            cache=cache
        )
        
        ffn_output = self.ffn(norm_x)
        
        # دمج المخرجات
        combined_output = attn_output + ffn_output
        
        # الاتصال المتبقي
        x = x + self.dropout(combined_output)
        
        # تطبيع نهائي
        x = self.norm2(x)
        
        return x, attn_weights, new_cache


class GLUFeedForward(FeedForward):
    """شبكة تغذية أمامية مع بوابة خطية (GLU)"""
    
    def __init__(self, 
                 d_model: int, 
                 ffn_dim: int, 
                 dropout: float = 0.1,
                 activation: str = "silu",
                 bias: bool = True):
        """تهيئة GLU FFN"""
        super().__init__(d_model, ffn_dim, dropout, activation, bias)
        
        # طبقة بوابة إضافية
        self.gate = nn.Linear(d_model, ffn_dim, bias=bias)
        
        # إعادة تهيئة الأوزان
        self._init_glu_weights()
    
    def _init_glu_weights(self) -> None:
        """تهيئة أوزان GLU"""
        nn.init.kaiming_normal_(self.gate.weight, nonlinearity='relu')
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """تمرير للأمام مع GLU"""
        # الطبقة الأولى والبوابة
        hidden = self.fc1(x)
        gate = self.gate(x)
        
        # تطبيق التنشيط والبوابة
        hidden = self.activation_fn(hidden) * gate.sigmoid()
        hidden = self.dropout(hidden)
        
        # الطبقة الثانية
        output = self.fc2(hidden)
        output = self.dropout(output)
        
        return output


class MoEBlock(TransformerBlock):
    """كتلة Transformer مع خبراء متعددين (MoE)"""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 ffn_dim: int,
                 num_experts: int = 8,
                 top_k: int = 2,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 bias: bool = True,
                 use_rmsnorm: bool = False,
                 rotary_emb: bool = False,
                 max_seq_len: int = 2048):
        """
        تهيئة كتلة MoE
        
        Args:
            num_experts: عدد الخبراء
            top_k: عدد الخبراء المراد اختيارهم
        """
        super().__init__(d_model, n_heads, ffn_dim, dropout, activation, 
                        bias, use_rmsnorm, rotary_emb, max_seq_len)
        
        # استبدال FFN العادية بـ MoE
        self.num_experts = num_experts
        self.top_k = top_k
        
        # إنشاء الخبراء
        self.experts = nn.ModuleList([
            FeedForward(d_model, ffn_dim, dropout, activation, bias)
            for _ in range(num_experts)
        ])
        
        # بوابة التوجيه
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # التسرب للبوابة
        self.gate_dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                start_pos: int = 0) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """تمرير للأمام مع MoE"""
        # الاهتمام (كما في الكتلة العادية)
        norm_x = self.norm1(x)
        
        rotary_pos_emb = None
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb._compute_rotary_matrix(start_pos + x.size(1))
            rotary_pos_emb = (cos, sin)
        
        attn_output, attn_weights, new_cache = self.attention(
            norm_x, 
            mask=mask,
            rotary_pos_emb=rotary_pos_emb,
            cache=cache
        )
        
        x = x + self.dropout(attn_output)
        
        # MoE بدلاً من FFN
        norm_x = self.norm2(x)
        
        # حساب أوزان البوابة
        gate_logits = self.gate(norm_x)  # [batch_size, seq_len, num_experts]
        gate_logits = self.gate_dropout(gate_logits)
        
        # اختيار أفضل k خبراء
        top_k_gate_logits, top_k_indices = torch.topk(
            gate_logits, 
            k=self.top_k, 
            dim=-1
        )
        
        # تطبيق softmax على top-k
        gate_weights = F.softmax(top_k_gate_logits, dim=-1)
        
        # تجميع مخرجات الخبراء
        moe_output = torch.zeros_like(norm_x)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[..., i]  # [batch_size, seq_len]
            weights = gate_weights[..., i].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # تطبيق كل خبير على المدخلات المناسبة
            for expert_idx in range(self.num_experts):
                # قناع للمواضع التي تستخدم هذا الخبير
                mask = (expert_indices == expert_idx)
                
                if mask.any():
                    # الحصول على مخرجات الخبير
                    expert_output = self.experts[expert_idx](
                        norm_x * mask.unsqueeze(-1).float()
                    )
                    
                    # إضافة إلى المخرجات مع الأوزان
                    moe_output = moe_output + expert_output * weights * mask.unsqueeze(-1).float()
        
        # الاتصال المتبقي
        x = x + self.dropout(moe_output)
        
        return x, attn_weights, new_cache


class TransformerBlockFactory:
    """مصنع لإنشاء أنواع مختلفة من كتل Transformer"""
    
    @staticmethod
    def create_block(block_type: str, **kwargs):
        """
        إنشاء كتلة Transformer
        
        Args:
            block_type: نوع الكتلة
            **kwargs: معاملات الكتلة
        
        Returns:
            كتلة Transformer
        """
        if block_type == "standard":
            return TransformerBlock(**kwargs)
        elif block_type == "parallel":
            return ParallelTransformerBlock(**kwargs)
        elif block_type == "moe":
            return MoEBlock(**kwargs)
        elif block_type == "glu":
            # استبدال FFN بـ GLU FFN
            kwargs_copy = kwargs.copy()
            ffn_dim = kwargs_copy.pop('ffn_dim')
            d_model = kwargs_copy.pop('d_model')
            
            # إنشاء كتلة عادية ثم استبدال FFN
            block = TransformerBlock(**kwargs_copy)
            block.ffn = GLUFeedForward(d_model, ffn_dim, kwargs_copy.get('dropout', 0.1))
            return block
        else:
            raise ValueError(f"نوع كتلة غير معروف: {block_type}")


class TransformerStack(nn.Module):
    """كومة من كتل Transformer"""
    
    def __init__(self, 
                 n_layers: int,
                 block_config: Dict[str, Any]):
        """
        تهيئة كومة Transformer
        
        Args:
            n_layers: عدد الطبقات
            block_config: إعدادات الكتل
        """
        super().__init__()
        
        self.n_layers = n_layers
        
        # إنشاء الكتل
        self.blocks = nn.ModuleList([
            TransformerBlockFactory.create_block(**block_config)
            for _ in range(n_layers)
        ])
        
        # ذاكرة التخزين المؤقت للتوليد
        self.cache = [None] * n_layers
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                start_pos: int = 0) -> torch.Tensor:
        """
        تمرير للأمام عبر جميع الكتل
        
        Args:
            x: Tensor المدخلات
            mask: قناع الاهتمام
            use_cache: استخدام ذاكرة التخزين المؤقت
            start_pos: موضع البداية
        
        Returns:
            Tensor المخرجات
        """
        all_attn_weights = []
        
        # إعادة تعيين الذاكرة المؤقتة إذا لم نستخدمها
        if not use_cache:
            self.cache = [None] * self.n_layers
        
        for i, block in enumerate(self.blocks):
            # الحصول على الذاكرة المؤقتة لهذه الطبقة
            layer_cache = self.cache[i] if use_cache else None
            
            # تمرير عبر الكتلة
            x, attn_weights, new_cache = block(
                x, 
                mask=mask,
                cache=layer_cache,
                start_pos=start_pos
            )
            
            # تحديث الذاكرة المؤقتة
            if use_cache and new_cache is not None:
                self.cache[i] = new_cache
            
            # تخزين أوزان الاهتمام
            if attn_weights is not None:
                all_attn_weights.append(attn_weights)
        
        return x, all_attn_weights
    
    def reset_cache(self) -> None:
        """إعادة تعيين ذاكرة التخزين المؤقت"""
        self.cache = [None] * self.n_layers
    
    def get_cache(self) -> list:
        """الحصول على ذاكرة التخزين المؤقت"""
        return self.cache
    
    def set_cache(self, cache: list) -> None:
        """تعيين ذاكرة التخزين المؤقت"""
        self.cache = cache


def test_transformer_blocks():
    """اختبار كتل Transformer"""
    print("🧪 اختبار كتل Transformer...")
    
    # معلمات اختبار
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    ffn_dim = 2048
    
    # إنشاء بيانات اختبار
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len).bool()
    
    # اختبار TransformerBlock العادية
    print("\n1. اختبار TransformerBlock:")
    block = TransformerBlock(d_model, n_heads, ffn_dim)
    output, attn_weights, _ = block(x, mask)
    print(f"   الشكل: {output.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار ParallelTransformerBlock
    print("\n2. اختبار ParallelTransformerBlock:")
    parallel_block = ParallelTransformerBlock(d_model, n_heads, ffn_dim)
    output, attn_weights, _ = parallel_block(x, mask)
    print(f"   الشكل: {output.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار كتلة مع GLU
    print("\n3. اختبار كتلة مع GLU:")
    glu_block = TransformerBlockFactory.create_block(
        block_type="glu",
        d_model=d_model,
        n_heads=n_heads,
        ffn_dim=ffn_dim
    )
    output, attn_weights, _ = glu_block(x, mask)
    print(f"   الشكل: {output.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار TransformerStack
    print("\n4. اختبار TransformerStack:")
    stack = TransformerStack(
        n_layers=4,
        block_config={
            "block_type": "standard",
            "d_model": d_model,
            "n_heads": n_heads,
            "ffn_dim": ffn_dim
        }
    )
    output, all_weights = stack(x, mask)
    print(f"   الشكل: {output.shape}")
    print(f"   عدد أوزان الاهتمام: {len(all_weights)}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار مع ذاكرة التخزين المؤقت
    print("\n5. اختبار مع ذاكرة التخزين المؤقت:")
    stack.reset_cache()
    
    # تمرير أول (توليد التسلسل)
    output1, _ = stack(x[:, :5, :], use_cache=True)
    
    # تمرير ثانٍ (استمرار التسلسل)
    output2, _ = stack(x[:, 5:, :], use_cache=True, start_pos=5)
    
    print(f"   الشكل 1: {output1.shape}")
    print(f"   الشكل 2: {output2.shape}")
    print(f"   ✓ تم بنجاح")
    
    print("\n✅ تم اختبار جميع كتل Transformer بنجاح!")


if __name__ == "__main__":
    test_transformer_blocks()