# -*- coding: utf-8 -*-
"""
النموذج اللغوي الصغير - النموذج الرئيسي
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from .embedding import EmbeddingLayer
from .transformer_block import TransformerStack
from .attention import apply_rotary_pos_emb


class TinyLLM(nn.Module):
    """النموذج اللغوي العصبي الصغير"""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 768,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 max_seq_len: int = 2048,
                 dropout: float = 0.1,
                 ffn_dim: int = 3072,
                 use_bias: bool = True,
                 positional_encoding: str = "sinusoidal",
                 use_rmsnorm: bool = False,
                 rotary_emb: bool = False,
                 tie_weights: bool = True):
        """
        تهيئة النموذج اللغوي
        
        Args:
            vocab_size: حجم المفردات
            d_model: بعد النموذج
            n_heads: عدد رؤوس الاهتمام
            n_layers: عدد طبقات Transformer
            max_seq_len: الحد الأقصى لطول التسلسل
            dropout: نسبة التسرب
            ffn_dim: بعد شبكة التغذية الأمامية
            use_bias: استخدام التحيز
            positional_encoding: نوع ترميز المواضع
            use_rmsnorm: استخدام RMSNorm
            rotary_emb: استخدام ترميز المواضع الدوارة
            tie_weights: ربط أوزان المدخلات والمخرجات
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # طبقة التضمين
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            positional_encoding=positional_encoding,
            padding_idx=0
        )
        
        # كومة Transformer
        self.transformer = TransformerStack(
            n_layers=n_layers,
            block_config={
                "block_type": "standard",
                "d_model": d_model,
                "n_heads": n_heads,
                "ffn_dim": ffn_dim,
                "dropout": dropout,
                "activation": "gelu",
                "bias": use_bias,
                "use_rmsnorm": use_rmsnorm,
                "rotary_emb": rotary_emb,
                "max_seq_len": max_seq_len
            }
        )
        
        # تطبيع نهائي
        if use_rmsnorm:
            from .transformer_block import RMSNorm
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)
        
        # رأس اللغة (Language Model Head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # ربط الأوزان إذا طلب
        if tie_weights:
            self.embedding.tie_weights(self.lm_head)
        
        # التسرب النهائي
        self.dropout = nn.Dropout(dropout)
        
        # ذاكرة تخزين مؤقت للتوليد
        self._cache = None
        
        # تهيئة الأوزان
        self._init_weights()
        
        # إحصائيات
        self.total_params = self._count_parameters()
        
        print(f"✅ تم إنشاء TinyLLM:")
        print(f"   المعلمات: {self.total_params:,}")
        print(f"   الطبقات: {n_layers}")
        print(f"   البعد: {d_model}")
        print(f"   الرؤوس: {n_heads}")
    
    def _init_weights(self) -> None:
        """تهيئة أوزان النموذج"""
        # تهيئة رأس اللغة
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # تهيئة التطبيع النهائي
        if isinstance(self.norm, nn.LayerNorm):
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)
    
    def _count_parameters(self) -> int:
        """عد معلمات النموذج"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                start_pos: int = 0) -> Dict[str, torch.Tensor]:
        """
        تمرير للأمام
        
        Args:
            input_ids: رموز المدخلات [batch_size, seq_len]
            attention_mask: قناع الانتباه [batch_size, seq_len]
            labels: تسميات للتدريب [batch_size, seq_len]
            use_cache: استخدام ذاكرة التخزين المؤقت
            start_pos: موضع البداية (للتوليد المتزايد)
        
        Returns:
            قاموس يحتوي على logits وخسائر
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. التضمين
        x = self.embedding(input_ids, start_pos=start_pos)
        
        # 2. قناع الاهتمام السببي
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, 
                                       device=input_ids.device).bool()
        
        # إنشاء قناع سببي
        causal_mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # دمج مع قناع الانتباه
        if attention_mask is not None:
            # توسيع الأبعاد
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            combined_mask = causal_mask & attention_mask
        else:
            combined_mask = causal_mask
        
        # 3. طبقات Transformer
        x, all_attn_weights = self.transformer(
            x, 
            mask=combined_mask,
            use_cache=use_cache,
            start_pos=start_pos
        )
        
        # 4. التطبيع النهائي
        x = self.norm(x)
        
        # 5. رأس اللغة
        logits = self.lm_head(x)
        
        # حساب الخسارة إذا كانت التسميات موجودة
        loss = None
        if labels is not None:
            # اقتصاص logits للتسميات
            logits_for_loss = logits[..., :-1, :].contiguous()
            labels_for_loss = labels[..., 1:].contiguous()
            
            # حساب خسارة التقاطع الإنتروبي
            loss = F.cross_entropy(
                logits_for_loss.view(-1, logits_for_loss.size(-1)),
                labels_for_loss.view(-1),
                ignore_index=0  # تجاهل فهرس الحشو
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "attention_weights": all_attn_weights,
            "hidden_states": x
        }
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """إنشاء قناع سببي"""
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    
    def generate(self, 
                input_ids: torch.Tensor,
                max_new_tokens: int = 100,
                temperature: float = 1.0,
                top_p: float = 1.0,
                top_k: int = 0,
                repetition_penalty: float = 1.0,
                do_sample: bool = True,
                use_cache: bool = True) -> torch.Tensor:
        """
        توليد نص
        
        Args:
            input_ids: رموز البدء [batch_size, seq_len]
            max_new_tokens: الحد الأقصى للرموز المولدة
            temperature: درجة الحرارة للعينة
            top_p: عينة nucleus
            top_k: عينة top-k
            repetition_penalty: عقاب التكرار
            do_sample: أخذ عينات أو اختيار جشع
            use_cache: استخدام ذاكرة التخزين المؤقت
        
        Returns:
            رموز مولدة
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        # إعداد ذاكرة التخزين المؤقت
        if use_cache:
            self.transformer.reset_cache()
        
        with torch.no_grad():
            # توليد الرموز
            generated = input_ids
            
            for i in range(max_new_tokens):
                # الحصول على logits للرموز الحالية
                outputs = self.forward(
                    input_ids=generated if i == 0 else generated[:, -1:],
                    use_cache=use_cache,
                    start_pos=generated.shape[1] - 1 if i > 0 else 0
                )
                
                logits = outputs["logits"]
                
                # الحصول على logits للرمز التالي
                next_token_logits = logits[:, -1, :] / temperature
                
                # تطبيق عقاب التكرار
                if repetition_penalty != 1.0:
                    self._apply_repetition_penalty(next_token_logits, generated, repetition_penalty)
                
                # تطبيق top-k
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    min_logits = top_k_logits[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_logits,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                # تطبيق top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # إزافة الرموز بعد top-p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for idx in range(batch_size):
                        indices_to_remove = sorted_indices[idx][sorted_indices_to_remove[idx]]
                        next_token_logits[idx][indices_to_remove] = float('-inf')
                
                # أخذ العينات أو الاختيار الجشع
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # إضافة الرمز الجديد
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                 generated: torch.Tensor, 
                                 penalty: float) -> None:
        """تطبيق عقاب التكرار"""
        for batch_idx in range(generated.shape[0]):
            for token in generated[batch_idx].unique():
                if token.item() != 0:  # تجاهل الحشو
                    logits[batch_idx, token] /= penalty
    
    def save_pretrained(self, save_path: str) -> None:
        """
        حفظ النموذج
        
        Args:
            save_path: مسار الحفظ
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # حفظ حالة النموذج
        model_path = os.path.join(save_path, "model.pt")
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, model_path)
        
        # حفظ الإعدادات
        config_path = os.path.join(save_path, "config.json")
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_config(), f, indent=2, ensure_ascii=False)
        
        print(f"✅ تم حفظ النموذج في {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> 'TinyLLM':
        """
        تحميل النموذج
        
        Args:
            load_path: مسار التحميل
        
        Returns:
            النموذج المحمل
        """
        import json
        import os
        
        # تحميل الإعدادات
        config_path = os.path.join(load_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # إنشاء النموذج
        model = cls(**config)
        
        # تحميل الأوزان
        model_path = os.path.join(load_path, "model.pt")
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ تم تحميل النموذج من {load_path}")
        return model
    
    def get_config(self) -> Dict[str, Any]:
        """الحصول على إعدادات النموذج"""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout.p if hasattr(self.dropout, 'p') else 0.1,
            "ffn_dim": getattr(self.transformer.blocks[0].ffn, 'fc1', None).out_features 
                      if hasattr(self.transformer.blocks[0], 'ffn') else 3072,
            "use_bias": True,  # يمكن جلبها من الطبقات
            "positional_encoding": self.embedding.positional_encoding_type,
            "use_rmsnorm": isinstance(self.norm, type(self.transformer.blocks[0].norm1)),
            "rotary_emb": hasattr(self.transformer.blocks[0], 'rotary_emb') 
                         and self.transformer.blocks[0].rotary_emb is not None,
            "tie_weights": self.lm_head.weight is self.embedding.token_embedding.embedding.weight
        }
    
    def print_model_info(self) -> None:
        """طباعة معلومات النموذج"""
        print("=" * 60)
        print("🧠 معلومات النموذج:")
        print("=" * 60)
        
        config = self.get_config()
        for key, value in config.items():
            print(f"{key}: {value}")
        
        print(f"\nالمعلمات الإجمالية: {self.total_params:,}")
        
        # تحليل حسب المكونات
        print("\n📊 تحليل المعلمات:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,}")
        
        print("=" * 60)


class TinyLLMForSequenceClassification(TinyLLM):
    """نسخة من النموذج لتصنيف التسلسل"""
    
    def __init__(self, 
                 vocab_size: int,
                 num_labels: int = 2,
                 **kwargs):
        """
        تهيئة النموذج للتصنيف
        
        Args:
            vocab_size: حجم المفردات
            num_labels: عدد الفئات
            **kwargs: معاملات إضافية لـ TinyLLM
        """
        super().__init__(vocab_size, **kwargs)
        
        # رأس التصنيف
        self.classifier = nn.Linear(self.d_model, num_labels)
        
        # رأس الكشف
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        
        # إعادة تهيئة رأس التصنيف
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        تمرير للأمام للتصنيف
        
        Args:
            input_ids: رموز المدخلات
            attention_mask: قناع الانتباه
            labels: تسميات الفئات
        
        Returns:
            قاموس يحتوي على logits وخسائر
        """
        # الحصول على مخرجات النموذج الأساسي
        outputs = super().forward(input_ids, attention_mask)
        
        # الحصول على تمثيل أول رمز ([CLS] أو أول رمز)
        pooled_output = outputs["hidden_states"][:, 0, :]
        
        # تطبيق التسرب والتصنيف
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # حساب الخسارة إذا كانت التسميات موجودة
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs["hidden_states"],
            "attention_weights": outputs["attention_weights"]
        }


class TinyLLMForQuestionAnswering(TinyLLM):
    """نسخة من النموذج للإجابة على الأسئلة"""
    
    def __init__(self, 
                 vocab_size: int,
                 **kwargs):
        """
        تهيئة النموذج للإجابة على الأسئلة
        
        Args:
            vocab_size: حجم المفردات
            **kwargs: معاملات إضافية لـ TinyLLM
        """
        super().__init__(vocab_size, **kwargs)
        
        # رأس بداية ونهاية الإجابة
        self.qa_outputs = nn.Linear(self.d_model, 2)
        
        # إعادة تهيئة
        nn.init.normal_(self.qa_outputs.weight, mean=0.0, std=0.02)
        if self.qa_outputs.bias is not None:
            nn.init.zeros_(self.qa_outputs.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        تمرير للأمام للإجابة على الأسئلة
        
        Args:
            input_ids: رموز المدخلات
            attention_mask: قناع الانتباه
            start_positions: مواقع بداية الإجابة
            end_positions: مواقع نهاية الإجابة
        
        Returns:
            قاموس يحتوي على logits وخسائر
        """
        # الحصول على مخرجات النموذج الأساسي
        outputs = super().forward(input_ids, attention_mask)
        
        # الحصول على logits للبداية والنهاية
        seq_output = outputs["hidden_states"]
        logits = self.qa_outputs(seq_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # حساب الخسارة إذا كانت المواقع موجودة
        loss = None
        if start_positions is not None and end_positions is not None:
            # تجاهل الحشو في حساب الخسارة
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "loss": loss,
            "hidden_states": outputs["hidden_states"],
            "attention_weights": outputs["attention_weights"]
        }


def create_model_from_config(config: Dict[str, Any]) -> TinyLLM:
    """
    إنشاء نموذج من إعدادات
    
    Args:
        config: إعدادات النموذج
    
    Returns:
        نموذج TinyLLM
    """
    model_config = config.get("model", {})
    
    return TinyLLM(
        vocab_size=model_config.get("vocab_size", 50000),
        d_model=model_config.get("d_model", 768),
        n_heads=model_config.get("n_heads", 12),
        n_layers=model_config.get("n_layers", 12),
        max_seq_len=model_config.get("max_seq_len", 2048),
        dropout=model_config.get("dropout", 0.1),
        ffn_dim=model_config.get("ffn_dim", 3072),
        use_bias=model_config.get("use_bias", True),
        positional_encoding=model_config.get("positional_encoding", "sinusoidal"),
        use_rmsnorm=model_config.get("use_rmsnorm", False),
        rotary_emb=model_config.get("rotary_emb", False),
        tie_weights=model_config.get("tie_weights", True)
    )


def test_tiny_llm():
    """اختبار النموذج اللغوي"""
    print("🧪 اختبار TinyLLM...")
    
    # معلمات اختبار
    batch_size = 2
    seq_len = 20
    vocab_size = 5000
    
    # إنشاء بيانات اختبار
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len).bool()
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # إنشاء نموذج صغير للاختبار
    print("\n1. اختبار TinyLLM الأساسي:")
    model = TinyLLM(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_seq_len=512,
        dropout=0.0  # إيقاف التسرب للاختبار
    )
    
    # اختبار التمرير الأمامي
    outputs = model(input_ids, attention_mask, labels)
    print(f"   logits shape: {outputs['logits'].shape}")
    print(f"   loss: {outputs['loss']}")
    print(f"   عدد أوزان الاهتمام: {len(outputs['attention_weights'])}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار التوليد
    print("\n2. اختبار توليد النص:")
    generated = model.generate(
        input_ids=input_ids[:, :5],  # 5 رموز بداية
        max_new_tokens=10,
        temperature=0.8,
        do_sample=True
    )
    print(f"   الشكل المولد: {generated.shape}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار حفظ وتحميل النموذج
    print("\n3. اختبار حفظ وتحميل النموذج:")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # حفظ النموذج
        model.save_pretrained(tmpdir)
        
        # تحميل النموذج
        loaded_model = TinyLLM.from_pretrained(tmpdir)
        
        # مقارنة المخرجات
        outputs1 = model(input_ids, attention_mask)
        outputs2 = loaded_model(input_ids, attention_mask)
        
        # التحقق من التطابق
        logits_diff = torch.abs(outputs1['logits'] - outputs2['logits']).max().item()
        print(f"   الفرق الأقصى في logits: {logits_diff:.6f}")
        
        if logits_diff < 1e-5:
            print(f"   ✓ تم بنجاح")
        else:
            print(f"   ✗ خطأ: الفرق كبير جداً")
    
    # طباعة معلومات النموذج
    print("\n4. معلومات النموذج:")
    model.print_model_info()
    
    print("\n✅ تم اختبار TinyLLM بنجاح!")


if __name__ == "__main__":
    test_tiny_llm()