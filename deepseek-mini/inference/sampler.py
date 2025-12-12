# -*- coding: utf-8 -*-
"""
أخذ العينات - استراتيجيات أخذ العينات للتوليد
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import math


class Sampler:
    """فئة أساسية لأخذ العينات"""
    
    def __init__(self, temperature: float = 1.0):
        """
        تهيئة أخذ العينات
        
        Args:
            temperature: درجة الحرارة
        """
        self.temperature = temperature
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        أخذ عينات من logits
        
        Args:
            logits: logits النموذج [batch_size, vocab_size]
        
        Returns:
            رموز مختارة [batch_size, 1]
        """
        raise NotImplementedError
    
    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """
        تطبيق درجة الحرارة على logits
        
        Args:
            logits: logits النموذج
        
        Returns:
            logits بعد تطبيق درجة الحرارة
        """
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits


class GreedySampler(Sampler):
    """أخذ عينات جشع (اختيار أعلى احتمال)"""
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """أخذ عينات جشع"""
        logits = self.apply_temperature(logits)
        return torch.argmax(logits, dim=-1, keepdim=True)


class RandomSampler(Sampler):
    """أخذ عينات عشوائية"""
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """أخذ عينات عشوائية"""
        logits = self.apply_temperature(logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class TopKSampler(Sampler):
    """أخذ عينات top-k"""
    
    def __init__(self, k: int = 50, temperature: float = 1.0):
        """
        تهيئة top-k
        
        Args:
            k: عدد العناصر العلوية
            temperature: درجة الحرارة
        """
        super().__init__(temperature)
        self.k = k
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """أخذ عينات top-k"""
        logits = self.apply_temperature(logits)
        
        # الحصول على أعلى k قيمة
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # تطبيق softmax على top-k فقط
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # أخذ عينات من top-k
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
        
        # الحصول على الرموز الفعلية
        sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices)
        
        return sampled_tokens


class TopPSampler(Sampler):
    """أخذ عينات nucleus (top-p)"""
    
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        """
        تهيئة top-p
        
        Args:
            p: احتمالية nucleus
            temperature: درجة الحرارة
        """
        super().__init__(temperature)
        self.p = p
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """أخذ عينات top-p"""
        logits = self.apply_temperature(logits)
        
        # ترتيب logits تنازلياً
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # حساب الاحتمالات التراكمية
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # إزالة الرموز بعد nucleus
        sorted_indices_to_remove = cumulative_probs > self.p
        
        # تأكد من اختيار رمز واحد على الأقل
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # تطبيق القناع
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[..., indices_to_remove] = float('-inf')
        
        # أخذ عينات من الرموز المتبقية
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1)
        
        return sampled_tokens


class TypicalSampler(Sampler):
    """أخذ عينات نموذجي"""
    
    def __init__(self, mass: float = 0.9, temperature: float = 1.0):
        """
        تهيئة أخذ العينات النموذجي
        
        Args:
            mass: كتلة الاحتمال
            temperature: درجة الحرارة
        """
        super().__init__(temperature)
        self.mass = mass
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """أخذ عينات نموذجي"""
        logits = self.apply_temperature(logits)
        
        # حساب الاحتمالات اللوغاريتمية
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # حساب الانتروبيا
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        
        # حساب الانحراف المطلق
        abs_dev = torch.abs(log_probs + entropy)
        
        # ترتيب حسب الانحراف
        sorted_abs_dev, sorted_indices = torch.sort(abs_dev, dim=-1)
        
        # حساب الاحتمالات التراكمية
        sorted_probs = torch.gather(probs, -1, sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # تحديد النطاق النموذجي
        mask = cumulative_probs <= self.mass
        
        # إنشاء قناع للرموز النموذجية
        typical_indices = sorted_indices[mask]
        
        # إذا لم يكن هناك رموز نموذجية، استخدم كل الرموز
        if typical_indices.numel() == 0:
            typical_indices = sorted_indices
        
        # إنشاء قناع logits
        typical_mask = torch.zeros_like(logits, dtype=torch.bool)
        typical_mask.scatter_(-1, typical_indices.unsqueeze(-1), True)
        
        # تطبيق القناع
        masked_logits = torch.where(
            typical_mask,
            logits,
            torch.full_like(logits, float('-inf'))
        )
        
        # أخذ عينات من الرموز النموذجية
        probs = F.softmax(masked_logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1)
        
        return sampled_tokens


class MirostatSampler(Sampler):
    """أخذ عينات Mirostat (للتحكم في الجدة)"""
    
    def __init__(self, tau: float = 3.0, learning_rate: float = 0.1, 
                 temperature: float = 1.0):
        """
        تهيئة Mirostat
        
        Args:
            tau: هدف الانتروبيا
            learning_rate: معدل التعلم
            temperature: درجة الحرارة
        """
        super().__init__(temperature)
        self.tau = tau
        self.learning_rate = learning_rate
        self.error = 0.0
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """أخذ عينات Mirostat"""
        logits = self.apply_temperature(logits)
        
        # حساب الانتروبيا الحالية
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()
        
        # حساب الخطأ
        error = self.tau - entropy
        self.error += error
        
        # تحديث درجة الحرارة
        self.temperature += self.learning_rate * self.error
        
        # إعادة تطبيق درجة الحرارة الجديدة
        logits = logits / self.temperature
        
        # أخذ عينات عشوائية
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1)
        
        return sampled_tokens
    
    def reset(self):
        """إعادة تعيين حالة Mirostat"""
        self.error = 0.0


class RepetitionPenaltySampler(Sampler):
    """أخذ عينات مع عقاب التكرار"""
    
    def __init__(self, penalty: float = 1.1, temperature: float = 1.0):
        """
        تهيئة عقاب التكرار
        
        Args:
            penalty: قوة العقاب (>1 للعقاب، <1 للتشجيع)
            temperature: درجة الحرارة
        """
        super().__init__(temperature)
        self.penalty = penalty
        self.generated_tokens = []
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """أخذ عينات مع عقاب التكرار"""
        logits = self.apply_temperature(logits)
        
        # تطبيق العقاب على الرموز المولدة مسبقاً
        for token in self.generated_tokens:
            logits[0, token] /= self.penalty
        
        # أخذ عينات
        probs = F.softmax(logits, dim=-1)
        sampled_token = torch.multinomial(probs, num_samples=1)
        
        # تحديث الرموز المولدة
        self.generated_tokens.append(sampled_token.item())
        
        return sampled_token
    
    def reset(self):
        """إعادة تعيين الرموز المولدة"""
        self.generated_tokens = []


class BeamSampler:
    """أخذ عينات بالحزمة"""
    
    def __init__(self, num_beams: int = 5, length_penalty: float = 1.0):
        """
        تهيئة أخذ العينات بالحزمة
        
        Args:
            num_beams: عدد الحزم
            length_penalty: عقاب الطول
        """
        self.num_beams = num_beams
        self.length_penalty = length_penalty
    
    def sample(self, logits: torch.Tensor, 
               beam_scores: torch.Tensor,
               beam_sequences: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        أخذ عينات بالحزمة
        
        Args:
            logits: logits النموذج [batch_size * num_beams, vocab_size]
            beam_scores: درجات الحزم الحالية [batch_size * num_beams]
            beam_sequences: تسلسلات الحزم الحالية
        
        Returns:
            تسلسلات ودرجات الحزم الجديدة
        """
        batch_size = beam_scores.size(0) // self.num_beams
        
        # حساب احتمالات اللوغاريتم
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size * num_beams, vocab_size]
        
        # توسيع درجات الحزم
        beam_scores_expanded = beam_scores.unsqueeze(-1)  # [batch_size * num_beams, 1]
        
        # حساب درجات المرشحين
        candidate_scores = beam_scores_expanded + log_probs  # [batch_size * num_beams, vocab_size]
        
        # إعادة التشكيل للمعالجة
        candidate_scores = candidate_scores.view(
            batch_size, self.num_beams * logits.size(-1)
        )
        
        # اختيار أفضل مرشحين
        topk_scores, topk_indices = torch.topk(
            candidate_scores, self.num_beams, dim=-1
        )
        
        # تحويل الفهارس إلى فهارس الحزمة والرمز
        beam_indices = topk_indices // logits.size(-1)
        token_indices = topk_indices % logits.size(-1)
        
        # تحديث تسلسلات الحزم
        new_beam_sequences = []
        for batch_idx in range(batch_size):
            batch_sequences = []
            for beam_idx in range(self.num_beams):
                # فهرس الحزمة الأصلية
                original_beam_idx = beam_indices[batch_idx, beam_idx]
                
                # التسلسل الأصلي
                original_sequence = beam_sequences[batch_idx * self.num_beams + original_beam_idx]
                
                # الرمز الجديد
                new_token = token_indices[batch_idx, beam_idx].unsqueeze(0)
                
                # التسلسل الجديد
                new_sequence = torch.cat([original_sequence, new_token])
                batch_sequences.append(new_sequence)
            
            new_beam_sequences.extend(batch_sequences)
        
        # تطبيق عقاب الطول
        for i, sequence in enumerate(new_beam_sequences):
            length = sequence.size(0)
            topk_scores.view(-1)[i] = topk_scores.view(-1)[i] / (length ** self.length_penalty)
        
        return new_beam_sequences, topk_scores.view(-1)


class SamplerFactory:
    """مصنع لإنشاء استراتيجيات أخذ العينات"""
    
    @staticmethod
    def create_sampler(sampler_type: str, **kwargs) -> Sampler:
        """
        إنشاء أخذ عينات
        
        Args:
            sampler_type: نوع أخذ العينات
            **kwargs: معاملات أخذ العينات
        
        Returns:
            أخذ عينات
        """
        if sampler_type == 'greedy':
            return GreedySampler(**kwargs)
        elif sampler_type == 'random':
            return RandomSampler(**kwargs)
        elif sampler_type == 'top_k':
            return TopKSampler(**kwargs)
        elif sampler_type == 'top_p':
            return TopPSampler(**kwargs)
        elif sampler_type == 'typical':
            return TypicalSampler(**kwargs)
        elif sampler_type == 'mirostat':
            return MirostatSampler(**kwargs)
        elif sampler_type == 'repetition_penalty':
            return RepetitionPenaltySampler(**kwargs)
        else:
            raise ValueError(f"نوع أخذ عينات غير معروف: {sampler_type}")


class DynamicSampler:
    """أخذ عينات ديناميكي يغير الاستراتيجية أثناء التوليد"""
    
    def __init__(self, initial_sampler: Sampler, 
                 change_steps: List[int] = None,
                 sampler_types: List[str] = None):
        """
        تهيئة أخذ العينات الديناميكي
        
        Args:
            initial_sampler: أخذ العينات الأولي
            change_steps: الخطوات لتغيير الاستراتيجية
            sampler_types: أنواع أخذ العينات لكل مرحلة
        """
        self.current_sampler = initial_sampler
        self.change_steps = change_steps or [10, 20, 30]
        self.sampler_types = sampler_types or ['top_k', 'top_p', 'greedy']
        self.step = 0
        
        # إنشاء أخذ العينات لكل مرحلة
        self.samplers = []
        for sampler_type in self.sampler_types:
            sampler = SamplerFactory.create_sampler(sampler_type)
            self.samplers.append(sampler)
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """أخذ عينات مع استراتيجية ديناميكية"""
        # تحديث أخذ العينات إذا لزم الأمر
        for i, change_step in enumerate(self.change_steps):
            if self.step == change_step and i < len(self.samplers):
                self.current_sampler = self.samplers[i]
                break
        
        # أخذ العينات
        result = self.current_sampler.sample(logits)
        
        # زيادة العداد
        self.step += 1
        
        return result
    
    def reset(self):
        """إعادة تعيين حالة أخذ العينات"""
        self.step = 0
        self.current_sampler = self.samplers[0] if self.samplers else None


def test_samplers():
    """اختبار استراتيجيات أخذ العينات"""
    print("🧪 اختبار أخذ العينات...")
    
    # إنشاء logits اختبار
    vocab_size = 100
    logits = torch.randn(1, vocab_size)
    
    # اختبار أخذ العينات الجشع
    print("\n1. اختبار أخذ العينات الجشع:")
    greedy_sampler = GreedySampler()
    greedy_token = greedy_sampler.sample(logits)
    print(f"   الرمز المختار: {greedy_token.item()}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار top-k
    print("\n2. اختبار أخذ العينات top-k:")
    topk_sampler = TopKSampler(k=10)
    topk_token = topk_sampler.sample(logits)
    print(f"   الرمز المختار: {topk_token.item()}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار top-p
    print("\n3. اختبار أخذ العينات top-p:")
    top_p_sampler = TopPSampler(p=0.9)
    top_p_token = top_p_sampler.sample(logits)
    print(f"   الرمز المختار: {top_p_token.item()}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار عقاب التكرار
    print("\n4. اختبار عقاب التكرار:")
    penalty_sampler = RepetitionPenaltySampler(penalty=1.5)
    
    # توليد عدة رموز
    tokens = []
    for _ in range(5):
        token = penalty_sampler.sample(logits)
        tokens.append(token.item())
    
    print(f"   الرموز المولدة: {tokens}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار SamplerFactory
    print("\n5. اختبار SamplerFactory:")
    factory = SamplerFactory()
    
    sampler_types = ['greedy', 'top_k', 'top_p']
    for sampler_type in sampler_types:
        sampler = factory.create_sampler(
            sampler_type,
            k=10,
            p=0.9,
            temperature=0.8
        )
        token = sampler.sample(logits)
        print(f"   {sampler_type}: {token.item()}")
    
    print(f"   ✓ تم بنجاح")
    
    # اختبار أخذ العينات الديناميكي
    print("\n6. اختبار أخذ العينات الديناميكي:")
    dynamic_sampler = DynamicSampler(
        initial_sampler=TopKSampler(k=10),
        change_steps=[2, 4],
        sampler_types=['top_k', 'top_p', 'greedy']
    )
    
    tokens = []
    for i in range(6):
        token = dynamic_sampler.sample(logits)
        tokens.append(token.item())
        print(f"   الخطوة {i}: {token.item()}")
    
    print(f"   ✓ تم بنجاح")
    
    print("\n✅ تم اختبار جميع استراتيجيات أخذ العينات بنجاح!")


if __name__ == "__main__":
    test_samplers()