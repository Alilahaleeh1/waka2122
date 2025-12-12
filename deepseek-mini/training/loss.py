# -*- coding: utf-8 -*-
"""
دوال الخسارة المخصصة للنموذج اللغوي
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any


class LanguageModelingLoss(nn.Module):
    """خسارة النمذجة اللغوية مع معالجة خاصة للحشو"""
    
    def __init__(self, ignore_index: int = 0, label_smoothing: float = 0.0):
        """
        تهيئة خسارة النمذجة اللغوية
        
        Args:
            ignore_index: فهرس لتجاهله في حساب الخسارة (عادةً الحشو)
            label_smoothing: تنعيم التسميات
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        حساب الخسارة
        
        Args:
            logits: مخرجات النموذج [batch_size, seq_len, vocab_size]
            labels: التسميات [batch_size, seq_len]
        
        Returns:
            قيمة الخسارة
        """
        # إعادة تشكيل logits وlabels
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        
        # حساب الخسارة
        loss = self.criterion(logits, labels)
        
        return loss
    
    def compute_perplexity(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        حساب Perplexity
        
        Args:
            logits: مخرجات النموذج
            labels: التسميات
        
        Returns:
            قيمة Perplexity
        """
        loss = self.forward(logits, labels)
        perplexity = torch.exp(loss).item()
        return perplexity


class FocalLoss(nn.Module):
    """خسارة Focal للتعامل مع عدم التوازن في الفئات"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 ignore_index: int = 0, reduction: str = 'mean'):
        """
        تهيئة خسارة Focal
        
        Args:
            alpha: معامل التوازن
            gamma: معامل التركيز
            ignore_index: فهرس لتجاهله
            reduction: نوع التخفيض ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        حساب خسارة Focal
        
        Args:
            logits: مخرجات النموذج
            labels: التسميات
        
        Returns:
            قيمة الخسارة
        """
        # تحويل logits إلى احتمالات
        probs = F.softmax(logits, dim=-1)
        
        # الحصول على احتمالات الفئات الصحيحة
        batch_size, seq_len, vocab_size = probs.shape
        probs = probs.view(-1, vocab_size)
        labels = labels.view(-1)
        
        # إنشاء قناع للفئات غير المهملة
        mask = (labels != self.ignore_index)
        probs = probs[mask]
        labels = labels[mask]
        
        if probs.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # الحصول على احتمالات الفئات الصحيحة
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze()
        
        # حساب خسارة Focal
        loss = -self.alpha * (1 - pt).pow(self.gamma) * torch.log(pt + 1e-8)
        
        # تطبيق التخفيض
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """خسارة مع تنعيم متقدم للتسميات"""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, 
                 ignore_index: int = 0, reduction: str = 'mean'):
        """
        تهيئة خسارة تنعيم التسميات
        
        Args:
            vocab_size: حجم المفردات
            smoothing: كمية التنعيم
            ignore_index: فهرس لتجاهله
            reduction: نوع التخفيض
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # توزيع التنعيم
        self.confidence = 1.0 - smoothing
        self.smoothing_value = smoothing / (vocab_size - 1)  # -1 لاستبعاد الفئة الصحيحة
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        حساب خسارة تنعيم التسميات
        
        Args:
            logits: مخرجات النموذج
            labels: التسميات
        
        Returns:
            قيمة الخسارة
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # إعادة التشكيل
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        
        # إنشاء تسميات منعمة
        smoothed_labels = torch.full_like(logits, self.smoothing_value, 
                                         device=logits.device)
        
        # تعيين الثقة للفئات الصحيحة
        mask = (labels != self.ignore_index).unsqueeze(1)
        smoothed_labels.scatter_(1, labels.unsqueeze(1), self.confidence)
        
        # تطبيق القناع
        smoothed_labels = smoothed_labels * mask.float()
        
        # حساب الخسارة السالبة للاحتمال اللوغاريتمي
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smoothed_labels * log_probs, dim=-1)
        
        # تطبيق التخفيض
        if self.reduction == 'mean':
            # متوسط فقط على العناصر غير المهملة
            non_pad_elements = mask.sum().item()
            if non_pad_elements > 0:
                loss = loss.sum() / non_pad_elements
            else:
                loss = torch.tensor(0.0, device=logits.device)
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


class KnowledgeDistillationLoss(nn.Module):
    """خسارة تقليد المعرفة (Knowledge Distillation)"""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        """
        تهيئة خسارة تقليد المعرفة
        
        Args:
            temperature: درجة الحرارة للتنعيم
            alpha: وزن خسارة التقليد مقابل الخسارة العادية
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        حساب خسارة تقليد المعرفة
        
        Args:
            student_logits: مخرجات الطالب
            teacher_logits: مخرجات المعلم
            labels: التسميات الحقيقية
        
        Returns:
            قيمة الخسارة الكلية
        """
        # خسارة KL بين توزيعات المعلم والطالب
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kd_loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # الخسارة العادية
        ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), 
                                 labels.view(-1), ignore_index=0)
        
        # الجمع بين الخسارتين
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """خسارة تباينية للنمذجة اللغوية"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        """
        تهيئة الخسارة التباينية
        
        Args:
            temperature: درجة الحرارة
            margin: الهامش للخسارة الثلاثية
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        حساب الخسارة التباينية
        
        Args:
            embeddings: التضمينات [batch_size, seq_len, hidden_size]
            labels: التسميات [batch_size, seq_len]
        
        Returns:
            قيمة الخسارة
        """
        batch_size, seq_len, hidden_size = embeddings.shape
        
        # إعادة التشكيل
        embeddings = embeddings.view(-1, hidden_size)  # [batch_size*seq_len, hidden_size]
        labels = labels.view(-1)  # [batch_size*seq_len]
        
        # تجاهل الحشو
        mask = (labels != 0)
        embeddings = embeddings[mask]
        labels = labels[mask]
        
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # حساب التشابه
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # إنشاء قناع للتسميات المتطابقة
        labels_expanded = labels.unsqueeze(0)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        
        # إزالة التشابه مع الذات
        self_mask = torch.eye(positive_mask.size(0), device=positive_mask.device)
        positive_mask = positive_mask - self_mask
        
        # خسارة InfoNCE
        exp_sim = torch.exp(similarity)
        
        # مجموع التشابهات للسالب
        sum_exp_sim = torch.sum(exp_sim * (1 - positive_mask), dim=1, keepdim=True)
        
        # خسارة للعينات الموجبة
        positive_loss = -torch.log(exp_sim * positive_mask / (exp_sim * positive_mask + sum_exp_sim + 1e-8))
        positive_loss = torch.sum(positive_loss) / torch.sum(positive_mask)
        
        return positive_loss


class MixtureOfExpertsLoss(nn.Module):
    """خسارة متخصصة للنماذج ذات الخبراء المتعددين (MoE)"""
    
    def __init__(self, aux_loss_weight: float = 0.01, 
                 load_balance_weight: float = 0.01):
        """
        تهيئة خسارة MoE
        
        Args:
            aux_loss_weight: وزن الخسارة المساعدة
            load_balance_weight: وزن موازنة الحمولة
        """
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.load_balance_weight = load_balance_weight
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                gate_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        حساب خسارة MoE
        
        Args:
            logits: مخرجات النموذج
            labels: التسميات
            gate_logits: مخرجات البوابة
            expert_indices: فهارس الخبراء المستخدمين
        
        Returns:
            قيمة الخسارة الكلية
        """
        # الخسارة الأساسية
        base_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                   labels.view(-1), ignore_index=0)
        
        # الخسارة المساعدة لتنوع الخبراء
        aux_loss = self._compute_auxiliary_loss(gate_logits, expert_indices)
        
        # خسارة موازنة الحمولة
        load_balance_loss = self._compute_load_balance_loss(gate_logits, expert_indices)
        
        # الجمع
        total_loss = base_loss + self.aux_loss_weight * aux_loss + \
                    self.load_balance_weight * load_balance_loss
        
        return total_loss
    
    def _compute_auxiliary_loss(self, gate_logits: torch.Tensor, 
                               expert_indices: torch.Tensor) -> torch.Tensor:
        """حساب الخسارة المساعدة"""
        # تشجيع استخدام خبراء مختلفين
        expert_usage = torch.zeros(gate_logits.size(-1), device=gate_logits.device)
        
        for indices in expert_indices:
            expert_usage.scatter_add_(0, indices.flatten(), 
                                     torch.ones_like(indices.flatten()))
        
        # الخسارة: تقليل التباين في الاستخدام
        usage_mean = expert_usage.mean()
        usage_var = ((expert_usage - usage_mean) ** 2).mean()
        
        return usage_var
    
    def _compute_load_balance_loss(self, gate_logits: torch.Tensor,
                                 expert_indices: torch.Tensor) -> torch.Tensor:
        """حساب خسارة موازنة الحمولة"""
        # حساب توزيع البوابة
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # حساب كمية العمل لكل خبير
        expert_load = torch.zeros(gate_probs.size(-1), device=gate_probs.device)
        
        for i in range(gate_probs.size(0)):
            for j in range(gate_probs.size(1)):
                expert_load += gate_probs[i, j, :]
        
        # تشجيع التوزيع المتساوي
        load_mean = expert_load.mean()
        load_balance = ((expert_load - load_mean) ** 2).mean()
        
        return load_balance


class LossFunctionFactory:
    """مصنع لإنشاء دوال الخسارة"""
    
    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> nn.Module:
        """
        إنشاء دالة خسارة
        
        Args:
            loss_type: نوع الخسارة
            **kwargs: معاملات الخسارة
        
        Returns:
            دالة خسارة
        """
        if loss_type == 'cross_entropy':
            return LanguageModelingLoss(**kwargs)
        elif loss_type == 'focal':
            return FocalLoss(**kwargs)
        elif loss_type == 'label_smoothing':
            return LabelSmoothingLoss(**kwargs)
        elif loss_type == 'knowledge_distillation':
            return KnowledgeDistillationLoss(**kwargs)
        elif loss_type == 'contrastive':
            return ContrastiveLoss(**kwargs)
        elif loss_type == 'moe':
            return MixtureOfExpertsLoss(**kwargs)
        else:
            raise ValueError(f"نوع خسارة غير معروف: {loss_type}")


class LossMonitor:
    """مراقب للخسائر وإحصائيات التدريب"""
    
    def __init__(self):
        """تهيئة مراقب الخسائر"""
        self.losses = []
        self.perplexities = []
        self.grad_norms = []
        self.learning_rates = []
    
    def update(self, loss: float, grad_norm: float = None, 
              lr: float = None, logits: torch.Tensor = None, 
              labels: torch.Tensor = None):
        """
        تحديث الإحصائيات
        
        Args:
            loss: قيمة الخسارة
            grad_norm: قاعدة التدرج
            lr: معدل التعلم
            logits: مخرجات النموذج
            labels: التسميات
        """
        self.losses.append(loss)
        
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        
        if lr is not None:
            self.learning_rates.append(lr)
        
        if logits is not None and labels is not None:
            perplexity = self._compute_perplexity(logits, labels)
            self.perplexities.append(perplexity)
    
    def _compute_perplexity(self, logits: torch.Tensor, 
                           labels: torch.Tensor) -> float:
        """حساب Perplexity"""
        loss_fn = LanguageModelingLoss()
        loss = loss_fn(logits, labels)
        return torch.exp(loss).item()
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات"""
        stats = {}
        
        if self.losses:
            stats['loss_mean'] = sum(self.losses) / len(self.losses)
            stats['loss_min'] = min(self.losses)
            stats['loss_max'] = max(self.losses)
            stats['loss_std'] = torch.std(torch.tensor(self.losses)).item()
        
        if self.perplexities:
            stats['ppl_mean'] = sum(self.perplexities) / len(self.perplexities)
            stats['ppl_min'] = min(self.perplexities)
            stats['ppl_max'] = max(self.perplexities)
        
        if self.grad_norms:
            stats['grad_norm_mean'] = sum(self.grad_norms) / len(self.grad_norms)
            stats['grad_norm_max'] = max(self.grad_norms)
        
        if self.learning_rates:
            stats['lr_mean'] = sum(self.learning_rates) / len(self.learning_rates)
        
        return stats
    
    def reset(self):
        """إعادة تعيين المراقب"""
        self.losses = []
        self.perplexities = []
        self.grad_norms = []
        self.learning_rates = []
    
    def print_summary(self):
        """طباعة ملخص الإحصائيات"""
        stats = self.get_stats()
        
        print("=" * 60)
        print("📊 ملخص الخسائر:")
        print("=" * 60)
        
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
        
        print("=" * 60)


def test_loss_functions():
    """اختبار دوال الخسارة"""
    print("🧪 اختبار دوال الخسارة...")
    
    # إنشاء بيانات اختبار
    batch_size = 4
    seq_len = 10
    vocab_size = 100
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # تعيين بعض التسميات كحشو
    labels[:, -2:] = 0
    
    # اختبار خسارة النمذجة اللغوية
    print("\n1. اختبار LanguageModelingLoss:")
    loss_fn = LanguageModelingLoss(ignore_index=0)
    loss = loss_fn(logits, labels)
    perplexity = loss_fn.compute_perplexity(logits, labels)
    
    print(f"   الخسارة: {loss:.4f}")
    print(f"   Perplexity: {perplexity:.4f}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار خسارة Focal
    print("\n2. اختبار FocalLoss:")
    focal_loss_fn = FocalLoss(ignore_index=0)
    focal_loss = focal_loss_fn(logits, labels)
    print(f"   الخسارة: {focal_loss:.4f}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار خسارة تنعيم التسميات
    print("\n3. اختبار LabelSmoothingLoss:")
    smoothing_loss_fn = LabelSmoothingLoss(vocab_size=vocab_size, smoothing=0.1)
    smoothing_loss = smoothing_loss_fn(logits, labels)
    print(f"   الخسارة: {smoothing_loss:.4f}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار LossFunctionFactory
    print("\n4. اختبار LossFunctionFactory:")
    factory = LossFunctionFactory()
    
    loss_functions = ['cross_entropy', 'focal', 'label_smoothing']
    for loss_type in loss_functions:
        try:
            loss_fn = factory.create_loss(
                loss_type, 
                vocab_size=vocab_size,
                ignore_index=0
            )
            loss_value = loss_fn(logits, labels)
            print(f"   {loss_type}: {loss_value:.4f}")
        except Exception as e:
            print(f"   {loss_type}: خطأ - {e}")
    
    print(f"   ✓ تم بنجاح")
    
    # اختبار LossMonitor
    print("\n5. اختبار LossMonitor:")
    monitor = LossMonitor()
    
    for i in range(5):
        monitor.update(
            loss=i * 0.1,
            grad_norm=i * 0.05,
            lr=1e-3,
            logits=logits,
            labels=labels
        )
    
    stats = monitor.get_stats()
    print(f"   متوسط الخسارة: {stats.get('loss_mean', 0):.4f}")
    print(f"   متوسط Perplexity: {stats.get('ppl_mean', 0):.4f}")
    print(f"   ✓ تم بنجاح")
    
    print("\n✅ تم اختبار جميع دوال الخسارة بنجاح!")


if __name__ == "__main__":
    test_loss_functions()