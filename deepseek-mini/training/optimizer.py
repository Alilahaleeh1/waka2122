# -*- coding: utf-8 -*-
"""
المحسنات المخصصة للنموذج اللغوي
"""

import torch
import torch.optim as optim
import math
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict


class AdamW(optim.AdamW):
    """AdamW محسن مع تسخين مخصص وتقليل معدل التعلم"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, warmup_steps=0,
                 total_steps=100000, min_lr=1e-6):
        """
        تهيئة AdamW مع تسخين
        
        Args:
            params: معاملات النموذج
            lr: معدل التعلم الأساسي
            betas: معاملات بيتا لـ Adam
            eps: قيمة epsilon للاستقرار العددي
            weight_decay: تسلل الوزن
            amsgrad: استخدام AMSGrad
            warmup_steps: خطوات التسخين
            total_steps: إجمالي خطوات التدريب
            min_lr: الحد الأدنى لمعدل التعلم
        """
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # حفظ معدل التعلم الأساسي
        self.base_lr = lr
        
        # تسجيل معدلات التعلم الأصلية
        for group in self.param_groups:
            group['initial_lr'] = lr
    
    def step(self, closure=None):
        """خطوة تحديث مع ضبط معدل التعلم"""
        # ضبط معدل التعلم قبل الخطوة
        self._adjust_learning_rate()
        
        # تنفيذ الخطوة الأصلية
        loss = super().step(closure)
        
        # زيادة عداد الخطوة
        self.current_step += 1
        
        return loss
    
    def _adjust_learning_rate(self):
        """ضبط معدل التعلم بناءً على الخطوة الحالية"""
        if self.current_step < self.warmup_steps:
            # مرحلة التسخين: زيادة خطية
            lr_mult = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            # مرحلة التبريد: تناقص جيب التمام
            progress = float(self.current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps))
            lr_mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # حساب معدل التعلم الجديد
        new_lr = self.min_lr + (self.base_lr - self.min_lr) * lr_mult
        
        # تحديث معدل التعلم لجميع مجموعات المعاملات
        for group in self.param_groups:
            group['lr'] = new_lr
    
    def get_lr(self) -> float:
        """الحصول على معدل التعلم الحالي"""
        return self.param_groups[0]['lr']
    
    def state_dict(self):
        """الحصول على حالة المحسن"""
        state = super().state_dict()
        state['current_step'] = self.current_step
        state['warmup_steps'] = self.warmup_steps
        state['total_steps'] = self.total_steps
        state['min_lr'] = self.min_lr
        state['base_lr'] = self.base_lr
        return state
    
    def load_state_dict(self, state_dict):
        """تحميل حالة المحسن"""
        self.current_step = state_dict.pop('current_step', 0)
        self.warmup_steps = state_dict.pop('warmup_steps', 0)
        self.total_steps = state_dict.pop('total_steps', 100000)
        self.min_lr = state_dict.pop('min_lr', 1e-6)
        self.base_lr = state_dict.pop('base_lr', self.param_groups[0]['lr'])
        super().load_state_dict(state_dict)


class Lion(optim.Optimizer):
    """محسن Lion (Evolved Sign Momentum)"""
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        تهيئة محسن Lion
        
        Args:
            params: معاملات النموذج
            lr: معدل التعلم
            betas: معاملات بيتا للزخم
            weight_decay: تسلل الوزن
        """
        if not 0.0 <= lr:
            raise ValueError(f"معدل تعلم غير صالح: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"بيتا 1 غير صالح: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"بيتا 2 غير صالح: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """خطوة تحديث"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # الحالة
                state = self.state[p]
                
                # تهيئة الحالة
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                
                # تحديث الزخم
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # حساب التحديث
                update = exp_avg.sign().add_(p, alpha=weight_decay)
                
                # تحديث المعاملات
                p.add_(update, alpha=-lr)
        
        return loss


class Adafactor(optim.Optimizer):
    """محسن Adafactor (مناسب للنماذج الكبيرة)"""
    
    def __init__(self, params, lr=None, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0, scale_parameter=True, relative_step=True,
                 warmup_init=False):
        """
        تهيئة محسن Adafactor
        
        Args:
            params: معاملات النموذج
            lr: معدل التعلم (استخدم None للخطوة النسبية)
            beta1: معامل بيتا للزخم
            beta2: معامل بيتا للقيم التربيعية
            eps: قيمة epsilon للاستقرار
            weight_decay: تسلل الوزن
            scale_parameter: قياس المعاملات
            relative_step: استخدام الخطوة النسبية
            warmup_init: تهيئة التسخين
        """
        if lr is not None and lr < 0.0:
            raise ValueError(f"معدل تعلم غير صالح: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"بيتا 1 غير صالح: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"بيتا 2 غير صالح: {beta2}")
        
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, scale_parameter=scale_parameter,
            relative_step=relative_step, warmup_init=warmup_init
        )
        super().__init__(params, defaults)
        
        # عداد الخطوة
        self.state['step'] = 0
    
    @torch.no_grad()
    def step(self, closure=None):
        """خطوة تحديث"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.state['step'] += 1
        step = self.state['step']
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adafactor لا يدعم التدرجات المتناثرة')
                
                # الحالة
                state = self.state[p]
                
                # تهيئة الحالة
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] = 0
                
                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']
                
                # تحديث القيم التربيعية
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # حساب RMS
                rms = exp_avg_sq.sqrt().add_(group['eps'])
                
                # حساب معدل التعلم
                if group['relative_step']:
                    # الخطوة النسبية
                    step_size = 1.0 / max(1, state['step'])
                else:
                    step_size = group['lr']
                
                # قياس المعاملات إذا طلب
                if group['scale_parameter']:
                    param_rms = p.data.pow(2).mean().sqrt().clamp(min=group['eps'])
                    step_size = step_size * param_rms
                
                # تحديث الزخم
                exp_avg.mul_(beta1).add_(grad.div(rms), alpha=1 - beta1)
                
                # تحديث المعاملات
                p.data.add_(exp_avg, alpha=-step_size)
                
                # تسلل الوزن
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss


class Sophia(optim.Optimizer):
    """محسن Sophia (ثانوي من الدرجة الثانية)"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, hessian_update_interval=10, 
                 hessian_approx='diagonal'):
        """
        تهيئة محسن Sophia
        
        Args:
            params: معاملات النموذج
            lr: معدل التعلم
            betas: معاملات بيتا
            eps: قيمة epsilon
            weight_decay: تسلل الوزن
            hessian_update_interval: فاصل تحديث Hessian
            hessian_approx: تقريب Hessian ('diagonal' أو 'kfac')
        """
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            hessian_update_interval=hessian_update_interval,
            hessian_approx=hessian_approx
        )
        super().__init__(params, defaults)
        
        # عداد الخطوة
        self.state['step'] = 0
    
    @torch.no_grad()
    def step(self, closure=None):
        """خطوة تحديث"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.state['step'] += 1
        step = self.state['step']
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # الحالة
                state = self.state[p]
                
                # تهيئة الحالة
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)
                
                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                hessian = state['hessian']
                beta1, beta2 = group['betas']
                
                # تحديث الزخم
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # تحديث Hessian بشكل دوري
                if step % group['hessian_update_interval'] == 0:
                    if group['hessian_approx'] == 'diagonal':
                        # تقريب قطري لـ Hessian
                        hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # حساب الخطوة
                hessian_clamped = hessian.clamp(min=group['eps'])
                update = exp_avg / hessian_clamped
                
                # تحديث المعاملات
                p.data.add_(update, alpha=-group['lr'])
                
                # تسلل الوزن
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss


class GradientClipper:
    """أداة قص التدرج"""
    
    @staticmethod
    def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
        """
        قص تدرج المعاملات
        
        Args:
            parameters: معاملات النموذج
            max_norm: الحد الأقصى للقاعدة
            norm_type: نوع القاعدة (2 لـ L2)
        
        Returns:
            القاعدة الأصلية
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    
    @staticmethod
    def clip_grad_value(parameters, clip_value: float):
        """
        قص قيم التدرج
        
        Args:
            parameters: معاملات النموذج
            clip_value: قيمة القص
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)
    
    @staticmethod
    def adaptive_clip(parameters, percentile: float = 90.0):
        """
        قص تدرج تكيفي بناءً على النسبة المئوية
        
        Args:
            parameters: معاملات النموذج
            percentile: النسبة المئوية للقص
        
        Returns:
            قيمة القص المستخدمة
        """
        all_grads = []
        for p in parameters:
            if p.grad is not None:
                all_grads.append(p.grad.abs().flatten())
        
        if not all_grads:
            return 0.0
        
        all_grads = torch.cat(all_grads)
        clip_value = torch.quantile(all_grads, percentile / 100.0).item()
        
        torch.nn.utils.clip_grad_value_(parameters, clip_value)
        return clip_value


class OptimizerFactory:
    """مصنع لإنشاء المحسنات"""
    
    @staticmethod
    def create_optimizer(optimizer_type: str, model: torch.nn.Module, **kwargs):
        """
        إنشاء محسن
        
        Args:
            optimizer_type: نوع المحسن
            model: النموذج
            **kwargs: معاملات المحسن
        
        Returns:
            محسن مكون
        """
        # الحصول على المعاملات
        params = OptimizerFactory._get_parameter_groups(model, kwargs)
        
        # إنشاء المحسن المناسب
        if optimizer_type.lower() == 'adamw':
            return AdamW(params, **kwargs)
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(params, **kwargs)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(params, **kwargs)
        elif optimizer_type.lower() == 'lion':
            return Lion(params, **kwargs)
        elif optimizer_type.lower() == 'adafactor':
            return Adafactor(params, **kwargs)
        elif optimizer_type.lower() == 'sophia':
            return Sophia(params, **kwargs)
        elif optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(params, **kwargs)
        else:
            raise ValueError(f"نوع محسن غير معروف: {optimizer_type}")
    
    @staticmethod
    def _get_parameter_groups(model: torch.nn.Module, config: Dict[str, Any]) -> List[Dict]:
        """
        تجميع معاملات النموذج لمعدلات تعلم مختلفة
        
        Args:
            model: النموذج
            config: إعدادات المحسن
        
        Returns:
            قائمة مجموعات المعاملات
        """
        # معدلات التعلم المختلفة لطبقات مختلفة
        no_decay = ['bias', 'LayerNorm.weight', 'RMSNorm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': config.get('weight_decay', 0.01),
                'lr': config.get('learning_rate', 3e-4)
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
                'lr': config.get('learning_rate', 3e-4)
            }
        ]
        
        # إضافة مجموعات إضافية إذا طلب
        if config.get('layerwise_lr', False):
            # معدلات تعلم مختلفة لكل طبقة
            optimizer_grouped_parameters = OptimizerFactory._create_layerwise_groups(model, config)
        
        return optimizer_grouped_parameters
    
    @staticmethod
    def _create_layerwise_groups(model: torch.nn.Module, config: Dict[str, Any]) -> List[Dict]:
        """إنشاء مجموعات طبقة بطبقة"""
        layers = []
        
        # تجميع الطبقات حسب النوع
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # تحديد نوع الطبقة
            if 'embedding' in name:
                lr = config.get('embedding_lr', config.get('learning_rate', 3e-4) * 0.5)
                weight_decay = config.get('embedding_weight_decay', config.get('weight_decay', 0.01))
            elif 'attention' in name:
                lr = config.get('attention_lr', config.get('learning_rate', 3e-4))
                weight_decay = config.get('attention_weight_decay', config.get('weight_decay', 0.01))
            elif 'norm' in name or 'ln' in name:
                lr = config.get('norm_lr', config.get('learning_rate', 3e-4) * 2.0)
                weight_decay = 0.0  # لا تسلل للطبقات المعيارية
            elif 'bias' in name:
                lr = config.get('bias_lr', config.get('learning_rate', 3e-4))
                weight_decay = 0.0
            else:
                lr = config.get('learning_rate', 3e-4)
                weight_decay = config.get('weight_decay', 0.01)
            
            layers.append({
                'params': [param],
                'lr': lr,
                'weight_decay': weight_decay
            })
        
        return layers


def create_optimizer(model: torch.nn.Module, 
                    learning_rate: float = 3e-4,
                    weight_decay: float = 0.01,
                    optimizer_type: str = 'adamw',
                    **kwargs) -> torch.optim.Optimizer:
    """
    دالة مساعدة لإنشاء محسن
    
    Args:
        model: النموذج
        learning_rate: معدل التعلم
        weight_decay: تسلل الوزن
        optimizer_type: نوع المحسن
        **kwargs: معاملات إضافية
    
    Returns:
        محسن مكون
    """
    config = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        **kwargs
    }
    
    return OptimizerFactory.create_optimizer(optimizer_type, model, **config)


def test_optimizers():
    """اختبار المحسنات"""
    print("🧪 اختبار المحسنات...")
    
    # إنشاء نموذج اختبار صغير
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10)
    )
    
    # اختبار AdamW
    print("\n1. اختبار AdamW:")
    optimizer = AdamW(model.parameters(), lr=1e-3, warmup_steps=10, total_steps=100)
    
    # خطوات تدريب وهمية
    for step in range(5):
        # تدرج وهمي
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param)
        
        optimizer.step()
        print(f"   الخطوة {step}: lr = {optimizer.get_lr():.2e}")
    
    print(f"   ✓ تم بنجاح")
    
    # اختبار Lion
    print("\n2. اختبار Lion:")
    optimizer = Lion(model.parameters(), lr=1e-3)
    
    for step in range(3):
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param)
        
        optimizer.step()
    
    print(f"   ✓ تم بنجاح")
    
    # اختبار OptimizerFactory
    print("\n3. اختبار OptimizerFactory:")
    optimizer = create_optimizer(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_type='adamw',
        warmup_steps=10,
        total_steps=100
    )
    
    print(f"   النوع: {type(optimizer).__name__}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار قص التدرج
    print("\n4. اختبار قص التدرج:")
    clipper = GradientClipper()
    
    # إنشاء تدرجات كبيرة
    for param in model.parameters():
        param.grad = torch.ones_like(param) * 10.0
    
    original_norm = clipper.clip_grad_norm(model.parameters(), max_norm=1.0)
    print(f"   القاعدة الأصلية: {original_norm:.2f}")
    
    # التحقق من القص
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm().item() ** 2
    
    total_norm = total_norm ** 0.5
    print(f"   القاعدة بعد القص: {total_norm:.2f}")
    
    if total_norm <= 1.0 + 1e-6:
        print(f"   ✓ تم بنجاح")
    else:
        print(f"   ✗ فشل القص")
    
    print("\n✅ تم اختبار جميع المحسنات بنجاح!")


if __name__ == "__main__":
    test_optimizers()