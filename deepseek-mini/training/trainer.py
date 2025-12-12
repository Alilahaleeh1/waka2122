# -*- coding: utf-8 -*-
"""
المدرب - للتدريب الفعلي للنموذج
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import wandb

from ..model.tiny_llm import TinyLLM
from .optimizer import create_optimizer
from .loss import LanguageModelingLoss
from ..utils.device_manager import DeviceManager


class Trainer:
    """مدرب النموذج اللغوي"""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset: torch.utils.data.Dataset,
                 val_dataset: Optional[torch.utils.data.Dataset] = None,
                 config: Dict[str, Any] = None,
                 device: torch.device = None):
        """
        تهيئة المدرب
        
        Args:
            model: النموذج المراد تدريبه
            train_dataset: مجموعة بيانات التدريب
            val_dataset: مجموعة بيانات التحقق
            config: إعدادات التدريب
            device: جهاز التدريب
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # إعدادات التدريب
        self.batch_size = self.config.get('batch_size', 32)
        self.micro_batch_size = self.config.get('micro_batch_size', 4)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 8)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.total_steps = self.config.get('total_steps', 100000)
        self.warmup_steps = self.config.get('warmup_steps', 2000)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.grad_clip = self.config.get('grad_clip', 1.0)
        self.checkpoint_steps = self.config.get('checkpoint_steps', 5000)
        self.eval_steps = self.config.get('eval_steps', 1000)
        self.save_dir = self.config.get('save_dir', './checkpoints')
        
        # إعدادات إضافية
        self.use_amp = self.config.get('use_amp', True)
        self.log_interval = self.config.get('log_interval', 10)
        self.eval_interval = self.config.get('eval_interval', 1000)
        self.save_interval = self.config.get('save_interval', 5000)
        
        # نقل النموذج إلى الجهاز
        self.model = self.model.to(self.device)
        
        # إنشاء DataLoaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        if val_dataset:
            self.val_loader = self._create_dataloader(val_dataset, shuffle=False)
        else:
            self.val_loader = None
        
        # إنشاء المحسن
        self.optimizer = create_optimizer(
            model=self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer_type='adamw'
        )
        
        # إنشاء مجدول معدل التعلم
        self.scheduler = self._create_scheduler()
        
        # دالة الخسارة
        self.criterion = LanguageModelingLoss(ignore_index=0)
        
        # إعداد AMP (Automatic Mixed Precision)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None
        
        # متغيرات التدريب
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # سجل التدريب
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # إعداد حفظ النقاط
        os.makedirs(self.save_dir, exist_ok=True)
        
        # إعداد WandB (اختياري)
        self.use_wandb = self.config.get('use_wandb', False)
        if self.use_wandb:
            self._init_wandb()
        
        print(f"✅ تم تهيئة المدرب")
        print(f"   الجهاز: {self.device}")
        print(f"   حجم الدفعة: {self.batch_size}")
        print(f"   خطوات تراكم التدرج: {self.gradient_accumulation_steps}")
        print(f"   إجمالي الخطوات: {self.total_steps}")
    
    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """إنشاء DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=getattr(dataset, 'collate_fn', None)
        )
    
    def _create_scheduler(self):
        """إنشاء مجدول معدل التعلم"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            # Warmup ثم تناقص خطي
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # تناقص خطي بعد Warmup
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return max(0.0, 1.0 - progress)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _init_wandb(self):
        """تهيئة WandB"""
        try:
            wandb.init(
                project=self.config.get('wandb_project', 'deepseek-mini'),
                name=self.config.get('wandb_name', 'training-run'),
                config=self.config
            )
            print(f"✅ تم تهيئة WandB")
        except Exception as e:
            print(f"⚠️  فشل تهيئة WandB: {e}")
            self.use_wandb = False
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        خطوة تدريب واحدة
        
        Args:
            batch: دفعة البيانات
        
        Returns:
            قيمة الخسارة
        """
        # نقل البيانات إلى الجهاز
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
        labels = batch['labels'].to(self.device) if 'labels' in batch else None
        
        # AMP forward pass
        with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
            
            # تطبيع الخسارة لتراكم التدرج
            loss = loss / self.gradient_accumulation_steps
        
        # AMP backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps  # إعادة القياس
    
    def train_epoch(self) -> float:
        """
        دورة تدريب واحدة
        
        Returns:
            متوسط خسارة الدورة
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # خطوة التدريب
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # تحديث التدرج إذا وصلنا إلى خطوات تراكم التدرج
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # قص التدرج
                if self.grad_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # تحديث الأوزان
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # تحديث مجدول معدل التعلم
                self.scheduler.step()
                
                # إعادة تعيين التدرجات
                self.optimizer.zero_grad()
                
                # تحديث الخطوة العالمية
                self.global_step += 1
                
                # تسجيل الخسارة
                current_lr = self.scheduler.get_last_lr()[0]
                self.learning_rates.append(current_lr)
                
                avg_loss = total_loss / num_batches
                self.train_losses.append(avg_loss)
                
                # تحديث شريط التقدم
                progress_bar.set_postfix({
                    'loss': avg_loss,
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
                
                # تسجيل إلى WandB
                if self.use_wandb and self.global_step % self.log_interval == 0:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': current_lr,
                        'train/step': self.global_step
                    })
                
                # التقييم الدوري
                if self.val_loader and self.global_step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    self.val_losses.append(val_loss)
                    
                    if self.use_wandb:
                        wandb.log({
                            'val/loss': val_loss,
                            'val/step': self.global_step
                        })
                    
                    # حفظ أفضل نموذج
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                
                # حفظ نقطة تفتيش دورية
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint()
                
                # التحقق من إكمال التدريب
                if self.global_step >= self.total_steps:
                    break
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self) -> float:
        """
        تقييم النموذج على مجموعة التحقق
        
        Returns:
            متوسط خسارة التحقق
        """
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # نقل البيانات إلى الجهاز
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
                labels = batch['labels'].to(self.device) if 'labels' in batch else None
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        print(f"\n📊 تقييم - خطوة {self.global_step}: خسارة = {avg_loss:.4f}")
        
        self.model.train()
        return avg_loss
    
    def train(self) -> None:
        """بدء التدريب"""
        print("🚀 بدء التدريب...")
        start_time = time.time()
        
        try:
            while self.global_step < self.total_steps:
                # تدريب دورة واحدة
                train_loss = self.train_epoch()
                
                print(f"\n📈 الدورة {self.epoch} - متوسط الخسارة: {train_loss:.4f}")
                
                # زيادة العداد
                self.epoch += 1
                
                # التحقق من إكمال التدريب
                if self.global_step >= self.total_steps:
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  التدريب توقف بواسطة المستخدم")
        
        finally:
            # التدريب الكامل
            end_time = time.time()
            training_time = end_time - start_time
            
            print(f"\n✅ التدريب مكتمل!")
            print(f"   الوقت الإجمالي: {training_time:.2f} ثانية")
            print(f"   الخطوات: {self.global_step}")
            print(f"   الدورة: {self.epoch}")
            print(f"   أفضل خسارة تحقق: {self.best_val_loss:.4f}")
            
            # حفظ النموذج النهائي
            self.save_checkpoint(is_final=True)
            
            # إغلاق WandB
            if self.use_wandb:
                wandb.finish()
    
    def save_checkpoint(self, 
                       is_best: bool = False, 
                       is_final: bool = False) -> None:
        """
        حفظ نقطة تفتيش
        
        Args:
            is_best: إذا كانت أفضل نقطة تفتيش
            is_final: إذا كانت النقطة النهائية
        """
        checkpoint_name = 'checkpoint'
        if is_best:
            checkpoint_name = 'best_model'
        elif is_final:
            checkpoint_name = 'final_model'
        else:
            checkpoint_name = f'checkpoint_step_{self.global_step}'
        
        checkpoint_path = os.path.join(self.save_dir, f'{checkpoint_name}.pt')
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        # إضافة Scaler إذا كان مستخدمًا
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # حفظ الإعدادات بشكل منفصل
        config_path = os.path.join(self.save_dir, f'{checkpoint_name}_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 تم حفظ {checkpoint_name} إلى {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        تحميل نقطة تفتيش
        
        Args:
            checkpoint_path: مسار نقطة التفتيش
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"نقطة التفتيش غير موجودة: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # تحميل حالة النموذج
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # تحميل حالة المحسن
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # تحميل حالة المجدول
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # تحميل المتغيرات الأخرى
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        # تحميل Scaler إذا كان موجودًا
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ تم تحميل نقطة التفتيش من {checkpoint_path}")
        print(f"   الخطوة: {self.global_step}, الدورة: {self.epoch}")
        print(f"   أفضل خسارة: {self.best_val_loss:.4f}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التدريب"""
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """رسم تاريخ التدريب"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # رسم خسارة التدريب
            axes[0, 0].plot(self.train_losses)
            axes[0, 0].set_title('خسارة التدريب')
            axes[0, 0].set_xlabel('الخطوة')
            axes[0, 0].set_ylabel('الخسارة')
            axes[0, 0].grid(True, alpha=0.3)
            
            # رسم خسارة التحقق
            if self.val_losses:
                axes[0, 1].plot(self.val_losses)
                axes[0, 1].set_title('خسارة التحقق')
                axes[0, 1].set_xlabel('الخطوة')
                axes[0, 1].set_ylabel('الخسارة')
                axes[0, 1].grid(True, alpha=0.3)
            
            # رسم معدل التعلم
            if self.learning_rates:
                axes[1, 0].plot(self.learning_rates)
                axes[1, 0].set_title('معدل التعلم')
                axes[1, 0].set_xlabel('الخطوة')
                axes[1, 0].set_ylabel('معدل التعلم')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_yscale('log')
            
            # رسم Perplexity (إذا كان متاحًا)
            if self.train_losses:
                perplexities = [np.exp(loss) for loss in self.train_losses]
                axes[1, 1].plot(perplexities)
                axes[1, 1].set_title('Perplexity التدريب')
                axes[1, 1].set_xlabel('الخطوة')
                axes[1, 1].set_ylabel('Perplexity')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_yscale('log')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 تم حفظ الرسم إلى {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("⚠️  Matplotlib غير مثبت، لا يمكن رسم الرسوم البيانية")


class DistributedTrainer(Trainer):
    """مدرب موزع للتدريب على أجهزة متعددة"""
    
    def __init__(self, *args, **kwargs):
        """تهيئة المدرب الموزع"""
        super().__init__(*args, **kwargs)
        
        # إعداد التدريب الموزع
        self.local_rank = kwargs.get('local_rank', 0)
        self.world_size = kwargs.get('world_size', 1)
        
        if self.world_size > 1:
            self._setup_distributed()
    
    def _setup_distributed(self):
        """إعداد التدريب الموزع"""
        try:
            import torch.distributed as dist
            dist.init_process_group(backend='nccl')
            
            # نقل النموذج إلى DDP
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            
            print(f"✅ تم إعداد التدريب الموزع (العالم: {self.world_size})")
            
        except Exception as e:
            print(f"⚠️  فشل إعداد التدريب الموزع: {e}")
            self.world_size = 1


def create_trainer(model_config: Dict[str, Any], 
                  data_config: Dict[str, Any],
                  training_config: Dict[str, Any]) -> Trainer:
    """
    إنشاء مدرب من الإعدادات
    
    Args:
        model_config: إعدادات النموذج
        data_config: إعدادات البيانات
        training_config: إعدادات التدريب
    
    Returns:
        مدرب جاهز
    """
    from ..model.tiny_llm import create_model_from_config
    from ..data.dataset import TextDataset
    from ..data.tokenizer import Tokenizer
    
    # إنشاء Tokenizer
    tokenizer_config = data_config.get('tokenizer', {})
    tokenizer = Tokenizer(tokenizer_config)
    
    # إنشاء مجموعات البيانات
    train_dataset = TextDataset(
        data_path=data_config.get('train_path', './data/processed/train.pt'),
        tokenizer=tokenizer,
        max_length=data_config.get('max_length', 2048)
    )
    
    val_dataset = None
    if data_config.get('val_path'):
        val_dataset = TextDataset(
            data_path=data_config.get('val_path'),
            tokenizer=tokenizer,
            max_length=data_config.get('max_length', 2048)
        )
    
    # إنشاء النموذج
    model = create_model_from_config({'model': model_config})
    
    # إنشاء المدرب
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config
    )
    
    return trainer


if __name__ == "__main__":
    # اختبار المدرب
    print("🧪 اختبار المدرب...")
    
    # إنشاء بيانات اختبار
    from ..data.dataset import create_sample_dataset
    from ..model.tiny_llm import TinyLLM
    
    # إنشاء بيانات عينة
    create_sample_dataset(num_samples=100)
    
    # إنشاء نموذج صغير
    model = TinyLLM(
        vocab_size=5000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=256
    )
    
    # إنشاء Tokenizer
    from ..data.tokenizer import Tokenizer
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
    
    # إنشاء مجموعة بيانات
    from ..data.dataset import TextDataset
    dataset = TextDataset(
        data_path="./data/processed/sample.pt",
        tokenizer=tokenizer,
        max_length=128
    )
    
    # تقسيم البيانات
    train_dataset, val_dataset, _ = dataset.split(train_ratio=0.8, val_ratio=0.1)
    
    # إنشاء المدرب
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config={
            'batch_size': 8,
            'micro_batch_size': 2,
            'gradient_accumulation_steps': 4,
            'total_steps': 100,
            'eval_steps': 50,
            'save_interval': 100
        }
    )
    
    # تدريب قصير للاختبار
    print("\n🚀 بدء تدريب اختباري...")
    try:
        trainer.train()
        print("\n✅ تم اختبار المدرب بنجاح!")
    except Exception as e:
        print(f"\n❌ خطأ في اختبار المدرب: {e}")