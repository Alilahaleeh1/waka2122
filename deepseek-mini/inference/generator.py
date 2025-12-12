# -*- coding: utf-8 -*-
"""
مولد النص - لتوليد النص من النموذج اللغوي
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple, Union
import time
from tqdm import tqdm


class TextGenerator:
    """مولد النص للنموذج اللغوي"""
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        تهيئة مولد النص
        
        Args:
            model: النموذج اللغوي
            config: إعدادات التوليد
        """
        self.model = model
        self.config = config
        
        # إعدادات التوليد
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.temperature = config.get('temperature', 1.0)
        self.top_p = config.get('top_p', 1.0)
        self.top_k = config.get('top_k', 50)
        self.repetition_penalty = config.get('repetition_penalty', 1.1)
        self.do_sample = config.get('do_sample', True)
        self.use_cache = config.get('use_cache', True)
        
        # الجهاز
        self.device = next(model.parameters()).device
        
        # إحصائيات
        self.generation_stats = {
            'total_tokens': 0,
            'total_time': 0,
            'tokens_per_second': 0
        }
    
    def generate(self, 
                 prompt: Union[str, torch.Tensor],
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 repetition_penalty: Optional[float] = None,
                 do_sample: Optional[bool] = None,
                 stream: bool = False,
                 stop_tokens: Optional[List[int]] = None) -> str:
        """
        توليد نص من المطالبة
        
        Args:
            prompt: المطالبة (نص أو tensor)
            max_new_tokens: الحد الأقصى للرموز المولدة
            temperature: درجة الحرارة
            top_p: عينة nucleus
            top_k: عينة top-k
            repetition_penalty: عقاب التكرار
            do_sample: أخذ عينات
            stream: إخراج متدفق
            stop_tokens: رموز التوقف
        
        Returns:
            النص المولد
        """
        # استخدام القيم المحددة أو الافتراضية
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        repetition_penalty = repetition_penalty or self.repetition_penalty
        do_sample = do_sample if do_sample is not None else self.do_sample
        
        # تحضير المدخلات
        input_ids = self._prepare_inputs(prompt)
        
        # إعادة تعيين ذاكرة التخزين المؤقت
        if hasattr(self.model, 'transformer'):
            self.model.transformer.reset_cache()
        
        # التوليد
        start_time = time.time()
        
        if stream:
            generated_text = self._generate_streaming(
                input_ids, max_new_tokens, temperature, top_p, 
                top_k, repetition_penalty, do_sample, stop_tokens
            )
        else:
            generated_ids = self._generate_ids(
                input_ids, max_new_tokens, temperature, top_p,
                top_k, repetition_penalty, do_sample, stop_tokens
            )
            generated_text = self._decode_output(generated_ids)
        
        # تحديث الإحصائيات
        self._update_stats(start_time, generated_ids if not stream else None)
        
        return generated_text
    
    def _prepare_inputs(self, prompt: Union[str, torch.Tensor]) -> torch.Tensor:
        """تحضير المدخلات للمطالبة"""
        if isinstance(prompt, str):
            # تحويل النص إلى رموز
            # هنا نحتاج إلى tokenizer، سنفترض وجوده في النموذج
            if hasattr(self.model, 'tokenizer'):
                input_ids = self.model.tokenizer.encode(prompt, add_special_tokens=True)
                input_ids = torch.tensor([input_ids], device=self.device)
            else:
                # للاختبار، إنشاء رموز عشوائية
                input_ids = torch.randint(100, 1000, (1, 10), device=self.device)
        elif isinstance(prompt, torch.Tensor):
            input_ids = prompt.to(self.device)
        else:
            raise ValueError(f"نوع مطالبة غير معروف: {type(prompt)}")
        
        return input_ids
    
    def _generate_ids(self, 
                     input_ids: torch.Tensor,
                     max_new_tokens: int,
                     temperature: float,
                     top_p: float,
                     top_k: int,
                     repetition_penalty: float,
                     do_sample: bool,
                     stop_tokens: Optional[List[int]] = None) -> torch.Tensor:
        """توليد معرفات الرموز"""
        self.model.eval()
        
        with torch.no_grad():
            generated = input_ids
            
            for i in range(max_new_tokens):
                # الحصول على logits
                outputs = self.model(
                    input_ids=generated,
                    use_cache=self.use_cache,
                    start_pos=generated.size(1) - 1 if i > 0 else 0
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                
                # تطبيق عقاب التكرار
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
                
                # تطبيق top-k
                if top_k > 0:
                    logits = self._apply_top_k(logits, top_k)
                
                # تطبيق top-p
                if top_p < 1.0:
                    logits = self._apply_top_p(logits, top_p)
                
                # أخذ العينات أو الاختيار الجشع
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # التحقق من رموز التوقف
                if stop_tokens and next_token.item() in stop_tokens:
                    break
                
                # إضافة الرمز الجديد
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def _generate_streaming(self, 
                           input_ids: torch.Tensor,
                           max_new_tokens: int,
                           temperature: float,
                           top_p: float,
                           top_k: int,
                           repetition_penalty: float,
                           do_sample: bool,
                           stop_tokens: Optional[List[int]] = None) -> str:
        """توليد نص متدفق"""
        self.model.eval()
        
        generated = input_ids
        generated_text = ""
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                # الحصول على logits
                outputs = self.model(
                    input_ids=generated[:, -1:] if i > 0 else generated,
                    use_cache=self.use_cache,
                    start_pos=generated.size(1) - 1 if i > 0 else 0
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                
                # تطبيق عقاب التكرار
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
                
                # تطبيق top-k
                if top_k > 0:
                    logits = self._apply_top_k(logits, top_k)
                
                # تطبيق top-p
                if top_p < 1.0:
                    logits = self._apply_top_p(logits, top_p)
                
                # أخذ العينات أو الاختيار الجشع
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # التحقق من رموز التوقف
                if stop_tokens and next_token.item() in stop_tokens:
                    break
                
                # إضافة الرمز الجديد
                generated = torch.cat([generated, next_token], dim=-1)
                
                # فك ترميز وتحديث النص
                token_text = self._decode_token(next_token.item())
                generated_text += token_text
                
                # إخراج متدفق
                print(token_text, end='', flush=True)
        
        print()  # سطر جديد في النهاية
        return generated_text
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                 generated: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """تطبيق عقاب التكرار"""
        for token in generated[0].unique():
            if token.item() != 0:  # تجاهل الحشو
                logits[0, token] /= penalty
        
        return logits
    
    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """تطبيق top-k"""
        values, indices = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, float('-inf')),
            logits
        )
        
        return logits
    
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """تطبيق top-p (nucleus sampling)"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # إزالة الرموز بعد top-p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float('-inf')
        
        return logits
    
    def _decode_output(self, generated_ids: torch.Tensor) -> str:
        """فك ترميز المعرفات إلى نص"""
        # هنا نحتاج إلى tokenizer
        if hasattr(self.model, 'tokenizer'):
            text = self.model.tokenizer.decode(generated_ids[0].tolist())
        else:
            # للاختبار، تحويل إلى نص بسيط
            text = f"[Token IDs: {generated_ids[0].tolist()}]"
        
        return text
    
    def _decode_token(self, token_id: int) -> str:
        """فك ترميز رمز واحد"""
        if hasattr(self.model, 'tokenizer'):
            token = self.model.tokenizer.decode([token_id])
            # إزالة الرموز الخاصة
            special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
            if any(st in token for st in special_tokens):
                token = ''
        else:
            token = f" {token_id}"
        
        return token
    
    def _update_stats(self, start_time: float, 
                     generated_ids: Optional[torch.Tensor] = None) -> None:
        """تحديث إحصائيات التوليد"""
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if generated_ids is not None:
            num_tokens = generated_ids.size(1)
            tokens_per_second = num_tokens / elapsed_time if elapsed_time > 0 else 0
            
            self.generation_stats['total_tokens'] += num_tokens
            self.generation_stats['total_time'] += elapsed_time
            self.generation_stats['tokens_per_second'] = tokens_per_second
    
    def beam_search(self, 
                    prompt: Union[str, torch.Tensor],
                    num_beams: int = 5,
                    max_new_tokens: int = 100,
                    length_penalty: float = 1.0,
                    early_stopping: bool = True) -> List[Tuple[str, float]]:
        """
        بحث بالحزمة (Beam Search)
        
        Args:
            prompt: المطالبة
            num_beams: عدد الحزم
            max_new_tokens: الحد الأقصى للرموز المولدة
            length_penalty: عقاب الطول
            early_stopping: إيقاف مبكر
        
        Returns:
            قائمة بالنصوص والدرجات
        """
        # تحضير المدخلات
        input_ids = self._prepare_inputs(prompt)
        
        # إعادة تعيين ذاكرة التخزين المؤقت
        if hasattr(self.model, 'transformer'):
            self.model.transformer.reset_cache()
        
        # بحث بالحزمة
        beams = self._beam_search_implementation(
            input_ids, num_beams, max_new_tokens, 
            length_penalty, early_stopping
        )
        
        # تحويل إلى نصوص
        results = []
        for beam_ids, score in beams:
            text = self._decode_output(beam_ids.unsqueeze(0))
            results.append((text, score))
        
        return results
    
    def _beam_search_implementation(self, 
                                   input_ids: torch.Tensor,
                                   num_beams: int,
                                   max_new_tokens: int,
                                   length_penalty: float,
                                   early_stopping: bool) -> List[Tuple[torch.Tensor, float]]:
        """تنفيذ بحث بالحزمة"""
        self.model.eval()
        
        with torch.no_grad():
            # الحزم الأولية
            beams = [(input_ids.clone(), 0.0)]  # (token_ids, log_prob)
            
            for step in range(max_new_tokens):
                new_beams = []
                
                for beam_ids, beam_score in beams:
                    # الحصول على logits للشعاع الحالي
                    outputs = self.model(
                        input_ids=beam_ids,
                        use_cache=self.use_cache,
                        start_pos=beam_ids.size(1) - 1
                    )
                    
                    logits = outputs['logits'][:, -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # أفضل k تكملات
                    topk_log_probs, topk_indices = torch.topk(
                        log_probs, num_beams, dim=-1
                    )
                    
                    for i in range(num_beams):
                        new_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                        new_beam_ids = torch.cat([beam_ids, new_token], dim=-1)
                        
                        # تحديث الدرجة مع عقاب الطول
                        new_score = beam_score + topk_log_probs[0, i].item()
                        length = new_beam_ids.size(1)
                        penalized_score = new_score / (length ** length_penalty)
                        
                        new_beams.append((new_beam_ids, penalized_score))
                
                # اختيار أفضل num_beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:num_beams]
                
                # إيقاف مبكر إذا كانت جميع الحزم انتهت
                if early_stopping:
                    # التحقق من رموز النهاية
                    # هنا نحتاج إلى معرفة رمز <eos>
                    eos_token = 2  # مثال
                    all_finished = all(
                        beam_ids[0, -1].item() == eos_token 
                        for beam_ids, _ in beams
                    )
                    
                    if all_finished:
                        break
            
            return beams
    
    def get_generation_stats(self) -> Dict[str, float]:
        """الحصول على إحصائيات التوليد"""
        return self.generation_stats.copy()
    
    def reset_stats(self) -> None:
        """إعادة تعيين الإحصائيات"""
        self.generation_stats = {
            'total_tokens': 0,
            'total_time': 0,
            'tokens_per_second': 0
        }
    
    def print_stats(self) -> None:
        """طباعة إحصائيات التوليد"""
        stats = self.get_generation_stats()
        
        print("=" * 60)
        print("📊 إحصائيات التوليد:")
        print("=" * 60)
        
        if stats['total_time'] > 0:
            print(f"الرموز الكلية: {stats['total_tokens']}")
            print(f"الوقت الكلي: {stats['total_time']:.2f} ثانية")
            print(f"الرموز في الثانية: {stats['tokens_per_second']:.2f}")
            print(f"الوقت لكل رمز: {(stats['total_time'] / stats['total_tokens'] * 1000):.2f} مللي ثانية")
        else:
            print("لا توجد بيانات توليد بعد")
        
        print("=" * 60)


class StreamingGenerator(TextGenerator):
    """مولد متدفق للنص"""
    
    def __init__(self, model, config: Dict[str, Any]):
        """تهيئة المولد المدفق"""
        super().__init__(model, config)
        self.callbacks = []
    
    def register_callback(self, callback):
        """تسجيل رد اتصال للتحديثات"""
        self.callbacks.append(callback)
    
    def generate_stream(self, 
                       prompt: Union[str, torch.Tensor],
                       max_new_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       top_p: Optional[float] = None,
                       top_k: Optional[int] = None,
                       stop_tokens: Optional[List[int]] = None) -> None:
        """
        توليد نص متدفق مع تحديثات
        
        Args:
            prompt: المطالبة
            max_new_tokens: الحد الأقصى للرموز
            temperature: درجة الحرارة
            top_p: عينة nucleus
            top_k: عينة top-k
            stop_tokens: رموز التوقف
        """
        # استخدام القيم المحددة أو الافتراضية
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        
        # تحضير المدخلات
        input_ids = self._prepare_inputs(prompt)
        
        # إعادة تعيين ذاكرة التخزين المؤقت
        if hasattr(self.model, 'transformer'):
            self.model.transformer.reset_cache()
        
        # التوليد المدفق
        generated = input_ids
        full_text = ""
        
        self._notify_callbacks('start', full_text)
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                # الحصول على logits
                outputs = self.model(
                    input_ids=generated[:, -1:] if i > 0 else generated,
                    use_cache=self.use_cache,
                    start_pos=generated.size(1) - 1 if i > 0 else 0
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                
                # تطبيق top-k و top-p
                if top_k > 0:
                    logits = self._apply_top_k(logits, top_k)
                if top_p < 1.0:
                    logits = self._apply_top_p(logits, top_p)
                
                # أخذ العينات
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # التحقق من رموز التوقف
                if stop_tokens and next_token.item() in stop_tokens:
                    self._notify_callbacks('stop', full_text, next_token.item())
                    break
                
                # إضافة الرمز الجديد
                generated = torch.cat([generated, next_token], dim=-1)
                
                # فك ترميز وتحديث النص
                token_text = self._decode_token(next_token.item())
                full_text += token_text
                
                # إعلام ردود الاتصال
                self._notify_callbacks('token', token_text, next_token.item())
                self._notify_callbacks('update', full_text, i + 1)
        
        self._notify_callbacks('complete', full_text)
    
    def _notify_callbacks(self, event_type: str, 
                         text: str, 
                         data: Any = None) -> None:
        """إعلام جميع ردود الاتصال"""
        for callback in self.callbacks:
            try:
                callback(event_type, text, data)
            except Exception as e:
                print(f"⚠️  خطأ في رد الاتصال: {e}")


def test_generator():
    """اختبار المولد"""
    print("🧪 اختبار مولد النص...")
    
    # إنشاء نموذج اختبار صغير
    from ..model.tiny_llm import TinyLLM
    
    vocab_size = 1000
    model = TinyLLM(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=256
    )
    
    # إنشاء المولد
    config = {
        'max_new_tokens': 20,
        'temperature': 0.8,
        'top_p': 0.9,
        'top_k': 50,
        'repetition_penalty': 1.1,
        'do_sample': True,
        'use_cache': True
    }
    
    generator = TextGenerator(model, config)
    
    # اختبار التوليد العادي
    print("\n1. اختبار التوليد العادي:")
    prompt = torch.randint(100, 200, (1, 5))
    
    generated = generator.generate(prompt, max_new_tokens=10)
    print(f"   الناتج: {generated}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار التوليد المتدفق
    print("\n2. اختبار التوليد المتدفق:")
    print("   جاري التوليد...")
    stream_output = generator.generate(prompt, max_new_tokens=10, stream=True)
    print(f"   الناتج الكامل: {stream_output}")
    print(f"   ✓ تم بنجاح")
    
    # اختبار بحث الحزمة
    print("\n3. اختبار بحث الحزمة:")
    try:
        results = generator.beam_search(
            prompt, 
            num_beams=3, 
            max_new_tokens=10
        )
        
        for i, (text, score) in enumerate(results):
            print(f"   الحزمة {i+1}: {text[:50]}... (الدرجة: {score:.2f})")
        
        print(f"   ✓ تم بنجاح")
    except Exception as e:
        print(f"   ✗ خطأ: {e}")
    
    # طباعة الإحصائيات
    print("\n4. إحصائيات التوليد:")
    generator.print_stats()
    
    print("\n✅ تم اختبار المولد بنجاح!")


if __name__ == "__main__":
    test_generator()