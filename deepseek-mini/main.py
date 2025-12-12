#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek Mini - النقطة الرئيسية للتطبيق
"""

import sys
import os
import argparse
from pathlib import Path

# إضافة مسار المشروع إلى Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.device_manager import DeviceManager
from gui.app import DeepSeekApp
from model.tiny_llm import TinyLLM
from inference.generator import TextGenerator
import torch
from PyQt5.QtWidgets import QApplication

def main():
    """الدالة الرئيسية للتطبيق"""
    parser = argparse.ArgumentParser(description="DeepSeek Mini - نموذج لغوي عصبي")
    parser.add_argument("--mode", type=str, choices=["gui", "cli", "train"], 
                       default="gui", help="وضع التشغيل")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="مسار ملف الإعدادات")
    parser.add_argument("--model-path", type=str, default=None,
                       help="مسار تحميل النموذج")
    parser.add_argument("--text", type=str, default=None,
                       help="نص للإكمال (في وضع CLI)")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="الحد الأقصى للرموز المولدة")
    
    args = parser.parse_args()
    
    # تحميل الإعدادات
    config = load_config(args.config)
    
    # إدارة الجهاز
    device_manager = DeviceManager()
    device = device_manager.get_device(config["system"]["device"])
    
    print(f"🚀 بدء تشغيل DeepSeek Mini v{config['project']['version']}")
    print(f"📱 الجهاز: {device}")
    print(f"🎮 الوضع: {args.mode}")
    
    if args.mode == "gui":
        # تشغيل واجهة المستخدم الرسومية
        app = QApplication(sys.argv)
        window = DeepSeekApp(config)
        window.show()
        sys.exit(app.exec_())
    
    elif args.mode == "cli":
        # وضع سطر الأوامر
        if args.text:
            run_cli_mode(config, device, args.text, args.max_tokens, args.model_path)
        else:
            run_interactive_cli(config, device, args.model_path)
    
    elif args.mode == "train":
        # وضع التدريب
        run_training(config, device)


def run_cli_mode(config, device, text, max_tokens, model_path):
    """تشغيل وضع سطر الأوامر"""
    print("\n🤖 وضع سطر الأوامر - DeepSeek Mini")
    print(f"📝 النص المدخل: {text}")
    
    # تحميل النموذج
    model = load_model(config, device, model_path)
    generator = TextGenerator(model, config["inference"])
    
    # توليد النص
    print("\n🔄 جاري التوليد...")
    generated = generator.generate(text, max_tokens=max_tokens)
    
    print(f"\n✅ الناتج:\n{generated}\n")


def run_interactive_cli(config, device, model_path):
    """تشغيل وضع المحادثة التفاعلية"""
    print("\n💬 وضع المحادثة التفاعلية")
    print("أدخل 'quit' للخروج، 'clear' لمسح الذاكرة")
    
    model = load_model(config, device, model_path)
    generator = TextGenerator(model, config["inference"])
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 أنت: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 وداعاً!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 تم مسح الذاكرة")
                continue
            
            # إضافة النص إلى تاريخ المحادثة
            conversation_history.append(f"👤 أنت: {user_input}")
            
            # تحضير النص مع التاريخ
            context = "\n".join(conversation_history[-6:])  # آخر 6 رسائل
            full_prompt = f"{context}\n🤖 المساعد:"
            
            # توليد الرد
            print("🤖 المساعد: ", end="", flush=True)
            response = generator.generate(full_prompt, max_tokens=200, stream=True)
            
            # إضافة الرد إلى التاريخ
            conversation_history.append(f"🤖 المساعد: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 وداعاً!")
            break
        except Exception as e:
            print(f"\n❌ خطأ: {e}")


def run_training(config, device):
    """تشغيل التدريب"""
    print("\n🎯 بدء التدريب...")
    
    from training.trainer import Trainer
    from data.dataset import TextDataset
    from data.tokenizer import Tokenizer
    
    # تحميل Tokenizer
    tokenizer = Tokenizer(config["tokenizer"])
    
    # تحميل البيانات
    print("📂 جاري تحميل البيانات...")
    train_dataset = TextDataset(config["data"]["train_path"], tokenizer, config["data"]["max_length"])
    val_dataset = TextDataset(config["data"]["val_path"], tokenizer, config["data"]["max_length"])
    
    # إنشاء النموذج
    print("🧠 جاري إنشاء النموذج...")
    model = TinyLLM(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"],
        ffn_dim=config["model"]["ffn_dim"],
        use_bias=config["model"]["use_bias"]
    ).to(device)
    
    # إنشاء المدرب
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config["training"],
        device=device
    )
    
    # البدء في التدريب
    print("🚀 بدء التدريب...")
    trainer.train()


def load_model(config, device, model_path=None):
    """تحميل النموذج"""
    print("🧠 جاري تحميل النموذج...")
    
    model = TinyLLM(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"],
        ffn_dim=config["model"]["ffn_dim"],
        use_bias=config["model"]["use_bias"]
    ).to(device)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ تم تحميل النموذج من {model_path}")
    else:
        print("⚠️  لم يتم تحميل أي نقطة حفظ، استخدام النموذج الأولي")
    
    model.eval()
    return model


if __name__ == "__main__":
    main()