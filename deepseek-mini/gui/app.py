# -*- coding: utf-8 -*-
"""
التطبيق الرئيسي للواجهة الرسومية
"""

import sys
import os
from pathlib import Path

# إضافة مسار المشروع إلى Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QStatusBar, QMessageBox,
                             QSystemTrayIcon, QMenu, QAction, QStyle)
from PyQt5.QtCore import Qt, QTimer, QSettings, QSize, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
import qdarkstyle

from .chat_window import ChatWindow
from .sidebar import Sidebar
from .styles import apply_stylesheet, get_stylesheet
from utils.config_loader import load_config
from utils.device_manager import DeviceManager
from model.tiny_llm import TinyLLM
from inference.generator import TextGenerator
import torch


class ModelLoaderThread(QThread):
    """خيط لتحميل النموذج في الخلفية"""
    
    model_loaded = pyqtSignal(object, object)  # إشارة عند تحميل النموذج
    error_occurred = pyqtSignal(str)  # إشارة عند حدوث خطأ
    
    def __init__(self, config, model_path=None):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.device_manager = DeviceManager()
    
    def run(self):
        """تحميل النموذج في الخلفية"""
        try:
            # تحديد الجهاز
            device = self.device_manager.get_device(
                self.config["system"]["device"]
            )
            
            # إنشاء النموذج
            model = TinyLLM(
                vocab_size=self.config["model"]["vocab_size"],
                d_model=self.config["model"]["d_model"],
                n_heads=self.config["model"]["n_heads"],
                n_layers=self.config["model"]["n_layers"],
                max_seq_len=self.config["model"]["max_seq_len"],
                dropout=self.config["model"]["dropout"],
                ffn_dim=self.config["model"]["ffn_dim"],
                use_bias=self.config["model"]["use_bias"]
            ).to(device)
            
            # تحميل النموذج إذا كان مسار محدد
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            
            model.eval()
            
            # إنشاء مولد النص
            generator = TextGenerator(model, self.config["inference"])
            
            # إرسال الإشارة أن النموذج جاهز
            self.model_loaded.emit(model, generator)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class DeepSeekApp(QMainWindow):
    """النافذة الرئيسية للتطبيق"""
    
    def __init__(self, config=None):
        super().__init__()
        
        # تحميل الإعدادات
        self.config = config or load_config()
        
        # تهيئة المتغيرات
        self.model = None
        self.generator = None
        self.current_model_path = None
        self.conversation_history = []
        
        # إعدادات الواجهة
        self.settings = QSettings("DeepSeek", "Mini")
        
        # تهيئة واجهة المستخدم
        self.init_ui()
        
        # تحميل النموذج في الخلفية
        self.load_model_in_background()
        
        # إعداد المؤقت للتحديثات الدورية
        self.setup_timers()
        
        # إعداد صينية النظام (اختياري)
        self.setup_system_tray()
    
    def init_ui(self):
        """تهيئة واجهة المستخدم"""
        # إعداد النافذة الرئيسية
        self.setWindowTitle(f"DeepSeek Mini v{self.config['project']['version']}")
        self.setGeometry(100, 100, 1200, 800)
        
        # تعيين الأيقونة
        icon_path = Path(__file__).parent / "assets" / "icons" / "logo.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # إنشاء القالب المركزي
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # التخطيط الرئيسي
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # إنشاء مقسم
        splitter = QSplitter(Qt.Horizontal)
        
        # الشريط الجانبي
        self.sidebar = Sidebar(self)
        self.sidebar.setMinimumWidth(250)
        self.sidebar.setMaximumWidth(350)
        
        # نافذة المحادثة
        self.chat_window = ChatWindow(self)
        
        # إضافة المكونات إلى المقسم
        if self.config["gui"]["show_sidebar"]:
            splitter.addWidget(self.sidebar)
        splitter.addWidget(self.chat_window)
        
        # تعيين نسب المقسم
        splitter.setSizes([250, 750])
        
        # إضافة المقسم إلى التخطيط
        main_layout.addWidget(splitter)
        
        # إنشاء شريط الحالة
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # إضافة عناصر إلى شريط الحالة
        #self.status_label = self.status_bar.addWidget("🚀 جاري تحميل النموذج...")
        #self.memory_label = self.status_bar.addWidget("")
        #self.device_label = self.status_bar.addWidget("")
        self.status_bar.showMessage("🚀 جاري تحميل النموذج...")
        
        # تطبيق الأنماط
        self.apply_styles()
        
        # إعداد قائمة الملف
        self.setup_menu_bar()
        
        # توصيل الإشارات
        self.connect_signals()
    
    def apply_styles(self):
        """تطبيق الأنماط على الواجهة"""
        # تطبيق النمط الداكن أو الفاتح
        theme = self.config["gui"]["theme"]
        stylesheet = get_stylesheet(theme)
        self.setStyleSheet(stylesheet)
        
        # تطبيق الخطوط
        font_family = self.config["gui"]["font_family"]
        font_size = self.config["gui"]["font_size"]
        
        font = QFont(font_family, font_size)
        QApplication.setFont(font)
        
        # تطبيق أنماط إضافية
        apply_stylesheet(self)
    
    def setup_menu_bar(self):
        """إعداد قائمة الملف"""
        menubar = self.menuBar()
        
        # قائمة الملف
        file_menu = menubar.addMenu("ملف")
        
        new_chat_action = QAction("محادثة جديدة", self)
        new_chat_action.setShortcut("Ctrl+N")
        new_chat_action.triggered.connect(self.new_chat)
        file_menu.addAction(new_chat_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("حفظ المحادثة", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_conversation)
        file_menu.addAction(save_action)
        
        load_action = QAction("تحميل المحادثة", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_conversation)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("خروج", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # قائمة الإعدادات
        settings_menu = menubar.addMenu("إعدادات")
        
        theme_action = QAction("تبديل السمة", self)
        theme_action.triggered.connect(self.toggle_theme)
        settings_menu.addAction(theme_action)
        
        model_action = QAction("تغيير النموذج", self)
        model_action.triggered.connect(self.change_model)
        settings_menu.addAction(model_action)
        
        settings_menu.addSeparator()
        
        config_action = QAction("الإعدادات المتقدمة", self)
        config_action.triggered.connect(self.show_settings_dialog)
        settings_menu.addAction(config_action)
        
        # قائمة المساعدة
        help_menu = menubar.addMenu("مساعدة")
        
        about_action = QAction("حول", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        docs_action = QAction("التوثيق", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
    
    def connect_signals(self):
        """توصيل الإشارات"""
        # إشارات من الشريط الجانبي
        self.sidebar.model_changed.connect(self.on_model_changed)
        self.sidebar.settings_changed.connect(self.on_settings_changed)
        self.sidebar.conversation_selected.connect(self.on_conversation_selected)
        
        # إشارات من نافذة المحادثة
        self.chat_window.message_sent.connect(self.on_message_sent)
        self.chat_window.generation_stopped.connect(self.on_generation_stopped)
    
    def setup_timers(self):
        """إعداد المؤقتات للتحديثات الدورية"""
        # مؤقت لتحديث حالة النظام
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self.update_system_status)
        self.system_timer.start(5000)  # كل 5 ثواني
        
        # مؤقت لحفظ المحادثة تلقائياً
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_conversation)
        self.auto_save_timer.start(60000)  # كل دقيقة
    
    def setup_system_tray(self):
        """إعداد صينية النظام"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            
            # تعيين الأيقونة
            icon_path = Path(__file__).parent / "assets" / "icons" / "logo.png"
            if icon_path.exists():
                self.tray_icon.setIcon(QIcon(str(icon_path)))
            
            # إنشاء قائمة السياق
            tray_menu = QMenu()
            
            show_action = QAction("إظهار", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)
            
            hide_action = QAction("إخفاء", self)
            hide_action.triggered.connect(self.hide)
            tray_menu.addAction(hide_action)
            
            tray_menu.addSeparator()
            
            quit_action = QAction("خروج", self)
            quit_action.triggered.connect(self.close)
            tray_menu.addAction(quit_action)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()
            
            # إشارات صينية النظام
            self.tray_icon.activated.connect(self.on_tray_activated)
    
    def load_model_in_background(self):
        """تحميل النموذج في الخلفية"""
        # الحصول على مسار النموذج من الإعدادات
        model_path = self.settings.value("model_path", None)
        
        # تحديث شريط الحالة
        self.status_bar.showMessage("🔄 جاري تحميل النموذج...")
        
        # إنشاء وتشغيل خيط التحميل
        self.loader_thread = ModelLoaderThread(self.config, model_path)
        self.loader_thread.model_loaded.connect(self.on_model_loaded)
        self.loader_thread.error_occurred.connect(self.on_model_load_error)
        self.loader_thread.start()
    
    def on_model_loaded(self, model, generator):
        """عند تحميل النموذج بنجاح"""
        self.model = model
        self.generator = generator
        
        # تحديث واجهة المستخدم
        self.status_bar.showMessage("✅ النموذج جاهز للاستخدام")
        
        # تحديث الشريط الجانبي
        self.sidebar.set_model_info(self.model.get_config())
        
        # تمكين إرسال الرسائل
        self.chat_window.set_enabled(True)
        
        # تحديث تسمية الجهاز
        device_name = "GPU" if next(model.parameters()).is_cuda else "CPU"
        #self.device_label.setText(f"📱 {device_name}")
        self.status_bar.showMessage(f"✅ النموذج جاهز على {device_name}")
        
        print("✅ تم تحميل النموذج بنجاح")
    
    def on_model_load_error(self, error_message):
        """عند فشل تحميل النموذج"""
        QMessageBox.critical(
            self,
            "خطأ في تحميل النموذج",
            f"فشل تحميل النموذج:\n{error_message}"
        )
        
        self.status_bar.showMessage("❌ فشل تحميل النموذج")
        
        # عرض نموذج وهمي للاختبار
        self.chat_window.set_enabled(True)
        
        print(f"❌ خطأ في تحميل النموذج: {error_message}")
    
    def on_message_sent(self, message):
        """عند إرسال رسالة من المستخدم"""
        # إضافة الرسالة إلى تاريخ المحادثة
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": QTimer().remainingTime()
        })
        
        # تحديث الشريط الجانبي
        self.sidebar.update_conversation_list(self.conversation_history)
        
        # توليد الرد إذا كان النموذج جاهز
        if self.generator:
            self.generate_response(message)
        else:
            # عرض رسالة وهمية إذا لم يكن النموذج جاهز
            self.chat_window.add_message(
                "assistant",
                "النموذج غير جاهز بعد. جاري التحميل..."
            )
    
    def generate_response(self, user_message):
        """توليد رد من النموذج"""
        # إعداد السياق
        context = self.prepare_context(user_message)
        
        # تحديث واجهة المستخدم
        self.status_bar.showMessage("🤖 جاري توليد الرد...")
        self.chat_window.start_thinking()
        
        # توليد الرد في خلفية منفصلة
        QTimer.singleShot(100, lambda: self._generate_in_thread(context))
    
    def _generate_in_thread(self, context):
        """توليد الرد في خلفية منفصلة"""
        try:
            # توليد الرد
            response = self.generator.generate(
                prompt=context,
                max_new_tokens=self.config["inference"]["max_new_tokens"],
                temperature=self.config["inference"]["temperature"],
                top_p=self.config["inference"]["top_p"],
                top_k=self.config["inference"]["top_k"],
                repetition_penalty=self.config["inference"]["repetition_penalty"],
                do_sample=self.config["inference"]["do_sample"]
            )
            
            # إضافة الرد إلى نافذة المحادثة
            self.chat_window.add_message("assistant", response)
            
            # إضافة الرد إلى تاريخ المحادثة
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": QTimer().remainingTime()
            })
            
            # تحديث الشريط الجانبي
            self.sidebar.update_conversation_list(self.conversation_history)
            
            # تحديث شريط الحالة
            self.status_bar.showMessage("✅ جاهز")
            
        except Exception as e:
            error_msg = f"خطأ في توليد الرد: {str(e)}"
            self.chat_window.add_message("assistant", error_msg)
            self.status_bar.showMessage("❌ خطأ في التوليد")
            
            print(f"❌ خطأ في توليد الرد: {e}")
    
    def prepare_context(self, user_message):
        """تحضير السياق للتوليد"""
        # تجميع تاريخ المحادثة
        context_parts = []
        
        # إضافة الرسائل السابقة (محدود بالعدد)
        max_history = self.config["gui"]["max_history"]
        for msg in self.conversation_history[-max_history:]:
            role = "أنت" if msg["role"] == "user" else "المساعد"
            context_parts.append(f"{role}: {msg['content']}")
        
        # إضافة الرسالة الحالية
        context_parts.append(f"أنت: {user_message}")
        context_parts.append("المساعد:")
        
        # تجميع السياق
        context = "\n".join(context_parts)
        
        return context
    
    def on_generation_stopped(self):
        """عند إيقاف التوليد"""
        self.status_bar.showMessage("⏹️  تم إيقاف التوليد")
    
    def on_model_changed(self, model_path):
        """عند تغيير النموذج"""
        self.current_model_path = model_path
        
        # حفظ المسار في الإعدادات
        self.settings.setValue("model_path", model_path)
        
        # إعادة تحميل النموذج
        self.load_model_in_background()
    
    def on_settings_changed(self, settings):
        """عند تغيير الإعدادات"""
        # تحديث الإعدادات
        for key, value in settings.items():
            if key in self.config:
                self.config[key] = value
        
        # تطبيق التغييرات
        self.apply_styles()
        
        # إعادة تحميل النموذج إذا تغيرت إعدادات الجهاز
        if "device" in settings.get("system", {}):
            self.load_model_in_background()
    
    def on_conversation_selected(self, conversation_id):
        """عند اختيار محادثة من القائمة"""
        # هنا يمكن تحميل المحادثة المحددة
        # هذا تنفيذ مبسط
        print(f"تم اختيار المحادثة: {conversation_id}")
    
    def on_tray_activated(self, reason):
        """عند التفاعل مع صينية النظام"""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()
    
    def update_system_status(self):
        """تحديث حالة النظام"""
        if self.model is not None:
            # تحديث استخدام الذاكرة
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                memory_percent = (memory_used / memory_total) * 100
                
                self.memory_label.setText(f"💾 {memory_used:.1f}/{memory_total:.1f} GB ({memory_percent:.0f}%)")
    
    def new_chat(self):
        """بدء محادثة جديدة"""
        # حفظ المحادثة الحالية إذا كانت موجودة
        if self.conversation_history:
            reply = QMessageBox.question(
                self,
                "محادثة جديدة",
                "هل تريد حفظ المحادثة الحالية قبل البدء بمحادثة جديدة؟",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.save_conversation()
            elif reply == QMessageBox.Cancel:
                return
        
        # مسح المحادثة الحالية
        self.conversation_history = []
        self.chat_window.clear_chat()
        
        # تحديث الشريط الجانبي
        self.sidebar.update_conversation_list(self.conversation_history)
        
        self.status_bar.showMessage("✅ محادثة جديدة جاهزة")
    
    def save_conversation(self):
        """حفظ المحادثة الحالية"""
        # هنا يمكن تنفيذ حفظ المحادثة إلى ملف
        # هذا تنفيذ مبسط
        QMessageBox.information(
            self,
            "حفظ المحادثة",
            "سيتم تنفيذ حفظ المحادثة في إصدار لاحق"
        )
    
    def load_conversation(self):
        """تحميل محادثة"""
        # هنا يمكن تنفيذ تحميل المحادثة من ملف
        # هذا تنفيذ مبسط
        QMessageBox.information(
            self,
            "تحميل المحادثة",
            "سيتم تنفيذ تحميل المحادثة في إصدار لاحق"
        )
    
    def auto_save_conversation(self):
        """حفظ المحادثة تلقائياً"""
        if self.conversation_history:
            # هنا يمكن تنفيذ الحفظ التلقائي
            # هذا تنفيذ مبسط
            pass
    
    def toggle_theme(self):
        """تبديل السمة بين الداكن والفاتح"""
        current_theme = self.config["gui"]["theme"]
        new_theme = "light" if current_theme == "dark" else "dark"
        
        # تحديث الإعدادات
        self.config["gui"]["theme"] = new_theme
        
        # تطبيق النمط الجديد
        self.apply_styles()
        
        # حفظ الإعدادات
        self.settings.setValue("theme", new_theme)
        
        self.status_bar.showMessage(f"✅ تم التبديل إلى السمة {new_theme}")
    
    def change_model(self):
        """تغيير النموذج"""
        # هنا يمكن تنفيذ اختيار نموذج من نافذة حوار
        # هذا تنفيذ مبسط
        QMessageBox.information(
            self,
            "تغيير النموذج",
            "سيتم تنفيذ اختيار النموذج في إصدار لاحق"
        )
    
    def show_settings_dialog(self):
        """عرض نافذة الإعدادات المتقدمة"""
        # هنا يمكن تنفيذ نافذة الإعدادات
        # هذا تنفيذ مبسط
        QMessageBox.information(
            self,
            "الإعدادات المتقدمة",
            "سيتم تنفيذ الإعدادات المتقدمة في إصدار لاحق"
        )
    
    def show_about_dialog(self):
        """عرض نافذة حول التطبيق"""
        about_text = f"""
        <h2>DeepSeek Mini</h2>
        <p>إصدار: {self.config['project']['version']}</p>
        <p>وصف: {self.config['project']['description']}</p>
        <p>المؤلف: {self.config['project']['author']}</p>
        <p>نموذج لغوي عصبي مع واجهة رسومية متكاملة</p>
        <hr>
        <p>تم التطوير باستخدام:</p>
        <ul>
            <li>PyTorch</li>
            <li>PyQt5</li>
            <li>Transformers Architecture</li>
        </ul>
        """
        
        QMessageBox.about(self, "حول DeepSeek Mini", about_text)
    
    def show_documentation(self):
        """عرض التوثيق"""
        # هنا يمكن فتح التوثيق في المتصفح
        # هذا تنفيذ مبسط
        QMessageBox.information(
            self,
            "التوثيق",
            "سيتم فتح التوثيق في إصدار لاحق"
        )
    
    def closeEvent(self, event):
        """عند محاولة إغلاق النافذة"""
        # التحقق من وجود محادثة غير محفوظة
        if self.conversation_history:
            reply = QMessageBox.question(
                self,
                "تأكيد الإغلاق",
                "هل تريد حفظ المحادثة الحالية قبل الإغلاق؟",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.save_conversation()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def run_app(config=None):
    """تشغيل التطبيق"""
    app = QApplication(sys.argv)
    
    # تعيين اسم التطبيق
    app.setApplicationName("DeepSeek Mini")
    app.setApplicationVersion("1.0.0")
    
    # إنشاء النافذة الرئيسية
    window = DeepSeekApp(config)
    window.show()
    
    # تشغيل حلقة الأحداث
    sys.exit(app.exec_())


if __name__ == "__main__":
    # اختبار التطبيق
    print("🧪 اختبار واجهة المستخدم...")
    
    # تشغيل التطبيق
    run_app()