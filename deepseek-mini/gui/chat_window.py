# -*- coding: utf-8 -*-
"""
نافذة المحادثة - للتفاعل مع النموذج اللغوي
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
                             QFrame, QLabel, QPushButton, QTextEdit, QSplitter,
                             QComboBox, QProgressBar, QGroupBox, QToolButton)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QTextCursor, QPixmap, QIcon
import markdown
from datetime import datetime

from .message_widgets import MessageWidget, ThinkingWidget
from .input_widget import InputWidget


class ChatWindow(QWidget):
    """نافذة المحادثة الرئيسية"""
    
    message_sent = pyqtSignal(str)
    generation_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # المتغيرات
        self.messages = []
        self.is_generating = False
        self.current_animation = None
        
        # تهيئة واجهة المستخدم
        self.init_ui()
        
        # تحميل الأيقونات
        self.load_icons()
    
    def init_ui(self):
        """تهيئة واجهة المستخدم"""
        # التخطيط الرئيسي
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # شريط الأدوات العلوي
        self.create_toolbar()
        main_layout.addWidget(self.toolbar_widget)
        
        # منطقة المحادثة
        self.create_chat_area()
        main_layout.addWidget(self.chat_scroll)
        
        # أدنى جزء (إدخال + تحكم)
        self.create_bottom_panel()
        main_layout.addWidget(self.bottom_panel)
        
        # إعداد الأنماط
        self.setup_styles()
    
    def create_toolbar(self):
        """إنشاء شريط الأدوات العلوي"""
        self.toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        # زر المحادثة الجديدة
        self.new_chat_btn = QPushButton("محادثة جديدة")
        self.new_chat_btn.setIcon(QIcon(":/icons/new_chat.svg"))
        self.new_chat_btn.clicked.connect(self.new_chat)
        toolbar_layout.addWidget(self.new_chat_btn)
        
        # زر مسح المحادثة
        self.clear_chat_btn = QPushButton("مسح المحادثة")
        self.clear_chat_btn.setIcon(QIcon(":/icons/clear.svg"))
        self.clear_chat_btn.clicked.connect(self.clear_chat)
        toolbar_layout.addWidget(self.clear_chat_btn)
        
        # فاصل
        toolbar_layout.addStretch()
        
        # زر النسخ
        self.copy_btn = QPushButton("نسخ المحادثة")
        self.copy_btn.setIcon(QIcon(":/icons/copy.svg"))
        self.copy_btn.clicked.connect(self.copy_chat)
        toolbar_layout.addWidget(self.copy_btn)
        
        # زر التصدير
        self.export_btn = QPushButton("تصدير")
        self.export_btn.setIcon(QIcon(":/icons/export.svg"))
        self.export_btn.clicked.connect(self.export_chat)
        toolbar_layout.addWidget(self.export_btn)
        
        # زر الإعدادات
        self.settings_btn = QToolButton()
        self.settings_btn.setIcon(QIcon(":/icons/settings.svg"))
        self.settings_btn.setText("الإعدادات")
        self.settings_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.settings_btn.clicked.connect(self.show_settings)
        toolbar_layout.addWidget(self.settings_btn)
    
    def create_chat_area(self):
        """إنشاء منطقة المحادثة"""
        # منطقة التمرير
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # القالب المركزي
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setContentsMargins(20, 20, 20, 20)
        self.chat_layout.setSpacing(15)
        
        # إضافة قالب المرن
        self.chat_layout.addStretch()
        
        # تعيين القالب لمنطقة التمرير
        self.chat_scroll.setWidget(self.chat_widget)
        
        # رسالة ترحيب
        self.add_welcome_message()
    
    def add_welcome_message(self):
        """إضافة رسالة ترحيب"""
        welcome_text = """
        <h3>مرحباً بك في DeepSeek Mini! 🤖</h3>
        <p>أنا مساعد ذكي يمكنه مساعدتك في:</p>
        <ul>
            <li>الإجابة على الأسئلة العامة</li>
            <li>كتابة النصوص والمقالات</li>
            <li>ترجمة النصوص</li>
            <li>تلخيص المحتوى</li>
            <li>كتابة التعليمات البرمجية</li>
        </ul>
        <p>اكتب رسالتك أدناه للبدء...</p>
        """
        
        welcome_widget = MessageWidget("assistant", welcome_text, timestamp="الآن")
        self.chat_layout.insertWidget(0, welcome_widget)
    
    def create_bottom_panel(self):
        """إنشاء اللوحة السفلية"""
        self.bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_panel)
        bottom_layout.setContentsMargins(10, 5, 10, 10)
        
        # شريط التحكم
        self.create_control_bar()
        bottom_layout.addWidget(self.control_bar)
        
        # منطقة الإدخال
        self.input_widget = InputWidget()
        self.input_widget.message_ready.connect(self.send_message)
        self.input_widget.stop_generation.connect(self.stop_generation)
        bottom_layout.addWidget(self.input_widget)
        
        # شريط الحالة السفلي
        self.create_status_bar()
        bottom_layout.addWidget(self.status_bar)
    
    def create_control_bar(self):
        """إنشاء شريط التحكم"""
        self.control_bar = QWidget()
        control_layout = QHBoxLayout(self.control_bar)
        control_layout.setContentsMargins(0, 0, 0, 5)
        
        # نموذج النموذج
        self.model_label = QLabel("النموذج: DeepSeek Mini")
        control_layout.addWidget(self.model_label)
        
        # فاصل
        control_layout.addStretch()
        
        # إعدادات التوليد
        self.gen_settings_btn = QPushButton("إعدادات التوليد")
        self.gen_settings_btn.setIcon(QIcon(":/icons/tune.svg"))
        self.gen_settings_btn.clicked.connect(self.show_generation_settings)
        control_layout.addWidget(self.gen_settings_btn)
    
    def create_status_bar(self):
        """إنشاء شريط الحالة"""
        self.status_bar = QWidget()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(0, 5, 0, 0)
        
        # تسمية الحالة
        self.status_label = QLabel("جاهز")
        status_layout.addWidget(self.status_label)
        
        # فاصل
        status_layout.addStretch()
        
        # عداد الرموز
        self.token_count_label = QLabel("الرموز: 0")
        status_layout.addWidget(self.token_count_label)
        
        # سرعة التوليد
        self.speed_label = QLabel("السرعة: 0 رمز/ثانية")
        status_layout.addWidget(self.speed_label)
    
    def load_icons(self):
        """تحميل الأيقونات"""
        # هنا يمكن تحميل الأيقونات من الملفات
        # هذه قيم افتراضية
        pass
    
    def setup_styles(self):
        """إعداد الأنماط"""
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QPushButton {
                padding: 5px 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                background-color: white;
            }
            
            QPushButton:hover {
                background-color: #e8e8e8;
            }
            
            QToolButton {
                padding: 5px 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                background-color: white;
            }
            
            QLabel {
                color: #333;
            }
        """)
    
    def add_message(self, role, content):
        """إضافة رسالة إلى المحادثة"""
        # إيقاف رسالة التفكير إذا كانت موجودة
        self.stop_thinking()
        
        # إنشاء طابع زمني
        timestamp = datetime.now().strftime("%H:%M")
        
        # إنشاء عنصر الرسالة
        message_widget = MessageWidget(role, content, timestamp)
        
        # إضافة الرسالة إلى القالب
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)
        
        # إضافة إلى قائمة الرسائل
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
        
        # تمرير إلى الأسفل
        self.scroll_to_bottom()
        
        # تحديث عداد الرموز
        self.update_token_count()
    
    def start_thinking(self):
        """بدء عرض رسالة التفكير"""
        # إيقاف أي تفكير سابق
        self.stop_thinking()
        
        # إنشاء عنصر التفكير
        self.thinking_widget = ThinkingWidget()
        
        # إضافته إلى القالب
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, self.thinking_widget)
        
        # تعيين حالة التوليد
        self.is_generating = True
        
        # تحديث شريط الحالة
        self.status_label.setText("🤖 جاري التفكير...")
        
        # تمرير إلى الأسفل
        self.scroll_to_bottom()
        
        # تمكين زر الإيقاف
        self.input_widget.set_stop_enabled(True)
    
    def stop_thinking(self):
        """إيقاف رسالة التفكير"""
        if hasattr(self, 'thinking_widget') and self.thinking_widget:
            # إزالة عنصر التفكير
            self.thinking_widget.hide()
            self.chat_layout.removeWidget(self.thinking_widget)
            self.thinking_widget.deleteLater()
            del self.thinking_widget
        
        # تحديث حالة التوليد
        self.is_generating = False
        
        # تحديث شريط الحالة
        self.status_label.setText("✅ جاهز")
        
        # تعطيل زر الإيقاف
        self.input_widget.set_stop_enabled(False)
    
    def send_message(self, message):
        """إرسال رسالة"""
        if not message.strip():
            return
        
        # إضافة رسالة المستخدم
        self.add_message("user", message)
        
        # إرسال إشارة للوالد
        self.message_sent.emit(message)
    
    def stop_generation(self):
        """إيقاف التوليد"""
        if self.is_generating:
            self.stop_thinking()
            self.generation_stopped.emit()
    
    def scroll_to_bottom(self):
        """التمرير إلى أسفل المحادثة"""
        # استخدام مؤقت للتأكد من أن العرض تم تحديثه
        QTimer.singleShot(100, self._scroll_to_bottom_impl)
    
    def _scroll_to_bottom_impl(self):
        """تنفيذ التمرير إلى الأسفل"""
        scrollbar = self.chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_token_count(self):
        """تحديث عداد الرموز"""
        # حساب الرموز التقريبي
        total_tokens = 0
        for msg in self.messages:
            # تقريب: 4 رموز لكل كلمة
            words = len(str(msg['content']).split())
            total_tokens += words * 4
        
        self.token_count_label.setText(f"الرموز: {total_tokens}")
    
    def clear_chat(self):
        """مسح المحادثة"""
        # تأكيد المسح
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "مسح المحادثة",
            "هل أنت متأكد من مسح المحادثة؟",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # إزالة جميع الرسائل عدا رسالة الترحيب
            for i in reversed(range(self.chat_layout.count())):
                widget = self.chat_layout.itemAt(i).widget()
                if widget and hasattr(widget, 'role'):
                    if widget.role != "welcome":
                        widget.hide()
                        self.chat_layout.removeWidget(widget)
                        widget.deleteLater()
            
            # مسح قائمة الرسائل
            self.messages = []
            
            # تحديث العدادات
            self.update_token_count()
            self.speed_label.setText("السرعة: 0 رمز/ثانية")
    
    def new_chat(self):
        """بدء محادثة جديدة"""
        # طلب حفظ المحادثة الحالية
        from PyQt5.QtWidgets import QMessageBox
        if self.messages:
            reply = QMessageBox.question(
                self,
                "محادثة جديدة",
                "هل تريد حفظ المحادثة الحالية؟",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.export_chat()
            elif reply == QMessageBox.Cancel:
                return
        
        # مسح المحادثة
        self.clear_chat()
        
        # إضافة رسالة ترحيب جديدة
        self.add_welcome_message()
    
    def copy_chat(self):
        """نسخ المحادثة"""
        import pyperclip
        
        # تجميع المحادثة
        chat_text = "محادثة DeepSeek Mini\n" + "="*30 + "\n\n"
        
        for msg in self.messages:
            role = "👤 أنت" if msg['role'] == 'user' else "🤖 المساعد"
            chat_text += f"{role} ({msg['timestamp']}):\n{msg['content']}\n\n"
        
        # النسخ إلى الحافظة
        pyperclip.copy(chat_text)
        
        # إشعار
        self.status_label.setText("✅ تم نسخ المحادثة")
        QTimer.singleShot(2000, lambda: self.status_label.setText("جاهز"))
    
    def export_chat(self):
        """تصدير المحادثة"""
        from PyQt5.QtWidgets import QFileDialog
        
        # اختيار الملف
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "تصدير المحادثة",
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "ملفات النص (*.txt);;ملفات Markdown (*.md);;جميع الملفات (*)"
        )
        
        if file_path:
            # تحديد التنسيق
            if file_path.endswith('.md'):
                self.export_markdown(file_path)
            else:
                self.export_text(file_path)
    
    def export_text(self, file_path):
        """تصدير كمستند نصي"""
        # تجميع المحادثة
        chat_text = "محادثة DeepSeek Mini\n" + "="*30 + "\n\n"
        
        for msg in self.messages:
            role = "👤 أنت" if msg['role'] == 'user' else "🤖 المساعد"
            chat_text += f"{role} ({msg['timestamp']}):\n{msg['content']}\n\n"
        
        # الحفظ
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(chat_text)
        
        self.status_label.setText(f"✅ تم التصدير إلى {file_path}")
        QTimer.singleShot(3000, lambda: self.status_label.setText("جاهز"))
    
    def export_markdown(self, file_path):
        """تصدير كمستند Markdown"""
        # تجميع المحادثة
        chat_md = f"# محادثة DeepSeek Mini\n\n"
        chat_md += f"**التاريخ:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        chat_md += "---\n\n"
        
        for msg in self.messages:
            role = "**👤 أنت**" if msg['role'] == 'user' else "**🤖 المساعد**"
            chat_md += f"### {role} ({msg['timestamp']})\n\n"
            chat_md += f"{msg['content']}\n\n"
            chat_md += "---\n\n"
        
        # الحفظ
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(chat_md)
        
        self.status_label.setText(f"✅ تم التصدير إلى {file_path}")
        QTimer.singleShot(3000, lambda: self.status_label.setText("جاهز"))
    
    def show_settings(self):
        """عرض إعدادات النافذة"""
        # هنا يمكن إضافة نافذة الإعدادات
        self.status_label.setText("⚙️  الإعدادات غير متاحة حالياً")
        QTimer.singleShot(2000, lambda: self.status_label.setText("جاهز"))
    
    def show_generation_settings(self):
        """عرض إعدادات التوليد"""
        # هنا يمكن إضافة نافذة إعدادات التوليد
        self.status_label.setText("⚙️  إعدادات التوليد غير متاحة حالياً")
        QTimer.singleShot(2000, lambda: self.status_label.setText("جاهز"))
    
    def set_enabled(self, enabled):
        """تعطيل/تمكين النافذة"""
        self.input_widget.setEnabled(enabled)
        self.new_chat_btn.setEnabled(enabled)
        self.clear_chat_btn.setEnabled(enabled)
        self.copy_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)
        
        if not enabled:
            self.status_label.setText("⏳ جاري التحميل...")
        else:
            self.status_label.setText("✅ جاهز")
    
    def update_generation_speed(self, tokens_per_second):
        """تحديث سرعة التوليد"""
        self.speed_label.setText(f"السرعة: {tokens_per_second:.1f} رمز/ثانية")
    
    def animate_message(self, widget):
        """تحريك ظهور الرسالة"""
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(300)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()


if __name__ == "__main__":
    # اختبار نافذة المحادثة
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    window = ChatWindow()
    window.setWindowTitle("اختبار نافذة المحادثة")
    window.resize(800, 600)
    window.show()
    
    # إضافة بعض الرسائل للاختبار
    window.add_message("user", "مرحباً، كيف حالك؟")
    window.add_message("assistant", "أهلاً! أنا بخير، شكراً لسؤالك. كيف يمكنني مساعدتك اليوم؟")
    
    sys.exit(app.exec_())