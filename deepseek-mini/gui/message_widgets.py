# -*- coding: utf-8 -*-
"""
عناصر عرض الرسائل - لعرض رسائل المستخدم والمساعد
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QTextBrowser, QPushButton, QToolButton,
                             QMenu, QAction, QProgressBar, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import (QFont, QPixmap, QPainter, QColor, QLinearGradient, 
                         QPalette, QTextCursor, QIcon)
import html
from datetime import datetime


class MessageWidget(QWidget):
    """عنصر عرض رسالة"""
    
    copy_requested = pyqtSignal(str)
    edit_requested = pyqtSignal(str)
    delete_requested = pyqtSignal()
    
    def __init__(self, role, content, timestamp="", parent=None):
        super().__init__(parent)
        
        self.role = role
        self.content = content
        self.timestamp = timestamp
        
        # متغيرات التنسيق
        self.is_expanded = True
        
        # تهيئة واجهة المستخدم
        self.init_ui()
        
        # تحميل المحتوى
        self.set_content(content)
        
        # إعداد الأنماط
        self.setup_styles()
    
    def init_ui(self):
        """تهيئة واجهة المستخدم"""
        # التخطيط الرئيسي
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        
        # صف الرأس (الأيقونة + الدور + الوقت)
        self.create_header()
        main_layout.addWidget(self.header_widget)
        
        # منطقة المحتوى
        self.create_content_area()
        main_layout.addWidget(self.content_frame)
        
        # صف الأدوات (أزرار الإجراءات)
        self.create_toolbar()
        main_layout.addWidget(self.toolbar_widget)
    
    def create_header(self):
        """إنشاء رأس الرسالة"""
        self.header_widget = QWidget()
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(5, 0, 5, 0)
        
        # الأيقونة
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(24, 24)
        
        # تعيين الأيقونة حسب الدور
        if self.role == "user":
            self.icon_label.setText("👤")
            self.icon_label.setToolTip("أنت")
        else:
            self.icon_label.setText("🤖")
            self.icon_label.setToolTip("المساعد الذكي")
        
        header_layout.addWidget(self.icon_label)
        
        # تسمية الدور
        self.role_label = QLabel()
        role_text = "أنت" if self.role == "user" else "المساعد الذكي"
        self.role_label.setText(f"<b>{role_text}</b>")
        self.role_label.setStyleSheet("color: #666;")
        
        header_layout.addWidget(self.role_label)
        
        # فاصل
        header_layout.addStretch()
        
        # الوقت
        self.time_label = QLabel(self.timestamp)
        self.time_label.setStyleSheet("color: #888; font-size: 11px;")
        header_layout.addWidget(self.time_label)
    
    def create_content_area(self):
        """إنشاء منطقة المحتوى"""
        self.content_frame = QFrame()
        self.content_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.content_frame.setLineWidth(1)
        
        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(15, 15, 15, 15)
        
        # عرض المحتوى
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        self.content_browser.setReadOnly(True)
        self.content_browser.setMaximumHeight(400)  # ارتفاع قصوي
        self.content_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # إعداد سياسة الحجم
        size_policy = self.content_browser.sizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Preferred)
        self.content_browser.setSizePolicy(size_policy)
        
        content_layout.addWidget(self.content_browser)
        
        # زر التوسيع/الطي
        self.toggle_button = QPushButton("عرض أقل")
        self.toggle_button.setFixedHeight(20)
        self.toggle_button.clicked.connect(self.toggle_expand)
        self.toggle_button.hide()  # مخفي حتى نحتاجه
        content_layout.addWidget(self.toggle_button)
    
    def create_toolbar(self):
        """إنشاء شريط الأدوات"""
        self.toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(5, 0, 5, 0)
        toolbar_layout.setSpacing(5)
        
        # فاصل
        toolbar_layout.addStretch()
        
        # زر النسخ
        self.copy_button = QToolButton()
        self.copy_button.setText("نسخ")
        self.copy_button.setIcon(QIcon(":/icons/copy.svg"))
        self.copy_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.copy_button.clicked.connect(self.copy_content)
        toolbar_layout.addWidget(self.copy_button)
        
        # زر إعادة الصياغة
        self.rewrite_button = QToolButton()
        self.rewrite_button.setText("إعادة صياغة")
        self.rewrite_button.setIcon(QIcon(":/icons/refresh.svg"))
        self.rewrite_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.rewrite_button.clicked.connect(self.rewrite_content)
        toolbar_layout.addWidget(self.rewrite_button)
        
        # زر المزيد
        self.more_button = QToolButton()
        self.more_button.setText("المزيد")
        self.more_button.setIcon(QIcon(":/icons/more.svg"))
        self.more_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.more_button.clicked.connect(self.show_more_menu)
        toolbar_layout.addWidget(self.more_button)
    
    def setup_styles(self):
        """إعداد الأنماط حسب الدور"""
        if self.role == "user":
            # نمط رسالة المستخدم
            self.content_frame.setStyleSheet("""
                QFrame {
                    background-color: #e3f2fd;
                    border: 1px solid #bbdefb;
                    border-radius: 10px;
                    border-top-left-radius: 0px;
                }
            """)
        else:
            # نمط رسالة المساعد
            self.content_frame.setStyleSheet("""
                QFrame {
                    background-color: #f1f8e9;
                    border: 1px solid #dcedc8;
                    border-radius: 10px;
                    border-top-right-radius: 0px;
                }
            """)
        
        # أنماط مشتركة
        self.toolbar_widget.setStyleSheet("""
            QToolButton {
                padding: 2px 8px;
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: white;
                font-size: 12px;
            }
            
            QToolButton:hover {
                background-color: #f0f0f0;
            }
            
            QToolButton:pressed {
                background-color: #e0e0e0;
            }
        """)
        
        self.toggle_button.setStyleSheet("""
            QPushButton {
                border: none;
                color: #666;
                background-color: transparent;
                font-size: 11px;
            }
            
            QPushButton:hover {
                color: #333;
                text-decoration: underline;
            }
        """)
        
        self.content_browser.setStyleSheet("""
            QTextBrowser {
                background-color: transparent;
                border: none;
                font-size: 14px;
                line-height: 1.6;
            }
            
            QScrollBar:vertical {
                background-color: transparent;
                width: 8px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #888;
                border-radius: 4px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
        """)
    
    def set_content(self, content):
        """تعيين محتوى الرسالة"""
        self.content = content
        
        # تنظيف النص من HTML غير آمن
        safe_content = html.escape(content)
        
        # تحويل Markdown إلى HTML إذا لزم
        formatted_content = self.format_content(safe_content)
        
        # تعيين المحتوى
        self.content_browser.setHtml(formatted_content)
        
        # التحقق إذا كان المحتوى طويلاً ويحتاج زر توسيع
        self.check_content_length()
    
    def format_content(self, content):
        """تنسيق المحتوى للعرض"""
        # دعم بسيط لـ Markdown
        formatted = content
        
        # العناوين
        formatted = formatted.replace('### ', '<h3>').replace('\n', '</h3>\n')
        formatted = formatted.replace('## ', '<h2>').replace('\n', '</h2>\n')
        formatted = formatted.replace('# ', '<h1>').replace('\n', '</h1>\n')
        
        # النقاط
        lines = formatted.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    formatted_lines.append('<ul>')
                    in_list = True
                formatted_lines.append(f'<li>{line[2:]}</li>')
            elif line.strip().startswith('* '):
                if not in_list:
                    formatted_lines.append('<ul>')
                    in_list = True
                formatted_lines.append(f'<li>{line[2:]}</li>')
            else:
                if in_list:
                    formatted_lines.append('</ul>')
                    in_list = False
                formatted_lines.append(line)
        
        if in_list:
            formatted_lines.append('</ul>')
        
        formatted = '\n'.join(formatted_lines)
        
        # الفقرات
        paragraphs = formatted.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            if not para.startswith('<') and not para.endswith('>'):
                formatted_paragraphs.append(f'<p>{para}</p>')
            else:
                formatted_paragraphs.append(para)
        
        formatted = '\n'.join(formatted_paragraphs)
        
        # أكواد
        if '```' in formatted:
            parts = formatted.split('```')
            for i in range(1, len(parts), 2):
                if i < len(parts):
                    code = parts[i].strip()
                    if '\n' in code:
                        # كتلة كود
                        parts[i] = f'<pre><code>{code}</code></pre>'
                    else:
                        # كود سطري
                        parts[i] = f'<code>{code}</code>'
            formatted = ''.join(parts)
        
        # الروابط
        import re
        url_pattern = r'(https?://[^\s]+)'
        formatted = re.sub(url_pattern, r'<a href="\1">\1</a>', formatted)
        
        # إضافة تنسيقات CSS
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    color: #333;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                }}
                
                p {{
                    margin-bottom: 10px;
                }}
                
                h1, h2, h3 {{
                    color: #2c3e50;
                    margin-top: 15px;
                    margin-bottom: 10px;
                }}
                
                ul, ol {{
                    margin-left: 20px;
                    margin-bottom: 10px;
                }}
                
                li {{
                    margin-bottom: 5px;
                }}
                
                pre {{
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 5px;
                    padding: 10px;
                    overflow-x: auto;
                    font-family: 'Courier New', monospace;
                    font-size: 13px;
                }}
                
                code {{
                    background-color: #f8f9fa;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                
                a {{
                    color: #3498db;
                    text-decoration: none;
                }}
                
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            {formatted}
        </body>
        </html>
        """
        
        return html_content
    
    def check_content_length(self):
        """التحقق من طول المحتوى وعرض زر التوسيع إذا لزم"""
        # حساب ارتفاع النص
        doc_height = self.content_browser.document().size().height()
        view_height = self.content_browser.height()
        
        # إذا كان النص أطول من الارتفاع الحالي
        if doc_height > view_height:
            self.toggle_button.show()
            
            # تعديل نص الزر
            if self.is_expanded:
                self.toggle_button.setText("عرض أقل")
            else:
                self.toggle_button.setText("عرض المزيد")
        else:
            self.toggle_button.hide()
    
    def toggle_expand(self):
        """تبديل توسيع/طي المحتوى"""
        if self.is_expanded:
            # طي المحتوى
            self.content_browser.setMaximumHeight(150)
            self.toggle_button.setText("عرض المزيد")
            self.is_expanded = False
        else:
            # توسيع المحتوى
            self.content_browser.setMaximumHeight(10000)  # قيمة كبيرة
            self.toggle_button.setText("عرض أقل")
            self.is_expanded = True
        
        # إعادة ضبط المحتوى
        QTimer.singleShot(100, self.check_content_length)
    
    def copy_content(self):
        """نسخ المحتوى"""
        import pyperclip
        pyperclip.copy(self.content)
        
        # تغيير نص الزر مؤقتاً
        original_text = self.copy_button.text()
        self.copy_button.setText("تم النسخ!")
        QTimer.singleShot(2000, lambda: self.copy_button.setText(original_text))
        
        # إرسال إشارة
        self.copy_requested.emit(self.content)
    
    def rewrite_content(self):
        """إعادة صياغة المحتوى"""
        # هنا يمكن إضافة منطق إعادة الصياغة
        print(f"إعادة صياغة الرسالة: {self.content[:50]}...")
    
    def show_more_menu(self):
        """عرض قائمة المزيد"""
        menu = QMenu(self)
        
        # إجراءات القائمة
        edit_action = QAction("تعديل", self)
        edit_action.triggered.connect(lambda: self.edit_requested.emit(self.content))
        menu.addAction(edit_action)
        
        delete_action = QAction("حذف", self)
        delete_action.triggered.connect(self.delete_requested)
        menu.addAction(delete_action)
        
        menu.addSeparator()
        
        save_action = QAction("حفظ كملف", self)
        save_action.triggered.connect(self.save_to_file)
        menu.addAction(save_action)
        
        # عرض القائمة
        menu.exec_(self.more_button.mapToGlobal(self.more_button.rect().bottomLeft()))
    
    def save_to_file(self):
        """حفظ المحتوى كملف"""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "حفظ الرسالة",
            f"message_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "ملفات النص (*.txt);;ملفات HTML (*.html);;جميع الملفات (*)"
        )
        
        if file_path:
            if file_path.endswith('.html'):
                content = self.content_browser.toHtml()
            else:
                content = self.content
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def resizeEvent(self, event):
        """معالج حدث تغيير الحجم"""
        super().resizeEvent(event)
        self.check_content_length()
    
    def enterEvent(self, event):
        """عند دخول الماوس"""
        self.toolbar_widget.show()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """عند خروج الماوس"""
        # إخفاء شريط الأدوات بعد تأخير
        QTimer.singleShot(500, self._hide_toolbar_if_needed)
        super().leaveEvent(event)
    
    def _hide_toolbar_if_needed(self):
        """إخفاء شريط الأدوات إذا لم يكن الماوس فوق العنصر"""
        if not self.underMouse():
            self.toolbar_widget.hide()
    
    def showEvent(self, event):
        """عند عرض العنصر"""
        super().showEvent(event)
        
        # تحريك ظهور العنصر
        self.animate_appearance()
    
    def animate_appearance(self):
        """تحريك ظهور الرسالة"""
        self.setWindowOpacity(0)
        
        animation = QPropertyAnimation(self, b"windowOpacity")
        animation.setDuration(300)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()


class ThinkingWidget(QWidget):
    """عنصر عرض حالة التفكير"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # تهيئة واجهة المستخدم
        self.init_ui()
        
        # بدء الرسوم المتحركة
        self.start_animation()
    
    def init_ui(self):
        """تهيئة واجهة المستخدم"""
        # التخطيط الرئيسي
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # إطار المحتوى
        content_frame = QFrame()
        content_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                border-top-right-radius: 0px;
            }
        """)
        
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # صف الأيقونة والنص
        top_layout = QHBoxLayout()
        
        # الأيقونة
        icon_label = QLabel("🤖")
        icon_label.setStyleSheet("font-size: 24px;")
        top_layout.addWidget(icon_label)
        
        # النص
        text_label = QLabel("جاري التفكير...")
        text_label.setStyleSheet("font-size: 14px; color: #666;")
        top_layout.addWidget(text_label)
        
        top_layout.addStretch()
        content_layout.addLayout(top_layout)
        
        # شريط التقدم المتحرك
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # شريط غير محدد
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #e0e0e0;
                border-radius: 1px;
            }
            
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 1px;
            }
        """)
        content_layout.addWidget(self.progress_bar)
        
        # زر الإيقاف
        self.stop_button = QPushButton("إيقاف")
        self.stop_button.setFixedHeight(30)
        self.stop_button.clicked.connect(self.stop_thinking)
        self.stop_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px 15px;
                background-color: white;
                color: #666;
            }
            
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        content_layout.addWidget(self.stop_button, 0, Qt.AlignCenter)
        
        main_layout.addWidget(content_frame)
    
    def start_animation(self):
        """بدء الرسوم المتحركة لشريط التقدم"""
        # استخدام مؤقت لتحديث شريط التقدم
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_progress)
        self.animation_timer.start(100)  # تحديث كل 100 مللي ثانية
        
        # قيمة التقدم الحالية
        self.progress_value = 0
    
    def update_progress(self):
        """تحديث شريط التقدم"""
        self.progress_value = (self.progress_value + 10) % 100
        
        # تحريك لون شريط التقدم
        style = f"""
            QProgressBar::chunk {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, 
                    stop:{self.progress_value/100} #8BC34A,
                    stop:1 #4CAF50
                );
            }}
        """
        self.progress_bar.setStyleSheet(style)
    
    def stop_thinking(self):
        """إيقاف التفكير"""
        self.animation_timer.stop()
        self.hide()
        
        # إرسال إشارة للوالد
        if self.parent():
            self.parent().stop_generation()
    
    def hideEvent(self, event):
        """عند إخفاء العنصر"""
        self.animation_timer.stop()
        super().hideEvent(event)


if __name__ == "__main__":
    # اختبار عناصر الرسائل
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            layout = QVBoxLayout(central_widget)
            
            # رسالة المستخدم
            user_msg = MessageWidget(
                "user",
                "مرحباً! هذا مثال على رسالة من المستخدم. يمكن أن تحتوي على نص طويل يتضمن:\n\n"
                "- نقاط\n- كود: `print('Hello')`\n- روابط: https://example.com\n\n"
                "وهذا فقرة أخرى من النص لتوضيح كيفية عرض المحتوى الطويل.",
                "10:30"
            )
            layout.addWidget(user_msg)
            
            # رسالة المساعد
            assistant_msg = MessageWidget(
                "assistant",
                "أهلاً! هذه رسالة من المساعد الذكي.\n\n"
                "يمكنني تقديم مساعدة في مواضيع متعددة:\n\n"
                "1. البرمجة\n2. الكتابة\n3. البحث\n\n"
                "```python\ndef hello():\n    print('مرحباً بالعالم!')\n```\n\n"
                "هذا مثال على كتلة كود.",
                "10:31"
            )
            layout.addWidget(assistant_msg)
            
            # عنصر التفكير
            thinking_widget = ThinkingWidget()
            layout.addWidget(thinking_widget)
            
            self.setWindowTitle("اختبار عناصر الرسائل")
            self.resize(500, 600)
    
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())