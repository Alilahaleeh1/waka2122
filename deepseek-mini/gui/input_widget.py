# -*- coding: utf-8 -*-
"""
عنصر الإدخال - لإدخال النص وإرسال الرسائل
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QComboBox, QLabel, QFrame, QSizeGrip,
                             QScrollBar, QMenu, QAction, QFileDialog, QToolButton)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QPoint
from PyQt5.QtGui import (QFont, QTextCursor, QKeyEvent, QIcon, QPixmap,
                         QTextCharFormat, QColor, QFontMetrics, QPainter)
import os
from datetime import datetime


class InputWidget(QWidget):
    """عنصر إدخال النص"""
    
    message_ready = pyqtSignal(str)
    stop_generation = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # المتغيرات
        self.is_generating = False
        self.text_history = []
        self.history_index = -1
        self.current_text = ""
        
        # تهيئة واجهة المستخدم
        self.init_ui()
        
        # إعداد الأنماط
        self.setup_styles()
        
        # إعداد المؤقتات
        self.setup_timers()
    
    def init_ui(self):
        """تهيئة واجهة المستخدم"""
        # التخطيط الرئيسي
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # شريط الأدوات العلوي
        self.create_toolbar()
        main_layout.addWidget(self.toolbar_widget)
        
        # منطقة الإدخال
        self.create_input_area()
        main_layout.addWidget(self.input_area)
        
        # شريط الأدوات السفلي
        self.create_bottom_bar()
        main_layout.addWidget(self.bottom_bar)
    
    def create_toolbar(self):
        """إنشاء شريط الأدوات العلوي"""
        self.toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar_widget)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar_layout.setSpacing(5)
        
        # زر تحميل الملف
        self.file_button = QToolButton()
        self.file_button.setIcon(QIcon(":/icons/attach.svg"))
        self.file_button.setToolTip("إرفاق ملف")
        self.file_button.clicked.connect(self.attach_file)
        toolbar_layout.addWidget(self.file_button)
        
        # زر الصور
        self.image_button = QToolButton()
        self.image_button.setIcon(QIcon(":/icons/image.svg"))
        self.image_button.setToolTip("إرفاق صورة")
        self.image_button.clicked.connect(self.attach_image)
        toolbar_layout.addWidget(self.image_button)
        
        # فاصل
        toolbar_layout.addStretch()
        
        # زر التنسيق
        self.format_button = QToolButton()
        self.format_button.setIcon(QIcon(":/icons/format.svg"))
        self.format_button.setToolTip("تنسيق النص")
        self.format_button.clicked.connect(self.show_format_menu)
        toolbar_layout.addWidget(self.format_button)
        
        # زر الإعدادات
        self.settings_button = QToolButton()
        self.settings_button.setIcon(QIcon(":/icons/settings.svg"))
        self.settings_button.setToolTip("إعدادات الإدخال")
        self.settings_button.clicked.connect(self.show_settings)
        toolbar_layout.addWidget(self.settings_button)
    
    def create_input_area(self):
        """إنشاء منطقة الإدخال"""
        self.input_area = QFrame()
        self.input_area.setFrameStyle(QFrame.Panel | QFrame.Raised)
        
        input_layout = QVBoxLayout(self.input_area)
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        # منطقة النص القابلة للتحرير
        self.text_edit = EnhancedTextEdit()
        self.text_edit.setPlaceholderText("اكتب رسالتك هنا... (اضغط Ctrl+Enter للإرسال)")
        self.text_edit.textChanged.connect(self.on_text_changed)
        self.text_edit.enter_pressed.connect(self.send_message)
        
        input_layout.addWidget(self.text_edit)
    
    def create_bottom_bar(self):
        """إنشاء الشريط السفلي"""
        self.bottom_bar = QWidget()
        bottom_layout = QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(10, 5, 10, 5)
        
        # عداد الحروف
        self.char_count_label = QLabel("0 حرف")
        bottom_layout.addWidget(self.char_count_label)
        
        # فاصل
        bottom_layout.addStretch()
        
        # زر الإيقاف
        self.stop_button = QPushButton("إيقاف التوليد")
        self.stop_button.setIcon(QIcon(":/icons/stop.svg"))
        self.stop_button.clicked.connect(self.stop_generation.emit)
        self.stop_button.setEnabled(False)
        bottom_layout.addWidget(self.stop_button)
        
        # زر الإرسال
        self.send_button = QPushButton("إرسال")
        self.send_button.setIcon(QIcon(":/icons/send.svg"))
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setDefault(True)
        self.send_button.setMinimumWidth(100)
        bottom_layout.addWidget(self.send_button)
        
        # زر الميكروفون
        self.mic_button = QToolButton()
        self.mic_button.setIcon(QIcon(":/icons/mic.svg"))
        self.mic_button.setToolTip("إدخال صوتي")
        self.mic_button.clicked.connect(self.start_voice_input)
        bottom_layout.addWidget(self.mic_button)
    
    def setup_styles(self):
        """إعداد الأنماط"""
        self.setStyleSheet("""
            QWidget {
                background-color: white;
            }
            
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            
            QTextEdit {
                border: none;
                background-color: transparent;
                font-size: 14px;
                padding: 10px;
                selection-background-color: #b3d9ff;
            }
            
            QPushButton {
                padding: 8px 16px;
                border-radius: 5px;
                border: 1px solid #4CAF50;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #45a049;
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #aaaaaa;
                color: #666666;
            }
            
            QPushButton#stop_button {
                background-color: #f44336;
                border-color: #f44336;
            }
            
            QPushButton#stop_button:hover {
                background-color: #d32f2f;
            }
            
            QToolButton {
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            
            QToolButton:hover {
                background-color: #f0f0f0;
            }
            
            QLabel {
                color: #666;
                font-size: 12px;
            }
        """)
        
        # تعيين ID لزر الإيقاف
        self.stop_button.setObjectName("stop_button")
    
    def setup_timers(self):
        """إعداد المؤقتات"""
        # مؤقت لحفظ النص الحالي
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_text)
        self.auto_save_timer.start(5000)  # كل 5 ثواني
    
    def on_text_changed(self):
        """عند تغيير النص"""
        # تحديث عداد الحروف
        text = self.text_edit.toPlainText()
        char_count = len(text)
        word_count = len(text.split())
        
        self.char_count_label.setText(f"{char_count} حرف، {word_count} كلمة")
        
        # حفظ النص الحالي
        self.current_text = text
        
        # تمكين/تعطيل زر الإرسال
        self.send_button.setEnabled(char_count > 0)
    
    def send_message(self):
        """إرسال الرسالة"""
        text = self.text_edit.toPlainText().strip()
        
        if not text:
            return
        
        # إضافة إلى التاريخ
        self.text_history.append(text)
        self.history_index = len(self.text_history)
        
        # إرسال الإشارة
        self.message_ready.emit(text)
        
        # مسح منطقة الإدخال
        self.text_edit.clear()
        
        # حفظ التاريخ إلى الملف
        self.save_history()
        
        # تحديث التركيز
        self.text_edit.setFocus()
    
    def attach_file(self):
        """إرفاق ملف"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "اختر ملف",
            "",
            "جميع الملفات (*);;ملفات النص (*.txt *.md *.py *.js *.html *.css);;"
            "ملفات PDF (*.pdf);;ملفات الوثائق (*.doc *.docx *.odt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # إضافة اسم الملف كمرفق
                filename = os.path.basename(file_path)
                text = f"[ملف مرفق: {filename}]\n```\n{content[:1000]}\n```"
                
                if len(content) > 1000:
                    text += f"\n... (تم اقتصاص الملف، الطول الأصلي: {len(content)} حرف)"
                
                self.text_edit.insertPlainText(text)
                
            except Exception as e:
                self.text_edit.insertPlainText(f"[خطأ في قراءة الملف: {str(e)}]")
    
    def attach_image(self):
        """إرفاق صورة"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "اختر صورة",
            "",
            "الصور (*.png *.jpg *.jpeg *.gif *.bmp);;جميع الملفات (*)"
        )
        
        if file_path:
            # إضافة مرجع للصورة
            filename = os.path.basename(file_path)
            text = f"![{filename}]({file_path})"
            self.text_edit.insertPlainText(text)
    
    def show_format_menu(self):
        """عرض قائمة التنسيق"""
        menu = QMenu(self)
        
        # العناوين
        h1_action = QAction("عنوان رئيسي (H1)", self)
        h1_action.triggered.connect(lambda: self.insert_format("# ", ""))
        menu.addAction(h1_action)
        
        h2_action = QAction("عنوان فرعي (H2)", self)
        h2_action.triggered.connect(lambda: self.insert_format("## ", ""))
        menu.addAction(h2_action)
        
        menu.addSeparator()
        
        # النص الغامق والمائل
        bold_action = QAction("غامق", self)
        bold_action.triggered.connect(lambda: self.insert_format("**", "**"))
        menu.addAction(bold_action)
        
        italic_action = QAction("مائل", self)
        italic_action.triggered.connect(lambda: self.insert_format("*", "*"))
        menu.addAction(italic_action)
        
        menu.addSeparator()
        
        # القوائم
        list_action = QAction("قائمة نقطية", self)
        list_action.triggered.connect(lambda: self.insert_format("- ", ""))
        menu.addAction(list_action)
        
        num_action = QAction("قائمة مرقمة", self)
        num_action.triggered.connect(lambda: self.insert_format("1. ", ""))
        menu.addAction(num_action)
        
        menu.addSeparator()
        
        # الكود
        code_action = QAction("كود سطري", self)
        code_action.triggered.connect(lambda: self.insert_format("`", "`"))
        menu.addAction(code_action)
        
        block_action = QAction("كتلة كود", self)
        block_action.triggered.connect(lambda: self.insert_format("```\n", "\n```"))
        menu.addAction(block_action)
        
        menu.addSeparator()
        
        # رابط
        link_action = QAction("رابط", self)
        link_action.triggered.connect(self.insert_link)
        menu.addAction(link_action)
        
        # عرض القائمة
        menu.exec_(self.format_button.mapToGlobal(self.format_button.rect().bottomLeft()))
    
    def insert_format(self, prefix, suffix):
        """إدراج تنسيق حول النص المحدد"""
        cursor = self.text_edit.textCursor()
        
        if cursor.hasSelection():
            # هناك نص محدد
            selected_text = cursor.selectedText()
            cursor.insertText(f"{prefix}{selected_text}{suffix}")
        else:
            # لا يوجد نص محدد، إدراج النص مع التنسيق
            cursor.insertText(f"{prefix}نص{suffix}")
            # تحديد النص المدرج للاستبدال السريع
            cursor.movePosition(QTextCursor.Left, QTextCursor.MoveAnchor, len(suffix))
            cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len("نص"))
            self.text_edit.setTextCursor(cursor)
    
    def insert_link(self):
        """إدراج رابط"""
        cursor = self.text_edit.textCursor()
        
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            link_text = f"[{selected_text}](رابط)"
            cursor.insertText(link_text)
            
            # تحديد "رابط" للاستبدال
            cursor.movePosition(QTextCursor.Left, QTextCursor.MoveAnchor, 1)
            cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len("رابط"))
            self.text_edit.setTextCursor(cursor)
        else:
            cursor.insertText("[نص الرابط](رابط)")
            
            # تحديد "نص الرابط" للاستبدال
            cursor.movePosition(QTextCursor.Left, QTextCursor.MoveAnchor, len("](رابط)"))
            cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len("نص الرابط"))
            self.text_edit.setTextCursor(cursor)
    
    def show_settings(self):
        """عرض إعدادات الإدخال"""
        menu = QMenu(self)
        
        # حجم الخط
        font_menu = menu.addMenu("حجم الخط")
        
        font_sizes = [10, 12, 14, 16, 18, 20]
        current_size = self.text_edit.font().pointSize()
        
        for size in font_sizes:
            size_action = QAction(f"{size}pt", self)
            size_action.setCheckable(True)
            size_action.setChecked(size == current_size)
            size_action.triggered.connect(lambda checked, s=size: self.set_font_size(s))
            font_menu.addAction(size_action)
        
        menu.addSeparator()
        
        # مسح التاريخ
        clear_action = QAction("مسح تاريخ الإدخال", self)
        clear_action.triggered.connect(self.clear_history)
        menu.addAction(clear_action)
        
        # عرض التاريخ
        history_action = QAction("عرض التاريخ", self)
        history_action.triggered.connect(self.show_history)
        menu.addAction(history_action)
        
        # عرض القائمة
        menu.exec_(self.settings_button.mapToGlobal(self.settings_button.rect().bottomLeft()))
    
    def set_font_size(self, size):
        """تعيين حجم الخط"""
        font = self.text_edit.font()
        font.setPointSize(size)
        self.text_edit.setFont(font)
    
    def clear_history(self):
        """مسح تاريخ الإدخال"""
        from PyQt5.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "مسح التاريخ",
            "هل أنت متأكد من مسح تاريخ الإدخال؟",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.text_history.clear()
            self.history_index = -1
            self.save_history()
    
    def show_history(self):
        """عرض تاريخ الإدخال"""
        if not self.text_history:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "التاريخ", "لا يوجد تاريخ إدخال.")
            return
        
        # هنا يمكن عرض التاريخ في نافذة منفصلة
        # هذا تنفيذ مبسط
        history_text = "\n".join([f"{i+1}. {text[:50]}..." 
                                 for i, text in enumerate(self.text_history[-10:])])
        
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "آخر 10 رسائل", history_text)
    
    def start_voice_input(self):
        """بدء الإدخال الصوتي"""
        # هنا يمكن تنفيذ الإدخال الصوتي
        # هذا تنفيذ مبسط
        self.text_edit.insertPlainText("[الإدخال الصوتي غير متاح حالياً]")
    
    def auto_save_text(self):
        """حفظ النص تلقائياً"""
        if self.current_text:
            # حفظ النص الحالي مؤقتاً
            pass
    
    def save_history(self):
        """حفظ التاريخ إلى ملف"""
        try:
            history_dir = os.path.join(os.path.expanduser("~"), ".deepseek_mini")
            os.makedirs(history_dir, exist_ok=True)
            
            history_file = os.path.join(history_dir, "input_history.txt")
            
            with open(history_file, 'w', encoding='utf-8') as f:
                for text in self.text_history[-100:]:  # حفظ آخر 100 رسالة
                    f.write(f"{text}\n{'='*50}\n")
                    
        except Exception as e:
            print(f"خطأ في حفظ التاريخ: {e}")
    
    def load_history(self):
        """تحميل التاريخ من ملف"""
        try:
            history_file = os.path.join(os.path.expanduser("~"), 
                                       ".deepseek_mini", "input_history.txt")
            
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    entries = content.split('\n' + '='*50 + '\n')
                    self.text_history = [entry.strip() for entry in entries if entry.strip()]
                    self.history_index = len(self.text_history)
                    
        except Exception as e:
            print(f"خطأ في تحميل التاريخ: {e}")
    
    def set_stop_enabled(self, enabled):
        """تعطيل/تمكين زر الإيقاف"""
        self.stop_button.setEnabled(enabled)
        self.is_generating = enabled
    
    def keyPressEvent(self, event):
        """معالج حدث الضغط على المفتاح"""
        # التنقل في التاريخ باستخدام السهمين لأعلى ولأسفل
        if event.key() == Qt.Key_Up and self.text_history:
            if self.history_index > 0:
                self.history_index -= 1
                self.text_edit.setPlainText(self.text_history[self.history_index])
        elif event.key() == Qt.Key_Down and self.text_history:
            if self.history_index < len(self.text_history) - 1:
                self.history_index += 1
                self.text_edit.setPlainText(self.text_history[self.history_index])
            else:
                self.history_index = len(self.text_history)
                self.text_edit.clear()
        else:
            super().keyPressEvent(event)


class EnhancedTextEdit(QTextEdit):
    """محرر نص محسن مع ميزات إضافية"""
    
    enter_pressed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # إعدادات إضافية
        self.setAcceptRichText(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        
        # تخصيص شريط التمرير
        self.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical {
                background-color: transparent;
                width: 10px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #888;
                border-radius: 5px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
        """)
    
    def keyPressEvent(self, event):
        """معالج حدث الضغط على المفتاح"""
        # Ctrl+Enter للإرسال
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
            self.enter_pressed.emit()
            return
        
        # Ctrl+Z للتراجع (مع زيادة الحد)
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            if event.modifiers() & Qt.ShiftModifier:
                self.redo()
            else:
                self.undo()
            return
        
        # Ctrl+Y لإعادة
        if event.key() == Qt.Key_Y and event.modifiers() == Qt.ControlModifier:
            self.redo()
            return
        
        # Ctrl+D لنسخ السطر
        if event.key() == Qt.Key_D and event.modifiers() == Qt.ControlModifier:
            self.duplicate_line()
            return
        
        # Tab للمسافة البادئة
        if event.key() == Qt.Key_Tab:
            cursor = self.textCursor()
            if cursor.hasSelection():
                # زيادة مسافة النص المحدد
                self.indent_selection()
            else:
                # إدراج مسافة
                cursor.insertText("    ")
            return
        
        # Shift+Tab لإزالة المسافة البادئة
        if event.key() == Qt.Key_Backtab:
            self.unindent_selection()
            return
        
        super().keyPressEvent(event)
    
    def duplicate_line(self):
        """تكرار السطر الحالي"""
        cursor = self.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        line_text = cursor.selectedText()
        
        cursor.movePosition(QTextCursor.EndOfLine)
        cursor.insertText("\n" + line_text)
    
    def indent_selection(self):
        """زيادة مسافة النص المحدد"""
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.StartOfLine)
        
        new_text = ""
        while cursor.position() <= end:
            cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
            line_text = cursor.selectedText()
            new_text += "    " + line_text + "\n"
            
            if cursor.atEnd():
                break
            
            cursor.movePosition(QTextCursor.Down)
            cursor.movePosition(QTextCursor.StartOfLine)
        
        # إزالة السطر الجديد الأخير
        new_text = new_text.rstrip('\n')
        
        # استبدال النص
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.insertText(new_text)
    
    def unindent_selection(self):
        """إزالة المسافة البادئة من النص المحدد"""
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.StartOfLine)
        
        new_text = ""
        while cursor.position() <= end:
            cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
            line_text = cursor.selectedText()
            
            # إزالة 4 مسافات أو تاب
            if line_text.startswith("    "):
                line_text = line_text[4:]
            elif line_text.startswith("\t"):
                line_text = line_text[1:]
            
            new_text += line_text + "\n"
            
            if cursor.atEnd():
                break
            
            cursor.movePosition(QTextCursor.Down)
            cursor.movePosition(QTextCursor.StartOfLine)
        
        # إزالة السطر الجديد الأخير
        new_text = new_text.rstrip('\n')
        
        # استبدال النص
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.insertText(new_text)
    
    def wheelEvent(self, event):
        """حدث عجلة الماوس للتحكم بحجم الخط"""
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            font = self.font()
            current_size = font.pointSize()
            
            if delta > 0:
                new_size = min(current_size + 1, 30)
            else:
                new_size = max(current_size - 1, 8)
            
            font.setPointSize(new_size)
            self.setFont(font)
            event.accept()
        else:
            super().wheelEvent(event)


if __name__ == "__main__":
    # اختبار عنصر الإدخال
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            layout = QVBoxLayout(central_widget)
            
            # إنشاء عنصر الإدخال
            self.input_widget = InputWidget()
            self.input_widget.message_ready.connect(self.on_message_ready)
            
            layout.addWidget(self.input_widget)
            
            self.setWindowTitle("اختبار عنصر الإدخال")
            self.resize(600, 300)
        
        def on_message_ready(self, message):
            print(f"رسالة مرسلة: {message}")
    
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())