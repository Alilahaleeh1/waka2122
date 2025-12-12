# -*- coding: utf-8 -*-
"""
الشريط الجانبي - للتحكم بالنموذج وعرض المعلومات
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFrame, QListWidget, QListWidgetItem,
                             QGroupBox, QComboBox, QCheckBox, QSlider,
                             QSpinBox, QDoubleSpinBox, QTreeWidget,
                             QTreeWidgetItem, QScrollArea, QSplitter,
                             QToolButton, QMenu, QAction, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QDateTime
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPainter, QBrush
import os
from datetime import datetime


class Sidebar(QWidget):
    """الشريط الجانبي للتحكم بالنموذج"""
    
    model_changed = pyqtSignal(str)
    settings_changed = pyqtSignal(dict)
    conversation_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # المتغيرات
        self.current_model = "DeepSeek Mini"
        self.conversations = []
        self.model_config = {}
        
        # تهيئة واجهة المستخدم
        self.init_ui()
        
        # تحميل المحادثات
        self.load_conversations()
        
        # إعداد الأنماط
        self.setup_styles()
    
    def init_ui(self):
        """تهيئة واجهة المستخدم"""
        # التخطيط الرئيسي
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # منطقة التمرير
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # القالب المركزي
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)
        
        # شعار التطبيق
        self.create_app_logo()
        content_layout.addWidget(self.logo_widget)
        
        # معلومات النموذج
        self.create_model_info()
        content_layout.addWidget(self.model_group)
        
        # إعدادات التوليد
        self.create_generation_settings()
        content_layout.addWidget(self.generation_group)
        
        # قائمة المحادثات
        self.create_conversation_list()
        content_layout.addWidget(self.conversation_group)
        
        # إحصائيات النظام
        self.create_system_stats()
        content_layout.addWidget(self.system_group)
        
        # فاصل مرن
        content_layout.addStretch()
        
        # زر الإعدادات المتقدمة
        self.create_advanced_settings()
        content_layout.addWidget(self.advanced_button)
        
        # تعيين القالب لمنطقة التمرير
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
    
    def create_app_logo(self):
        """إنشاء شعار التطبيق"""
        self.logo_widget = QWidget()
        logo_layout = QVBoxLayout(self.logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        
        # الشعار
        logo_label = QLabel("DeepSeek Mini")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #3498db;
                color: white;
                border-radius: 10px;
            }
        """)
        logo_layout.addWidget(logo_label)
        
        # الإصدار
        version_label = QLabel("الإصدار 1.0.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        logo_layout.addWidget(version_label)
    
    def create_model_info(self):
        """إنشاء معلومات النموذج"""
        self.model_group = QGroupBox("معلومات النموذج")
        model_layout = QVBoxLayout()
        
        # اسم النموذج
        self.model_name_label = QLabel("DeepSeek Mini")
        self.model_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        model_layout.addWidget(self.model_name_label)
        
        # حجم النموذج
        self.model_size_label = QLabel("الحجم: 125M معلمات")
        model_layout.addWidget(self.model_size_label)
        
        # حالة النموذج
        self.model_status_label = QLabel("الحالة: ⚪ غير محمل")
        model_layout.addWidget(self.model_status_label)
        
        # زر تغيير النموذج
        self.change_model_button = QPushButton("تغيير النموذج")
        self.change_model_button.clicked.connect(self.change_model)
        model_layout.addWidget(self.change_model_button)
        
        # زر تحديث النموذج
        self.refresh_model_button = QPushButton("تحديث النموذج")
        self.refresh_model_button.clicked.connect(self.refresh_model)
        model_layout.addWidget(self.refresh_model_button)
        
        self.model_group.setLayout(model_layout)
    
    def create_generation_settings(self):
        """إنشاء إعدادات التوليد"""
        self.generation_group = QGroupBox("إعدادات التوليد")
        gen_layout = QVBoxLayout()
        
        # درجة الحرارة
        temp_layout = QHBoxLayout()
        temp_label = QLabel("درجة الحرارة:")
        temp_layout.addWidget(temp_label)
        
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(80)
        self.temperature_slider.valueChanged.connect(self.on_settings_changed)
        temp_layout.addWidget(self.temperature_slider)
        
        self.temperature_label = QLabel("0.8")
        temp_layout.addWidget(self.temperature_label)
        gen_layout.addLayout(temp_layout)
        
        # Top-p
        top_p_layout = QHBoxLayout()
        top_p_label = QLabel("Top-p:")
        top_p_layout.addWidget(top_p_label)
        
        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setRange(0, 100)
        self.top_p_slider.setValue(90)
        self.top_p_slider.valueChanged.connect(self.on_settings_changed)
        top_p_layout.addWidget(self.top_p_slider)
        
        self.top_p_label = QLabel("0.9")
        top_p_layout.addWidget(self.top_p_label)
        gen_layout.addLayout(top_p_layout)
        
        # Top-k
        top_k_layout = QHBoxLayout()
        top_k_label = QLabel("Top-k:")
        top_k_layout.addWidget(top_k_label)
        
        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setRange(0, 200)
        self.top_k_spinbox.setValue(50)
        self.top_k_spinbox.valueChanged.connect(self.on_settings_changed)
        top_k_layout.addWidget(self.top_k_spinbox)
        gen_layout.addLayout(top_k_layout)
        
        # عقاب التكرار
        penalty_layout = QHBoxLayout()
        penalty_label = QLabel("عقاب التكرار:")
        penalty_layout.addWidget(penalty_label)
        
        self.penalty_spinbox = QDoubleSpinBox()
        self.penalty_spinbox.setRange(1.0, 2.0)
        self.penalty_spinbox.setSingleStep(0.1)
        self.penalty_spinbox.setValue(1.1)
        self.penalty_spinbox.valueChanged.connect(self.on_settings_changed)
        penalty_layout.addWidget(self.penalty_spinbox)
        gen_layout.addLayout(penalty_layout)
        
        # أقصى رموز
        max_tokens_layout = QHBoxLayout()
        max_tokens_label = QLabel("أقصى رموز:")
        max_tokens_layout.addWidget(max_tokens_label)
        
        self.max_tokens_spinbox = QSpinBox()
        self.max_tokens_spinbox.setRange(10, 4096)
        self.max_tokens_spinbox.setValue(512)
        self.max_tokens_spinbox.valueChanged.connect(self.on_settings_changed)
        max_tokens_layout.addWidget(self.max_tokens_spinbox)
        gen_layout.addLayout(max_tokens_layout)
        
        # أخذ العينات
        self.sampling_checkbox = QCheckBox("أخذ عينات عشوائية")
        self.sampling_checkbox.setChecked(True)
        self.sampling_checkbox.stateChanged.connect(self.on_settings_changed)
        gen_layout.addWidget(self.sampling_checkbox)
        
        self.generation_group.setLayout(gen_layout)
    
    def create_conversation_list(self):
        """إنشاء قائمة المحادثات"""
        self.conversation_group = QGroupBox("المحادثات الحديثة")
        conv_layout = QVBoxLayout()
        
        # قائمة المحادثات
        self.conversation_list = QListWidget()
        self.conversation_list.itemClicked.connect(self.on_conversation_selected)
        conv_layout.addWidget(self.conversation_list)
        
        # أزرار التحكم
        button_layout = QHBoxLayout()
        
        self.new_conversation_button = QPushButton("جديد")
        self.new_conversation_button.clicked.connect(self.new_conversation)
        button_layout.addWidget(self.new_conversation_button)
        
        self.delete_conversation_button = QPushButton("حذف")
        self.delete_conversation_button.clicked.connect(self.delete_conversation)
        button_layout.addWidget(self.delete_conversation_button)
        
        self.rename_conversation_button = QPushButton("إعادة تسمية")
        self.rename_conversation_button.clicked.connect(self.rename_conversation)
        button_layout.addWidget(self.rename_conversation_button)
        
        conv_layout.addLayout(button_layout)
        
        self.conversation_group.setLayout(conv_layout)
    
    def create_system_stats(self):
        """إنشاء إحصائيات النظام"""
        self.system_group = QGroupBox("إحصائيات النظام")
        system_layout = QVBoxLayout()
        
        # استخدام الذاكرة
        self.memory_label = QLabel("ذاكرة GPU: --/-- GB")
        system_layout.addWidget(self.memory_label)
        
        # استخدام المعالج
        self.cpu_label = QLabel("المعالج: --%")
        system_layout.addWidget(self.cpu_label)
        
        # درجة حرارة GPU
        self.temp_label = QLabel("حرارة GPU: --°C")
        system_layout.addWidget(self.temp_label)
        
        # سرعة التوليد
        self.speed_label = QLabel("السرعة: -- رمز/ثانية")
        system_layout.addWidget(self.speed_label)
        
        # شريط تقدم الذاكرة
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        system_layout.addWidget(self.memory_progress)
        
        # زر تحديث
        self.refresh_stats_button = QPushButton("تحديث الإحصائيات")
        self.refresh_stats_button.clicked.connect(self.update_system_stats)
        system_layout.addWidget(self.refresh_stats_button)
        
        self.system_group.setLayout(system_layout)
        
        # مؤقت لتحديث الإحصائيات
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_system_stats)
        self.stats_timer.start(5000)  # كل 5 ثواني
    
    def create_advanced_settings(self):
        """إنشاء الإعدادات المتقدمة"""
        self.advanced_button = QPushButton("⚙️ الإعدادات المتقدمة")
        self.advanced_button.clicked.connect(self.show_advanced_settings)
        self.advanced_button.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #34495e;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #2c3e50;
            }
        """)
    
    def setup_styles(self):
        """إعداد الأنماط"""
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #495057;
            }
            
            QListWidget {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
                font-size: 12px;
            }
            
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
            }
            
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
            
            QPushButton {
                padding: 5px 10px;
                border-radius: 5px;
                border: 1px solid #ced4da;
                background-color: white;
                color: #495057;
            }
            
            QPushButton:hover {
                background-color: #e9ecef;
            }
            
            QLabel {
                color: #495057;
            }
            
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 3px;
                text-align: center;
            }
            
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            
            QSlider::groove:horizontal {
                height: 6px;
                background: #dee2e6;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #3498db;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            QSpinBox, QDoubleSpinBox {
                padding: 3px;
                border: 1px solid #ced4da;
                border-radius: 3px;
                background-color: white;
            }
            
            QCheckBox {
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
    
    def set_model_info(self, config):
        """تعيين معلومات النموذج"""
        self.model_config = config
        
        # تحديث العلامات
        vocab_size = config.get('vocab_size', 0)
        d_model = config.get('d_model', 0)
        n_layers = config.get('n_layers', 0)
        
        # حساب الحجم التقريبي
        param_count = (vocab_size * d_model) + (n_layers * 12 * d_model * d_model)
        param_count_m = param_count / 1_000_000
        
        self.model_size_label.setText(f"الحجم: {param_count_m:.1f}M معلمات")
        self.model_status_label.setText("الحالة: ✅ محمل وجاهز")
        
        # تحديث اسم النموذج
        model_name = config.get('model_name', 'DeepSeek Mini')
        self.model_name_label.setText(model_name)
    
    def update_conversation_list(self, conversation_history):
        """تحديث قائمة المحادثات"""
        # حفظ المحادثة الحالية
        if conversation_history:
            self.save_current_conversation(conversation_history)
        
        # تحديث القائمة
        self.load_conversations()
    
    def save_current_conversation(self, conversation_history):
        """حفظ المحادثة الحالية"""
        try:
            conversations_dir = os.path.join(os.path.expanduser("~"), 
                                           ".deepseek_mini", "conversations")
            os.makedirs(conversations_dir, exist_ok=True)
            
            # اسم الملف
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conv_{timestamp}.json"
            filepath = os.path.join(conversations_dir, filename)
            
            # تحويل إلى JSON
            import json
            conversation_data = {
                "id": timestamp,
                "title": f"محادثة {datetime.now().strftime('%H:%M')}",
                "timestamp": timestamp,
                "message_count": len(conversation_history),
                "preview": conversation_history[-1]['content'][:50] if conversation_history else "",
                "data": conversation_history
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"خطأ في حفظ المحادثة: {e}")
    
    def load_conversations(self):
        """تحميل المحادثات المحفوظة"""
        self.conversation_list.clear()
        self.conversations = []
        
        try:
            conversations_dir = os.path.join(os.path.expanduser("~"), 
                                           ".deepseek_mini", "conversations")
            
            if not os.path.exists(conversations_dir):
                return
            
            # تحميل الملفات
            import json
            conv_files = sorted(os.listdir(conversations_dir), reverse=True)[:20]  # آخر 20 محادثة
            
            for filename in conv_files:
                if filename.endswith('.json'):
                    filepath = os.path.join(conversations_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)
                        
                        # إضافة إلى القائمة
                        item_text = f"{conv_data['title']}\n{conv_data['preview']}..."
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.UserRole, conv_data['id'])
                        
                        self.conversation_list.addItem(item)
                        self.conversations.append(conv_data)
                        
                    except Exception as e:
                        print(f"خطأ في تحميل المحادثة {filename}: {e}")
                        
        except Exception as e:
            print(f"خطأ في تحميل المحادثات: {e}")
    
    def on_settings_changed(self):
        """عند تغيير الإعدادات"""
        settings = {
            "temperature": self.temperature_slider.value() / 100.0,
            "top_p": self.top_p_slider.value() / 100.0,
            "top_k": self.top_k_spinbox.value(),
            "repetition_penalty": self.penalty_spinbox.value(),
            "max_tokens": self.max_tokens_spinbox.value(),
            "do_sample": self.sampling_checkbox.isChecked()
        }
        
        # تحديث العلامات
        self.temperature_label.setText(f"{settings['temperature']:.2f}")
        self.top_p_label.setText(f"{settings['top_p']:.2f}")
        
        # إرسال الإشارة
        self.settings_changed.emit(settings)
    
    def on_conversation_selected(self, item):
        """عند اختيار محادثة"""
        conv_id = item.data(Qt.UserRole)
        self.conversation_selected.emit(conv_id)
        
        # هنا يمكن تحميل المحادثة المحددة
        print(f"تم اختيار المحادثة: {conv_id}")
    
    def change_model(self):
        """تغيير النموذج"""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "اختر نموذج",
            "",
            "ملفات النموذج (*.pt *.pth *.bin);;جميع الملفات (*)"
        )
        
        if file_path:
            self.model_changed.emit(file_path)
            
            # تحديث العلامة
            model_name = os.path.basename(file_path)
            self.model_name_label.setText(model_name)
            self.model_status_label.setText("الحالة: ⏳ جاري التحميل...")
    
    def refresh_model(self):
        """تحديث النموذج"""
        # إعادة تحميل النموذج الحالي
        self.model_status_label.setText("الحالة: ⏳ جاري التحديث...")
        QTimer.singleShot(1000, lambda: self.model_status_label.setText("الحالة: ✅ محمل وجاهز"))
    
    def new_conversation(self):
        """محادثة جديدة"""
        from PyQt5.QtWidgets import QInputDialog
        
        title, ok = QInputDialog.getText(
            self,
            "محادثة جديدة",
            "أدخل عنوان المحادثة:"
        )
        
        if ok and title:
            # إنشاء محادثة جديدة
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            conversation_data = {
                "id": timestamp,
                "title": title,
                "timestamp": timestamp,
                "preview": "محادثة جديدة",
                "data": []
            }
            
            # إضافة إلى القائمة
            item_text = f"{title}\nمحادثة جديدة..."
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, timestamp)
            
            self.conversation_list.insertItem(0, item)
            self.conversations.insert(0, conversation_data)
    
    def delete_conversation(self):
        """حذف المحادثة المحددة"""
        current_item = self.conversation_list.currentItem()
        
        if not current_item:
            return
        
        conv_id = current_item.data(Qt.UserRole)
        
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "حذف المحادثة",
            "هل أنت متأكد من حذف المحادثة المحددة؟",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # حذف الملف
            try:
                conversations_dir = os.path.join(os.path.expanduser("~"), 
                                               ".deepseek_mini", "conversations")
                filepath = os.path.join(conversations_dir, f"conv_{conv_id}.json")
                
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                # إزالة من القائمة
                row = self.conversation_list.row(current_item)
                self.conversation_list.takeItem(row)
                
            except Exception as e:
                QMessageBox.critical(self, "خطأ", f"فشل حذف المحادثة: {e}")
    
    def rename_conversation(self):
        """إعادة تسمية المحادثة"""
        current_item = self.conversation_list.currentItem()
        
        if not current_item:
            return
        
        from PyQt5.QtWidgets import QInputDialog
        
        old_title = current_item.text().split('\n')[0]
        new_title, ok = QInputDialog.getText(
            self,
            "إعادة تسمية",
            "أدخل العنوان الجديد:",
            text=old_title
        )
        
        if ok and new_title:
            # تحديث النص
            preview = current_item.text().split('\n')[1]
            current_item.setText(f"{new_title}\n{preview}")
    
    def update_system_stats(self):
        """تحديث إحصائيات النظام"""
        try:
            import psutil
            import torch
            
            # استخدام الذاكرة النظامية
            memory = psutil.virtual_memory()
            self.cpu_label.setText(f"المعالج: {psutil.cpu_percent()}%")
            
            # استخدام الذاكرة GPU
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                memory_percent = (memory_used / memory_total) * 100
                
                self.memory_label.setText(f"ذاكرة GPU: {memory_used:.1f}/{memory_total:.1f} GB")
                self.memory_progress.setValue(int(memory_percent))
                
                # درجة الحرارة (محاولة قراءتها)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.temp_label.setText(f"حرارة GPU: {temp}°C")
                    pynvml.nvmlShutdown()
                except:
                    self.temp_label.setText("حرارة GPU: --°C")
            else:
                self.memory_label.setText("ذاكرة GPU: غير متاحة")
                self.temp_label.setText("حرارة GPU: --°C")
                self.memory_progress.setValue(0)
            
        except Exception as e:
            print(f"خطأ في تحديث إحصائيات النظام: {e}")
    
    def show_advanced_settings(self):
        """عرض الإعدادات المتقدمة"""
        from PyQt5.QtWidgets import QDialog, QTabWidget, QFormLayout, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("الإعدادات المتقدمة")
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # علامات التبويب
        tabs = QTabWidget()
        
        # تبويب النموذج
        model_tab = QWidget()
        model_layout = QFormLayout(model_tab)
        
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp32", "fp16", "bf16"])
        model_layout.addRow("الدقة:", self.precision_combo)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "mps"])
        model_layout.addRow("الجهاز:", self.device_combo)
        
        self.cache_checkbox = QCheckBox("استخدام ذاكرة التخزين المؤقت")
        self.cache_checkbox.setChecked(True)
        model_layout.addRow("الذاكرة المؤقتة:", self.cache_checkbox)
        
        tabs.addTab(model_tab, "النموذج")
        
        # تبويب الواجهة
        ui_tab = QWidget()
        ui_layout = QFormLayout(ui_tab)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["فاتح", "داكن", "تلقائي"])
        ui_layout.addRow("السمة:", self.theme_combo)
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setValue(12)
        ui_layout.addRow("حجم الخط:", self.font_size_spin)
        
        self.animation_checkbox = QCheckBox("تمكين التحريك")
        self.animation_checkbox.setChecked(True)
        ui_layout.addRow("التحريك:", self.animation_checkbox)
        
        tabs.addTab(ui_tab, "الواجهة")
        
        # تبويب النظام
        system_tab = QWidget()
        system_layout = QFormLayout(system_tab)
        
        self.auto_save_checkbox = QCheckBox("الحفظ التلقائي")
        self.auto_save_checkbox.setChecked(True)
        system_layout.addRow("الحفظ:", self.auto_save_checkbox)
        
        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(1, 60)
        self.auto_save_interval.setValue(5)
        self.auto_save_interval.setSuffix(" دقائق")
        system_layout.addRow("فاصل الحفظ:", self.auto_save_interval)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        system_layout.addRow("مستوى التسجيل:", self.log_level_combo)
        
        tabs.addTab(system_tab, "النظام")
        
        layout.addWidget(tabs)
        
        # أزرار الحوار
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_advanced_settings)
        
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            self.apply_advanced_settings()
    
    def apply_advanced_settings(self):
        """تطبيق الإعدادات المتقدمة"""
        # هنا يمكن تطبيق الإعدادات
        print("تم تطبيق الإعدادات المتقدمة")


if __name__ == "__main__":
    # اختبار الشريط الجانبي
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            layout = QHBoxLayout(central_widget)
            
            # إنشاء الشريط الجانبي
            self.sidebar = Sidebar()
            self.sidebar.model_changed.connect(self.on_model_changed)
            self.sidebar.settings_changed.connect(self.on_settings_changed)
            
            layout.addWidget(self.sidebar)
            
            # معلومات وهمية للنموذج
            model_config = {
                'vocab_size': 50000,
                'd_model': 768,
                'n_layers': 12,
                'model_name': 'DeepSeek Mini'
            }
            self.sidebar.set_model_info(model_config)
            
            self.setWindowTitle("اختبار الشريط الجانبي")
            self.resize(900, 600)
        
        def on_model_changed(self, model_path):
            print(f"تم تغيير النموذج إلى: {model_path}")
        
        def on_settings_changed(self, settings):
            print(f"تم تغيير الإعدادات: {settings}")
    
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())