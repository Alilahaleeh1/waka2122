# -*- coding: utf-8 -*-
"""
أنماط وتنسيقات الواجهة الرسومية
"""

from PyQt5.QtGui import QPalette, QColor, QFont, QIcon, QPixmap, QPainter
from PyQt5.QtCore import Qt
import os
from pathlib import Path


def get_stylesheet(theme="dark"):
    """
    الحصول على أنماط CSS للواجهة
    
    Args:
        theme: السمة ("dark", "light", "blue")
    
    Returns:
        سلسلة CSS
    """
    if theme == "dark":
        return dark_stylesheet()
    elif theme == "light":
        return light_stylesheet()
    elif theme == "blue":
        return blue_stylesheet()
    else:
        return light_stylesheet()


def dark_stylesheet():
    """أنماط السمة الداكنة"""
    return """
    /* النافذة الرئيسية */
    QMainWindow {
        background-color: #1e1e1e;
    }
    
    QWidget {
        background-color: #2d2d30;
        color: #d4d4d4;
        font-family: 'Segoe UI', 'Tahoma', 'Geneva', 'Verdana', sans-serif;
        font-size: 12px;
        border: none;
    }
    
    /* الأزرار */
    QPushButton {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px 12px;
        color: #d4d4d4;
        font-weight: normal;
    }
    
    QPushButton:hover {
        background-color: #46464a;
        border-color: #666666;
    }
    
    QPushButton:pressed {
        background-color: #007acc;
        border-color: #1c97ea;
        color: white;
    }
    
    QPushButton:disabled {
        background-color: #2d2d30;
        border-color: #3e3e42;
        color: #6e6e6e;
    }
    
    /* أزرار خاصة */
    QPushButton#primary_button {
        background-color: #007acc;
        border-color: #1c97ea;
        color: white;
        font-weight: bold;
    }
    
    QPushButton#primary_button:hover {
        background-color: #1c97ea;
    }
    
    QPushButton#danger_button {
        background-color: #d13438;
        border-color: #e81123;
        color: white;
    }
    
    /* أزرار الأدوات */
    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 4px;
    }
    
    QToolButton:hover {
        background-color: #3e3e42;
        border-color: #555555;
    }
    
    QToolButton:pressed {
        background-color: #007acc;
    }
    
    QToolButton:checked {
        background-color: #007acc;
        border-color: #1c97ea;
        color: white;
    }
    
    /* مربعات النص */
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px;
        color: #d4d4d4;
        selection-background-color: #007acc;
        selection-color: white;
    }
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #007acc;
    }
    
    QTextEdit, QPlainTextEdit {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    
    /* القوائم */
    QMenuBar {
        background-color: #2d2d30;
        color: #d4d4d4;
    }
    
    QMenuBar::item {
        padding: 5px 10px;
        background-color: transparent;
    }
    
    QMenuBar::item:selected {
        background-color: #3e3e42;
    }
    
    QMenu {
        background-color: #2d2d30;
        border: 1px solid #555555;
        color: #d4d4d4;
    }
    
    QMenu::item {
        padding: 6px 30px 6px 20px;
    }
    
    QMenu::item:selected {
        background-color: #007acc;
        color: white;
    }
    
    QMenu::separator {
        height: 1px;
        background-color: #555555;
        margin: 4px 0px;
    }
    
    /* القوائم المنسدلة */
    QComboBox {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px;
        color: #d4d4d4;
        min-height: 22px;
    }
    
    QComboBox:hover {
        border-color: #666666;
    }
    
    QComboBox:focus {
        border-color: #007acc;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #d4d4d4;
        width: 0;
        height: 0;
    }
    
    QComboBox QAbstractItemView {
        background-color: #2d2d30;
        border: 1px solid #555555;
        color: #d4d4d4;
        selection-background-color: #007acc;
        selection-color: white;
    }
    
    /* مربعات الاختيار */
    QCheckBox {
        spacing: 8px;
        color: #d4d4d4;
    }
    
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    
    QCheckBox::indicator:unchecked {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-radius: 3px;
    }
    
    QCheckBox::indicator:checked {
        background-color: #007acc;
        border: 1px solid #1c97ea;
        border-radius: 3px;
        image: url(':/icons/check.svg');
    }
    
    QCheckBox::indicator:indeterminate {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-radius: 3px;
        image: url(':/icons/minus.svg');
    }
    
    /* أزرار الاختيار */
    QRadioButton {
        spacing: 8px;
        color: #d4d4d4;
    }
    
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border-radius: 8px;
        border: 1px solid #555555;
        background-color: #3e3e42;
    }
    
    QRadioButton::indicator:checked {
        background-color: #007acc;
        border: 4px solid #3e3e42;
    }
    
    /* أشرطة التمرير */
    QScrollBar:vertical {
        background-color: #2d2d30;
        width: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #3e3e42;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #46464a;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        background: none;
        height: 0px;
    }
    
    QScrollBar:horizontal {
        background-color: #2d2d30;
        height: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #3e3e42;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #46464a;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        background: none;
        width: 0px;
    }
    
    /* أشرطة التقدم */
    QProgressBar {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-radius: 4px;
        text-align: center;
        color: #d4d4d4;
    }
    
    QProgressBar::chunk {
        background-color: #007acc;
        border-radius: 4px;
    }
    
    /* المجموعات */
    QGroupBox {
        font-weight: bold;
        border: 1px solid #555555;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 12px;
        background-color: #252526;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px 0 8px;
        color: #d4d4d4;
    }
    
    /* الأطر */
    QFrame[frameShape="4"] { /* QFrame::Panel */
        background-color: #252526;
        border: 1px solid #555555;
        border-radius: 4px;
    }
    
    QFrame[frameShape="5"] { /* QFrame::StyledPanel */
        background-color: #2d2d30;
        border: 1px solid #555555;
        border-radius: 4px;
    }
    
    /* أشرطة الأدوات */
    QToolBar {
        background-color: #2d2d30;
        border-bottom: 1px solid #555555;
        spacing: 4px;
        padding: 2px;
    }
    
    QToolBar::separator {
        width: 1px;
        background-color: #555555;
        margin: 0 4px;
    }
    
    /* أشرطة الحالة */
    QStatusBar {
        background-color: #007acc;
        color: white;
        padding: 4px;
    }
    
    QStatusBar::item {
        border: none;
    }
    
    /* علامات التبويب */
    QTabWidget::pane {
        background-color: #2d2d30;
        border: 1px solid #555555;
        border-radius: 4px;
    }
    
    QTabBar::tab {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 6px 12px;
        margin-right: 2px;
        color: #d4d4d4;
    }
    
    QTabBar::tab:selected {
        background-color: #2d2d30;
        border-color: #555555;
        border-bottom-color: #2d2d30;
    }
    
    QTabBar::tab:hover:!selected {
        background-color: #46464a;
    }
    
    /* قوائم العناصر */
    QListWidget, QTreeWidget, QTableWidget {
        background-color: #3e3e42;
        border: 1px solid #555555;
        border-radius: 4px;
        color: #d4d4d4;
        outline: none;
    }
    
    QListWidget::item, QTreeWidget::item, QTableWidget::item {
        padding: 4px;
        border: none;
    }
    
    QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
        background-color: #007acc;
        color: white;
    }
    
    QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected, QTableWidget::item:hover:!selected {
        background-color: #46464a;
    }
    
    /* شريط التمرير في المحرر */
    QScrollArea {
        background-color: transparent;
        border: none;
    }
    
    /* العناوين */
    QLabel {
        color: #d4d4d4;
    }
    
    QLabel[heading="h1"] {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
    }
    
    QLabel[heading="h2"] {
        font-size: 16px;
        font-weight: bold;
        color: #e6e6e6;
    }
    
    QLabel[heading="h3"] {
        font-size: 14px;
        font-weight: bold;
        color: #cccccc;
    }
    
    /* الرسائل */
    QLabel#message_user {
        background-color: #0e639c;
        color: white;
        padding: 8px;
        border-radius: 8px;
        border-bottom-right-radius: 0;
    }
    
    QLabel#message_assistant {
        background-color: #388a34;
        color: white;
        padding: 8px;
        border-radius: 8px;
        border-bottom-left-radius: 0;
    }
    
    /* أشرطة التمرير للمنطقة القابلة للتمرير */
    QScrollBar:vertical {
        background: #2d2d30;
        width: 10px;
        border-radius: 5px;
    }
    
    QScrollBar::handle:vertical {
        background: #555555;
        border-radius: 5px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: #666666;
    }
    
    /* التأثيرات الخاصة */
    .gradient_bg {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #1e1e1e, stop:1 #2d2d30);
    }
    
    .glass_effect {
        background-color: rgba(45, 45, 48, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    """


def light_stylesheet():
    """أنماط السمة الفاتحة"""
    return """
    /* النافذة الرئيسية */
    QMainWindow {
        background-color: #f5f5f5;
    }
    
    QWidget {
        background-color: #ffffff;
        color: #333333;
        font-family: 'Segoe UI', 'Tahoma', 'Geneva', 'Verdana', sans-serif;
        font-size: 12px;
        border: none;
    }
    
    /* الأزرار */
    QPushButton {
        background-color: #f0f0f0;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 6px 12px;
        color: #333333;
        font-weight: normal;
    }
    
    QPushButton:hover {
        background-color: #e8e8e8;
        border-color: #b3b3b3;
    }
    
    QPushButton:pressed {
        background-color: #0078d4;
        border-color: #005a9e;
        color: white;
    }
    
    QPushButton:disabled {
        background-color: #f5f5f5;
        border-color: #e0e0e0;
        color: #a6a6a6;
    }
    
    /* أزرار خاصة */
    QPushButton#primary_button {
        background-color: #0078d4;
        border-color: #005a9e;
        color: white;
        font-weight: bold;
    }
    
    QPushButton#primary_button:hover {
        background-color: #106ebe;
    }
    
    QPushButton#danger_button {
        background-color: #d13438;
        border-color: #a4262c;
        color: white;
    }
    
    /* أزرار الأدوات */
    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 4px;
    }
    
    QToolButton:hover {
        background-color: #f0f0f0;
        border-color: #cccccc;
    }
    
    QToolButton:pressed {
        background-color: #0078d4;
        color: white;
    }
    
    QToolButton:checked {
        background-color: #0078d4;
        border-color: #005a9e;
        color: white;
    }
    
    /* مربعات النص */
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 6px;
        color: #333333;
        selection-background-color: #0078d4;
        selection-color: white;
    }
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #0078d4;
    }
    
    QTextEdit, QPlainTextEdit {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    
    /* القوائم */
    QMenuBar {
        background-color: #f5f5f5;
        color: #333333;
    }
    
    QMenuBar::item {
        padding: 5px 10px;
        background-color: transparent;
    }
    
    QMenuBar::item:selected {
        background-color: #e8e8e8;
    }
    
    QMenu {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        color: #333333;
    }
    
    QMenu::item {
        padding: 6px 30px 6px 20px;
    }
    
    QMenu::item:selected {
        background-color: #0078d4;
        color: white;
    }
    
    QMenu::separator {
        height: 1px;
        background-color: #cccccc;
        margin: 4px 0px;
    }
    
    /* القوائم المنسدلة */
    QComboBox {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 6px;
        color: #333333;
        min-height: 22px;
    }
    
    QComboBox:hover {
        border-color: #b3b3b3;
    }
    
    QComboBox:focus {
        border-color: #0078d4;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #333333;
        width: 0;
        height: 0;
    }
    
    QComboBox QAbstractItemView {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        color: #333333;
        selection-background-color: #0078d4;
        selection-color: white;
    }
    
    /* مربعات الاختيار */
    QCheckBox {
        spacing: 8px;
        color: #333333;
    }
    
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    
    QCheckBox::indicator:unchecked {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 3px;
    }
    
    QCheckBox::indicator:checked {
        background-color: #0078d4;
        border: 1px solid #005a9e;
        border-radius: 3px;
        image: url(':/icons/check.svg');
    }
    
    QCheckBox::indicator:indeterminate {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 3px;
        image: url(':/icons/minus.svg');
    }
    
    /* أزرار الاختيار */
    QRadioButton {
        spacing: 8px;
        color: #333333;
    }
    
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border-radius: 8px;
        border: 1px solid #cccccc;
        background-color: #ffffff;
    }
    
    QRadioButton::indicator:checked {
        background-color: #0078d4;
        border: 4px solid #ffffff;
    }
    
    /* أشرطة التمرير */
    QScrollBar:vertical {
        background-color: #f5f5f5;
        width: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #cccccc;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #b3b3b3;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        background: none;
        height: 0px;
    }
    
    QScrollBar:horizontal {
        background-color: #f5f5f5;
        height: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #cccccc;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #b3b3b3;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        background: none;
        width: 0px;
    }
    
    /* أشرطة التقدم */
    QProgressBar {
        background-color: #f0f0f0;
        border: 1px solid #cccccc;
        border-radius: 4px;
        text-align: center;
        color: #333333;
    }
    
    QProgressBar::chunk {
        background-color: #0078d4;
        border-radius: 4px;
    }
    
    /* المجموعات */
    QGroupBox {
        font-weight: bold;
        border: 1px solid #cccccc;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 12px;
        background-color: #f9f9f9;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px 0 8px;
        color: #333333;
    }
    
    /* الأطر */
    QFrame[frameShape="4"] { /* QFrame::Panel */
        background-color: #f9f9f9;
        border: 1px solid #cccccc;
        border-radius: 4px;
    }
    
    QFrame[frameShape="5"] { /* QFrame::StyledPanel */
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
    }
    
    /* أشرطة الأدوات */
    QToolBar {
        background-color: #f5f5f5;
        border-bottom: 1px solid #cccccc;
        spacing: 4px;
        padding: 2px;
    }
    
    QToolBar::separator {
        width: 1px;
        background-color: #cccccc;
        margin: 0 4px;
    }
    
    /* أشرطة الحالة */
    QStatusBar {
        background-color: #0078d4;
        color: white;
        padding: 4px;
    }
    
    QStatusBar::item {
        border: none;
    }
    
    /* علامات التبويب */
    QTabWidget::pane {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
    }
    
    QTabBar::tab {
        background-color: #f0f0f0;
        border: 1px solid #cccccc;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 6px 12px;
        margin-right: 2px;
        color: #333333;
    }
    
    QTabBar::tab:selected {
        background-color: #ffffff;
        border-color: #cccccc;
        border-bottom-color: #ffffff;
    }
    
    QTabBar::tab:hover:!selected {
        background-color: #e8e8e8;
    }
    
    /* قوائم العناصر */
    QListWidget, QTreeWidget, QTableWidget {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
        color: #333333;
        outline: none;
    }
    
    QListWidget::item, QTreeWidget::item, QTableWidget::item {
        padding: 4px;
        border: none;
    }
    
    QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
        background-color: #0078d4;
        color: white;
    }
    
    QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected, QTableWidget::item:hover:!selected {
        background-color: #f0f0f0;
    }
    
    /* شريط التمرير في المحرر */
    QScrollArea {
        background-color: transparent;
        border: none;
    }
    
    /* العناوين */
    QLabel {
        color: #333333;
    }
    
    QLabel[heading="h1"] {
        font-size: 18px;
        font-weight: bold;
        color: #000000;
    }
    
    QLabel[heading="h2"] {
        font-size: 16px;
        font-weight: bold;
        color: #1a1a1a;
    }
    
    QLabel[heading="h3"] {
        font-size: 14px;
        font-weight: bold;
        color: #333333;
    }
    
    /* الرسائل */
    QLabel#message_user {
        background-color: #0078d4;
        color: white;
        padding: 8px;
        border-radius: 8px;
        border-bottom-right-radius: 0;
    }
    
    QLabel#message_assistant {
        background-color: #107c10;
        color: white;
        padding: 8px;
        border-radius: 8px;
        border-bottom-left-radius: 0;
    }
    
    /* أشرطة التمرير للمنطقة القابلة للتمرير */
    QScrollBar:vertical {
        background: #f5f5f5;
        width: 10px;
        border-radius: 5px;
    }
    
    QScrollBar::handle:vertical {
        background: #cccccc;
        border-radius: 5px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: #b3b3b3;
    }
    
    /* التأثيرات الخاصة */
    .gradient_bg {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #f5f5f5, stop:1 #ffffff);
    }
    
    .glass_effect {
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    """


def blue_stylesheet():
    """أنماط السمة الزرقاء"""
    return """
    /* النافذة الرئيسية */
    QMainWindow {
        background-color: #e3f2fd;
    }
    
    QWidget {
        background-color: #f5fbff;
        color: #1565c0;
        font-family: 'Segoe UI', 'Tahoma', 'Geneva', 'Verdana', sans-serif;
        font-size: 12px;
        border: none;
    }
    
    /* الأزرار */
    QPushButton {
        background-color: #bbdefb;
        border: 1px solid #64b5f6;
        border-radius: 4px;
        padding: 6px 12px;
        color: #0d47a1;
        font-weight: normal;
    }
    
    QPushButton:hover {
        background-color: #90caf9;
        border-color: #42a5f5;
    }
    
    QPushButton:pressed {
        background-color: #1976d2;
        border-color: #1565c0;
        color: white;
    }
    
    QPushButton:disabled {
        background-color: #e3f2fd;
        border-color: #bbdefb;
        color: #64b5f6;
    }
    
    /* أزرار خاصة */
    QPushButton#primary_button {
        background-color: #1976d2;
        border-color: #1565c0;
        color: white;
        font-weight: bold;
    }
    
    QPushButton#primary_button:hover {
        background-color: #1565c0;
    }
    
    QPushButton#danger_button {
        background-color: #d32f2f;
        border-color: #c62828;
        color: white;
    }
    
    /* أزرار الأدوات */
    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 4px;
    }
    
    QToolButton:hover {
        background-color: #bbdefb;
        border-color: #64b5f6;
    }
    
    QToolButton:pressed {
        background-color: #1976d2;
        color: white;
    }
    
    QToolButton:checked {
        background-color: #1976d2;
        border-color: #1565c0;
        color: white;
    }
    
    /* مربعات النص */
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        border-radius: 4px;
        padding: 6px;
        color: #1565c0;
        selection-background-color: #1976d2;
        selection-color: white;
    }
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #1976d2;
    }
    
    QTextEdit, QPlainTextEdit {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    
    /* القوائم */
    QMenuBar {
        background-color: #bbdefb;
        color: #0d47a1;
    }
    
    QMenuBar::item {
        padding: 5px 10px;
        background-color: transparent;
    }
    
    QMenuBar::item:selected {
        background-color: #90caf9;
    }
    
    QMenu {
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        color: #1565c0;
    }
    
    QMenu::item {
        padding: 6px 30px 6px 20px;
    }
    
    QMenu::item:selected {
        background-color: #1976d2;
        color: white;
    }
    
    QMenu::separator {
        height: 1px;
        background-color: #bbdefb;
        margin: 4px 0px;
    }
    
    /* القوائم المنسدلة */
    QComboBox {
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        border-radius: 4px;
        padding: 6px;
        color: #1565c0;
        min-height: 22px;
    }
    
    QComboBox:hover {
        border-color: #90caf9;
    }
    
    QComboBox:focus {
        border-color: #1976d2;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #1565c0;
        width: 0;
        height: 0;
    }
    
    QComboBox QAbstractItemView {
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        color: #1565c0;
        selection-background-color: #1976d2;
        selection-color: white;
    }
    
    /* مربعات الاختيار */
    QCheckBox {
        spacing: 8px;
        color: #1565c0;
    }
    
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    
    QCheckBox::indicator:unchecked {
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        border-radius: 3px;
    }
    
    QCheckBox::indicator:checked {
        background-color: #1976d2;
        border: 1px solid #1565c0;
        border-radius: 3px;
        image: url(':/icons/check.svg');
    }
    
    QCheckBox::indicator:indeterminate {
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        border-radius: 3px;
        image: url(':/icons/minus.svg');
    }
    
    /* أزرار الاختيار */
    QRadioButton {
        spacing: 8px;
        color: #1565c0;
    }
    
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border-radius: 8px;
        border: 1px solid #bbdefb;
        background-color: #ffffff;
    }
    
    QRadioButton::indicator:checked {
        background-color: #1976d2;
        border: 4px solid #ffffff;
    }
    
    /* أشرطة التمرير */
    QScrollBar:vertical {
        background-color: #e3f2fd;
        width: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #bbdefb;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #90caf9;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        background: none;
        height: 0px;
    }
    
    QScrollBar:horizontal {
        background-color: #e3f2fd;
        height: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #bbdefb;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #90caf9;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        background: none;
        width: 0px;
    }
    
    /* أشرطة التقدم */
    QProgressBar {
        background-color: #bbdefb;
        border: 1px solid #64b5f6;
        border-radius: 4px;
        text-align: center;
        color: #0d47a1;
    }
    
    QProgressBar::chunk {
        background-color: #1976d2;
        border-radius: 4px;
    }
    
    /* المجموعات */
    QGroupBox {
        font-weight: bold;
        border: 1px solid #bbdefb;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 12px;
        background-color: #e3f2fd;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px 0 8px;
        color: #1565c0;
    }
    
    /* الأطر */
    QFrame[frameShape="4"] { /* QFrame::Panel */
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 4px;
    }
    
    QFrame[frameShape="5"] { /* QFrame::StyledPanel */
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        border-radius: 4px;
    }
    
    /* أشرطة الأدوات */
    QToolBar {
        background-color: #bbdefb;
        border-bottom: 1px solid #64b5f6;
        spacing: 4px;
        padding: 2px;
    }
    
    QToolBar::separator {
        width: 1px;
        background-color: #64b5f6;
        margin: 0 4px;
    }
    
    /* أشرطة الحالة */
    QStatusBar {
        background-color: #1976d2;
        color: white;
        padding: 4px;
    }
    
    QStatusBar::item {
        border: none;
    }
    
    /* علامات التبويب */
    QTabWidget::pane {
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        border-radius: 4px;
    }
    
    QTabBar::tab {
        background-color: #bbdefb;
        border: 1px solid #64b5f6;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 6px 12px;
        margin-right: 2px;
        color: #0d47a1;
    }
    
    QTabBar::tab:selected {
        background-color: #ffffff;
        border-color: #bbdefb;
        border-bottom-color: #ffffff;
    }
    
    QTabBar::tab:hover:!selected {
        background-color: #e3f2fd;
    }
    
    /* قوائم العناصر */
    QListWidget, QTreeWidget, QTableWidget {
        background-color: #ffffff;
        border: 1px solid #bbdefb;
        border-radius: 4px;
        color: #1565c0;
        outline: none;
    }
    
    QListWidget::item, QTreeWidget::item, QTableWidget::item {
        padding: 4px;
        border: none;
    }
    
    QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
        background-color: #1976d2;
        color: white;
    }
    
    QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected, QTableWidget::item:hover:!selected {
        background-color: #e3f2fd;
    }
    
    /* شريط التمرير في المحرر */
    QScrollArea {
        background-color: transparent;
        border: none;
    }
    
    /* العناوين */
    QLabel {
        color: #1565c0;
    }
    
    QLabel[heading="h1"] {
        font-size: 18px;
        font-weight: bold;
        color: #0d47a1;
    }
    
    QLabel[heading="h2"] {
        font-size: 16px;
        font-weight: bold;
        color: #1565c0;
    }
    
    QLabel[heading="h3"] {
        font-size: 14px;
        font-weight: bold;
        color: #1976d2;
    }
    
    /* الرسائل */
    QLabel#message_user {
        background-color: #1976d2;
        color: white;
        padding: 8px;
        border-radius: 8px;
        border-bottom-right-radius: 0;
    }
    
    QLabel#message_assistant {
        background-color: #388e3c;
        color: white;
        padding: 8px;
        border-radius: 8px;
        border-bottom-left-radius: 0;
    }
    
    /* أشرطة التمرير للمنطقة القابلة للتمرير */
    QScrollBar:vertical {
        background: #e3f2fd;
        width: 10px;
        border-radius: 5px;
    }
    
    QScrollBar::handle:vertical {
        background: #bbdefb;
        border-radius: 5px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: #90caf9;
    }
    
    /* التأثيرات الخاصة */
    .gradient_bg {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #e3f2fd, stop:1 #ffffff);
    }
    
    .glass_effect {
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(187, 222, 251, 0.5);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    """


def apply_stylesheet(widget):
    """
    تطبيق الأنماط على عنصر واجهة
    
    Args:
        widget: عنصر واجهة المستخدم
    """
    # تعيين الخط الافتراضي
    font = QFont()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    widget.setFont(font)
    
    # تعيين سياسات الحجم
    widget.setSizePolicy(widget.sizePolicy())


def get_color_palette(theme="dark"):
    """
    الحصول على لوحة الألوان للسمة
    
    Args:
        theme: السمة ("dark", "light", "blue")
    
    Returns:
        QPalette
    """
    palette = QPalette()
    
    if theme == "dark":
        # السمة الداكنة
        palette.setColor(QPalette.Window, QColor(45, 45, 48))
        palette.setColor(QPalette.WindowText, QColor(212, 212, 212))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
        palette.setColor(QPalette.ToolTipBase, QColor(212, 212, 212))
        palette.setColor(QPalette.ToolTipText, QColor(212, 212, 212))
        palette.setColor(QPalette.Text, QColor(212, 212, 212))
        palette.setColor(QPalette.Button, QColor(45, 45, 48))
        palette.setColor(QPalette.ButtonText, QColor(212, 212, 212))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.white)
    
    elif theme == "light":
        # السمة الفاتحة
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(233, 231, 227))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.white)
    
    elif theme == "blue":
        # السمة الزرقاء
        palette.setColor(QPalette.Window, QColor(227, 242, 253))
        palette.setColor(QPalette.WindowText, QColor(21, 101, 192))
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(240, 248, 255))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, QColor(21, 101, 192))
        palette.setColor(QPalette.Text, QColor(21, 101, 192))
        palette.setColor(QPalette.Button, QColor(187, 222, 251))
        palette.setColor(QPalette.ButtonText, QColor(21, 101, 192))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(25, 118, 210))
        palette.setColor(QPalette.Highlight, QColor(25, 118, 210))
        palette.setColor(QPalette.HighlightedText, Qt.white)
    
    return palette


def get_icon_path(icon_name):
    """
    الحصول على مسار الأيقونة
    
    Args:
        icon_name: اسم الأيقونة
    
    Returns:
        مسار الأيقونة
    """
    # مجلد الأيقونات
    icon_dir = Path(__file__).parent / "assets" / "icons"
    
    # الأيقونات المدمجة
    builtin_icons = {
        "logo": "logo.png",
        "send": "send_icon.png",
        "user": "user_icon.png",
        "bot": "bot_icon.png",
        "settings": "settings_icon.png",
        "new_chat": "new_chat_icon.png",
        "attach": "attach.svg",
        "image": "image.svg",
        "format": "format.svg",
        "mic": "mic.svg",
        "stop": "stop.svg",
        "copy": "copy.svg",
        "export": "export.svg",
        "refresh": "refresh.svg",
        "more": "more.svg",
        "tune": "tune.svg",
        "clear": "clear.svg",
        "check": "check.svg",
        "minus": "minus.svg"
    }
    
    if icon_name in builtin_icons:
        icon_file = builtin_icons[icon_name]
        icon_path = icon_dir / icon_file
        
        if icon_path.exists():
            return str(icon_path)
    
    # إذا لم توجد الأيقونة، إنشاء أيقونة بسيطة
    return create_fallback_icon(icon_name)


def create_fallback_icon(icon_name):
    """
    إنشاء أيقونة بديلة إذا لم توجد الأيقونة
    
    Args:
        icon_name: اسم الأيقونة
    
    Returns:
        أيقونة QIcon
    """
    # إنشاء أيقونة بسيطة بناءً على الاسم
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # اختيار اللون بناءً على نوع الأيقونة
    if "send" in icon_name:
        color = QColor(76, 175, 80)
        # رسم سهم
        painter.setBrush(color)
        painter.drawEllipse(4, 4, 24, 24)
        painter.setPen(Qt.white)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "▶")
    
    elif "settings" in icon_name:
        color = QColor(33, 150, 243)
        # رسم ترس
        painter.setBrush(color)
        painter.drawEllipse(4, 4, 24, 24)
        painter.setPen(Qt.white)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "⚙")
    
    elif "user" in icon_name:
        color = QColor(244, 67, 54)
        # رسم وجه
        painter.setBrush(color)
        painter.drawEllipse(4, 4, 24, 24)
        painter.setPen(Qt.white)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "👤")
    
    elif "bot" in icon_name:
        color = QColor(103, 58, 183)
        # رسم روبوت
        painter.setBrush(color)
        painter.drawEllipse(4, 4, 24, 24)
        painter.setPen(Qt.white)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "🤖")
    
    else:
        # أيقونة افتراضية
        color = QColor(96, 125, 139)
        painter.setBrush(color)
        painter.drawEllipse(4, 4, 24, 24)
        painter.setPen(Qt.white)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "?" if len(icon_name) == 0 else icon_name[0].upper())
    
    painter.end()
    
    icon = QIcon(pixmap)
    return icon


def get_font(font_name="", size=0):
    """
    الحصول على الخط
    
    Args:
        font_name: اسم الخط
        size: حجم الخط
    
    Returns:
        QFont
    """
    font = QFont()
    
    if font_name:
        font.setFamily(font_name)
    else:
        # خطوط افتراضية (بدعم العربية)
        arabic_fonts = [
            "Segoe UI",
            "Tahoma",
            "Arial",
            "Microsoft Sans Serif",
            "DejaVu Sans"
        ]
        
        for f in arabic_fonts:
            if QFont(f).exactMatch():
                font.setFamily(f)
                break
    
    if size > 0:
        font.setPointSize(size)
    
    return font


def create_gradient(color1, color2, horizontal=True):
    """
    إنشاء تدرج لوني
    
    Args:
        color1: اللون الأول
        color2: اللون الثاني
        horizontal: اتجاه أفقي
    
    Returns:
        سلسلة CSS للتدرج
    """
    if isinstance(color1, str):
        color1 = QColor(color1)
    if isinstance(color2, str):
        color2 = QColor(color2)
    
    if horizontal:
        gradient = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color1.name()}, stop:1 {color2.name()})"
    else:
        gradient = f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {color1.name()}, stop:1 {color2.name()})"
    
    return gradient


def apply_animation_style(widget):
    """
    تطبيق أنماط التحريك على عنصر
    
    Args:
        widget: عنصر واجهة المستخدم
    """
    # يمكن إضافة أنماط للتحريك هنا
    widget.setStyleSheet(widget.styleSheet() + """
        QPushButton {
            transition: background-color 0.2s, border-color 0.2s;
        }
        
        QPushButton:hover {
            transition: background-color 0.1s, border-color 0.1s;
        }
    """)


if __name__ == "__main__":
    # اختبار الأنماط
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            layout = QVBoxLayout(central_widget)
            
            # زر لتبديل السمات
            self.theme_button = QPushButton("تبديل السمة")
            self.theme_button.clicked.connect(self.toggle_theme)
            layout.addWidget(self.theme_button)
            
            # عناصر اختبار
            test_button = QPushButton("زر اختبار")
            layout.addWidget(test_button)
            
            test_label = QLabel("نص اختبار")
            layout.addWidget(test_label)
            
            # تطبيق السمة الداكنة أولاً
            self.current_theme = "dark"
            self.apply_theme()
            
            self.setWindowTitle("اختبار الأنماط")
            self.resize(400, 300)
        
        def toggle_theme(self):
            """تبديل السمة"""
            if self.current_theme == "dark":
                self.current_theme = "light"
            elif self.current_theme == "light":
                self.current_theme = "blue"
            else:
                self.current_theme = "dark"
            
            self.apply_theme()
        
        def apply_theme(self):
            """تطبيق السمة الحالية"""
            stylesheet = get_stylesheet(self.current_theme)
            self.setStyleSheet(stylesheet)
            
            # تطبيق لوحة الألوان
            palette = get_color_palette(self.current_theme)
            self.setPalette(palette)
            
            self.theme_button.setText(f"السمة: {self.current_theme}")
    
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())