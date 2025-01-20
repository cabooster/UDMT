# shared_state.py
from PySide6.QtCore import QObject, Signal

class SharedState(QObject):
    test_spin_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self._test_spin = 0.8

    def set_test_spin(self, value):
        """设置新的值，并发出信号"""
        if self._test_spin != value:
            self._test_spin = value
            self.test_spin_changed.emit(value)

    def get_test_spin(self):
        """获取当前值"""
        return self._test_spin



shared_state = SharedState()
