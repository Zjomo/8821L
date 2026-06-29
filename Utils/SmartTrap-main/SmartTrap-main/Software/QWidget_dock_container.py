from PyQt6.QtWidgets import QDockWidget

class QWidgetWindowDocker(QDockWidget):
    """
    A helper class which makes widgets "dockable" so they can be docked in the main user interface
    window.
    """
    def __init__(self, Qwidget, Title="Widget container"):
        super().__init__(Title)
        self.widget = Qwidget
        self.setWidget(self.widget)  # Embed the original widget inside the QDockWidget


    def closeEvent(self, event):
        # Instead of closing, hide the dock widget
        event.ignore()
        self.hide()