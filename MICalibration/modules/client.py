# --------------------------------------------------------
# Calibration stage
# Written by Jiaheng Wang
# --------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath('../../'))
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QMutex, Qt
from PyQt5.QtGui import QKeyEvent
from MICalibration.cfg import *
from MICalibration.modules.MI import Player, Stimulator


class Client(QMainWindow):
    def __init__(self):
        super(Client, self).__init__()

        self.player = Player(self)
        self.setCentralWidget(self.player)

        self.stimulator = Stimulator(self)

        self.initMetaData()
        self.initSignalSlot()
        self.stimulator.start()

    def initMetaData(self):
        self.pause()

    def initSignalSlot(self):
        self.stimulator.stimulation.connect(self.process)

    def process(self, stimulation: dict):
        state = stimulation['state']
        if state == mState.STATE_INDICATOR:
            self.player.process(state, TYPE=stimulation['TYPE'])
        elif state == mState.STATE_NULL:
            if self.isRunning:
                self.pause()
            self.player.process(state)
        else:
            self.player.process(state)

    def pause(self):
        # be cautious to use this function
        print(f'experiment pause')
        self.isRunning = False
        self.stimulator.mutex.lock()
        self.stimulator.event.set()

    def keyReleaseEvent(self, a0: QKeyEvent) -> None:
        if a0.key() == Qt.Key_Space:
            if self.isRunning:
                self.pause()
            else:
                print(f'experiment resume')
                self.isRunning = not self.isRunning
                self.stimulator.mutex.unlock()
        elif a0.key() == Qt.Key_Escape:
            self.close()
        a0.accept()

    def closeEvent(self, a0) -> None:
        self.stimulator.terminate()
        a0.accept()


if __name__ == '__main__':
    app = QApplication([])
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    client = Client()
    client.show()
    sys.exit(app.exec_())