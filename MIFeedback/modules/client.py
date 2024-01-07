# --------------------------------------------------------
# Feedback stage
# Written by Jiaheng Wang
# --------------------------------------------------------
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QMutex, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtNetwork import QTcpSocket, QHostAddress
from MIFeedback.cfg import *
from MIFeedback.modules.MI import Player, Stimulator, TrajectoryOutlet
from utils.tools import Depack, Pack
import sys


class Client(QMainWindow):
    def __init__(self):
        super(Client, self).__init__()

        self.player = Player(self)
        self.setCentralWidget(self.player)

        self.stimulator = Stimulator(self)
        self.trajectoryOutlet = TrajectoryOutlet(self.player)

        self.tcpClient = QTcpSocket(self)
        self.tcpClient.connectToHost(QHostAddress(IPADDRESS), PORT)
        assert self.tcpClient.waitForConnected()
        print('The server has been connected')

        self.depack = Depack()
        self.pack = Pack()

        self.initMetaData()
        self.initSignalSlot()
        self.stimulator.start()
        self.trajectoryOutlet.start()

    def initMetaData(self):
        self.pause()

    def initSignalSlot(self):
        self.stimulator.stimulation.connect(self.process)
        self.tcpClient.readyRead.connect(self.readMessage)
        self.player.collision.connect(self.stimulator.collide)

    def readMessage(self):
        if n := self.tcpClient.bytesAvailable():
            package = self.tcpClient.read(n)
            messages = self.depack.depack(package)
            for message in messages:
                # deal with network delay
                if message['state'] != self.player.state:
                    return
                self.player.process(**message)

    def process(self, stimulation: dict):
        state = stimulation['state']
        self.player.state = state
        if state == mState.STATE_PROMPT or state == mState.STATE_ACTION:
            self.player.process(state, TYPE=stimulation['TYPE'])
        elif state == mState.STATE_FEEDBACK:
            self.tcpClient.write(self.pack.pack(stimulation))
        elif state == mState.STATE_RESULT:
            self.tcpClient.write(self.pack.pack(stimulation))
            self.player.process(**stimulation)
        elif state == mState.STATE_NULL:
            self.tcpClient.write(self.pack.pack(stimulation))
            if self.isRunning:
                self.pause()
        else:
            self.player.process(**stimulation)

    def pause(self):
        # be cautious to use this function
        print(f'experiment pause')
        self.isRunning = False
        self.tcpClient.write(self.pack.pack({'state': mState.STATE_PAUSE}))
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
        self.trajectoryOutlet.terminate()
        self.tcpClient.disconnect()
        a0.accept()


if __name__ == '__main__':
    app = QApplication([])
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    client = Client()
    client.show()
    sys.exit(app.exec_())