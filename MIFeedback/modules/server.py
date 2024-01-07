# --------------------------------------------------------
# Feedback stage
# Written by Jiaheng Wang
# --------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath('../../'))
os.chdir(os.path.abspath('../../'))
from PyQt5.QtNetwork import QTcpServer, QTcpSocket, QHostAddress
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QTimer
from MIFeedback.modules.decoder import Decoder
from MIFeedback.cfg import *
from utils.pylsl_inlet import DataInlet
from utils.tools import Depack, Pack
import pylsl
from functools import partial


class Server(QMainWindow):
    def __init__(self):
        super(Server, self).__init__()

        self.tcpServer = QTcpServer(self)
        self.clientConnection = QTcpSocket()
        self.tcpServer.listen(QHostAddress.Any, PORT)
        assert self.tcpServer.isListening()

        self.streamInfo = [info for info in pylsl.resolve_streams() if info.type() == 'EEG' and info.name() == 'g.tec62'][0]
        self.depack = Depack()
        self.pack = Pack()
        self.dataInlet = DataInlet(self.streamInfo)
        self.decoder = Decoder(self.dataInlet)
        self.feedback_timer = QTimer(self)

        self.initMetaData()
        self.initSingalSlot()

    def initMetaData(self):
        self.lsl_name = self.streamInfo.name()
        self.lsl_cn = self.streamInfo.channel_count()
        self.lsl_fs = self.streamInfo.nominal_srate()
        self.feedback_timer.setInterval(mStateDur.STATE_FEEDBACK_STEP)

    def initSingalSlot(self):
        self.tcpServer.newConnection.connect(self.newConnect)

    def newConnect(self):
        self.clientConnection = self.tcpServer.nextPendingConnection()
        assert self.clientConnection.isOpen()
        self.clientConnection.readyRead.connect(self.readMessage)
        self.clientConnection.disconnected.connect(self.delConnect)

        self.dataInlet.inlet = pylsl.StreamInlet(self.streamInfo)
        self.dataInlet.start()
        print('The client has been connected!')

    def delConnect(self):
        self.pause()
        self.clientConnection.deleteLater()
        self.dataInlet.stop()
        MODEL_QUEUE.clear()
        for i in range(RUN_NUM // len(MODEL_SET)):
            random.shuffle(MODEL_SET)
            MODEL_QUEUE.extend(MODEL_SET)
        print('The client has been disconnected!')

    def readMessage(self):
        if n := self.clientConnection.bytesAvailable():
            package = self.clientConnection.read(n)
            messages = self.depack.depack(package)
            for message in messages:
                self.process(message)

    def process(self, message: dict):
        if message['state'] == mState.STATE_FEEDBACK:
            self.feedback_timer.timeout.connect(partial(self.on_feedback_timer_timeout, message))
            self.feedback_timer.start()
        elif message['state'] == mState.STATE_RESULT:
            self.pause()
        elif message['state'] == mState.STATE_NULL:
            if MODEL_QUEUE:
                model = MODEL_QUEUE.pop(0)
                self.decoder.setModel(MODEL_PATH[model])
                message['model'] = model
            else:
                print('End of experiment!')
            self.clientConnection.write(self.pack.pack(message))
        elif message['state'] == mState.STATE_PAUSE:
            self.pause()

    def on_feedback_timer_timeout(self, message: dict):
        pred = self.decoder.process()
        message['pred'] = pred
        self.clientConnection.write(self.pack.pack(message))

    def pause(self):
        if self.feedback_timer.isActive():
            self.feedback_timer.stop()
            self.feedback_timer.timeout.disconnect()

    def closeEvent(self, a0) -> None:
        self.dataInlet.stop()
        self.tcpServer.close()
        a0.accept()


if __name__ == '__main__':
    app = QApplication([])
    server = Server()
    server.show()
    #server.hide()
    sys.exit(app.exec_())