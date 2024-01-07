# --------------------------------------------------------
# Calibration stage
# Written by Jiaheng Wang
# --------------------------------------------------------
import time
from PyQt5.QtCore import QThread, QMutex, pyqtSignal
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
from ..cfg import *
import pylsl
import random
import threading
import winsound


class Player(pg.GraphicsView):
    def __init__(self, parent=None):
        pg.setConfigOptions(background='d')
        super(Player, self).__init__(parent)

        self.view = pg.ViewBox()
        self.setCentralWidget(self.view)

        self.view.enableAutoRange(x=False, y=False)
        self.view.setMouseEnabled(False, False)
        geometry = QApplication.primaryScreen().geometry()
        self.xRange = [-(geometry.width() // 2), geometry.width() // 2]
        self.yRange = [-(geometry.height() // 2), geometry.height() // 2]
        self.view.setXRange(*self.xRange, padding=0)
        self.view.setYRange(*self.yRange, padding=0)
        self.setFixedSize(geometry.width(), geometry.height())

        self.initMetaData()

    def initMetaData(self):
        self.state = mState.STATE_NULL
        self.cross_parms = [400, 250]   # horizontal, vertical
        self.arrow_parms = {'tipAngle': 60, 'baseAngle': 45, 'headLen': 40, 'tailLen': 100, 'tailWidth': 5}

        self.markerInfo = pylsl.StreamInfo('MI_Markers', 'Markers', 1, channel_format=pylsl.cf_int32)
        self.markerOutlet = pylsl.StreamOutlet(self.markerInfo)

    def process(self, state: mState = mState.STATE_NULL, **kwargs):
        self.state = state
        self.view.clear()

        if state == mState.STATE_CROSS_BEEP:
            t = threading.Thread(target=lambda :winsound.Beep(500, 500))
            t.start()
            self.draw_cross()
        elif state == mState.STATE_CROSS:
            self.draw_cross()
        elif state == mState.STATE_INDICATOR:
            self.draw_cross()
            if kwargs['TYPE'] == MIType.TYPE_LEFT:
                self.draw_arrow([-(self.cross_parms[0] // 2) - 10, 0], 0)
                self.markerOutlet.push_sample([MIType.TYPE_LEFT])
            elif kwargs['TYPE'] == MIType.TYPE_RIGHT:
                self.draw_arrow([self.cross_parms[0] // 2 + 10, 0], 180)
                self.markerOutlet.push_sample([MIType.TYPE_RIGHT])
            elif kwargs['TYPE'] == MIType.TYPE_UP:
                self.draw_arrow([0, self.cross_parms[1] // 2 + 80], 90)
                self.markerOutlet.push_sample([MIType.TYPE_UP])
            elif kwargs['TYPE'] == MIType.TYPE_DOWN:
                self.draw_arrow([0, -(self.cross_parms[1] // 2) - 80], -90)
                self.markerOutlet.push_sample([MIType.TYPE_DOWN])
            elif kwargs['TYPE'] == MIType.TYPE_REST:
                self.markerOutlet.push_sample([MIType.TYPE_REST])
        elif state == mState.STATE_RELAX:
            pass
        elif state == mState.STATE_NULL:
            self.markerOutlet.push_sample([mState.STATE_NULL])

    def draw_cross(self):
        lx, ly = self.cross_parms
        startX, endX = -(lx // 2), lx // 2
        hLine = pg.PlotCurveItem(x=[startX, endX], y=[0, 0], pen=pg.mkPen((255, 255, 255), width=2))
        startY, endY = -(ly // 2), ly // 2
        vLine = pg.PlotCurveItem(x=[0, 0], y=[startY, endY], pen=pg.mkPen((255, 255, 255), width=2))
        self.view.addItem(hLine)
        self.view.addItem(vLine)

    def draw_bar(self, pos, color = 'c'): #  r, g, b, c, m, y, k, w
        bar = pg.BarGraphItem(x=[pos[0]], y=[pos[1]], width=[self.bar_parms[0]], height=[self.bar_parms[1]], pen=color, brush=color)
        self.view.addItem(bar)

    def draw_arrow(self, pos, angle):
        arrow = pg.ArrowItem(angle=angle, **self.arrow_parms, pen=None, brush='r')
        arrow.setPos(*pos)
        self.view.addItem(arrow)

    def draw_image(self, array, TYPE):
        image = pg.ImageItem(array)
        pos_x, pos_y = self.image_parms
        if TYPE == MIType.TYPE_LEFT:
            pos_x *= -1
            pos_y *= -1
        image.setPos(pos_x - array.shape[0] // 2, pos_y - array.shape[1] // 2)
        self.view.addItem(image)

    def draw_text(self, text, x, y):
        from PyQt5.QtGui import QFont
        text = pg.TextItem(
            html=f'<div style="text-align: center">'
                 f'<span style="font-size: 100pt;">{text}</span></div>',
            anchor=(0.5, 0.5),)
        text.setColor((255, 0, 0))
        text.setPos(x, y)
        font = QFont('Times', weight=QFont.Bold)
        text.setFont(font)
        self.view.addItem(text)


class Stimulator(QThread):
    stimulation = pyqtSignal(dict)

    def __init__(self, parent = None):
        super(Stimulator, self).__init__(parent)
        self.mutex = QMutex()
        self.event = threading.Event()

    def run(self):
        time.sleep(1)

        for run in range(RUN_NUM):
            self.waiting(1)
            print('experiment start, new run!')
            self.stimulation.emit({'state': mState.STATE_NULL})
            self.waiting(1)
            self.sleep(3)

            # trials = sum(([TYPE] * (TRIAL_NUM // len(TARGET)) for TYPE in MIType if TYPE in TARGET), start=[])
            # trials = np.random.choice(trials, len(trials), replace=False)
            TYPES = list(TARGET)
            trials = []
            for i in range(TRIAL_NUM // len(TARGET)):
                random.shuffle(TYPES)
                trials += TYPES
            for idx, trial in enumerate(trials):
                print(f'run {run}, trial {idx}, type {[k for k, v in MIType.__dict__.items() if v is trial]}')

                self.stimulation.emit({'state':mState.STATE_CROSS_BEEP})
                self.waiting(mStateDur.STATE_CROSS_BEEP_DUR, )

                self.stimulation.emit({'state':mState.STATE_CROSS})
                self.waiting(mStateDur.STATE_CROSS_DUR, )

                self.stimulation.emit({'state':mState.STATE_INDICATOR, 'TYPE':trial})
                self.waiting(mStateDur.STATE_INDICATOR_DUR,)

                self.stimulation.emit({'state':mState.STATE_RELAX})
                self.waiting(mStateDur.STATE_RELAX_DUR + random.random())

    def waiting(self, duration):
        self.event.clear()
        self.event.wait(duration)
        # if the mutex is locked beforehand by the pause action, it will wait until resuming the program.
        self.mutex.lock()
        self.mutex.unlock()