# --------------------------------------------------------
# Feedback stage
# Written by Jiaheng Wang
# --------------------------------------------------------
from PyQt5.QtCore import QThread, QMutex, pyqtSignal, Qt, QRectF
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPen
import pyqtgraph as pg
from ..cfg import *
import time
from collections import defaultdict
import pylsl
import random
import threading
import winsound


class Player(pg.GraphicsView):
    '''
    Set the range of x and y axes as 1920 and 1080, so that the size of elements on the 2D plane is intuitive.
    '''
    collision = pyqtSignal(bool)

    def __init__(self, parent=None):
        pg.setConfigOptions(background='k')
        super(Player, self).__init__(parent)

        self.view = pg.ViewBox()
        self.setCentralWidget(self.view)

        self.view.enableAutoRange(x=False, y=False)
        self.view.setMouseEnabled(False, False)
        geometry = QApplication.primaryScreen().geometry()
        self.geometry = geometry

        self.xRange = [-(geometry.width() // 2), geometry.width() // 2]
        self.yRange = [-(geometry.height() // 2), geometry.height() // 2]
        self.view.setXRange(*self.xRange, padding=0)
        self.view.setYRange(*self.yRange, padding=0)
        self.setFixedSize(geometry.width(), geometry.height())
        self.sceneRect = QRectF(self.xRange[0], self.yRange[0], geometry.width(), geometry.height())

        self.initMetaData()
        self.mutex = QMutex()

    def initMetaData(self):
        self.markerInfo = pylsl.StreamInfo('MI_Markers', 'Markers', 1, channel_format=pylsl.cf_int32)
        self.markerOutlet = pylsl.StreamOutlet(self.markerInfo)

        self.state = None
        self.pstate = None
        self.hit_type = None
        self.result = mState.STATE_RESULT

        self.background = [0, 0, self.geometry.height(), self.geometry.height()]
        self.cursor_size = [60, 60]   # horizontal, vertical
        self.cursor_pos = [0, 0]
        self.xpred = 0
        self.ypred = 0

        self.bars = defaultdict(dict)
        self.hbar_parms = [60, 540] # width, height
        self.bars[MIType.TYPE_LEFT]['pos'] = [self.yRange[0] + self.hbar_parms[0] // 2 + 0, 0]
        self.bars[MIType.TYPE_RIGHT]['pos'] = [self.yRange[1] - self.hbar_parms[0] // 2 - 0, 0]
        self.bars[MIType.TYPE_LEFT]['size'] = self.hbar_parms
        self.bars[MIType.TYPE_RIGHT]['size'] = self.hbar_parms
        self.vbar_parms = [540, 60] # width, height
        self.bars[MIType.TYPE_UP]['pos'] = [0, self.yRange[1] - self.vbar_parms[1] // 2 - 0]
        self.bars[MIType.TYPE_DOWN]['pos'] = [0, self.yRange[0] + self.vbar_parms[1] // 2 + 0]
        self.bars[MIType.TYPE_UP]['size'] = self.vbar_parms
        self.bars[MIType.TYPE_DOWN]['size'] = self.vbar_parms

        self.bars[MIType.TYPE_LEFT]['rect'] = QRectF(self.yRange[0] + 0, -(self.hbar_parms[1] // 2), *self.hbar_parms)
        self.bars[MIType.TYPE_RIGHT]['rect'] = QRectF(self.yRange[1]-self.hbar_parms[0] - 0, -(self.hbar_parms[1] // 2), *self.hbar_parms)
        self.bars[MIType.TYPE_UP]['rect'] = QRectF(-(self.vbar_parms[0]//2), self.yRange[1]-self.vbar_parms[1], *self.vbar_parms)
        self.bars[MIType.TYPE_DOWN]['rect'] = QRectF(-(self.vbar_parms[0] // 2), self.yRange[0] + 0, *self.vbar_parms)

        self.defcol = (100, 100, 100)
        for TYPE in TARGET:
            self.bars[TYPE]['col'] = self.defcol

        self.num_hits = 0
        self.num_errors = 0
        self.num_misses = 0
        self.text_pos = [self.yRange[1] - 80, 480]

    def process(self, state: int = mState.STATE_NULL, **kwargs):
        self.view.clear()
        rect = self.draw_rectangle(*self.background)

        if state == mState.STATE_PROMPT:
            self.bars[kwargs['TYPE']]['col'] = 'b'
            for TYPE in TARGET:
                self.draw_bar(self.bars[TYPE]['pos'], self.bars[TYPE]['size'], self.bars[TYPE]['col'])
        elif state == mState.STATE_ACTION:
            self.draw_ball(self.cursor_pos, self.cursor_size)
            self.bars[kwargs['TYPE']]['col'] = 'b'
            for TYPE in TARGET:
                self.draw_bar(self.bars[TYPE]['pos'], self.bars[TYPE]['size'], self.bars[TYPE]['col'])
        elif state == mState.STATE_FEEDBACK:
            if kwargs['pred']['TYPE'] in [MIType.TYPE_LEFT, MIType.TYPE_RIGHT]:
                v = kwargs['pred']['value'] if kwargs['pred']['TYPE'] is MIType.TYPE_RIGHT else -kwargs['pred']['value']
                self.xpred = momentum * self.xpred + (1 - momentum) * v
                self.ypred = momentum * self.ypred
            elif kwargs['pred']['TYPE'] in [MIType.TYPE_UP, MIType.TYPE_DOWN]:
                v = kwargs['pred']['value'] if kwargs['pred']['TYPE'] is MIType.TYPE_UP else -kwargs['pred']['value']
                self.ypred = momentum * self.ypred + (1 - momentum) * v
                self.xpred = momentum * self.xpred
            else:
                self.ypred = 0 #momentum * self.ypred
                self.xpred = 0 #momentum * self.xpred
            self.mutex.lock()
            self.cursor_pos[0] += (self.xpred + bias)
            self.cursor_pos[1] += (self.ypred + bias)
            self.mutex.unlock()

            ball = self.draw_ball(self.cursor_pos, self.cursor_size)

            self.bars[kwargs['TYPE']]['col'] = 'b'
            for TYPE in TARGET:
                self.draw_bar(self.bars[TYPE]['pos'], self.bars[TYPE]['size'], self.bars[TYPE]['col'])

            x, y, w, h  = ball.boundingRect().getRect()
            cx, cy = x + w/2, y + h/2
            r = (w + h) / 4 * 0.7 # get the actual size of the ball
            ballRect = QRectF(cx-r, cy-r, r*2, r*2)
            for TYPE in TARGET:
                if ballRect.intersects(self.bars[TYPE]['rect']):
                    self.hit_type = TYPE
                    break
            else:
                if not QRectF(rect.boundingRect()).contains(cx, cy):
                    self.hit_type = True
            if self.hit_type is not None:
                self.collision.emit(True)
        elif state == mState.STATE_RESULT:
            for TYPE in TARGET:
                self.bars[TYPE]['col'] = self.defcol

            for TYPE in TARGET:
                if self.hit_type is TYPE:
                    self.bars[TYPE]['col'] = 'g' if kwargs['TYPE'] == TYPE else 'r'
                    break
            else:
                self.bars[kwargs['TYPE']]['col'] = 'y'

            for TYPE in TARGET:
                if self.bars[TYPE]['col'] == 'g':
                    self.num_hits += 1
                    self.result = mState.STATE_RESULT+ 1
                    break
                if self.bars[TYPE]['col'] == 'r':
                    self.num_errors += 1
                    self.result = mState.STATE_RESULT + 2
                    break
                if self.bars[TYPE]['col'] == 'y':
                    self.num_misses += 1
                    self.result = mState.STATE_RESULT + 3
                    break

            for TYPE in TARGET:
                self.draw_bar(self.bars[TYPE]['pos'], self.bars[TYPE]['size'], self.bars[TYPE]['col'])
            self.draw_ball(self.cursor_pos, self.cursor_size)

            self.mutex.lock()
            self.cursor_pos = [0, 0]
            self.mutex.unlock()
            for TYPE in TARGET:
                self.bars[TYPE]['col'] = self.defcol
            self.xpred = 0
            self.ypred = 0
            self.hit_type = None
        elif state == mState.STATE_RELAX:
            # also noted as the preparation period
            t = threading.Thread(target=lambda :winsound.Beep(500, 500))
            t.start()
            for TYPE in TARGET:
                self.draw_bar(self.bars[TYPE]['pos'], self.bars[TYPE]['size'], self.bars[TYPE]['col'])
        elif state == mState.STATE_NULL:
            if self.num_hits or self.num_misses or self.num_errors:
                print(f'Hit / Error / Miss: {self.num_hits} / {self.num_errors} / {self.num_misses}')
            self.num_hits, self.num_errors, self.num_misses = 0, 0, 0

        if state != mState.STATE_NULL:
            text = f'Hit: {self.num_hits}<br/> Error: {self.num_errors}<br/> Miss: {self.num_misses}'
            self.draw_text(text, *self.text_pos)

        if state != self.pstate:
            self.pstate = state
            if state == mState.STATE_PROMPT:
                self.markerOutlet.push_sample([kwargs['TYPE']])
            elif state == mState.STATE_RESULT:
                self.markerOutlet.push_sample([self.result])
            elif state == mState.STATE_NULL:
                if 'model' in kwargs:
                    self.markerOutlet.push_sample([kwargs['model']])
            else:
                self.markerOutlet.push_sample([state])

    def draw_ball(self, pos, size, color = (255, 0, 255)):
        x, y = pos
        lx, ly = size
        ball = pg.ScatterPlotItem(x=[x], y=[y], size= (lx + ly) // 2, pen=pg.mkPen(*color), brush=pg.mkBrush(*color))
        self.view.addItem(ball)
        return ball

    def draw_bar(self, pos, size, color = 'c'): #  color options: r, g, b, c, m, y, k, w
        bar = pg.BarGraphItem(x=[pos[0]], y=[pos[1]], width=[size[0]], height=[size[1]], pen= QPen(Qt.white, 5, Qt.DashLine), brush=color)
        self.view.addItem(bar)

    def draw_rectangle(self, x, y, w, h, color = 'd'):
        rect = pg.BarGraphItem(x=[x], y=[y], width=[w], height=[h], pen=color, brush=color)
        self.view.addItem(rect)
        return rect

    def draw_text(self, text, x, y):
        text = pg.TextItem(
            html=f'<div style="text-align: center">'
                 f'<span style="color: #000000; font-size: 15pt;">{text}</span></div>',
            anchor=(0.5, 0.5))
        text.setPos(x, y)
        self.view.addItem(text)


class TrajectoryOutlet(QThread):
    def __init__(self, player: Player):
        QThread.__init__(self)
        self.fs = int(1000 / mStateDur.STATE_FEEDBACK_STEP)
        self.trajectoryInfo = pylsl.StreamInfo('Trajectory', 'EEG', 2, self.fs, channel_format=pylsl.cf_float32)
        self.trajectoryOutlet = pylsl.StreamOutlet(self.trajectoryInfo)
        self.player = player

    def run(self):
        time.sleep(1)
        self.is_running = True
        print(f'TrajectoryOutlet has been opened.')

        next_time = pylsl.local_clock()
        while self.is_running:
            next_time += 1 / self.fs

            self.player.mutex.lock()
            self.trajectoryOutlet.push_sample([*self.player.cursor_pos])
            self.player.mutex.unlock()

            sleep = max(0, next_time - pylsl.local_clock())
            self.msleep(round(sleep * 1000))


class Stimulator(QThread):
    stimulation = pyqtSignal(dict)

    def __init__(self, parent = None):
        super(Stimulator, self).__init__(parent)
        self.mutex = QMutex()
        self.event = threading.Event()
        self.collision = False

    def run(self):
        time.sleep(1)

        for run in range(RUN_NUM):
            self.waiting(1) # it will be blocked at the inital state for the purpose of economical recording
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

                self.stimulation.emit({'state':mState.STATE_RELAX, 'TYPE':trial})
                self.waiting(mStateDur.STATE_RELAX_DUR)

                self.stimulation.emit({'state':mState.STATE_PROMPT, 'TYPE':trial})
                self.waiting(mStateDur.STATE_PROMPT_DUR)

                self.stimulation.emit({'state':mState.STATE_ACTION, 'TYPE':trial})
                self.waiting(mStateDur.STATE_ACTION_DUR)

                self.stimulation.emit({'state': mState.STATE_FEEDBACK, 'TYPE': trial})
                self.waiting(mStateDur.STATE_FEEDBACK_DUR)

                self.stimulation.emit({'state':mState.STATE_RESULT, 'TYPE':trial})
                self.waiting(mStateDur.STATE_RESULT_DUR)

        self.stimulation.emit({'state': mState.STATE_NULL})

    def waiting(self, duration):
        self.event.clear()
        self.event.wait(duration)
        # if the mutex is locked beforehand by the pause action, it will wait until the program resumes.
        self.mutex.lock()
        self.mutex.unlock()

    def collide(self):
        #self.collision = True
        self.event.set()