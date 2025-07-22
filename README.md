# MI-BCI using deep learning

## Overview

This is a python implementation of motor imagery (MI)-based brain computer interfaces (BCIs). In particular, traditional machine learning as well as deep learning approaches are implemented and investigated in real online settings.  The BCI system consists of two phases including calibration and feedback stages. In the calibration stage, subjects are instructed to perform cued MI tasks without feedback. In the feedback stage,  2D center-out tasks are presented and the cursor is continuously controlled through real-time MI-BCI operation.  The snapshots of the two stages are presented below. An demo video of feedback control is also provided.

The implementation is compact, modularized and easy-to-customized.

<div align=center>
  <img src="calibration snapshot.jpg" alt="Calibration snapshot" width="300"/>
  <img src="feedback snapshot.jpg" alt="Feedback snapshot" width="300"/>
</div>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Calibration snapshot &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Feedback snapshot**

## Dependencies

The system is mainly developed with the PyQt5 (5.15.6) framework along with mainstream data science and EEG-processing packages including numpy (1.22.3), scipy (1.8.0), pytorch (1.13.0), pylsl (1.16.0), mne (1.2.2), pyqtgraph (0.13.0), etc. A python interpreter version of 3.8 is recommended. 

## Usage

------

For a general case, a calibration stage is first performed to record subject-specific MI data.

- [ ] Set up a EEG acquisition procedure and make sure real-time EEG signals are sent by [LSL](https://github.com/labstreaminglayer) outlets. Otherwise, you can run ***./utils/LSL_sender.py*** for a stream simulation.
- [ ] Configure the file ***./MICalibration/cfg.py*** with your own experimental settings.
- [ ] Run the program with ***./MICalibration/modules/client.py***.
- [ ] Record EEG data and Markers using [LabRecoder](https://github.com/labstreaminglayer/App-LabRecorder), and the file is saved in the **.xdf** format.

------

Next,  a structed MI dataset should be made and decoding models are trained.

- [ ] Make a structed dataset using **dataProcess.m**. An example data of one subject is provided for a try.

- [ ] Train subject-specific models with the **algorithms** module. Specifically, FBCSP (benchmark) and IFNet (SOTA) are implemented. It is straightforward to train and just take note of the corresponding configurations.

- [ ] Put pretrained model files into ***./MIFeedback/resources/***. An example resource of one subject is provided.

------

Now, we are ready to conduct online cursor control through MI-BCI operation.

- [ ] Check ***./MIFeedback/cfg.py*** for a necessary modification.
- [ ] Run the file ***./MIFeedback/modules/server.py*** to start a decoding server.
- [ ] Run the file ***./MIFeedback/modules/client.py*** to perform real-time MI-BCI control!

## Cite:
J. Wang, L. Yao and Y. Wang, "Enhanced Online Continuous Brain-Control by Deep Learning-based EEG Decoding," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, doi: 10.1109/TNSRE.2025.3591254.

We are appreciated if you find this work to be beneficial for you.

Welcome to contact with us by Jiaheng-Wang@zju.edu.cn.
