import wx
import cv2

"""Class responsible for user graphic interface GUI.
 It processes the captured frame of webcam
 and  triggers the video capturing every given time"""
class BaseWindow(wx.Frame):
    def __init__(self, captured, title='Face expression recognition', parent=None):

        self.captured = captured

        success, frame = self.takeFrame()
        if not success:
            print 'Could not take frame from Camera'
            raise SystemExit

        # video parameters
        self.height, self.width, x = frame.shape
        # take frame from video buffer as bitmap
        self.bmp = wx.BitmapFromBuffer(self.width, self.height, frame)

        # init application window
        wx.Frame.__init__(self, parent, title=title, size=(self.width, self.height))

        # set timer to grab picture at every 100ms (10 fps)
        self.playTimer = wx.Timer(self)
        self.playTimer.Start(1000 / 10)
        self.Bind(wx.EVT_TIMER, self.onNextFrame)

        # create panel for camera stream
        self.panelcamera = wx.Panel(self, size=(self.width, self.height))
        self.panelcamera.Bind(wx.EVT_PAINT, self.onPaint)

        # create vertical panel and add horizontal panels to it
        self.verticalSizer = wx.BoxSizer(wx.VERTICAL)
        self.verticalSizer.Add(self.panelcamera, 1, flag=wx.EXPAND | wx.ALIGN_CENTER)
        self.verticalSizer.Fit(self)

        # Layout sizer
        self.SetSizer(self.verticalSizer)
        self.SetMinSize((self.width, self.height))
        self.SetMaxSize((self.width+10, self.height+10))
        self.Center()

    # function triggered by system timer. It takes a current frame, process it and displays in the window
    def onNextFrame(self, evt):
        success, frame = self.takeFrame()
        if success:
            frame = self.processFrame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.bmp.CopyFromBuffer(frame)
            self.Refresh(eraseBackground=False)

    def onPaint(self, evt):
        if self.bmp:
            dc = wx.BufferedPaintDC(self.panelcamera)
            dc.DrawBitmap(self.bmp, 0, 0)

    # take a frame
    def takeFrame(self):
        return self.captured.read()

    # process frame
    def processFrame(self, frame_RGB):
        pass

