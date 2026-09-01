 # -*- coding: utf-8 -*-
"""
PYATOMS ATOM SIMULATOR
Created on Mon Nov 15 14:45:06 2021
@author: Asari
"""

import sys
import time

import numpy as numpy

from pathlib import Path

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

# PyQT imports for creating widgets/etc
from PyQt5.QtWidgets import * #QApplication, QSplashScreen, QProgressBar, QWidget, QLabel, QPushButton, QSpinBox, QMenu, QComboBox, QMainWindow, QHBoxLayout, QVBoxLayout, QSlider, QGroupBox, QGridLayout, QRadioButton, QDialog, QLineEdit, QInputDialog
from PyQt5.QtCore import * #Qt
from PyQt5.QtGui import * #QPainter, QColor, QPixmap, QFont


# THESE TWO ARE FOR EMBEDDING MATPLOTLIB PLOTS INTO PYQT5 GUIs
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# FigureCanvasQTAgg : It will provide the canvas for the figure
# NavigationToolbar2QT : It will provide the tool bar for the matplotlib figs (zooming in, panning, saving image, etc)
# https://www.geeksforgeeks.org/how-to-embed-matplotlib-graph-in-pyqt5/




# IMPORT OTHER FILES:
from .PyAtoms_Widgets import SimulatorWidget
from .hexatoms import hexatoms
from .squareatoms import squareatoms
from .moirelattice import moirelattice


# main window
# which inherits QMainWindow
class Window(QMainWindow):
       
    # constructor - always include this
    def __init__(self, parent=None):
        super(Window, self).__init__(parent) # always need to do this because our class Window is inheriting from parent QDialog....
        self.initUI() # Call the function that initializes everything
        
        

    def initUI(self):
        # Define default instance variables 
        # Use 'self.' when defining basically any thing in the class - these are class member variables (?) they can be accessed in all functions throughout the class

        # Call functions to initialize everything else
        self.setWindowTitle('PyAtoms v. 1.0') # Sets the title on the external window that pops up when you run the code

        icon_path = Path(__file__).resolve().parent / "pyatoms.ico"
        self.setWindowIcon(QIcon(str(icon_path))) # Sets the icon on the external window

        self.initGeo() # Sets size of the popup gui window
        self.initWidgetsGrid() # For placing multiple widgets in the popup gui in a grid layout
        self.show()



    ## Tried to change the numbers here to change the size of the window but it doesnt work?
    def initGeo(self): 
        # Set geometry of popup gui window
        self.setGeometry(100, 100, 1250,800)#self.width(),self.height())#1200, 850)
        # self.setStyleSheet("background: gray;") # Change color of background in window
        # self.setStyleSheet("color:red") # Change color of allll the displayed text in the gui
        # self.setStyleSheet("color: magenta;background: gray")
        # self.setStyleSheet("border: 1px solid black;")

      
        # Hard-coded these numbers just by running the code and seeing how it looked when I changed the numbers
        # These numbers set the size of the window that pops up
        self.x = self.width() // 3 + 20 
        self.y = 40
        self.w = (2*self.width()) // 3 - 100
        self.h = self.height() - 100
        self.SimWidget = SimulatorWidget(self, self.x,self.y,self.w,self.h) # Define an instance of the SimulatorWidget class, which is being imported thus can be accessed in this file
        # self.showMaximized() # To open the window fully maximized https://www.geeksforgeeks.org/pyqt5-how-to-open-window-in-maximized-format/


    # Create a layout to place all the widgets/groupboxes in a grid layout
    # This is basically the function that calls all the other init_something functions in the child class to place them as widgets in the gui grid
    def initWidgetsGrid(self):
        # SYNTAX for adding widget to gui:
        #(addWidget(QWidget, int r, int c, int rowspan, int columnspan)) ---  Adds widget at specified row and column and having specified width and/or height
        #grid.addWidget(row, column, width, height)

        # build each existing widget once so the layout can place it in the appropriate section
        moireModelWidget = self.SimWidget.initMoireBtn()
        outputWidget = self.SimWidget.initOutputTabs()
        imageParametersWidget = self.SimWidget.initImageParameters()
        filteringWidget = self.SimWidget.initFiltering()
        timeEstimatorWidget = self.SimWidget.initTimeEstimatorTabs()
        moireCalculatorWidget = self.SimWidget.initMoireCalcWidget()
        moireCalculatorWidget.setMinimumWidth(350)

        # build the three lattice panels before creating the plot
        lattice1Widget = self.SimWidget.initLattice1Parameters()
        lattice2Widget = self.SimWidget.initLattice2Parameters()
        lattice3Widget = self.SimWidget.initLattice3Parameters()

        # create matplotlib last because initmatplotlibfig() immediately calls plotatoms which controls the initialized things above
        plotWidget = self.SimWidget.initMatplotlibFig()

        # keep the lattice panels consistent without forcing the whole window to be tall enough to display all three at once
        latticeHeight = 280
        lattice1Widget.setFixedHeight(latticeHeight)
        lattice2Widget.setFixedHeight(latticeHeight)
        lattice3Widget.setFixedHeight(latticeHeight)

        # left side vertically scrollable lattice controls
        self.latticeControlsContainer = QWidget()
        latticeLayout = QVBoxLayout(self.latticeControlsContainer)
        latticeLayout.setContentsMargins(4, 4, 4, 4)
        latticeLayout.setSpacing(11)
        latticeLayout.addWidget(lattice1Widget)
        latticeLayout.addWidget(lattice2Widget)
        latticeLayout.addWidget(lattice3Widget)
        latticeLayout.addStretch(1)

        # preserve the controls' natural size -> when the viewport is smaller, QScrollArea will show
        # scrollbars instead of squashing all the controls
        self.latticeControlsContainer.adjustSize()
        self.latticeControlsContainer.setMinimumSize(self.latticeControlsContainer.sizeHint())

        self.latticeScrollArea = QScrollArea()
        self.latticeScrollArea.setWidgetResizable(True)
        self.latticeScrollArea.setWidget(self.latticeControlsContainer)
        self.latticeScrollArea.setFrameShape(QFrame.NoFrame)
        self.latticeScrollArea.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.latticeScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.latticeScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.latticeScrollArea.setMinimumWidth(280)

        # top right section -> all global controls in one area that can scroll in both directions
        # when the window is too small
        self.topControlsContainer = QWidget()
        topGrid = QGridLayout(self.topControlsContainer)
        topGrid.setContentsMargins(4, 4, 4, 4)
        topGrid.setHorizontalSpacing(11)
        topGrid.setVerticalSpacing(11)

        topGrid.addWidget(moireModelWidget, 0, 0)
        topGrid.addWidget(outputWidget, 1, 0)
        topGrid.addWidget(imageParametersWidget, 0, 1, 2, 1)
        topGrid.addWidget(filteringWidget, 0, 2)
        topGrid.addWidget(timeEstimatorWidget, 1, 2)
        topGrid.addWidget(moireCalculatorWidget, 0, 3, 2, 1)

        # any extra room goes to blank space instead of stretching the control boxes
        topGrid.setColumnStretch(4, 1)
        topGrid.setRowStretch(2, 1)

        self.topControlsContainer.adjustSize()
        self.topControlsContainer.setMinimumSize(self.topControlsContainer.sizeHint())

        self.topControlsScrollArea = QScrollArea()
        self.topControlsScrollArea.setWidgetResizable(True)
        self.topControlsScrollArea.setWidget(self.topControlsContainer)
        self.topControlsScrollArea.setFrameShape(QFrame.NoFrame)
        self.topControlsScrollArea.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.topControlsScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.topControlsScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.topControlsScrollArea.setMinimumWidth(160)

        # dont let the top section grow taller than the height required to show all of its controls
        # any additional window height should go to matplotlib instead!
        horizontalBarHeight = self.topControlsScrollArea.horizontalScrollBar().sizeHint().height()
        self.topControlsPreferredHeight = self.topControlsContainer.sizeHint().height() + horizontalBarHeight + 16
        self.topControlsScrollArea.setMaximumHeight(self.topControlsPreferredHeight)

        # matplotlib stays outside both scroll areas so its toolbar, zooming, etc are unaffected
        plotWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plotWidget.setMinimumSize(0, 220)
        self.SimWidget.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.SimWidget.canvas.setMinimumSize(0, 0)

        # right side -> draggable divider between the scrollable controls and teh expanding matplotlib figure
        self.rightSplitter = QSplitter(Qt.Vertical)
        self.rightSplitter.setChildrenCollapsible(False)
        self.rightSplitter.setHandleWidth(6)
        self.rightSplitter.addWidget(self.topControlsScrollArea)
        self.rightSplitter.addWidget(plotWidget)

        # both sections share added height when the wndow is enlarged (until the controls reach their max height)
        self.rightSplitter.setStretchFactor(0, 2)
        self.rightSplitter.setStretchFactor(1, 3)

        # entire window has a draggable divider between lattice controls and the right side
        self.mainSplitter = QSplitter(Qt.Horizontal)
        self.mainSplitter.setChildrenCollapsible(False)
        self.mainSplitter.setHandleWidth(6)
        self.mainSplitter.addWidget(self.latticeScrollArea)
        self.mainSplitter.addWidget(self.rightSplitter)
        self.mainSplitter.setStretchFactor(0, 0)
        self.mainSplitter.setStretchFactor(1, 1)

        centralWidget = QWidget()
        centralLayout = QHBoxLayout(centralWidget)
        centralLayout.setContentsMargins(4, 4, 4, 4)
        centralLayout.addWidget(self.mainSplitter)
        self.setCentralWidget(centralWidget)

        # wait until Qt knows the windows real on-screen dimensions before choosing the starting splitter positions
        QTimer.singleShot(0, self.setInitialSplitterSizes)

    def setInitialSplitterSizes(self):
        # give the lattice column enough width for its natural layout BUT dont let it
        # consume a big fraction of a small screen
        totalWidth = max(self.mainSplitter.width(), 1)
        latticeHint = self.latticeControlsContainer.sizeHint().width()
        scrollBarWidth = self.latticeScrollArea.verticalScrollBar().sizeHint().width()
        preferredLatticeWidth = latticeHint + scrollBarWidth + 8
        maximumLatticeWidth = max(280, int(totalWidth * 0.32))
        latticeWidth = max(280, min(preferredLatticeWidth, maximumLatticeWidth))
        self.mainSplitter.setSizes([latticeWidth, max(totalWidth - latticeWidth, 1)])

        # show the full top controls when the screen is tall enough. on a shorter display
        # limit them to about 35 % of the right workspace so the most height goes to the plots
        totalHeight = max(self.rightSplitter.height(), 1)
        maximumControlsHeight = max(200, int(totalHeight * 0.40))
        controlsHeight = max(180, min(self.topControlsPreferredHeight, maximumControlsHeight))
        self.rightSplitter.setSizes([controlsHeight, max(totalHeight - controlsHeight, 1)])

        



    # Overriding keyPressEvent so that if the escape button is pressed, it doesn't automatically close the program
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # print("Esc clicked")
            self.esc_error = QMessageBox()
            self.esc_error.setWindowTitle("Warning!")
            self.esc_error.setText("Escape button pressed. Do you want to exit the program?")
            self.esc_error.setIcon(QMessageBox.Warning)
            self.esc_error.setStandardButtons(QMessageBox.Cancel)
            self.esc_error.addButton(QPushButton("Exit program"), QMessageBox.NoRole) # Add a custom button, from https://stackoverflow.com/questions/15682665/how-to-add-custom-button-to-a-qmessagebox-in-pyqt4
            x = self.esc_error.exec()
            # print((x))
            if x == 0: # if x =0, this means the Exit program button was clicked (found out by just printing the value of x for each button)
                sys.exit(app.exec_()) # Exit program if the exit button was clicked




# Handle high resolution displays:  https://stackoverflow.com/questions/39247342/pyqt-gui-size-on-high-resolution-screens
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
# To run, go to the file path location in terminal and type  'python PyAtoms_GUI.py'
# driver code
# if __name__ == '__main__':  # this won't be run when imported... https://stackoverflow.com/questions/6523791/why-is-python-running-my-module-when-i-import-it-and-how-do-i-stop-it
                            # # code here will only run when you invoke 'python main.py'

def main():
    # creating apyqt5 application
    app = QApplication(sys.argv)

    # Create splash screen (loading screen) from https://gist.github.com/345161974/8897f9230006d51803c987122b3d4f17
    # splash_pix = QPixmap("HH.png")
    # splash_pix = QPixmap("logo_magma_Small.png")

    logo_path = Path(__file__).resolve().parent / "logo_magma_Small.png"
    splash_pix = QPixmap(str(logo_path))
    
    # splash_pix.scaledToHeight(240, Qt.SmoothTransformation)
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint) # WindowStaysOnTopHint: to keep it above all the other windows on the desktop.
    # splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint) # I commented this out and it fixed the issues of it reappearing after clicking away and back from hte gui
    splash.setEnabled(False)

    fontsize=40
    font = splash.font()
    font.setPixelSize(fontsize)
    font.setWeight(QFont.Bold)
    splash.setFont(font)

    # adding progress bar
    progressBar = QProgressBar(splash)
    progressBar.setMaximum(10)
    progressBar.setGeometry(0, splash_pix.height() - 50, splash_pix.width(), 20)

    splash.show()
    splash.showMessage("<h1><font color='white'>Loading...</font></h1>",Qt.AlignBottom, Qt.black)
    

    for i in range(1, 12):
        progressBar.setValue(i)
        t = time.time()
        while time.time() < t + 0.1:
           app.processEvents()

    # Simulate something that takes time
    time.sleep(1)

    # creating a window object
    main = Window()
       
    # showing the window
    #    main.show()

    splash.finish(main) # Remove the splash when the_editor has finished setting itself up.

    splash.close()

    # loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

    
    




































