import numpy as np # Import numpy for mathematical operations and functions
import sys # Required for starting and exiting application
import copy # Need the deepcopy function to copy entire arrays
from PIL import Image # Required to read jpeg images
import cv2
import PIL
from matplotlib import pyplot as plt

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage, qRgb
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtCore, QtGui, QtWidgets

################################# GLOBAL VARIABLES #################################
imagename = ''
# ker_img_name = ''
# blur_img_name = ''
# org_img_name = ''

ker_img_mat = np.zeros(1)
blur_img_mat = np.zeros(1)
deblur_img_mat = np.zeros(1)
org_img_mat = np.zeros(1)

k_val = 0.2
rad_val = 50
gam_val = 0.01

##################################################DEFINING THE CLASS MAIN WINDOW##################################################

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        ################################# MAIN WINDOW #################################
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(682, 383)
        MainWindow.setMinimumSize(QtCore.QSize(682, 383))
        
        ################################# CENTRAL WIDGET #################################
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        
        ################################# NN  #################################
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout_4.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        
        #################################  NN #################################
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        
        ################################# KERNEL IMAGE  #################################
        self.kernel_img = QtWidgets.QLabel(self.centralWidget)
        self.kernel_img.setObjectName("kernel_img")
        self.gridLayout.addWidget(self.kernel_img, 0, 0, 1, 1)
        
        ################################# BLURREDD IMAGE  #################################
        self.blur_img = QtWidgets.QLabel(self.centralWidget)
        self.blur_img.setObjectName("blur_img")
        self.gridLayout.addWidget(self.blur_img, 0, 1, 1, 1)
    
        ################################# DEBLURRED IMAGE  #################################
        self.deblur_img = QtWidgets.QLabel(self.centralWidget)
        self.deblur_img.setObjectName("deblur_img")
        self.gridLayout.addWidget(self.deblur_img, 0, 2, 1, 1)
        
        #################################  ORIGINAL IMAGE #################################
        self.original_img = QtWidgets.QLabel(self.centralWidget)
        self.original_img.setObjectName("original_img")
        self.gridLayout.addWidget(self.original_img, 0, 3, 1, 1)
        
        ################################# KERNEL LOAD BUTTON  #################################
        self.load_kernel = QtWidgets.QPushButton(self.centralWidget)
        self.load_kernel.setObjectName("load_kernel")
        self.gridLayout.addWidget(self.load_kernel, 1, 0, 1, 1)
        self.load_kernel.clicked.connect(self.ker_button)                #Connecting the function to the button click
        
        ################################# BLURRED LOAD BUTTON  #################################
        self.load_blur = QtWidgets.QPushButton(self.centralWidget)
        self.load_blur.setObjectName("load_blur")
        self.gridLayout.addWidget(self.load_blur, 1, 1, 1, 1)
        self.load_blur.clicked.connect(self.blurload_button)
        
        ################################# SAVE BUTTON  #################################
        self.save_deblur = QtWidgets.QPushButton(self.centralWidget)
        self.save_deblur.setObjectName("save_deblur")
        self.gridLayout.addWidget(self.save_deblur, 1, 2, 1, 1)
        self.save_deblur.clicked.connect(self.save_button)
        
        ################################# ORIGINAL LOAD BUTTON  #################################
        self.load_original = QtWidgets.QPushButton(self.centralWidget)
        self.load_original.setObjectName("load_original")
        self.gridLayout.addWidget(self.load_original, 1, 3, 1, 1)
        self.load_original.clicked.connect(self.org_button)                #Connecting the function to the button click
        
        ################################# BLUR BUTTON  #################################
        self.blur = QtWidgets.QPushButton(self.centralWidget)
        self.blur.setObjectName("blur")
        self.gridLayout.addWidget(self.blur, 2, 0, 1, 1)
        self.blur.clicked.connect(self.blur_button)
        
        ################################# FULL INVERSE FILTER BUTTON  #################################
        self.full_inv = QtWidgets.QPushButton(self.centralWidget)
        self.full_inv.setObjectName("full_inv")
        self.gridLayout.addWidget(self.full_inv, 2, 1, 1, 1)
        self.full_inv.clicked.connect(self.full_inv__button)
        
        ################################# TRUNCATED INVERSE FILTER BUTTON  #################################
        self.trunc_inv = QtWidgets.QPushButton(self.centralWidget)
        self.trunc_inv.setObjectName("trunc_inv")
        self.gridLayout.addWidget(self.trunc_inv, 2, 2, 1, 1)
        self.trunc_inv.clicked.connect(self.trunc_inv_button)
        
        ################################# TRUNCATED INVERSE FILTER SLIDER  #################################
        self.slider_trunc_inv = QtWidgets.QSlider(self.centralWidget)
        self.slider_trunc_inv.setOrientation(QtCore.Qt.Horizontal)
        self.slider_trunc_inv.setObjectName("slider_trunc_inv")
        self.gridLayout.addWidget(self.slider_trunc_inv, 2, 3, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.slider_trunc_inv.setRange(0,99)                                                #Range of the slider
        self.slider_trunc_inv.setValue(rad_val)                                                   #Initialization
        self.slider_trunc_inv.valueChanged.connect(self.changeValue_rad)                       #Calling the function when value is changed

        ################################# WIENER FILTER BUTTON  #################################
        self.wiener = QtWidgets.QPushButton(self.centralWidget)
        self.wiener.setObjectName("wiener")
        self.gridLayout.addWidget(self.wiener, 3, 0, 1, 1)
        self.wiener.clicked.connect(self.wiener_button)
        
        ################################# K VALUE BOX #################################
        self.k_val_box = QtWidgets.QLineEdit(self.centralWidget)
        self.k_val_box.setObjectName("k_val_box")
        self.gridLayout.addWidget(self.k_val_box, 3, 1, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.k_val_box.setText(str(k_val))
        self.k_val_box.setAlignment(Qt.AlignCenter)
        self.k_val_box.show()
        
        ################################# CONSTRAINED LEAST SQUARES FILTER  #################################
        self.clsf = QtWidgets.QPushButton(self.centralWidget)
        self.clsf.setObjectName("clsf")
        self.gridLayout.addWidget(self.clsf, 3, 2, 1, 1)
        self.clsf.clicked.connect(self.clsf_button)
        
        ################################# gam VALUE BOX #################################
        self.gam_val_box = QtWidgets.QLineEdit(self.centralWidget)
        self.gam_val_box.setObjectName("gam_val_box")
        self.gridLayout.addWidget(self.gam_val_box, 3, 3, 1, 1, QtCore.Qt.AlignHCenter)
        
        self.gam_val_box.setText(str(gam_val))
        self.gam_val_box.setAlignment(Qt.AlignCenter)
        self.gam_val_box.show()

        
        ################################# PSNR BUTTON  #################################
        self.psnr = QtWidgets.QPushButton(self.centralWidget)
        self.psnr.setObjectName("psnr")
        self.gridLayout.addWidget(self.psnr, 4, 0, 1, 1)
        self.psnr.clicked.connect(self.psnr_button)
        
        ################################# SSIM BUTTON  #################################
        self.ssim = QtWidgets.QPushButton(self.centralWidget)
        self.ssim.setObjectName("ssim")
        self.gridLayout.addWidget(self.ssim, 4, 2, 1, 1)
        self.ssim.clicked.connect(self.ssim_button)
        



        
        
        
        #################################   #################################
        self.psnr_text = QtWidgets.QLabel(self.centralWidget)
        self.psnr_text.setObjectName("psnr_text")
        self.gridLayout.addWidget(self.psnr_text, 4, 1, 1, 1)
        
        #################################   #################################
        self.ssim_text = QtWidgets.QLabel(self.centralWidget)
        self.ssim_text.setObjectName("ssim_text")
        self.gridLayout.addWidget(self.ssim_text, 4, 3, 1, 1)
        
        #################################   #################################
        self.horizontalLayout_4.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralWidget)
        
        #################################   #################################
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 682, 17))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        
        #################################   #################################
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        
        #################################   #################################
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

     ##########################################################################################################
######################################################### BUTTON FUNCTIONS ################################################
     ##########################################################################################################

    ################################# KERNEL LOAD FUNCTION #################################
    def ker_button(self):
        print("LOAD KERNEL")
        self.openFileNameDialog_ker()       #Calling function to open file

    ################################# BLUR LOAD FUNCTION #################################
    def blurload_button(self):
        print("LOAD BLUR")
        self.openFileNameDialog_blur()       #Calling function to open file

        # self.openFileNameDialog_ker()       #Calling function to open file

    ################################# SAVE FUNCTION #################################
    def save_button(self):
        print("SAVE DEBLURRED")
        self.save_image()               #Calling function to save


    ################################# ORIGINAL LOAD FUNCTION #################################
    def org_button(self):
        print("LOAD ORIGINAL")
        self.openFileNameDialog_org()       #Calling function to open file


    ################################# BLUR FUNCTION #################################
    def blur_button(self):
        global deblur_img_mat
        print("BLUR")

        deblur_img_mat = self.blur_3d()
        self.display(deblur_img_mat)        #Display the image
        self.psnr_button()
        self.ssim_button()
        
    ################################# FULL INVERSE FILTER FUNCTION #################################
    def full_inv__button(self):
        global deblur_img_mat
        print("FULL INVERSE FILTER")
        
        deblur_img_mat = self.deblur_3d()
        self.display(deblur_img_mat)        #Display the image
        self.psnr_button()
        self.ssim_button()
        
    ################################# TRUNCATED INVERSE FILTER FUNCTION #################################
    def trunc_inv_button(self):
        global deblur_img_mat
        print("TRUNCATED INVERSE FILTER")
        
        deblur_img_mat = self.truncated_3d()
        self.display(deblur_img_mat)        #Display the image
        self.psnr_button()
        self.ssim_button()

    ################################# CHANGE RADIUS VALUE #################################
    def changeValue_rad(self,val):
        global rad_val
        rad_val = val          #Updating Blur value

    ################################# WIENER FILTER FUNCTION #################################
    def wiener_button(self):
        global deblur_img_mat,k_val
        print("WIENER FILTER")

        a = self.k_val_box.text()
        k_val=float(a)#Updating Blur value

        deblur_img_mat = self.wiener_filter3d()
        self.display(deblur_img_mat)        #Display the image
        self.psnr_button()
        self.ssim_button()


    ################################# CONSTRAINED LEAST SQUARES FILTER FUNCTION #################################
    def clsf_button(self):
        global deblur_img_mat,gam_val
        
        print("CONSTRAINED LEAST SQUARES FILTER")
        
        a = self.gam_val_box.text()
        gam_val = float(a)          #Updating Blur value

        deblur_img_mat = self.clsf3d()
        self.display(deblur_img_mat)        #Display the image
        self.psnr_button()
        self.ssim_button()


    ################################# PSNR FUNCTION #################################
    def psnr_button(self):
        global ker_img_mat,blur_img_mat,deblur_img_mat,org_img_mat
        print("PSNR")

        image = org_img_mat
        result = deblur_img_mat

        mse = np.mean( (image - result) ** 2 )
        if mse == 0:
            return 100
        ans = 20*np.log10(255/np.sqrt(mse))

        ans = float("{0:.2f}".format(ans))

        self.psnr_text.setText(str(ans))
        self.psnr_text.setAlignment(Qt.AlignCenter)
        self.psnr_text.show()

    ################################# SSIM FUNCTION #################################
    def ssim_button(self):
        global ker_img_mat,blur_img_mat,deblur_img_mat,org_img_mat
        print("SSIM")
        x = org_img_mat
        y = deblur_img_mat

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        sigma_x = np.mean((x-mean_x)**2)
        sigma_y = np.mean((y-mean_y)**2)
        sigma_xy = np.mean((x-mean_x)*(y-mean_y)) 
        k1 = 0.01
        k2 = 0.03
        l =255
        c1 =k1*l
        c2 = k2*l
        ans = ( (2*mean_x*mean_y+c1)*(2*sigma_xy+c2) )/( (mean_x**2+mean_y**2+c1)*(sigma_x+sigma_y+c2) )
      
        ans = float("{0:.2f}".format(ans))

        self.ssim_text.setText(str(ans))
        self.ssim_text.setAlignment(Qt.AlignCenter)
        self.ssim_text.show()


     ##########################################################################################################
######################################################### FUNCTIONS ################################################
     ##########################################################################################################



    ################################# OPEN KERNEL IMAGE #################################
    def openFileNameDialog_ker(self):    
        global ker_img_mat
        # global hsv_img_mat, hsv_modimg_mat, hsv_last_mat
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName()
        if fileName:
            imagename = fileName
            ker_img_mat = cv2.imread(imagename)
            # ker_img_mat = np.int_(ker_img_mat)
            
            # modimg_mat = img
            # img_mat = img
            # last_mat = img
            pixmap = QPixmap(fileName)
            pixmap = pixmap.scaled(155,220)
            self.kernel_img.setPixmap(pixmap)
            self.kernel_img.show()
            
            # hsv_img_mat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # self.update_hsv()
    

    ################################# OPEN BLURRED IMAGE #################################
    def openFileNameDialog_blur(self):    
        global blur_img_mat
        # global hsv_img_mat, hsv_modimg_mat, hsv_last_mat
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName()
        if fileName:
            imagename = fileName
            blur_img_mat = cv2.imread(imagename)
            # blur_img_mat = np.int_(blur_img_mat)
            
            # modimg_mat = img
            # img_mat = img
            # last_mat = img
            pixmap = QPixmap(fileName)
            pixmap = pixmap.scaled(155,220)
            self.blur_img.setPixmap(pixmap)
            self.blur_img.show()
            
            # hsv_img_mat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # self.update_hsv()
   
    ################################# SAVE IMAGE #################################
    def save_image(self): 
        global deblur_img_mat
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName()
        if fileName:
            namee = fileName + '.jpg'
            print(namee)
            cv2.imwrite(namee,deblur_img_mat)


    ################################# OPEN ORIGINAL IMAGE #################################
    def openFileNameDialog_org(self):    
        global org_img_mat
        # global hsv_img_mat, hsv_modimg_mat, hsv_last_mat
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName()
        if fileName:
            imagename = fileName
            org_img_mat = cv2.imread(imagename)
            # org_img_mat = (org_img_mat)
            
            # modimg_mat = img
            # img_mat = img
            # last_mat = img
            pixmap = QPixmap(fileName)
            pixmap = pixmap.scaled(155,220)
            self.original_img.setPixmap(pixmap)
            self.original_img.show()
            
            # hsv_img_mat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # self.update_hsv()
    ################################# BLURRING FUNCTION #################################
    
    def blur_1d(self,image,kernel):
        
        r,c = image.shape
        r1,c1 = kernel.shape
        rmax = max(r,r1)
        cmax = max(c,c1)
        pad_img = np.zeros((2*rmax,2*cmax))  
        pad_ker = np.zeros((2*rmax,2*cmax))
    
        kernel = kernel/np.sum(kernel)
        pad_img[0:r,0:c] = image
        pad_ker[0:r1,0:c1] = kernel

        fft_pad_img = np.fft.fft2(pad_img)
        fft_pad_ker = np.fft.fft2(pad_ker)

        fft_pad_img2 = np.fft.fftshift(fft_pad_img)
        fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)
    
        div = fft_pad_img2*(fft_pad_ker2)
        div2 = np.fft.ifftshift(div)
        div3 = np.fft.ifft2(div2).real

        divided = div3[0:r,0:c]
        # deblur_img_mat = divided
        return divided.astype(int)
    
    def blur_3d(self):
        global ker_img_mat,blur_img_mat,deblur_img_mat,org_img_mat
        
        image = org_img_mat
        kernel = ker_img_mat
        
        out_R = self.blur_1d(image[:,:,0],kernel[:,:,0])
        out_G = self.blur_1d(image[:,:,1],kernel[:,:,1])
        out_B = self.blur_1d(image[:,:,2],kernel[:,:,2])

        output = np.zeros((image.shape))
        output[:,:,0] = out_R
        output[:,:,1] = out_G
        output[:,:,2] = out_B

        # output[output<0] = 0
        # output[output>255] = 255
        # # deblur_img_mat = output
        return output.astype(int)

    ################################# FULL INVERSE FILTER FUNCTION ##################################
    
    def deblur_1d(self,image,kernel):
        r,c = image.shape
        r1,c1 = kernel.shape
        rmax = max(r,r1)
        cmax = max(c,c1)
        
        pad_img = np.zeros((2*rmax,2*cmax))  #dtype = int
        pad_ker = np.zeros((2*rmax,2*cmax))

        kernel = kernel/np.sum(kernel)
        pad_img[0:r,0:c] = image
        pad_ker[0:r1,0:c1] = kernel

        fft_pad_img = np.fft.fft2(pad_img)
        fft_pad_ker = np.fft.fft2(pad_ker)

        fft_pad_img2 = np.fft.fftshift(fft_pad_img)
        fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)
        
        lim = 0.01
        fft_pad_ker2[np.abs(fft_pad_ker2)<lim] = lim

        div = fft_pad_img2/(fft_pad_ker2)
        div2 = np.fft.ifftshift(div)
        div3 = np.fft.ifft2(div2).real

        divided = div3[0:r,0:c]
        
        return divided.astype(int)

    def deblur_3d(self):
        global ker_img_mat,blur_img_mat,deblur_img_mat,org_img_mat
        
        image = blur_img_mat
        kernel = ker_img_mat
        
        out_R = self.deblur_1d(image[:,:,0],kernel[:,:,0])
        out_G = self.deblur_1d(image[:,:,1],kernel[:,:,1])
        out_B = self.deblur_1d(image[:,:,2],kernel[:,:,2])

        output = np.zeros((image.shape))
        output[:,:,0] = out_R
        output[:,:,1] = out_G
        output[:,:,2] = out_B

        return output.astype(int)

    ################################# TRUNCATED INVERSE FILTER FUNCTION ##################################
    def truncated_1d(self,image,kernel):
        global rad_val

        r0 = rad_val
        # print("r =", r0)
        
        r,c = image.shape
        r1,c1 = kernel.shape
        rmax = max(r,r1)
        cmax = max(c,c1)
        x0 = rmax
        y0 = cmax
        rx = (r0/100*x0)
        ry = (r0/100*y0)
        
        pad_img = np.zeros((2*rmax,2*cmax))  #dtype = int
        pad_ker = np.zeros((2*rmax,2*cmax))
        fft_trunc = np.ones((2*rmax,2*cmax),dtype=complex)
        
        kernel = kernel/np.sum(kernel)
        pad_img[0:r,0:c] = image
        pad_ker[0:r1,0:c1] = kernel

        fft_pad_img = np.fft.fft2(pad_img)
        fft_pad_ker = np.fft.fft2(pad_ker)

        fft_pad_img2 = np.fft.fftshift(fft_pad_img)
        fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)

        lim = 0.1
        fft_pad_ker2[np.abs(fft_pad_ker2)<lim] = lim

        fft_trunc[np.int_(x0-rx):np.int_(x0+rx),np.int_(y0-ry):np.int_(y0+ry)] = fft_pad_ker2[np.int_(x0-rx):np.int_(x0+rx),np.int_(y0-ry):np.int_(y0+ry)]
        
        div = fft_pad_img2/(fft_trunc)
        div2 = np.fft.ifftshift(div)
        div3 = np.fft.ifft2(div2).real

        divided = div3[0:r,0:c]
        
        return divided.astype(int)

    def truncated_3d(self):
        
        global ker_img_mat,blur_img_mat,deblur_img_mat,org_img_mat,rad_val

        print("Radius =",rad_val)
        
        image = blur_img_mat
        kernel = ker_img_mat

        out_R = self.truncated_1d(image[:,:,0],kernel[:,:,0])
        out_G = self.truncated_1d(image[:,:,1],kernel[:,:,1])
        out_B = self.truncated_1d(image[:,:,2],kernel[:,:,2])

        output = np.zeros((image.shape))
        output[:,:,0] = out_R
        output[:,:,1] = out_G
        output[:,:,2] = out_B

        return output.astype(int)


    ################################# WIENER FILTER FUNCTION #################################


    def wiener_filter1d(self,image, kernel):
        global k_val
        K1 = k_val
        
        r,c = image.shape
        r1,c1 = kernel.shape
        rmax = max(r,r1)
        cmax = max(c,c1)
        pad_img = np.zeros((2*rmax,2*cmax))  #dtype = int
        pad_ker = np.zeros((2*rmax,2*cmax))

        kernel = kernel/np.sum(kernel)
        pad_img[0:r,0:c] = image
        pad_ker[0:r1,0:c1] = kernel

        fft_pad_img = np.fft.fft2(pad_img)
        fft_pad_ker = np.fft.fft2(pad_ker)

        fft_pad_img2 = np.fft.fftshift(fft_pad_img)
        fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)

        fft_pad_ker3 = (np.abs(fft_pad_ker2)**2) /( (fft_pad_ker2)*((np.abs(fft_pad_ker2)**2) + K1) )
        output = fft_pad_ker3*fft_pad_img2

        out = np.fft.ifftshift(output)
        out2 = np.fft.ifft2(out).real   
        out3 = out2[0:r,0:c]
        return out3.astype(int)

    def wiener_filter3d(self):
        
        global ker_img_mat,blur_img_mat,deblur_img_mat,org_img_mat
        
        image = blur_img_mat
        kernel = ker_img_mat

        out_R = self.wiener_filter1d(image[:,:,0],kernel[:,:,0])
        out_G = self.wiener_filter1d(image[:,:,1],kernel[:,:,1])
        out_B = self.wiener_filter1d(image[:,:,2],kernel[:,:,2])

        output = np.zeros((image.shape))
        output[:,:,0] = out_R
        output[:,:,1] = out_G
        output[:,:,2] = out_B

        return output.astype(int)

    ################################# CONSTRAINED LEAST SQUARES FILTER FUNCTION #################################

    def clsf1d(self,image,kernel):
        global gam_val
        gamma1 = gam_val

        r,c = image.shape
        r1,c1 = kernel.shape
        rmax = max(r,r1)
        cmax = max(c,c1)
        p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        
        pad_img = np.zeros((2*rmax,2*cmax))  #dtype = int
        pad_ker = np.zeros((2*rmax,2*cmax))
        pad_p = np.zeros((2*rmax,2*cmax))
        
        kernel = kernel/np.sum(kernel)
        pad_img[0:r,0:c] = image
        pad_ker[0:r1,0:c1] = kernel
        pad_p[0:3,0:3] = p

        fft_pad_img = np.fft.fft2(pad_img)
        fft_pad_ker = np.fft.fft2(pad_ker)
        fft_pad_p = np.fft.fft2(pad_p)

        fft_pad_img2 = np.fft.fftshift(fft_pad_img)
        fft_pad_ker2 = np.fft.fftshift(fft_pad_ker)
        fft_pad_p2 = np.fft.fftshift(fft_pad_p)

        fft_pad_ker3 = (np.conj(fft_pad_ker2)) /( (np.abs(fft_pad_ker2)**2) + gamma1*( np.abs(fft_pad_p2)**2 ) )
        
        output = fft_pad_ker3*fft_pad_img2

        out = np.fft.ifftshift(output)
        out2 = np.fft.ifft2(out).real
        
        out3 = out2[0:r,0:c]
        return out3.astype(int)

    def clsf3d(self):
        global ker_img_mat,blur_img_mat,deblur_img_mat,org_img_mat
        
        image = blur_img_mat
        kernel = ker_img_mat

        out_R = self.clsf1d(image[:,:,0],kernel[:,:,0])
        out_G = self.clsf1d(image[:,:,1],kernel[:,:,1])
        out_B = self.clsf1d(image[:,:,2],kernel[:,:,2])

        output = np.zeros((image.shape))
        output[:,:,0] = out_R
        output[:,:,1] = out_G
        output[:,:,2] = out_B

        return output

    ################################# DISPLAY IMAGE #################################
    def display(self, im):#Input im is a matrix

        im = np.uint8(im)
        im = np.require(im, np.uint8, 'C')
        gray_color_table = [qRgb(i, i, i) for i in range(256)]

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888).rgbSwapped()
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);

            pixmap = QPixmap(qim)
            pixmap = pixmap.scaled(155,220)
            self.deblur_img.setPixmap(pixmap)
            self.deblur_img.show()
        

    ##########################################################################################################
######################################################### NAMING TEXT BOXES ################################################
    ##########################################################################################################

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Restoration"))
        self.blur_img.setText(_translate("MainWindow", "Blurred Image"))
        self.load_original.setText(_translate("MainWindow", "Load Original Image"))
        self.kernel_img.setText(_translate("MainWindow", "Kernel Image"))
        self.deblur_img.setText(_translate("MainWindow", "Deblurred Image"))
        self.load_kernel.setText(_translate("MainWindow", "Load Kernel Image"))
        self.original_img.setText(_translate("MainWindow", "Original Image"))
        self.load_blur.setText(_translate("MainWindow", "Load Blurred Image"))
        self.blur.setText(_translate("MainWindow", "Blur"))
        self.save_deblur.setText(_translate("MainWindow", "Save Deblurred Image"))
        self.ssim.setText(_translate("MainWindow", "SSIM"))
        self.full_inv.setText(_translate("MainWindow", "Full Inverse Filter"))
        self.trunc_inv.setText(_translate("MainWindow", "Truncated Inverse Filter"))
        self.wiener.setText(_translate("MainWindow", "Wiener Filter"))
        self.clsf.setText(_translate("MainWindow", "Constrained Least Square Filter"))
        self.psnr.setText(_translate("MainWindow", "PSNR"))
        self.psnr_text.setText(_translate("MainWindow", "PSNR Value"))
        self.ssim_text.setText(_translate("MainWindow", "SSIM Value"))

######################################################### MAIN FUNCTION ################################################

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
