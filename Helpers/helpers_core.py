
import os
from PIL import Image

from keras.models import Model
from keras.layers import Input, Dense, Activation, Concatenate, \
							Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.regularizers import l1, l2, l1_l2

import matplotlib.pyplot as plt
from matplotlib import cm


#%% PLOTS

#%%

def pcaPlot(X3,col,title=''):
    '''
    To be cleaned
    
    '''
    f = plt.figure(figsize=(18,6))
    
    ax = f.add_subplot(1,3,1)
    ax.scatter(X3[:,0],X3[:,1],color=col)
    ax.set_xlabel(0, fontsize = 15)
    ax.set_ylabel(1, fontsize = 15)
    ax.set_title(title)
    
    ax = f.add_subplot(1,3,2)
    ax.scatter(X3[:,0],X3[:,2],color=col)
    ax.set_xlabel(0, fontsize = 15)
    ax.set_ylabel(2, fontsize = 15)
#     ax.set_xlim(-4,8)
#     ax.set_ylim(-2,2)

    ax = f.add_subplot(1,3,3)
    ax.scatter(X3[:,1],X3[:,2],color=col)
    ax.set_xlabel(1, fontsize = 15)
    ax.set_ylabel(2, fontsize = 15)
#     ax.set_xlim(-4,8)
#     ax.set_ylim(-2,2)

    plt.show()


#%% DATA HANDLERS

#%%

class OneClassPerDirImageHandler:
    
    def __init__(self, pathToDataDir):
        '''
        Inputs
        ------
        `pathToDataDir` : str
            path to the directory containing one directory per class
            and all images in each one of these subsirectories.
        '''

        self.pathToDataDir = pathToDataDir
                
        for _p in (self.pathToDataDir,):
            if not os.path.isdir(_p):
                raise ValueError('directory does not exist: {}'.format(_p))
        
        self.classDict = {}
        self.classDictInv = {}
        self.nbPerClass = {}
        self.filesPerClass = {}
        i = -1
        for root,folders,files in os.walk(self.pathToDataDir): # NB: this is walked in random order
            if folders == []:
                i += 1
                folder = os.path.split(root)[-1]
                self.classDict[i] = folder
                self.classDictInv[folder] = i
                self.nbPerClass[i] = len(files)
                self.filesPerClass[i] = files


    def getImage(self, classId, numId):
        '''
        returns the `numId` image for class `classId`.

        Inputs
        ------
        `classId` : int or str
            if int, index of the class as in classDict;
            if str, name of the self.classDictInv
        `numId` : int, list of int, or 'all'
            if int, index of the image in its class. Should be lower than self.nbPerClass[classId].
            if 'all', will return all images for that class.
        '''

        if isinstance(classId, int):
            classIdx_ = classId
            className_ = self.classDict[classId]
        elif isinstance(classId, str):
            classIdx_ = self.classDictInv[classId]
            className_ = classId
        else:
            raise TypeError('unkown type for classId: {}'.format(type(classId)))

        nbInClass = self.nbPerClass[classIdx_] 

        if isinstance(numId, int):
            numId_ = [numId]
        elif isinstance(numId, list):
            numId_ = numId
        elif 'all' == numId:
            numId_ = [i for i in range(nbInClass)]
        else:
            raise ValueError('numId not understood: {}'.format(numId))

        ls_im = []
        for n in numId_:
            if n > nbInClass:
                raise ValueError('numId ({}) larger than number of elements ({}) in class {}'.format(n,nbInClass,className_))
            path = os.path.join(self.pathToDataDir, className_, self.filesPerClass[classIdx_][n])
            im = Image.open(path)

            ls_im.append(im)

        if 1 == len(ls_im): # compatibility with numId == int
            ls_im = ls_im[0]

        return(ls_im)
     
#%%

#%% MODELS

#%%


def scaleConvNet1_image(edge, nbClass, nScale, filters):
    '''
    a convolutional network for square images, acting as a one layer filter
    for different scales in parralel, rather than acting in a cascade way.

    Inputs
    ------
    `edge` : int
        edge of the square images.
    `nbClass` : int
        number of outputs in the final layer.
    `nbScale` : int
		number of scales to be treated. Lower scales are obtained
		by subsampling using an average pooling layer.
	`filters` : int
		number of filters to apply at each scale (the `filters`
		parameters for layers like `Conv2D`).
    '''
    
    # NB: quantities to make input someday...
    kernel_size = 2**4  # medium-sized coverage
    strides = 2       # should be enough to keep ~ translation invariance (no interval left uncovered)
    l1Reg = 0.00      # to be tested...
    # multDense = 1     # used for the final MLP    

    #--- design   
    In = Input((edge,edge,1))
    Scale = []
    
    L = In
    for i in range(nScale):
        # convolution part (may use more fancy filters later)
        Conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      activation='relu',
                      # activation='tanh',
                      use_bias=False)(L)
#         Conv = AveragePooling2D(pool_size=int(Conv.shape[1]))(Conv)
        Conv = MaxPooling2D(pool_size=int(Conv.shape[1]))(Conv)
        Conv = Flatten()(Conv)
        Scale.append(Conv)
        # subsampling part
        L = AveragePooling2D(pool_size=2)(L)
    
    Out = Concatenate(axis=-1)(Scale) # note: does not work if len(Out)==1
    
    # Out = Dense(multDense*(nScale*filters),activation='relu')(Out)
    # Out = Dense(multDense*(nScale*filters),activation='relu')(Out)
    Out = Dense(nbClass,
                activation='softmax',
                kernel_regularizer=l1(l1Reg))(Out)
    
    model = Model(inputs=[In],outputs=[Out])
            
    #--- compilation
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', # IAH: change the loss later if possible
                  metrics=['categorical_accuracy'])

    #--- return
    model.summary()
    return model    
                