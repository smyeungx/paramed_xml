import numpy as np
import xml.etree.ElementTree as ET
import math
from matplotlib import pyplot as plt

def ReadRawFromXML(f:str):
    f_raw = f + '.raw'
    f_xml = f + '.xml'

    # Read xml configuration file of the raw data
    xml_root = ET.parse(f_xml).getroot()
    for x in xml_root.iter('AP_Samples'):
        NFE = int(x.text)
    for x in xml_root.iter('AP.Encodings'):
        NPE = int(x.text)
    for x in xml_root.iter('AP.Echoes'):
        NEchoes = int(x.text)
    for x in xml_root.iter('Receivers'):
        NCoil = int(x.text)
    for x in xml_root.iter('Slices'):
        NS = int(x.text)
        NS = int(NS/ 2)     # FIXME: Don't know why 16 slices from xml, but 8 slices in Paramed Signal Analysis
    
    header_size=1056
    data_size = NFE * NPE * NEchoes * NCoil * NS * 2 * 2
    #f_size = 10996512
    #data_size = f_size - header_size

    # Read in raw file as signed 16-bit integer
    with open(f_raw, 'rb') as file_obj:
        file_obj.seek(header_size)
        raw = np.frombuffer(file_obj.read(data_size),dtype=np.short)

    # Convert to complex number
    ksp = raw[0::2]+ raw[1::2] * np.complex64(1j)
    ksp = ksp.reshape(NCoil, NS, NPE, NEchoes, NFE)

    # Reorder looping
    ksp = np.transpose(ksp, (4,2,3,1,0))
    # Recon GE_STIR_
    img = np.zeros((NFE, NPE, NEchoes, NS, NCoil), dtype=np.complex64)
    for c in range(NCoil):
        for k in range(NS):
            for i in range(NEchoes):
                img[:,:,i,k,c] = np.fft.ifftshift(np.fft.ifft2(np.squeeze(ksp[:,:,i,k,c])))

    # Reorder looping
    #img = np.transpose(img, (4,2,3,1,0))
    # Crop NPE Oversampling
    img = img[int(NFE/4):int(NFE*3/4),:,:,:,:]
    # Combine 2 Rx
    if (NCoil==2):
        img = np.sqrt(np.power(img[:,:,:,:,0],2) + np.power(img[:,:,:,:,1],2))
    
    img = np.squeeze(img)
    img = np.transpose(img, (0,1,3,2))

    # Visualize    
    imshow4( np.squeeze(img[:,:,:,:]), outpng=f+'.png')

def imshow4(v, N_YX=[], block=True, title='', window_title='', min_border=True, outpng=''):
    if (N_YX==[]):
        if (len(v.shape)==2):
            N_YX = [1,1]
            NS = 1
        elif (len(v.shape)==3):
            #N_YX = [1,v.shape[2]]
            # Auto Square
            N = math.ceil(math.sqrt(v.shape[2]))
            N_YX = [N,N]
            N_YX[1] = N-1 if (N*(N-1)==v.shape[2]) else N
            NS = v.shape[2]
        elif (len(v.shape)==4):
            N_YX = [v.shape[3],v.shape[2]]
            NS = v.shape[3] * v.shape[2]
        else:
            # Cannot display image with 1 dimension
            return

        v = np.reshape(v, (v.shape[0], v.shape[1], NS))
    else:
        NS = N_YX[0] * N_YX[1]
        v = np.reshape(v, (v.shape[0], v.shape[1], NS))

    canvas = np.zeros((v.shape[0]*N_YX[0], v.shape[1]*N_YX[1]), dtype=np.complex64)

    for j in range(N_YX[1]):
        for i in range(N_YX[0]):
            os = j * N_YX[0] + i
            if (os < NS):
                canvas[ i*v.shape[0]:(i+1)*v.shape[0], j*v.shape[1]:(j+1)*v.shape[1]] = v[:,:,os]

    if (min_border):
        fig = plt.figure()
        fig.frameon = False
        ax = fig.add_axes((0, 0, 1, 1))
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.box(False)
    plt.imshow(abs(canvas), cmap='gray')
    plt.title(title)
    plt.pause(0.01)
    plt.draw()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(window_title)

    if (outpng!=''):
        plt.savefig(outpng, dpi=600)

    plt.show(block=block)

f0 = 'GE_STIR_20230306155419'
f1 = 'GE_STIR_20230307160331'
f2 = 'GE_STIR_20230307164134'
f3 = 'GE_STIR_20230308124857'
f4 = 'GE_STIR_20230310093024'
f5 = 'GE_STIR_20230310100210'
f6 = 'GE_STIR_20230310103208'

ReadRawFromXML(f = r'C:\temp\GESTIR2 TIME MEDICAL' + '\\' + f1)
