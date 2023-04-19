import numpy as np
import xml.etree.ElementTree as ET
import os
import math
from matplotlib import pyplot as plt

def ReadRawFromXML(f:str, verbose=False):
    f_raw = f + '.raw'
    f_xml = f + '.xml'
    f_points = f.replace('_0','AcqPoints').replace('_1','AcqPoints') + '.txt'

    # Read xml configuration file of the raw data
    xml_root = ET.parse(f_xml).getroot()
    for x in xml_root.iter('AP_Samples'):
        NFE = int(x.text)
    for x in xml_root.iter('AP.Encodings'):
        NPE = int(x.text)
    for x in xml_root.iter('AP.Echoes'):
        NEchoes = int(x.text)
    for x in xml_root.iter('DummyScans'):
        DummyScans = int(x.text)
    for x in xml_root.iter('Receivers'):
        NCoil = int(x.text)
    for x in xml_root.iter('Slices'):
        NS = int(x.text)
        #NS = int(NS/ 2)     # FIXME: Don't know why 16 slices from xml, but 8 slices in Paramed Signal Analysis
    
    # Hard coded
    #NPE = 144
    #NS = 42

    # Read permutation file
    with open(f_points) as f_pts:
        lines = f_pts.readlines()
        perm = np.zeros((len(lines)-1, 2), dtype=np.int16)
        print(len(lines))
        c = 0
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n','')
            if (lines[i] != 'AcqPoints'):
                # screen
                p = lines[i].split(' ')
                perm[c,0] = p[0]
                perm[c,1] = p[1]
                c += 1

    # analysis permutation file
    pe1_min, pe2_min = np.amin(perm,axis=0)
    pe1_max, pe2_max = np.amax(perm,axis=0)
    NPE1 = pe1_max - pe1_min + 1
    NPE2 = pe2_max - pe2_min + 1
    
    # File size analysis
    header_size=1056        # Fixed header size
    data_size = (NFE * NPE * NEchoes * NCoil) * 2 * 2   # NS is in NPE, Real/Imaginary, Int16 => 2x2
    print("XML deduced size: " + str(data_size))
    #f_size = 10996512
    #data_size = f_size - header_size

    # Raw file info
    f_raw_stats = os.stat(f_raw)
    print("Raw file size: " + str(f_raw_stats.st_size))
    print("Raw - deducted: " + str(f_raw_stats.st_size - data_size))

    # Read in raw file as signed 16-bit integer
    with open(f_raw, 'rb') as file_obj:
        file_obj.seek(header_size)
        raw = np.frombuffer(file_obj.read(data_size),dtype=np.short)

    # Convert to complex number
    ksp_1d = raw[0::2]+ raw[1::2] * np.complex64(1j)
    ksp_1d = ksp_1d.reshape(NCoil, NPE, NEchoes, NFE)
    # Strip out dummy
    ksp_1d = ksp_1d[:,DummyScans:,:,:]
    # Permute elliptical like k-space to data, according to permutation file
    ksp = np.zeros((NCoil,NPE2,NPE1,NEchoes,NFE), dtype=np.complex64)
    for i in range(perm.shape[0]):
        ksp[:,int(perm[i,1]-pe2_min), int(perm[i,0]-pe1_min),:,:] = ksp_1d[:,i,:,:]
    
    # Show 3D K-Space in a single 2D image
    if (False):
        #NS_ = 1
        #NPE_ = NPE-DummyScans
        ksp_ = ksp.reshape(NCoil, NPE2*NPE1, NEchoes, NFE)
        imshow4(np.squeeze(np.abs(ksp_[0,:4000,0,:])))
        #ksp_proj = np.squeeze(np.abs(ksp[0,0,:,0,:]))
        #plt.plot(ksp_proj)


    # Reorder looping
    ksp = np.transpose(ksp, (4,2,1,3,0))
    # Recon 3D GBASS
    img = np.zeros((NFE, NPE1, NPE2, NEchoes, NCoil), dtype=np.complex64)
    for c in range(NCoil):
            for i in range(NEchoes):
                img[:,:,:,i,c] = np.fft.ifftshift(np.fft.ifftn(np.squeeze(ksp[:,:,:,i,c])))

    # Crop NPE Oversampling
    #img = img[int(NFE/4):int(NFE*3/4),:,:,:,:]
    # Combine 2 Rx
    COMBINE_COIL = False
    if (NCoil==2):
        if COMBINE_COIL:
            img_ = np.sqrt(np.power(img[:,:,:,:,0],2) + np.power(img[:,:,:,:,1],2))
        else:
            img_ = np.squeeze(np.abs(img[:,:,:,:,0]))
    
    img_ = np.squeeze(img_)

    # Visualize    
    if (verbose):
        imshow4( np.squeeze(img_[:,:,:]), outpng=f+'.png')

    return np.squeeze(img)

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

def CISS_Recon(img0, img1):
    ax = len(img0.shape)
    # Combine with magnitude
    img01 = np.stack((np.abs(img0),np.abs(img1)),axis=ax)
    # Take the max
    r = np.amax(img01,axis=ax)
    # Average two channel
    r = np.mean(r, axis=ax-1)
    return r

def CISS(d,f0,f1):
    img0 = ReadRawFromXML(f = d + '\\' + f0)
    img1 = ReadRawFromXML(f = d + f1)
    return CISS_Recon(img0,img1)

f0 = 'GBASS3D_20230320163451_0'
f1 = 'GBASS3D_20230320163634_1'
f2 = 'GBASS3D_20230320183126_0'
f3 = 'GBASS3D_20230320183308_1'
f4 = 'GBASS3D_20230321090721_0'
f5 = 'GBASS3D_20230321090815_1'
f6 = 'GBASS3D_20230321093140_0'
f7 = 'GBASS3D_20230321093338_1'
f8 = 'GBASS3D_20230321104458_0'
f9 = 'GBASS3D_20230321104607_1'
f10 = 'GBASS3D_20230321153628_0'
f11 = 'GBASS3D_20230321153730_1'
f12 = 'GBASS3D_20230321160324_0'
f13 = 'GBASS3D_20230321160406_1'
f14 = 'GBASS3D_20230321165310_0'
f15 = 'GBASS3D_20230321165452_1'
f16 = 'GBASS3D_20230322112146_0'
f17 = 'GBASS3D_20230322112241_1'
f18 = 'GBASS3D_20230322150959_0'
f19 = 'GBASS3D_20230322151054_1'
f20 = 'GBASS3D_20230322155929_0'
f21 = 'GBASS3D_20230322160031_1'
f22 = 'GBASS3D_20230322163451_0'
f23 = 'GBASS3D_20230322163554_1'
f24 = 'GBASS3D_20230322174557_0'
f25 = 'GBASS3D_20230322174742_1'
f26 = 'GBASS3D_20230323093258_0'
f27 = 'GBASS3D_20230323093430_1'
f28 = 'GBASS3D_20230323093820_0'
f29 = 'GBASS3D_20230323093915_1'
f30 = 'GBASS3D_20230323152728_0'
f31 = 'GBASS3D_20230323152822_1'
f32 = 'GBASS3D_20230323165824_0'
f33 = 'GBASS3D_20230323165911_1'
f34 = 'GBASS3D_20230324085516_0'
f35 = 'GBASS3D_20230324085619_1'
f36 = 'GBASS3D_20230324085735_0'
f37 = 'GBASS3D_20230324085838_1'
f38 = 'GBASS3D_20230324101031_0'
f39 = 'GBASS3D_20230324101144_1'
f40 = 'GBASS3D_20230324112647_0'
f41 = 'GBASS3D_20230324112750_1'
f42 = 'GBASS3D_20230324112914_0'
f43 = 'GBASS3D_20230324113023_1'
f44 = 'GBASS3D_20230324174035_0'
f45 = 'GBASS3D_20230324174138_1'
f46 = 'GBASS3D_20230327093354_0'
f47 = 'GBASS3D_20230327093537_1'
f48 = 'GBASS3D_20230327102901_0'
f49 = 'GBASS3D_20230327102952_1'
f50 = 'GBASS3D_20230327103058_0'
f51 = 'GBASS3D_20230327103153_1'

d = r'C:\temp\SEQUENZE GUI E RAW DATA\GBASS TIME MEDICAL' + '\\'
imshow4(CISS(d,f0,f1),outpng=d+f0+'1.png')
imshow4(CISS(d,f2,f3),outpng=d+f2+'1.png')
imshow4(CISS(d,f4,f5),outpng=d+f4+'1.png')
imshow4(CISS(d,f6,f7),outpng=d+f6+'1.png')
imshow4(CISS(d,f8,f9),outpng=d+f8+'1.png')

imshow4(CISS(d,f10,f11),outpng=d+f10+'1.png')
imshow4(CISS(d,f12,f13),outpng=d+f12+'1.png')
imshow4(CISS(d,f14,f15),outpng=d+f14+'1.png')
imshow4(CISS(d,f16,f17),outpng=d+f16+'1.png')
imshow4(CISS(d,f18,f19),outpng=d+f18+'1.png')

imshow4(CISS(d,f20,f21),outpng=d+f20+'1.png')
imshow4(CISS(d,f22,f23),outpng=d+f22+'1.png')
imshow4(CISS(d,f24,f25),outpng=d+f24+'1.png')
imshow4(CISS(d,f26,f27),outpng=d+f26+'1.png')
imshow4(CISS(d,f28,f29),outpng=d+f28+'1.png')

imshow4(CISS(d,f30,f31),outpng=d+f30+'1.png')
imshow4(CISS(d,f32,f33),outpng=d+f32+'1.png')
imshow4(CISS(d,f34,f35),outpng=d+f34+'1.png')
imshow4(CISS(d,f36,f37),outpng=d+f36+'1.png')
imshow4(CISS(d,f38,f39),outpng=d+f38+'1.png')

imshow4(CISS(d,f40,f41),outpng=d+f40+'1.png')
imshow4(CISS(d,f42,f43),outpng=d+f42+'1.png')
imshow4(CISS(d,f44,f45),outpng=d+f44+'1.png')
imshow4(CISS(d,f46,f47),outpng=d+f46+'1.png')
imshow4(CISS(d,f48,f49),outpng=d+f48+'1.png')

imshow4(CISS(d,f50,f51),outpng=d+f50+'1.png')

