import cv2
import numpy as np
import numpy
from matplotlib import pyplot as plt
import tensorflow
import numpy as np
import numba as nb
import depthai as dai

from tensorflow.keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator()

targ = idg.flow_from_directory("./images",target_size=(200,200))

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Dropout

in_layer = Input(shape=(300,300,3))
o1 = Conv2D(16, 3, activation="relu")(in_layer)
s1 = MaxPooling2D(2)(o1)
o2 = Conv2D(32, 3, activation="relu")(s1)
s2 = MaxPooling2D(2)(o2)
o3 = Conv2D(64, 3, activation="relu")(s2)
s3 = MaxPooling2D(2)(o3)
flat = Flatten()(in_layer)
a2 = Dense(500, activation="relu")(flat)
a3 = Dense(300, activation="relu")(a2)
drop1 = Dropout(.2)(a3)
a4 = Dense(200, activation="relu")(a3)
a5 = Dense(100, activation="relu")(a4)
drop2 = Dropout(.2)(a4)
outLayer = Dense(2, activation="softmax")(a5)

model = Model(in_layer, outLayer)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(targ, epochs=25)

streams = []
streams.append('isp')
@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0
    for i in nb.prange(input.size // 5): 
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift

    return out

print("depthai version:", dai.__version__)
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)

device = dai.Device(pipeline)
device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    q_list.append(q)
    # Make window resizable, and configure initial size
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (960, 540))



def getClassName(index):
    if index == 0:
        return "object found"
    else:
        return "object not found"

font = cv2.FONT_HERSHEY_COMPLEX

org = (50, 50)
img_found = cv2.imread("./images/found/capture_isp_11.png")
res,coords = model.predict(np.array([img_found]))
(fX, fY, fW, fH) = coords
size = "(" + str(fW) + "," + str(fH) + ")"

fontScale = 6
color = (255, 0, 0)  
thickness = 2

def img_alignment(img, img1):
    img, img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 
    img_size = img.shape
    warp_mode = cv2.MOTION_TRANSLATION

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3,dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3,dtype=np.float32)
    
    n_iterations = 6000
    termination_eps = 1e-10

    criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, n_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(img, img1, warp_matrix, warp_mode, criteria )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img1_aligned = cv2.warpPerspective(img1, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img1_aligned = cv2.warpAffine(img1, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return img1_aligned

capt_flag = False
img_count = 0
while True:
    for index,q in enumerate(q_list):
        name = q.getName()
        data = q.get()
        if index+1 < len(q_list):
            name1 = q_list[index+1].getName()
        else:
            name1 = q_list[index].getName()
        
        if index+1 < len(q_list):
            data1 = q_list[index+1].get()
        else:
            data1 = q_list[index].get()
        width, height = data.getWidth(), data.getHeight()
        width1,height1 = data1.getWidth(),data1.getHeight()

        payload = data.getData()
        payload1 = data1.getData()
        capture_file_info_str = ('capture_' + name + '_' + str(width) + 'x' + str(height) + '_' + str(data.getSequenceNum()))
        capture_file_info_str = f"capture_{name}_{img_count}"
        capture_file_info_str1 = f"capture_{name}_{img_count + 1}"
        if name == 'isp':
            shape = (height * 3 // 2, width)
            yuv420p = payload.reshape(shape).astype(np.uint8)
            yuv420p1 = payload1.reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
            bgr1 = cv2.cvtColor(yuv420p1, cv2.COLOR_YUV2BGR_IYUV)
            gray_img =  cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            grays_img2 = cv2.cvtColor(bgr1,cv2.COLOR_BGR2GRAY)
        if capt_flag: 
            filename = capture_file_info_str + '.png'
            print("Saving to file:", filename)
            gray_img = np.ascontiguousarray(gray_img) 
            img1 = np.ascontiguousarray(img1) 
            cv2.imwrite(filename, gray_img)
        bgr = np.ascontiguousarray(bgr) 
        res = model.predict(np.array([cv2.resize(bgr, (200,200))]))
        bgr = cv2.putText(bgr, getClassName(res.argmax(axis=1)) , org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        if getClassName(res.argmax(axis=1))=="object found":
            bgr = cv2.putText(bgr,size,(100,100), font,fontScale, color, thickness, cv2.LINE_AA)

        diff = cv2.absdiff(bgr, bgr1)
        
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        diff_blur = cv2.GaussianBlur(diff_gray, (5,5,), 0)

        _, binary_img = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, b, l = cv2. boundingRect(contour)
            if cv2.contourArea(contour) > 300:
                cv2.rectangle(bgr, (x, y), (x+b, y+l), (0,255,0), 2)
        cv2.imshow(name, bgr)
    capt_flag = False
    input = cv2.waitKey(5)
    if input%256 == 27:
        print("Operation over")
        break
    elif input%256 == 32:
        capt_flag = True
        img_count += 1
