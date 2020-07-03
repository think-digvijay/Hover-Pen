# %%
import cv2
import numpy as np
import time

# %%
def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


while True:
    ret, frame = cap.read()
    
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    stacked = np.hstack((mask_3, frame, res))
    
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx = 0.4, fy = 0.4))
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord('s'):
        thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(thearray)
        
        np.save('penval', thearray)
        break
        
cap.release()
cv2.destroyAllWindows()

# %%
load_from_disk = True

if load_from_disk:
    penval = np.load('penval.npy')
    
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5), np.uint8)

while(1):
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
        
    else:
        lower_range = np.array([26, 80, 147])
        upper_range = np.array([81, 255, 255])
        
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    res = cv2.bitwise_and(frame, frame, mask = mask)
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    stacked = np.hstack((mask_3, frame, res))
    
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx = 0.4, fy = 0.4))
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()
        

# %%
load_from_disk = True

if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5, 5), np.uint8)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

noiseth = 500

while(1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
            
    else:             
        lower_range  = np.array([26,80,147])
        upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
        c = max(contours, key = cv2.contourArea)
        
        x,y,w,h = cv2.boundingRect(c)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,255),2)
    
    cv2.imshow('image',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

# %%
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

canvas = None

x1, y1 = 0, 0

noiseth = 800

while(1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1 )
    
    if canvas is None:
        canvas = np.zeros_like(frame)
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]

    else:             
        lower_range  = np.array([26,80,147])
        upper_range = np.array([81,255,255])
        
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
        
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4)
        x1,y1= x2,y2
        
    else:
        x1, y1 = 0, 0
        
    frame = cv2.add(frame,canvas)
    
    stacked = np.hstack((canvas,frame))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
    if k == ord('c'):
        canvas = None
        
cv2.destroyAllWindows()
cap.release()

# %%
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

canvas=None

x1,y1=0,0

noiseth = 800

wiper_thresh = 20000

clear = False

while(1):
    _ , frame = cap.read()
    frame = cv2.flip( frame, 1 )
    
    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
            
    else:             
        lower_range  = np.array([26,80,147])
        upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        area = cv2.contourArea(c)
         
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 5)
        
        x1,y1= x2,y2
        
        if area > wiper_thresh:
            cv2.putText(canvas, 'Clearing Canvas', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5, cv2.LINE_AA)
            
    else:
        x1, y1 = 0, 0
        
    _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)
    
    cv2.imshow('image',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
    if clear == True:
        
        time.sleep(1)
        canvas = None
        
        clear = False
        
    if k == ord('c'):
        canvas = None
        clear = False
        
cv2.destroyAllWindows()
cap.release()

# %%
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

pen_img = cv2.resize(cv2.imread('pen.png',1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.jpg',1), (50, 50))

kernel = np.ones((5,5),np.uint8)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

canvas = None

backgroundobject = cv2.createBackgroundSubtractorMOG2( detectShadows = False )

background_threshold = 600

switch = 'Pen'

last_switch = time.time()

x1, y1 = 0, 0

noiseth = 800

wiper_thresh = 10000

clear = False

while(1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1 )
    
    if canvas is None:
        canvas = np.zeros_like(frame)
        
    top_left = frame[0: 50, 0: 50]
    fgmask = backgroundobject.apply(top_left)
    
    switch_thresh = np.sum(fgmask == 255)
    
    if switch_thresh > background_threshold  and (time.time() - last_switch) > 1:
        
        last_switch = time.time()
        
        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'
            
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
            
    else:             
        lower_range  = np.array([26,80,147])
        upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
                
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        area = cv2.contourArea(c)
        
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
        
        else:
            if switch == 'Pen':
                canvas = cv2.line(canvas, (x1,y1), (x2,y2), [255,0,0], 5)
            else:
                cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
                
        x1,y1= x2,y2
        
        # Now if the area is greater than the wiper threshold then set the clear variable to True
        if area > wiper_thresh:
            cv2.putText(canvas,'Clearing Canvas',(0,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1, cv2.LINE_AA)
            clear = True 
    else:
        x1, y1 = 0, 0
        
    _,mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)

    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
        frame[0: 50, 0: 50] = eraser_img
    else:
        frame[0: 50, 0: 50] = pen_img

    
    cv2.imshow('image',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
    if clear == True:
        
        time.sleep(1)
        canvas = None
        
        clear = False
        
    if k == ord('c'):
        canvas = None
        clear = False
        
cv2.destroyAllWindows()
cap.release()


# %%


# %%
