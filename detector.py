import cv2
import numpy as np
import os

VIDEO_FILES = ["video1.mp4", "video2.mp4", "video3.mp4"]

MIN_AREA = 2000
MAX_AREA_RATIO = 0.05
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 3.5
MIN_SOLIDITY = 0.5

MIN_DET_FRAMES = 6
MIN_STABLE_FRAMES = 25
MIN_STATIC_SECONDS = 20

MAX_MATCH_DIST = 80
MOVEMENT_THRESHOLD = 20
MAX_MISSED_SEC = 14

LR_NORMAL = 0.001
LR_FAST = 0.02
WARMUP_SEC = 3

COLOR_MOVING = (0,255,0)
COLOR_FORGOTTEN = (0,0,255)

def center(b):
    x,y,w,h=b
    return (x+w//2,y+h//2)

def dist(a,b):
    return np.hypot(a[0]-b[0],a[1]-b[1])

def iou(a,b):
    ax,ay,aw,ah=a
    bx,by,bw,bh=b
    ax2,ay2=ax+aw,ay+ah
    bx2,by2=bx+bw,by+bh
    ix1=max(ax,bx)
    iy1=max(ay,by)
    ix2=min(ax2,bx2)
    iy2=min(ay2,by2)
    iw=max(0,ix2-ix1)
    ih=max(0,iy2-iy1)
    inter=iw*ih
    union=aw*ah+bw*bh-inter
    if union==0:
        return 0
    return inter/union

class Track:
    def __init__(self,id,bbox,t):
        self.id=id
        self.bbox=bbox
        self.center=center(bbox)
        self.first=t
        self.last_seen=t
        self.last_moved=t
        self.stable_frames=0
        self.is_stable=False
        self.missed=0

    def update(self,bbox,t):
        c=center(bbox)
        move=dist(self.center,c)
        if move>MOVEMENT_THRESHOLD:
            self.last_moved=t
            self.stable_frames=0
            self.is_stable=False
        else:
            self.stable_frames+=1
            if self.stable_frames>=MIN_STABLE_FRAMES:
                self.is_stable=True
        self.bbox=bbox
        self.center=c
        self.last_seen=t
        self.missed=0

def process_video(path):
    cap=cv2.VideoCapture(path)
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<1: fps=30
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area=W*H
    bg=cv2.createBackgroundSubtractorMOG2(history=800,varThreshold=35,detectShadows=False)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    tracks={}
    next_id=0
    frame=0
    warm=int(WARMUP_SEC*fps)
    candidates={}
    while True:
        ret,img=cap.read()
        if not ret: break
        frame+=1
        t=frame/fps
        fg=bg.apply(img,learningRate=LR_NORMAL)
        _,fg=cv2.threshold(fg,200,255,cv2.THRESH_BINARY)
        fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,kernel)
        fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,kernel)
        if frame<warm:
            cv2.imshow("Detection",img)
            cv2.imshow("FG",fg)
            if cv2.waitKey(1)==27: break
            continue
        contours,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        detections=[]
        for c in contours:
            area=cv2.contourArea(c)
            if area<MIN_AREA: continue
            if area>MAX_AREA_RATIO*frame_area: continue
            x,y,w,h=cv2.boundingRect(c)
            ar=w/float(h)
            if not (MIN_ASPECT_RATIO<=ar<=MAX_ASPECT_RATIO): continue
            hull=cv2.convexHull(c)
            sol=area/(cv2.contourArea(hull)+1)
            if sol<MIN_SOLIDITY: continue
            detections.append((x,y,w,h))
        new_cands={}
        for d in detections:
            c=center(d)
            matched=False
            for cid,(bbox,count) in candidates.items():
                if dist(center(bbox),c)<40:
                    new_cands[cid]=(d,count+1)
                    matched=True
                    break
            if not matched:
                new_cands[len(new_cands)]=(d,1)
        candidates=new_cands
        for cid,(bbox,count) in list(candidates.items()):
            if count>=MIN_DET_FRAMES:
                tracks[next_id]=Track(next_id,bbox,t)
                next_id+=1
                del candidates[cid]
        for bbox in detections:
            c=center(bbox)
            best=None
            best_d=999
            for tr in tracks.values():
                d=dist(tr.center,c)
                if d<best_d and d<MAX_MATCH_DIST:
                    best=tr
                    best_d=d
            if best:
                best.update(bbox,t)
        for tid in list(tracks.keys()):
            tr=tracks[tid]
            tr.missed=t-tr.last_seen
            if tr.missed>MAX_MISSED_SEC and not tr.is_stable:
                del tracks[tid]
        for tr in tracks.values():
            life = t - tr.first
            stationary = t - tr.last_moved
            if tr.stable_frames < 12: continue
            x, y, w, h = tr.bbox
            if tr.is_stable and stationary > MIN_STATIC_SECONDS:
                color = COLOR_FORGOTTEN
                label = f"ABANDONED {stationary:.1f}s"
            else:
                color = COLOR_MOVING
                label = None
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            if label:
                cv2.putText(img,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.imshow("Detection",img)
        cv2.imshow("FG",fg)
        if cv2.waitKey(1)==27: break
    cap.release()

if __name__=="__main__":
    for v in VIDEO_FILES:
        if not os.path.isfile(v):
            print("not found:",v)
            continue
        print("processing",v)
        process_video(v)
    cv2.destroyAllWindows()