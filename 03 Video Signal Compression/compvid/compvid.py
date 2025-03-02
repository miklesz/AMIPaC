import os, time
import threading
import numpy as np
import matplotlib.pyplot as plt
#import bqplot.pyplot as plt
from matplotlib import animation
import cv2 as cv
from IPython.display import HTML
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
matplotlib.rcParams['animation.html'] = 'html5'

def psnr(img1, img2):
    s1 = cv.absdiff(img1, img2) # |img1 - img2|
    s1 = np.float32(s1) # can not make a square in 8 bits
    s2 = s1*s1  # |img1 - img2|^2
    sse = s2.sum()
    p_snr = 0.0 # for small values return 0
    if sse >1e-10:
        shape = s1.shape
        mse = 1.0 *sse/(shape[0]*shape[1]*shape[2])
        p_snr = 10 * np.log10((255*255)/mse)
    return p_snr
    
class CoVidTool():    
    def __init__(self, conf):
        self.conf = conf
        self.vid = {}
        self.window = 5

    def get_frames(self, vid):
        return int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    
    def get_position(self, vid): 
        return int(vid.get(cv.CAP_PROP_POS_FRAMES))
    
    def get_geometry(self, vid):
        return (int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)))
    
    def load_vids(self):
        c = self.conf
        v = self.vid
        v['source'] = cv.VideoCapture(os.path.join(c['vpath'], c['source']))
        v['recons'] = cv.VideoCapture(os.path.join(c['vpath'], c['recons']))

    def close_vids(self):
        self.vid['source'].release()
        self.vid['recons'].release()
        self.data = None
        del self.data

    def check_geometry(self):
        s = self.vid['source']
        r = self.vid['recons']
        Ns = self.get_frames(s)
        Nr = self.get_frames(r)
        out = True
        if Ns != Nr:
            #print('source frames: ', Ns,
            #      'recons frames: ', Nr)
            out = True
        (Ws, Hs) = self.get_geometry(s)
        (Wr, Hr) = self.get_geometry(r)  
        if (Ws != Wr) or (Hs != Hr):
            #print('geometry mismatch')
            #print('source geometry: %dx%d'%(Ws,Hs))
            #print('recons geometry: %dx%d'%(Wr,Hr))
            out = False
        return out, (Ws,Hs), (Wr,Hr)
        
    def init_data(self):
        s = self.vid['source']
        N = self.get_frames(s)
        self.data = np.zeros((N,6), dtype=float)

    def running_mean(self, values, k):
        return sum(values[0:k+1])/(k+1)

    def moving_avg(self,values, window, k):
        start = max(0,k+1-window)
        return sum(values[start:k+1])/(k+1-start)
    
    def update_pnsr(self, frame, emarf, k):
        self.data[k,0] = psnr(frame, emarf)
        self.data[k,1] = self.running_mean(self.data[:,0],k)
        self.data[k,2] = self.moving_avg(self.data[:,0],self.window,k)

    def update_mae(self, frame, emarf, k):
        reduc = cv.cvtColor(cv.absdiff(frame,emarf), cv.COLOR_BGR2GRAY)
        self.data[k,3] = np.average(np.average(reduc, axis=0),axis=0)
        self.data[k,4] = self.running_mean(self.data[:,3],k)
        self.data[k,5] = self.moving_avg(self.data[:,3],self.window,k)

    def set_data(self, k):
        s = self.vid['source']
        r = self.vid['recons']
        s0, frame = s.read()
        s1, emarf = r.read()
        if (s0 and s1):
            self.update_pnsr(frame, emarf, k)
            self.update_mae(frame, emarf, k)

    def set_data_return_comp(self, k):
        s = self.vid['source']
        r = self.vid['recons']
        s0, frame = s.read()
        s1, emarf = r.read()
        out = None
        if (s0 and s1):
            self.update_pnsr(frame, emarf, k)
            self.update_mae(frame, emarf, k)
            f0 = cv.resize(frame, self.redu_geom, interpolation=cv.INTER_AREA)
            f1 = cv.resize(emarf, self.redu_geom, interpolation=cv.INTER_AREA)
            fdiff = cv.absdiff(f0,f1)
            fdiff = cv.normalize(fdiff, None, 0, 255, cv.NORM_MINMAX)
            ffull = cv.hconcat([f0,f1,fdiff])
            out = cv.cvtColor(ffull, cv.COLOR_BGR2RGB)
        return out
            
    
    def compare_frames(self):
        s = self.vid['source']
        r = self.vid['recons']
        N = self.get_frames(s)
        print(self.check_geometry)
        p_snr = np.zeros(N)
        for k in range(N):
            t_start = time.time()
            #s.set(cv.CAP_PROP_POS_FRAMES, k)
            success_s, ims = s.read()
            #r.set(cv.CAP_PROP_POS_FRAMES, k)
            success_r, imr = r.read()
            if success_s and success_r:
                p_snr[k] = psnr(ims, imr)
            t_delta = time.time() -t_start
            print(k, 'fps', "%0.2f"%(1/t_delta)) 
                
        plt.plot(p_snr)
        plt.show()

    def create_mpl_fig(self):
        title = 'Source and reconstructed video sequence comparison'

        fig, axs = plt.subplots(3, 1, num=title)
        #fig.canvas.manager.set_window_title(title)
        text_frame = 'frame:{:4d}'
        text_psnr ='running mean PSNR:{:6.2f}'
        text_mae = 'running mean MAE:{:6.2f}'
        self.figsuptext = '  |   '.join([text_frame, text_psnr, text_mae])
        fst = self.figsuptext.format(0,0,0)
        suptitle = fig.suptitle(fst, fontsize=9)
        axs[0].set_xticks(ticks=[80, 80*3, 80*5],
                          labels=['source', 'reconstructed', 'absdiff'])
        axs[0].set_yticks([])
                
        #axs[1].set_title('peak signal-to-noise ratio')
        axs[1].set_ylim(bottom=0, top=None)
        
        #axs[2].sharex(axs[1])
        axs[1].set_xticklabels([])
        axs[1].grid(axis='both')
        #axs[2].set_title('MAE')
        axs[2].set_ylim(bottom=0, top=None)
        axs[2].set_xlabel('frame index')        
        axs[2].grid(axis='both')
        
        self.fig = fig
        self.axs = axs
        self.figsuptitle = suptitle
        
    def set_animate(self, N):
        self.nodata = np.zeros(N)        
        fig = self.fig
        alpha= 0.4        
        W,H = self.get_geometry(self.vid['source'])
        w = 160        
        prop = w/W
        h = int(H*prop)
        self.redu_geom =(w,h)
                
        frame = np.ones((h,w*3,3), dtype=np.uint8)
        self.im = self.axs[0].imshow(frame)                        
        self.line_pnsr, = self.axs[1].plot(
            self.nodata,
            color='tab:blue',
            alpha=alpha,
            lw=1,
            label='PSNR')
        self.line_psnr_rm, = self.axs[1].plot(
            self.nodata,
             color='grey',
            ls='--',
            label='running mean PSNR')
        self.line_psnr_ma, = self.axs[1].plot(
            self.nodata,
            color='tab:blue',
            lw=1,
            label='moving avg PSNR')                
        self.axs[1].set_xlim(left=0, right=N-3)
        self.axs[1].legend(fontsize=8)
                
        self.line_mae, = self.axs[2].plot(
            self.nodata,
            color='tab:red',
            alpha=alpha,
            lw=1,
            label='MAE')
        self.line_mae_rm, = self.axs[2].plot(
            self.nodata,
            color='grey',
            ls='--',
            label='running mean MAE')
        self.line_mae_ma, = self.axs[2].plot(
            self.nodata,
            color='tab:red',
            lw=1,
            label='moving avg MAE')
        self.axs[2].set_xlim(left=0, right=N-3)
        self.axs[2].legend(fontsize=8)
        
    def draw_frame(self, k):
        if  k < (self.anim_steps-1):
            idata = self.set_data_return_comp(k)
            if idata is not None:
                self.im.set_data(idata)
            self.line_pnsr.set_ydata(np.hstack([self.data[:k,0].T, self.nodata[k:]]))
            self.line_psnr_rm.set_ydata(np.hstack([self.data[:k,1].T, self.nodata[k:]]))
            self.line_psnr_ma.set_ydata(np.hstack([self.data[:k,2].T, self.nodata[k:]]))
            self.line_mae.set_ydata(np.hstack([self.data[:k,3].T, self.nodata[k:]]))
            self.line_mae_rm.set_ydata(np.hstack([self.data[:k,4].T, self.nodata[k:]]))
            self.line_mae_ma.set_ydata(np.hstack([self.data[:k,5].T, self.nodata[k:]]))
            if (k+1)%10==0:
                self.axs[1].set_ylim(top=np.max(self.data[:k,0]),
                                bottom=np.min(self.data[:k,0]))
                self.axs[2].set_ylim(top=np.max(self.data[:k,3]),
                                bottom=np.min(self.data[:k,3]))
                fst = self.figsuptext.format(k,self.data[k,1], self.data[k,4])
                self.figsuptitle.set_text(fst)
            #self.fig.canvas.draw()
        if k == self.anim_steps-1:
            self.figsuptitle.set_text(self.figsuptext.format(k,self.data[k,1], self.data[k,4]))
            self.line_pnsr.set_ydata(self.data[:,0].T)
            self.line_psnr_rm.set_ydata(self.data[:,1].T)
            self.line_psnr_ma.set_ydata(self.data[:,2].T)
            self.line_mae.set_ydata(self.data[:,3].T)
            self.line_mae_rm.set_ydata(self.data[:,4].T)
            self.line_mae_ma.set_ydata(self.data[:,5].T)
            self.axs[1].set_ylim((np.min(self.data[:-3,0]),
                                  np.max(self.data[:-3,0])))
            self.axs[2].set_ylim((np.min(self.data[:-3,3]),
                                  np.max(self.data[:-3,3])))
            #self.fig.canvas.draw()
        return (
            self.im,
            self.line_pnsr,
            self.line_psnr_rm,
            self.line_psnr_ma,
            self.line_mae,
            self.line_mae_rm,
            self.line_mae_ma)
        
    def run_base(self):
        self.load_vids()
        self.compare_frames()
        
    def run_anim(self):
        self.load_vids()
        self.check_geometry()
        self.init_data()

        self.create_mpl_fig()
        data = self.data
        N = data.shape[0]
        self.set_animate(N)
        self.anim_steps = N-2
        anim = animation.FuncAnimation(self.fig, self.draw_frame, frames=self.anim_steps, interval=20, blit=True)
        HTML(anim.to_html5_video())

    def animate(self):
        self.create_mpl_fig()
        data = self.data
        N = data.shape[0]
        self.set_animate(N)
        self.anim_steps = N-2
        anim = animation.FuncAnimation(
            self.fig,
            self.draw_frame,
            frames=self.anim_steps,
            interval=20,
            blit=True
        )
        #
        # plt.show()
        return anim.to_html5_video()




        
        
        

if __name__ == '__main__':
    conf = {'vpath':'../src/vid/',
            'source': 'agh_src1_hrc0.m4v',
            'recons': 'agh_src1_hrc_h264.mp4'}

    conf = {'vpath':'../src/vid/',
            'source': 'source.mkv',
            'recons': 'recons.mkv'}
    
    v = CoVidTool(conf)
    v.run_anim()
