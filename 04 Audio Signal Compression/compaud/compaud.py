'''compaud/compaud.py
Compare compression codecs for audio signals using ffmpeg for I/O.
Add spectral features, librosa features will be enough
1) comparison of the signal ok 
2) fft
3) spectrogram ok
4) energy entropy
5) spectral entropy
https://raphaelvallat.com/entropy/build/html/generated/entropy.spectral_entropy.html
https://musicinformationretrieval.com/spectral_features.html
6) spectral roll off
https://librosa.org/doc/main/generated/librosa.feature.spectral_rolloff.html
https://librosa.org/doc/main/generated/librosa.feature.spectral_centroid.html
https://librosa.org/doc/main/generated/librosa.feature.spectral_bandwidth.html
7) short time energy? energy
https://superkogito.github.io/blog/2020/02/09/naive_vad.html
https://musicinformationretrieval.com/novelty_functions.html
8) zero crossing rate
https://colab.research.google.com/github/stevetjoa/musicinformationretrieval.com/blob/gh-pages/zcr.ipynb
https://librosa.org/doc/main/generated/librosa.feature.zero_crossing_rate.html

AV/AGH/2023
'''

import os, pprint, time
import numpy as np
import scipy
import ffmpeg
import matplotlib.pyplot as plt
import librosa, librosa.display
import IPython.display as ipd


def _probe(fname):
    return ffmpeg.probe(
        fname,
        hide_banner=None,
        select_streams='a',
        log_level='0')

def _get_info(probe):
    pf = probe['format']
    info = {}
    info['format'] = (pf['format_name'], pf['format_long_name'])
    info['duration'] = float(pf['duration'])
    # bit rate get
    info['fsize'] = float(pf['size'])/1024
    info['bit_rate_file'] = float(pf['bit_rate'])
    # file bit rate print(8*float(pf['size'])/info['duration']) [bit/sec]
    s = probe['streams'][0]
    if s['codec_type'] == 'audio':
        info['sample_format'] = s['sample_fmt']
        info['codec'] = (s['codec_name'], s['codec_long_name'])
        info['channels'] = s['channels']
        info['duration'] = float(s['duration'])
        info['sample_rate'] = int(s['sample_rate'])
        info['bit_rate_stream'] = float(s['bit_rate'])
    return info

def probe_info(fname):
    return _get_info(_probe(fname))

def transcode(fsrc, fout, acodec, bitrate):
    #CBR constant bit rate
    # 8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, or 320    
    global_args = ['-vn', '-sn','-dn', '-hide_banner','-loglevel', '0']
    proc_trans = (
        ffmpeg
        .input(fsrc)
        .audio
        .output(fout, audio_bitrate=bitrate, acodec=acodec)
        .overwrite_output()
        .global_args(*global_args)
        .run()
    )

def faudio_to_np(fname, sample_format, dtype, n_channels):
    out, err = (
        ffmpeg
        .input(fname)
        .audio            
        .output('pipe:', format=sample_format)
        .overwrite_output()
        .global_args(*['-vn', '-sn','-dn', '-hide_banner','-loglevel', '0'])
        .run(capture_stdout=True)
    )
    
    audio = (
        np
        .frombuffer(out, dtype)
        .reshape([-1, n_channels])
    )
    return audio
    

class CompAudio():
    def __init__(self):
        # only little endian
        self.sample_format = {
            'f32le': np.float32,
            'f64le': np.float64,
            's16le': np.int16,
            's32le': np.int32,
            'u16le': np.uint16,
            'u32le': np.uint32}
        self.src = {}
        self.rec = {}
        
    def get_sample_format(self, info):
        # Use one stream, default value to s16le if the source is not a raw pcm
        sample_format = 's16le'
        sf = info['codec'][0].split('_')[-1]
        if sf in self.sample_format.keys():
            sample_format = sf            
        return sample_format
        
    def get_data(self, fsource, frecons, info_source, info_recons):
        '''Get raw audio data in the same sample format for comparison.
        Default to float32 for librosa
        '''
        self.src['fname'] = fsource
        self.src['info'] = info_source
        self.rec['fname'] = frecons
        self.rec['info'] = info_recons
        sample_format = 'f32le'
        dtype = self.sample_format[sample_format]
        n_channels = info_source['channels']
        
        self.src['data'] = faudio_to_np(fsource, sample_format, dtype, n_channels)
        self.rec['data'] = faudio_to_np(frecons, sample_format, dtype, n_channels)


    def set_ae_mae(self):
        s = self.src['data']
        r = self.rec['data']
        N = min(s.shape[0],r.shape[0])
        data_shape = list(s.shape)
        data_shape[0] = N
        sae = np.zeros(data_shape, dtype=float)
        ae = np.abs(s[:N,:] -r[:N,:])
        
        sae[0,:] = ae[0,:]
        for k in range(1,N):        
            sae[k,:] = sae[k-1,:] + ae[k,:]
        #for k in range(N):        
        #    sae[k,:] = sae[k,:]/(k+1)
        #mae = sae
        mae = sae/np.array([range(1,N+1),range(1,N+1)]).T
        self.comp_data = {}
        self.comp_data['mae'] = mae 
        self.comp_data['ae'] = ae

    def set_ax_conf(self, ax, title):
        font_size = 8
        font_size_legend = 6
        legend_loc = 'lower right'
        ax.set_title(title, fontsize=font_size)
        ax.legend(fontsize=font_size_legend, loc=legend_loc)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.tick_params(axis='both', which='minor', labelsize=font_size)

    def display_wave(self, data, ax, channel='left'):
        sr = self.src['info']['sample_rate']
        labeltxt = 'Left channel'
        color ='blue'
        if channel != 'left':
            labeltxt = 'Right channel'
            color = 'red'
        librosa.display.waveshow(
            data,
            sr=sr,
            ax=ax,
            color=color,
            alpha=0.4,
            label=labeltxt)
        ax.tick_params(axis='x',which='both',bottom=False)
        

    def set_ax_spec(self, ax, title):
        font_size = 7
        ticks_font_size= 6
        ax.set_title(title, fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=ticks_font_size)
        ax.tick_params(axis='both', which='minor', labelsize=ticks_font_size)

    def plot_wave(self):
        #time related elements alltogether
        sr = self.src['info']['sample_rate']
        #fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4), dpi=100,
        #                        num='Source and reconstructed soundwaves')
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4), dpi=100)


        self.display_wave(self.src['data'][:,0], ax=axs[0], channel='left')
        self.display_wave(self.src['data'][:,1], ax=axs[0], channel='right')        
        self.set_ax_conf(axs[0],title='Source')
        axs[0].label_outer()

        self.display_wave(self.rec['data'][:,0], ax=axs[1], channel='left')
        self.display_wave(self.rec['data'][:,1], ax=axs[1], channel='right')
        self.set_ax_conf(axs[1],title='Reconstructed')
        
    
        axs[1].set_xlabel('Time (s)')
        #axs[2,1].set_xlabel('Time (s)')
        

    def plot_spec_stft(self):
        sr = self.src['info']['sample_rate']
        #fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8,6), dpi=100,
        #                        num="Spectrogram (STFT)")
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8,6), dpi=100)
         
        fig.subplots_adjust(hspace=0.35)
        D = librosa.stft(self.src['data'][:,0])
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)        
        im0 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=axs[0], sr=sr)
        self.set_ax_spec(axs[0],title='Source, Left Channel, STFT (log scale)')
        axs[0].label_outer()

        D = librosa.stft(self.src['data'][:,1])
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        im1 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=axs[1], sr=sr)
        self.set_ax_spec(axs[1],title='Source, Right Channel, STFT (log scale)')
        axs[1].label_outer()
        
        D = librosa.stft(self.rec['data'][:,0])
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        im2 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=axs[2], sr=sr)
        self.set_ax_spec(axs[2], title='Reconstructed, Left Channel, STFT (log scale)')
        axs[2].label_outer()
        
        D = librosa.stft(self.rec['data'][:,1])
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        im3 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=axs[3], sr=sr)
        self.set_ax_spec(axs[3], title='Reconstructed, Right Channel, STFT (log scale)')

        
        cbar = fig.colorbar(im0, ax=[axs[0],axs[1], axs[2], axs[3]], format="%+2.f dB")
        cbar.ax.tick_params(labelsize=8)

        axs[3].set_xlabel('Time (s)')

    def plot_spec_mel(self):
        sr = self.src['info']['sample_rate']
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8,6), dpi=100)
        #fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8,6), dpi=100,
        #                        num="Spectrogram (MEL)")
        fig.subplots_adjust(hspace=0.35)
        
        D = librosa.feature.melspectrogram(y=self.src['data'][:,0], sr=sr)
        M_db = librosa.power_to_db(np.abs(D), ref=np.max)
        im0 = librosa.display.specshow(M_db, x_axis='time', y_axis='mel', ax=axs[0], sr=sr)
        self.set_ax_spec(axs[0], title='Source, Left Channel, Mel spectrogram')
        axs[0].label_outer()


        D = librosa.feature.melspectrogram(y=self.src['data'][:,1], sr=sr)
        M_db = librosa.power_to_db(np.abs(D), ref=np.max)
        im1 = librosa.display.specshow(M_db, x_axis='time', y_axis='mel', ax=axs[1], sr=sr)
        self.set_ax_spec(axs[1],title='Source, Right Channel, Mel spectrogram')
        axs[1].label_outer()


        D = librosa.feature.melspectrogram(y=self.rec['data'][:,0], sr=sr)
        M_db = librosa.power_to_db(np.abs(D), ref=np.max)
        im2 = librosa.display.specshow(M_db, x_axis='time', y_axis='mel', ax=axs[2], sr=sr)
        self.set_ax_spec(axs[2],title='Reconstructed, Left Channel, Mel spectrogram')
        axs[2].label_outer()
        

        D = librosa.feature.melspectrogram(y=self.rec['data'][:,1], sr=sr)
        M_db = librosa.power_to_db(np.abs(D), ref=np.max)
        im3 = librosa.display.specshow(M_db, x_axis='time', y_axis='mel', ax=axs[3], sr=sr)
        self.set_ax_spec(axs[3], title='Reconstructed, Right Channel, Mel spectrogram')    
        
        cbar = fig.colorbar(im0, ax=[axs[0],axs[1], axs[2], axs[3]], format="%+2.f dB")
        cbar.ax.tick_params(labelsize=8)

        axs[3].set_xlabel('Time (s)')


    def plot_ae_mae(self):
        self.set_ae_mae()
        sr = self.src['info']['sample_rate']
        ae = self.comp_data['ae']
        mae = self.comp_data['mae']
        N = ae[:,0].shape[0]        
        t = np.arange(N)/sr
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4), dpi=100)
        #fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4), dpi=100,
        #                        num="AE/MAE")
        
        ptick = {'axis':'y','style':'sci','scilimits':(0,0),'useMathText':True}
        tick_fontsize = 7
        pbase = {'alpha':0.5}
        pleft = {'color':'blue', 'label':'Left channel'}
        pright= {'color':'red', 'label':'Right channel'}
        
        axs[0].plot(t, ae[:,0], **pbase, **pleft)
        axs[0].plot(t, ae[:,1], **pbase, **pright)
        self.set_ax_conf(axs[0],title='AE')
                
        axs[0].ticklabel_format(**ptick)
        axs[0].yaxis.get_offset_text().set_fontsize(tick_fontsize)
        axs[0].grid(axis='both')
        axs[0].set_ylim(bottom=0, top=None)
        axs[0].label_outer()

        pmae = {'ls':'--', 'lw':2}
        axs[1].plot(t, mae[:,0], **pmae, **pleft, **pbase)
        axs[1].plot(t, mae[:,1], **pmae, **pright, **pbase)

        self.set_ax_conf(axs[1],title='MAE')
        axs[1].ticklabel_format(**ptick)
        axs[1].grid(axis='both')
        axs[1].set_ylim(bottom=0, top=None)
        axs[1].yaxis.get_offset_text().set_fontsize(tick_fontsize)
        axs[1].set_xlabel('Time (s)')

    def plot_fft(self):
        sr = self.src['info']['sample_rate']        
        # FFT
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4), dpi=100)
        #fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4), dpi=100,
        #                        num='FFT')
        
        s0 = np.abs(scipy.fftpack.fft(self.src['data'][:,0]))
        f0 = np.linspace(0,sr, len(s0))
        s1 = np.abs(scipy.fftpack.fft(self.src['data'][:,1]))
        f1 = np.linspace(0,sr, len(s1))
        xmax = min(f0.max(), 5_000)
        
        axs[0].set_xlim((0,xmax))
        pbase = {'alpha':0.5}
        pleft = {'color':'blue', 'label':'Left channel'}
        pright = {'color':'red', 'label':'Right channel'} 
        axs[0].plot(f0,s0, **pbase, **pleft)
        axs[0].plot(f1,s1, **pbase, **pright)
        self.set_ax_conf(axs[0],title='Source, FFT')
        axs[0].label_outer()
        
        s0 = np.abs(scipy.fftpack.fft(self.rec['data'][:,0]))
        f0 = np.linspace(0,sr, len(s0))
        s1 = np.abs(scipy.fftpack.fft(self.rec['data'][:,1]))
        
        f1 = np.linspace(0,sr, len(s1))
        axs[1].plot(f0,s0, **pbase, **pleft)
        axs[1].plot(f1,s1, **pbase, **pright)        
        self.set_ax_conf(axs[1], title='Reconstructed, FFT')
        axs[1].set_xlim((0,xmax))
        axs[1].set_xlabel('Frequency (Hz)')
                                    
        
if __name__ == '__main__':
    pass
   


 
    
