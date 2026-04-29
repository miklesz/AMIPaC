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
    streams = [s for s in probe['streams'] if s.get('codec_type') == 'audio']
    if not streams:
        raise ValueError('No audio stream found in the selected file.')
    s = streams[0]
    info = {}
    info['format'] = (pf.get('format_name', 'unknown'),
                      pf.get('format_long_name', 'unknown'))
    duration = s.get('duration', pf.get('duration', 0))
    info['duration'] = float(duration)
    # bit rate get
    info['fsize'] = float(pf.get('size', 0))/1024
    info['bit_rate_file'] = float(pf.get('bit_rate') or 0)
    # file bit rate print(8*float(pf['size'])/info['duration']) [bit/sec]
    if info['bit_rate_file'] == 0 and info['duration'] > 0 and info['fsize'] > 0:
        info['bit_rate_file'] = 8 * (info['fsize'] * 1024) / info['duration']
    info['sample_format'] = s.get('sample_fmt', 'unknown')
    info['codec'] = (s.get('codec_name', 'unknown'),
                     s.get('codec_long_name', 'unknown'))
    info['channels'] = int(s.get('channels', 1))
    info['sample_rate'] = int(s.get('sample_rate', 0))
    info['bit_rate_stream'] = float(s.get('bit_rate') or info['bit_rate_file'])
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
        src_channels = info_source['channels']
        rec_channels = info_recons['channels']
        
        self.src['data'] = faudio_to_np(fsource, sample_format, dtype, src_channels)
        self.rec['data'] = faudio_to_np(frecons, sample_format, dtype, rec_channels)

    def channel_count(self):
        return min(self.src['data'].shape[1], self.rec['data'].shape[1])

    def channel_label(self, k):
        return ['Left channel', 'Right channel'][k] if k < 2 else 'Channel %d' % (k + 1)

    def channel_color(self, k):
        return ['blue', 'red', 'green', 'purple'][k % 4]


    def set_ae_mae(self):
        s = self.src['data']
        r = self.rec['data']
        N = min(s.shape[0],r.shape[0])
        C = self.channel_count()
        data_shape = [N, C]
        data_shape[0] = N
        sae = np.zeros(data_shape, dtype=float)
        ae = np.abs(s[:N, :C] - r[:N, :C])
        
        sae[0,:] = ae[0,:]
        for k in range(1,N):        
            sae[k,:] = sae[k-1,:] + ae[k,:]
        #for k in range(N):        
        #    sae[k,:] = sae[k,:]/(k+1)
        #mae = sae
        mae = sae/np.arange(1, N+1)[:, None]
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
        if isinstance(channel, int):
            labeltxt = self.channel_label(channel)
            color = self.channel_color(channel)
        else:
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


        for k in range(self.src['data'].shape[1]):
            self.display_wave(self.src['data'][:, k], ax=axs[0], channel=k)
        self.set_ax_conf(axs[0],title='Source')
        axs[0].label_outer()

        for k in range(self.rec['data'].shape[1]):
            self.display_wave(self.rec['data'][:, k], ax=axs[1], channel=k)
        self.set_ax_conf(axs[1],title='Reconstructed')
        
    
        axs[1].set_xlabel('Time (s)')
        #axs[2,1].set_xlabel('Time (s)')
        

    def plot_spec_stft(self):
        sr = self.src['info']['sample_rate']
        #fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8,6), dpi=100,
        #                        num="Spectrogram (STFT)")
        rows = self.src['data'].shape[1] + self.rec['data'].shape[1]
        fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(8, max(4, rows * 1.5)), dpi=100)
        axs = np.atleast_1d(axs)
         
        fig.subplots_adjust(hspace=0.35)
        images = []
        row = 0
        for k in range(self.src['data'].shape[1]):
            D = librosa.stft(self.src['data'][:, k])
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            images.append(librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=axs[row], sr=sr))
            self.set_ax_spec(axs[row], title='Source, %s, STFT (log scale)' % self.channel_label(k))
            axs[row].label_outer()
            row += 1

        for k in range(self.rec['data'].shape[1]):
            D = librosa.stft(self.rec['data'][:, k])
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            images.append(librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=axs[row], sr=sr))
            self.set_ax_spec(axs[row], title='Reconstructed, %s, STFT (log scale)' % self.channel_label(k))
            if row < rows - 1:
                axs[row].label_outer()
            row += 1

        
        cbar = fig.colorbar(images[0], ax=list(axs), format="%+2.f dB")
        cbar.ax.tick_params(labelsize=8)

        axs[-1].set_xlabel('Time (s)')

    def plot_spec_mel(self):
        sr = self.src['info']['sample_rate']
        rows = self.src['data'].shape[1] + self.rec['data'].shape[1]
        fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(8, max(4, rows * 1.5)), dpi=100)
        axs = np.atleast_1d(axs)
        #fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8,6), dpi=100,
        #                        num="Spectrogram (MEL)")
        fig.subplots_adjust(hspace=0.35)
        
        images = []
        row = 0
        for k in range(self.src['data'].shape[1]):
            D = librosa.feature.melspectrogram(y=self.src['data'][:, k], sr=sr)
            M_db = librosa.power_to_db(np.abs(D), ref=np.max)
            images.append(librosa.display.specshow(M_db, x_axis='time', y_axis='mel', ax=axs[row], sr=sr))
            self.set_ax_spec(axs[row], title='Source, %s, Mel spectrogram' % self.channel_label(k))
            axs[row].label_outer()
            row += 1

        for k in range(self.rec['data'].shape[1]):
            D = librosa.feature.melspectrogram(y=self.rec['data'][:, k], sr=sr)
            M_db = librosa.power_to_db(np.abs(D), ref=np.max)
            images.append(librosa.display.specshow(M_db, x_axis='time', y_axis='mel', ax=axs[row], sr=sr))
            self.set_ax_spec(axs[row], title='Reconstructed, %s, Mel spectrogram' % self.channel_label(k))
            if row < rows - 1:
                axs[row].label_outer()
            row += 1
        
        cbar = fig.colorbar(images[0], ax=list(axs), format="%+2.f dB")
        cbar.ax.tick_params(labelsize=8)

        axs[-1].set_xlabel('Time (s)')


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
        for k in range(ae.shape[1]):
            pch = {'color': self.channel_color(k), 'label': self.channel_label(k)}
            axs[0].plot(t, ae[:, k], **pbase, **pch)
        self.set_ax_conf(axs[0],title='AE')
                
        axs[0].ticklabel_format(**ptick)
        axs[0].yaxis.get_offset_text().set_fontsize(tick_fontsize)
        axs[0].grid(axis='both')
        axs[0].set_ylim(bottom=0, top=None)
        axs[0].label_outer()

        pmae = {'ls':'--', 'lw':2}
        for k in range(mae.shape[1]):
            pch = {'color': self.channel_color(k), 'label': self.channel_label(k)}
            axs[1].plot(t, mae[:, k], **pmae, **pbase, **pch)

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
        xmax = min(f0.max(), 5_000)
        
        axs[0].set_xlim((0,xmax))
        pbase = {'alpha':0.5}
        for k in range(self.src['data'].shape[1]):
            sk = np.abs(scipy.fftpack.fft(self.src['data'][:, k]))
            fk = np.linspace(0, sr, len(sk))
            pch = {'color': self.channel_color(k), 'label': self.channel_label(k)}
            axs[0].plot(fk, sk, **pbase, **pch)
        self.set_ax_conf(axs[0],title='Source, FFT')
        axs[0].label_outer()
        
        for k in range(self.rec['data'].shape[1]):
            sk = np.abs(scipy.fftpack.fft(self.rec['data'][:, k]))
            fk = np.linspace(0, sr, len(sk))
            pch = {'color': self.channel_color(k), 'label': self.channel_label(k)}
            axs[1].plot(fk, sk, **pbase, **pch)
        self.set_ax_conf(axs[1], title='Reconstructed, FFT')
        axs[1].set_xlim((0,xmax))
        axs[1].set_xlabel('Frequency (Hz)')
                                    
        
if __name__ == '__main__':
    pass
   


 
    
