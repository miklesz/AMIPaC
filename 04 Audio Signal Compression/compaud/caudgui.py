import os, shutil, time, pathlib
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, Audio
import ipywidgets as wg
import librosa
from .compaud import probe_info, transcode, faudio_to_np, CompAudio

def run_show():
    path_h = pathlib.Path(__file__).parent
    with open(path_h / 'run_show.html', 'r') as f:
        return HTML(f.read())

def _print_info(p_info):
    r = p_info
    sep = ' | '
    la = 'Format: [' + r['format'][0] + '] ' + r['format'][1] + sep
    la += 'Codec: [' + r['codec'][0] + '] ' + r['codec'][1] + sep
    la += 'Channels: ' + '%d'%r['channels']  
    lb = 'Duration: ' + '%.2f'%r['duration'] + ' s' + sep
    lb += 'Sample Rate: ' + '%.2f'%(r['sample_rate']/1e3) + ' kHz' + sep
    lb += 'Bit Rate File: ' + '%.2f'%(r['bit_rate_file']/1e3) + ' kbit/s' + sep
    lb += 'File Size: ' + '%.2f'%r['fsize'] + ' kB'
    return (la,lb)


class CAudGui():
    def __init__(self):
        self.out_codec = ['mp3', 'aac', 'vorbis']
        self.out_type = {'mp3':'mp3', 'vorbis':'ogg', 'aac':'m4a'}
        self.codec_implement = {'mp3':'mp3', 'vorbis':'libvorbis', 'aac':'aac'}
        self.audio_path = './tmp'
        self.components = {'probe_source':{},
                           'transcode':{},
                           'compare':{}}
        self.probe_info = {}
        self.setup_workspace()
        self.data = {'source':None,
                     'local':None,
                     'external':None}
        
    def setup_workspace(self):
        if os.path.isdir(self.audio_path):
            shutil.rmtree(self.audio_path)
            os.mkdir(self.audio_path)
        
    def add_path(self, fname):
        return os.path.join(self.audio_path, fname)
    
    def rm_file(self, fname):
        os.remove(self.add_path(fname))
        
    def write_file(self, fname, content):
        with open(self.add_path(fname), 'wb') as f:
            f.write(content)
            
    def check_file(self, fname):            
        return os.path.isfile(self.add_path(fname))
    
    def info_file(self, fname, ftype):
        pi = probe_info(self.add_path(fname))
        self.probe_info[ftype] = pi
        return _print_info(pi)

    def transcode(self, fsource, fout, acodec, bitrate):
        fsrc = self.add_path(fsource)
        fout = self.add_path(fout)
        bit_rate = '%dk'%bitrate
        acodec = self.codec_implement[acodec]
        transcode(fsrc, fout, acodec, bit_rate)
        
    def create_probe_source(self):
        c = self.components['probe_source']
        c['bn_upl'] = wg.FileUpload(description='Upload Source',
                                    tooltip='Upload source file')
        c['bn_rm'] = wg.Button(
            description='Remove',
            tooltip='Remove source file',
            icon='remove')
        c['f_desc'] = wg.Label('Source file name: ')
        c['f_name'] = wg.Label('')
        c['info1'] = wg.Label('')
        c['info2'] = wg.Label('')        
        line_bn = wg.HBox([c['bn_upl'], c['bn_rm']])
        c['line_fn'] = wg.HBox([c['f_desc'], c['f_name']])
        c['top'] = wg.VBox([line_bn, c['line_fn'], c['info1'], c['info2']])
        return c['top']
    
    def get_cr_info(self):
        c_comp = self.components['compare']
        p_src = self.probe_info['source']
        p_rec = self.probe_info[c_comp['sel_rec'].value]
        cr_txt = 'Compression ratio: %.2f'%(p_src['fsize']/p_rec['fsize'])
        return cr_txt

    def plot_data(self):
        c_src = self.components['probe_source']
        c_comp = self.components['compare']
        p_src = self.probe_info['source']
        p_rec = self.probe_info[c_comp['sel_rec'].value]
        plt.close('all')
        ca = CompAudio()
        ca.get_data(self.add_path(c_src['f_name'].value),
                    self.add_path(c_comp['f_name'].value),
                    p_src,
                    p_rec)
        
        c_comp['mpl_1'].clear_output()                
        with c_comp['mpl_1']:
            ca.plot_wave()
            plt.show()
        c_comp['mpl_2'].clear_output()                            
        with c_comp['mpl_2']:
            ca.plot_ae_mae()
            plt.show()
        c_comp['mpl_3'].clear_output()                            
        with c_comp['mpl_3']:
            ca.plot_fft()
            plt.show()
        c_comp['mpl_4'].clear_output()                
        with c_comp['mpl_4']:
            ca.plot_spec_stft()
            plt.show()
        c_comp['mpl_5'].clear_output()                
        with c_comp['mpl_5']:
            ca.plot_spec_mel()
            plt.show()
        c_comp['src_au'].clear_output()    
        with c_comp['src_au']:
            display(Audio(ca.src['data'].T, rate=ca.src['info']['sample_rate']))
        c_comp['rec_au'].clear_output()
        with c_comp['rec_au']:
            display(Audio(ca.rec['data'].T, rate=ca.src['info']['sample_rate']))

            
    def drive_probe_source(self):
        c = self.components['probe_source']
        c_trs = self.components['transcode']        
        def handle_upl_chg(change):
            if len(change.new) > 0:
                if len(change.old) > 0:
                    self.rm_file(change.old[0]['name'])
                c['f_name'].value = change.new[0]['name']
                self.write_file(c['f_name'].value, change.new[0]['content'])
                (la, lb) = self.info_file(c['f_name'].value, 'source')
                c['info1'].value = la
                c['info2'].value = lb
                c_trs['f_name'].value = c_trs['f_name_pre'].value + '.' + c_trs['f_type'].value
                c_trs['info1'].value = ''
                c_trs['info2'].value = ''
                c_trs['dl_lnk'].value = ''
            else:
                c['f_name'].value = ''
        c['bn_upl'].observe(handle_upl_chg, names='value')

        def on_bn_rm_ckd(b):
            if self.check_file(c['f_name'].value):
                self.rm_file(c['f_name'].value)
            c['f_name'].value = ''
            c['bn_upl'].value = ()
            c['info1'].value = ''
            c['info2'].value = ''
            if self.check_file(c_trs['f_name'].value):
                self.rm_file(c_trs['f_name'].value)
            c_trs['info1'].value = ''
            c_trs['info2'].value = ''
            c_trs['dl_lnk'].value = ''
        c['bn_rm'].on_click(on_bn_rm_ckd)


    def create_transcoder(self):
        c = self.components['transcode']
        c_src = self.components['probe_source']
        c['bn_trs'] = wg.Button(
            description='Transcode',
            tooltip='Transcode source file',
            icon='flash')
        
        c['bn_rm'] = wg.Button(
            description='Remove',
            tooltip='Remove transcoded file',
            icon='remove')

        c['sel_cod'] = wg.Dropdown(
            options=self.out_codec,
            value='mp3',
            description='Audio Codec')

        c['sel_btr'] = wg.IntSlider(
            value=256,
            min=8,
            max=384,
            step=8,
            description='Bit Rate',
            continous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d')
        c['f_desc'] = wg.Label('Transcoded file name: ')
        c['f_name'] = wg.Label('')
        c['dl_html'] = "<a href={tran_file} target='_blank'> (download) </a>"
        c['dl_lnk'] = wg.HTML(value='',)
        c['f_name_pre'] = wg.Text(
            description='New Name',
            value='transcoded',
            placeholder='transcoded file name')
        dot = wg.Label('.')
        c['f_type'] = wg.Label('')
        c['info1'] = wg.Label('')
        c['info2'] = wg.Label('')
        line_bn = wg.HBox([c['bn_trs'], c['bn_rm']])
        line_trs = wg.HBox([c['f_desc'], c['f_name'], c['dl_lnk']])
        line_new_name = wg.HBox([c['f_name_pre'], dot, c['f_type']])
        line_cfg = wg.HBox([c['sel_cod'], c['sel_btr'], wg.Label('kbit/s')])
        c['top'] = wg.VBox(
            [line_bn,
             c_src['line_fn'],
             line_trs,
             line_new_name,
             line_cfg,
             c['info1'],
             c['info2']])        
        return c['top']

    def drive_transcoder(self):
        c_src = self.components['probe_source']
        c = self.components['transcode']
        c_cmp = self.components['compare']
        c['f_type'].value = self.out_type[c['sel_cod'].value]
        
        def handle_fname_pre_chg(change):            
            c['f_name'].value = change.new + '.' + c['f_type'].value
            
        c['f_name_pre'].observe(handle_fname_pre_chg, names='value')
            
        def handle_sel_cod_chg(change):            
            c['f_type'].value = self.out_type[change.new]
            c['f_name'].value = c['f_name_pre'].value + '.' + c['f_type'].value
            
        c['sel_cod'].observe(handle_sel_cod_chg, names='value')
        
        def on_bn_trs_ckd(b):            
            if c_src['f_name'].value != '':
                if self.check_file(c['f_name'].value):
                    self.rm_file(c['f_name'].value)
                targs = [
                    c_src['f_name'].value,
                    c['f_name'].value,
                    c['sel_cod'].value,
                    c['sel_btr'].value]

                self.transcode(*targs)
                ht = c['dl_html'].format(tran_file=self.add_path(c['f_name'].value))
                c['dl_lnk'].value = ht
                (la, lb) = self.info_file(c['f_name'].value, 'local')
                c['info1'].value = la
                c['info2'].value = lb
                c_cmp['sel_rec'].value = 'local'
                c_cmp['f_name'].value = c['f_name'].value
        c['bn_trs'].on_click(on_bn_trs_ckd)

        def on_bn_rm_ckd(b):
            if self.check_file(c['f_name'].value):
                self.rm_file(c['f_name'].value)
            c['info1'].value = ''
            c['info2'].value = ''
            c['dl_lnk'].value = ''
            
        c['bn_rm'].on_click(on_bn_rm_ckd)
                        
    def create_comparer(self):
        c_src = self.components['probe_source']
        c = self.components['compare']
        c['bn_comp'] = wg.Button(
            description='Compare',
            tooltip='Compare source with reconstructed video',
            icon='play')
        
        c['bn_upl'] = wg.FileUpload(
            description='Upload external',
            tooltip='Upload external file',
            layout=wg.Layout(width='200px'))
        
        c['bn_rm'] = wg.Button(
            description='Remove external',
            tooltip='Remove external file',
            icon='remove')

        c['sel_rec'] = wg.Dropdown(
            options=[('transcoded','local'),
                     ('external', 'external')],
            value='local',
            description='Use:',
            layout=wg.Layout(width='200px'))

        c['f_desc'] = wg.Label('Reconstructed file name: ')
        c['f_name'] = wg.Label('')
        c['src_au'] = wg.Output()
        c['rec_au'] = wg.Output()
        
        c['info1'] = wg.Label('')
        c['info2'] = wg.Label('')
        c['info_cr'] = wg.Label('')

        mk_output = lambda : wg.Output(layout={
                'width':'100%',
                'height':'auto'})
        
        for k in range(5):
            c['mpl_%d'%(k+1)] = mk_output()
                
        
        line_bn = wg.HBox(
            [c['bn_comp'], c['sel_rec'], c['bn_upl'], c['bn_rm']])
        
        line_fn = wg.HBox([c['f_desc'], c['f_name']])

        sec_fn = wg.TwoByTwoLayout(top_left=c_src['line_fn'],
                                   top_right=c['src_au'],
                                   bottom_left=line_fn,
                                   bottom_right=c['rec_au'])

        plots_ac = wg.Accordion(
            children=[c['mpl_1'], c['mpl_2'], c['mpl_3'], c['mpl_4'], c['mpl_5']],
            titles=['Soundwaves',
                    'AE/MAE',
                    'FFT',
                    'Spectrogram(STFT)' ,
                    'Spectrogram(MEL)']
        )
        
        c['top'] = wg.VBox(
            [line_bn,
             sec_fn,
             c['info1'],
             c['info2'],
             c['info_cr'],
             plots_ac])
        return c['top']

    def clear_output_mpl(self):
        c = self.components['compare']
        for k in range(5):
            c['mpl_%d'%(k+1)].clear_output()
        c['src_au'].clear_output()
        c['rec_au'].clear_output()
        plt.close('all')
        

    def drive_comparer(self):
        c_src = self.components['probe_source']
        c_trs = self.components['transcode']
        c = self.components['compare']
        def handle_upl_chg(change):
            if len(change.new) > 0:
                if len(change.old) > 0:
                    self.rm_file(change.old[0]['name'])
                c['f_name'].value = change.new[0]['name']
                self.write_file(c['f_name'].value, change.new[0]['content'])
                c['sel_rec'].value = 'local'
                (la, lb) = self.info_file(c['f_name'].value, 'external')
                c['info1'].value = la
                c['info2'].value = lb                
                c['sel_rec'].value = 'external'
            else:
                c['f_name'].value = ''
            c['info_cr'].value = ''            
        c['bn_upl'].observe(handle_upl_chg, names='value')

        def handle_sel_rec_chg(change):
            if change.new == 'local':
                c['f_name'].value = c_trs['f_name'].value
                c['info1'].layout.visibility = 'hidden'
                c['info2'].layout.visibility = 'hidden'
                self.clear_output_mpl()
                
            elif change.new == 'external':
                if len(c['bn_upl'].value) > 0:
                    c['f_name'].value = c['bn_upl'].value[0]['name']
                    c['info1'].layout.visibility = 'visible'
                    c['info2'].layout.visibility = 'visible'
                    self.clear_output_mpl()
                    
                else:
                    c['f_name'].value = ''
                    self.clear_output_mpl()


        c['sel_rec'].observe(handle_sel_rec_chg, names='value')

        def on_bn_rm_ckd(b):
            if self.check_file(c['f_name'].value):
                self.rm_file(c['f_name'].value)
            c['f_name'].value = ''
            c['bn_upl'].value = ()
            c['info1'].value = ''
            c['info2'].value = ''
            c['info_cr'].value = ''
            self.clear_output_mpl()
        c['bn_rm'].on_click(on_bn_rm_ckd)

        def on_bn_comp_ckd(b):
            if (self.check_file(c['f_name'].value) and
                self.check_file(c_src['f_name'].value)):
                c['info_cr'].value = self.get_cr_info()
                self.plot_data()
            else:
                c['info_cr'].value = ''
                
                
        c['bn_comp'].on_click(on_bn_comp_ckd)
        
    def create_gui(self):
        self.top = wg.Tab()
        titles = ['Probe Source', 'Transcode', 'Compare']        
        ftab = [self.create_probe_source,
                self.create_transcoder,
                self.create_comparer]  
        self.top.children = [ft() for ft in ftab]        
        self.top.titles = titles
        
    def drive_gui(self):
        self.drive_probe_source()
        self.drive_transcoder()
        self.drive_comparer()
        
    def run(self):
        self.create_gui()
        self.drive_gui()
        display(self.top)        

def run_gui():
    return CAudGui().run()

if __name__ == '__main__':
    a = CAudGui()
    a.run()
