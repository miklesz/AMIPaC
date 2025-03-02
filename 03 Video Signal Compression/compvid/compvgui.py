import os, shutil, time
import ipywidgets as wg
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from .videotranc import TCoVid
from .compvid import CoVidTool


def print_info(info):
    r = info
    sep = ' | '
    cp = 'Codec: ' + r['codec'] + sep
    cp += 'Pixel format: ' + r['pixel format']  
    gfs = 'Geometry: ' + r['geometry'] + sep
    gfs += 'FPS: ' + r['fps'] + sep
    gfs += 'OS file size: ' + r['OS file size'] + sep
    gfs += 'Zip file size: ' + r['Zip file size']
    return [cp, gfs]


class CoViGui():
    
    def __init__(self):
        self.top = None
        self.output_type = ['avi', 'mp4', 'm4v', 'mkv', 'mov', 'mjpg', 'webm']
        self.video_codec = ['libx264', 'libx265', 'jpeg2000', 'mpeg4', 'zlib']
        self.pixel_format = ['yuv420p', 'rgb24',  'yuv411p', 'yuv444p']
        self.vid_tran = TCoVid()
        self.probe_info = {'source':None, 'local':None, 'external':None}

        self.video_path = './tmp'
        if os.path.isdir(self.video_path):
            shutil.rmtree(self.video_path)
            os.mkdir(self.video_path)
            
        self.vid_comp = CoVidTool({'vpath': self.video_path,
                                   'source': '',
                                   'recons': ''})                
        
    def add_path(self, fname):
        return os.path.join(self.video_path, fname)
            
    def check_file_vid(self, fname):            
        return os.path.isfile(self.add_path(fname))

    def remove_vid(self, fname):
        os.remove(self.add_path(fname))
        
    def write_vid(self, fname, content):
        with open(self.add_path(fname), 'wb') as f:
            f.write(content)    
    
    def _set_probe_info(self, fname, info1, info2):
        fn = self.add_path(fname)
        la, lb = print_info(self.vid_tran.probe_reduc(fn))
        info1.value = la
        info2.value = lb

    def set_probe_info(self, fname, info1, info2, name):
        fn = self.add_path(fname)
        info = self.vid_tran.probe(fn)
        self.probe_info[name] = info
        la, lb = print_info(self.vid_tran.reduc_probeinfo(info))
        info1.value = la
        info2.value = lb

    def transcode_vid(self, fsource, fdest, conpar):
        f0 = self.add_path(fsource)
        f1 = self.add_path(fdest)
        self.vid_tran.transcode(f0, f1, conpar)
                                   
    def create_probe_source(self):
        bn_upl_src = wg.FileUpload(description='Upload Source')
        bn_rm_src = wg.Button(
            description='Remove',
            tooltip='Remove source file',
            icon='remove')
        
        src_desc = wg.Label('Source file name: ')
        src_file = wg.Label('')
        probe_src1 = wg.Label('')
        probe_src2 = wg.Label('')

        bn_line = wg.HBox([bn_upl_src, bn_rm_src])
        src_fname = wg.HBox([src_desc, src_file])
        probe_top = wg.VBox([bn_line, src_fname, probe_src1, probe_src2])
        # pointers
        self.bn_upl_src = bn_upl_src
        self.src_file = src_file
        self.probe_src1 = probe_src1
        self.probe_src2 = probe_src2
        self.bn_rm_src = bn_rm_src
        self.src_file_line = src_fname  
        return probe_top

    def drive_probe_source(self):
        u = self.bn_upl_src
        s = self.src_file
        p1 = self.probe_src1
        p2 = self.probe_src2
        
        def handle_upl_src_change(change):
            if len(change.new) > 0:
                if len(change.old) > 0:
                    self.remove_vid(change.old[0]['name'])
                s.value = change.new[0]['name']
                self.write_vid(s.value, change.new[0]['content'])
                self.set_probe_info(s.value, p1, p2, 'source')
            else:
                s.value = ''
        u.observe(handle_upl_src_change, names='value')

        def on_bn_rm_src_clicked(b):
            if self.check_file_vid(s.value):
                self.remove_vid(s.value)
            else:
                s.value = ''
            u.value = ()
            p1.value = ''
            p2.value = ''
            self.cr_info.value = ''
                
        self.bn_rm_src.on_click(on_bn_rm_src_clicked)
                                    
    def create_transcoder(self):
        bn_transcode = wg.Button(
            description='Transcode',
            tooltip='Transcode',
            icon='flash')
        
        bn_rm_out = wg.Button(
            description='Remove',
            tooltip='Remove transcoded file',
            icon='remove')

        sel_output = wg.Dropdown(
            options=self.output_type,
            value='mp4',
            description='video format')

        sel_codec = wg.Dropdown(
            options=self.video_codec,
            value='libx264',
            description='codec')
        
        sel_pixel = wg.Dropdown(
            options=self.pixel_format,
            value='yuv420p',
            description='pixel format')
        
        crf = wg.IntSlider(
            value=17,
            min=0,
            max=51,
            step=1,
            description='crf',
            continous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d')

        tran_desc = wg.Label('Transcoded file name: ')
        tran_file = wg.Label('')
        dl_html = "<a href={tran_file} target='_blank'> (download) </a>"
        tran_download  = wg.HTML(value='',)
        tran_outname = wg.Text(
            description='new name',
            value='transcoded',
            placeholder='transcoded file name')
        dot = wg.Label('.')
        tran_type = wg.Label('')
                
        bn_line = wg.HBox([bn_transcode, bn_rm_out])
        tran_file_line = wg.HBox([tran_desc, tran_file, tran_download ])
        tran_new_name_line = wg.HBox([tran_outname, dot, tran_type])
        tran_cfg_line1 = wg.HBox([sel_output, sel_codec])
        tran_cfg_line2 = wg.HBox([sel_pixel, crf])

        probe_tran1 = wg.Label('')
        probe_tran2 = wg.Label('')

        out = wg.Output(
            layout={
                'width':'100%',
                'height':'auto',
                'max_height':'100px',
                'border': '1px solid gray',
                'overflow':'hidden scroll'})
        
        accor = wg.Accordion(children=[out], titles=['ffmpeg output'])
        
        tc_top = wg.VBox([
            bn_line,
            self.src_file_line,
            tran_file_line,
            tran_new_name_line,
            tran_cfg_line1,
            tran_cfg_line2,
            probe_tran1,
            probe_tran2,
            accor])
        
        self.sel_output = sel_output
        self.tran_type = tran_type
        self.tran_file = tran_file
        self.tran_outname = tran_outname
        self.sel_codec = sel_codec
        self.sel_pixel = sel_pixel
        self.crf = crf
        self.out = out
        self.dl_html = dl_html
        self.tran_download = tran_download
        self.bn_transcode = bn_transcode
        self.bn_rm_out = bn_rm_out
        self.probe_tran1 = probe_tran1
        self.probe_tran2 = probe_tran2
        return tc_top

    def drive_transcoder(self):
        f = self.tran_file
        t = self.tran_type
        o = self.tran_outname
        
        def clean_output():
            self.tran_download.value = ''
            self.probe_tran1.value = ''
            self.probe_tran2.value = ''
            self.out.clear_output()
        
        t.value = self.sel_output.value
        f.value = o.value + '.' + t.value
        
        dl = wg.dlink(
            (self.sel_output, 'value'),
            (t, 'value'))
        
        def handle_tran_type_change(change):
            f.value = o.value + '.' + change.new
            clean_output()

        def handle_tran_outname_change(change):
            f.value = change.new + '.' + t.value
            clean_output()

        def handle_src_file_change(change):
            if change.new == '':
                if self.check_file_vid(f.value):
                    self.remove_vid(f.value)
                f.value = ''     
                clean_output()
            else:
                f.value = o.value + '.' + t.value
                
        def handle_conf_change(change):
            clean_output()
            
        t.observe(handle_tran_type_change, names='value')
        o.observe(handle_tran_outname_change, names='value')
        self.src_file.observe(handle_src_file_change, names='value')
        self.sel_pixel.observe(handle_conf_change, names='value')
        self.sel_codec.observe(handle_conf_change, names='value')
        self.crf.observe(handle_conf_change, names='value')
    
        def on_bn_transcode_clicked(b):
            if self.src_file.value != '':
                conpar = {
                    'vcodec': self.sel_codec.value,
                    'preset': 'ultrafast',
                    'crf': self.crf.value,
                    'pix_fmt': self.sel_pixel.value}
                
                with self.out:
                    if self.check_file_vid(f.value):
                        self.remove_vid(f.value)
                    self.transcode_vid(self.src_file.value,
                                       f.value,
                                       conpar)
                    ht = self.dl_html.format(tran_file=self.add_path(f.value))
                    self.tran_download.value = ht
                    self.set_probe_info(f.value,
                                        self.probe_tran1,
                                        self.probe_tran2,
                                        'local')
                    self.sel_rec.value = 'local'
                    self.rec_file.value = self.tran_file.value
                    self.set_cr_info()
                    
        self.bn_transcode.on_click(on_bn_transcode_clicked)

        def on_bn_rm_out_clicked(b):
            if self.check_file_vid(f.value):
                self.remove_vid(f.value)
                clean_output()
            if self.sel_rec.value == 'local':
                self.rec_file.value = ''
                self.cr_info.value = ''
                
        self.bn_rm_out.on_click(on_bn_rm_out_clicked)
                                
    def create_comparer(self):
        bn_compare = wg.Button(
            description='Compare',
            tooltip='Compare source with reconstructed video',
            icon='play')
        bn_upl_ext = wg.FileUpload(description='Upload external')
        bn_rm_ext = wg.Button(
            description='Remove external',
            tooltip='Remove external video',
            icon='remove')
        sel_rec = wg.Dropdown(
            options=[('transcoded','local'),
                     ('external', 'external')],
            value='local',
            description='Use:',
            layout=wg.Layout(width='200px'))
        rec_desc = wg.Label('Reconstructed file name: ')
        rec_file = wg.Label('')        
        
        bn_line = wg.HBox([bn_compare, bn_upl_ext, bn_rm_ext])
        rec_line = wg.HBox([rec_desc, rec_file])
        probe_ext1 = wg.Label('')
        probe_ext2 = wg.Label('')
        '''
        mpl_win = wg.Output(layout={
                'width':'100%',
                'height':'auto'})
        '''
        mpl_win = wg.HTML()

        cr_info = wg.Label('')
        comp_line  = wg.HBox([sel_rec,cr_info])
        comp_top = wg.VBox([
            bn_line, self.src_file_line, rec_line, comp_line, probe_ext1, probe_ext2, mpl_win])
        self.sel_rec = sel_rec
        self.rec_file = rec_file
        self.bn_upl_ext = bn_upl_ext
        self.probe_ext1 = probe_ext1
        self.probe_ext2 = probe_ext2
        self.bn_rm_ext = bn_rm_ext
        self.mpl_win = mpl_win
        self.bn_compare = bn_compare
        self.cr_info = cr_info
        return comp_top
    
    def set_cr_info(self):
            if ((self.src_file.value != '') and
                (self.rec_file.value != '')):
                probe_src = self.probe_info['source']
                probe_rec = self.probe_info[self.sel_rec.value]
                cr_txt = 'Compression ratio: %.2f'%(probe_src['os_fsize']/probe_rec['os_fsize'])
                crzip_txt = 'Zip compression ratio: %.2f'%(probe_src['os_fsize']/probe_src['zip_fsize'])
                self.cr_info.value = ' | '.join([cr_txt, crzip_txt])
            else:
                self.cr_info.value = ''

    def drive_comparer(self):
        self.rec_file.value = ''
        self.probe_ext1.layout.visibility = 'hidden'
        self.probe_ext2.layout.visibility = 'hidden'


                
        def handle_sel_rec_change(change):
            if change.new == 'local':
                self.rec_file.value = self.tran_file.value
                self.probe_ext1.layout.visibility = 'hidden'
                self.probe_ext2.layout.visibility = 'hidden'                
            elif change.new == 'external':
                if len(self.bn_upl_ext.value) > 0:
                    self.rec_file.value = self.bn_upl_ext.value[0]['name']
                    self.probe_ext1.layout.visibility = 'visible'
                    self.probe_ext2.layout.visibility = 'visible'
                else:
                    self.rec_file.value = ''
            self.set_cr_info()
                    
        self.sel_rec.observe(handle_sel_rec_change, names='value')

        def handle_upl_ext_change(change):
            if len(change.new) > 0:
                if len(change.old) > 0:
                    self.remove_vid(change.old[0]['name'])
                self.write_vid(change.new[0]['name'],
                               change.new[0]['content'])
                self.set_probe_info(change.new[0]['name'],
                                    self.probe_ext1,
                                    self.probe_ext2,
                                    'external')
                self.sel_rec.value = 'external'
                self.probe_ext1.layout.visibility = 'visible'
                self.probe_ext2.layout.visibility = 'visible'
                self.rec_file.value = change.new[0]['name'] 
                
        self.bn_upl_ext.observe(handle_upl_ext_change, names='value')
                
        def on_bn_rm_ext_clicked(b):
            if len(self.bn_upl_ext.value)>0:
                if self.check_file_vid(self.bn_upl_ext.value[0]['name']):                
                    self.remove_vid(self.bn_upl_ext.value[0]['name'])
                    self.bn_upl_ext.value = ()
                    self.probe_ext1.value = ''
                    self.probe_ext2.value = ''
            if self.sel_rec.value == 'external': 
                self.rec_file.value = ''
                self.cr_info.value = ''
                                                
                                        
        self.bn_rm_src.on_click(on_bn_rm_ext_clicked)
        self.bn_rm_ext.on_click(on_bn_rm_ext_clicked)

        def handle_tran_file_change(change):
            if self.sel_rec.value == 'local':
                self.rec_file.value = change.new

                
        self.tran_file.observe(handle_tran_file_change, names='value')
        
            
        def on_bn_compare_clicked(b):
            print('bn_co_cl')
            if ((self.rec_file.value != '') or
                (self.src_file.value != '')):
                if hasattr(self.vid_comp, 'fig'):
                    self.vid_comp.fig.clear()
                    plt.close(self.vid_comp.fig)
                #self.mpl_win.clear_output()
                self.mpl_win.value = ''
                
            '''
            if ((self.rec_file.value == '') or
                (self.src_file.value == '')):
                with self.mpl_win:
                    txt = 'Warning: File missing, both source and reconstructed'
                    txt += ' files should be selected.'
                    print(txt)
            '''

            if ((self.rec_file.value != '') and
                (self.src_file.value != '')):
                self.set_cr_info()

                # with self.mpl_win:
                #    self.animate_plot()

                self.mpl_win.value = 'Creating animation... '
                ht5 = self.animate_plot()
                self.mpl_win.value = 'Animation done!'
                self.mpl_win.value = ht5

                self.vid_comp.close_vids()
                                                        
        self.bn_compare.on_click(on_bn_compare_clicked)

    def animate_plot(self):
        self.vid_comp.conf['source'] = self.src_file.value
        self.vid_comp.conf['recons'] = self.rec_file.value
        self.vid_comp.load_vids()
        g_success, geom_s, geom_r = self.vid_comp.check_geometry()
        if g_success:
            self.vid_comp.init_data()
            out = self.vid_comp.animate()
            #self.vid_comp.close_vids()
        else:
            out = None
            '''
            txt = 'Warning: Geometry mismatch, source and reconstructed'
            txt += ' files should have the same image size!'
            print(txt)
            print('source: %dx%d'%geom_s, '  reconstructed: %dx%d'%geom_r)
            '''
        #self.vid_comp.close_vids()
        return out 

        
                
    def create_gui(self):
        tab = wg.Tab()        
        titles = ['Probe Source','Transcode', 'Compare']
        ftab = [self.create_probe_source,
                self.create_transcoder,
                self.create_comparer]        
        tab.children = [ft() for ft in ftab]        
        tab.titles = titles
        self.top = tab
        
    def drive_gui(self):
        self.drive_probe_source()
        self.drive_transcoder()
        self.drive_comparer()
        
    def run(self):
        self.create_gui()
        self.drive_gui()
        display(self.top)

def run_gui():
    return CoViGui().run()

