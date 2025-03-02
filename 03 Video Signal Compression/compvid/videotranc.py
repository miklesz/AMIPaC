'''video_tranc.py

Video Transcoding and Compression Experiment

Andres Vejar, 2023

requirements
 - ffpmeg-python (ffmpeg installed)
    pip install ffmpeg-python
    info in https://www.streamingmedia.com/Articles/ReadArticle.aspx?ArticleID=152090
 - zlib (for zip)
 - opencv
'''
import os
import ffmpeg
import zipfile

class TCoVid():
    def __init__(self):
        self.keys_sel = [
            'codec_name',
            'codec_long_name',
            'width',
            'height',
            'pix_fmt',
            'r_frame_rate',
            'bit_rate',
            'duration',
            'nb_frames']
        self.conpar = {
            'vcodec':'libx265',
            'preset':'ultrafast',
            'crf':28, 
            'pix_fmt':'yuv420p'
            }

    def cal_file_size(self, video_info):
        # in MB
        vi = video_info
        br = float(vi['bit_rate'])
        dr = float(vi['duration'])                   
        return ((br*dr)/8.0)/(1024**2)

    def zip_file_size(self, vid_fname):
        zipname = vid_fname + '.zip'
        with zipfile.ZipFile(
                file=zipname,
                mode='w',
                compression=zipfile.ZIP_DEFLATED) as myzip:
            myzip.write(vid_fname)
        zipsize = os.stat(zipname).st_size/(1024**2)
        os.remove(zipname)
        return zipsize

    def os_file_size(self, vid_fname):
        return os.stat(vid_fname).st_size/(1024**2)
        
    def probe(self, vid_fname):
        probe = ffmpeg.probe(vid_fname)
        video_info = next(
            (stream
             for stream
             in probe['streams']
             if stream['codec_type'] == 'video'),
             None)        
        vid = {}
        for k in self.keys_sel:
            if k in video_info.keys():
                vid[k] = video_info[k]
        if 'bit_rate' in video_info.keys():
            vid['cal_fsize'] = self.cal_file_size(video_info)
        vid['os_fsize'] = self.os_file_size(vid_fname)
        vid['zip_fsize'] = self.zip_file_size(vid_fname)               
        return vid

    def probe_reduc(self, vid_fname):
        v = self.probe(vid_fname)
        return self.reduc_probeinfo(probe_reduc)
    
    def reduc_probeinfo(self, info):
        v = info
        r = {}
        cdx = tuple([v[x] for x in ['codec_name', 'codec_long_name']])
        r['codec'] = '[%s]  %s'%cdx
        r['pixel format'] = v['pix_fmt']
        r['geometry'] = "%sx%s"%(v['width'],v['height'])
        r['fps'] = v['r_frame_rate'].replace('/1','')
        r['OS file size'] = '%.2f MB'%v['os_fsize']        
        r['Zip file size'] = '%.2f MB'%v['zip_fsize']
        return r
        
    def print_probe(self, vid):
        for k in vid.keys():
            print(k.ljust(25), vid[k])

    def transcode(self, vid_fname, vid_outname, conpar=None):
        stream = ffmpeg.input(vid_fname)
        if conpar is None:
            conpar = self.conpar
        stream = ffmpeg.output(
            stream,
            vid_outname,
            **conpar)
        stream.run()
        
if __name__ == '__main__':
    v = CoVid()
    v.print_probe(v.probe('vid/agh_src1_hrc0.m4v'))
    v.transcode('vid/agh_src1_hrc0.m4v', 'vid/out.avi')
    v.print_probe(v.probe('vid/out.avi'))
