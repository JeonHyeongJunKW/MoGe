"""Compare a metric Large Teacher and three final Students on new rectified RGB."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from moge.model.v2 import MoGeModel
from moge.utils.io import read_image
from tools.v4l2_distillation import sha256
matplotlib.use('Agg')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--students',nargs=4,type=Path,required=True)
    p.add_argument('--student-names',nargs=4,default=['initial_student','lora_r4','lora_r8','direct_4'])
    p.add_argument('--student-titles',nargs=4,default=['Initial Student','LoRA r4 final','LoRA r8 final','direct last4 final'])
    p.add_argument('--video-name',default='front_left_initial_vs_final.mp4')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('test','train'),default='test')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    info=json.loads((a.dataset/'dataset.json').read_text())
    assert info['format']=='moge_v4l2_rgb_v1'
    rows=sorted((r for r in info['samples'] if r['camera']=='front_left' and r['split']==a.split),
                key=lambda r:int(Path(r['sample']).name))
    assert rows and len({r['session'] for r in rows})==1
    assert len(set(a.student_names))==4 and 'teacher' not in a.student_names
    session=rows[0]['session']
    torch.set_num_threads(2)
    paths=dict(teacher=a.teacher, **dict(zip(a.student_names,a.students)))
    models={};configs={}
    for name,path in paths.items():
        c=torch.load(path,map_location='cpu',weights_only=True)
        m=MoGeModel(**c['model_config']);m.load_state_dict(c['model'],strict=True)
        models[name]=m.cuda().eval()
        configs[name]=dict(path=str(path.resolve()),sha256=sha256(path),tokens=1200 if name=='teacher' else 400,
                           lora=c['model_config'].get('lora'))
    cmap=matplotlib.colormaps['turbo'];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19)
    def depth_panel(d,v):
        c=(cmap(np.clip(np.nan_to_num(d)/20,0,1))[...,:3]*255).astype(np.uint8)
        c[~v]=255
        return c
    def mask_panel(x,v):
        c=(np.where(v,x,0).clip(0,1)*255).astype(np.uint8)
        return np.repeat(c[...,None],3,-1)
    def normal_panel(n,v):
        n=np.nan_to_num(n);n=n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-6)
        c=((n*.5+.5).clip(0,1)*255).astype(np.uint8)
        c[~v]=0
        return c
    video=a.output/a.video_name
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24',
         '-s','2880x960','-r','2','-i','-','-an','-vf','fps=30','-c:v','libx264','-threads','2',
         '-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    metrics=[]
    try:
        with torch.inference_mode():
            for i,row in enumerate(rows):
                folder=a.dataset/row['sample'];rgb=read_image(folder/'image.jpg')
                rect=__import__('cv2').imread(str(folder/'rectification_valid.png'),0)>0
                x=torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255
                out={}
                for name,m in models.items():
                    pred=m.infer(x,num_tokens=configs[name]['tokens'],use_fp16=True,
                                 apply_mask=False,return_mask_probability=True)
                    out[name]={k:pred[k].cpu().numpy() for k in ('depth','mask','normal','mask_probability')}
                teacher=out['teacher'];td=teacher['depth'];tp=teacher['mask_probability'];tm=teacher['mask']
                tv=rect & tm & np.isfinite(td) & (td>=.1) & (td<=100)
                frame={'sample':row['sample'],'teacher_valid_pixels':int(tv.sum())}
                panels=[]
                for name,prd in out.items():
                    d=prd['depth'];mask=prd['mask'];normal=prd['normal'];prob=prd['mask_probability']
                    dv=rect & mask & np.isfinite(d) & (d>=.1) & (d<=100)
                    nv=rect & mask & np.isfinite(normal).all(-1)
                    panels.append((depth_panel(d,dv),mask_panel(prob,rect),normal_panel(normal,nv)))
                    if name=='teacher':continue
                    both=tv & np.isfinite(d) & (d>0)
                    if both.any():
                        ratio=np.maximum(d[both]/td[both],td[both]/d[both])
                        frame[name+'/depth_rel']=float(np.mean(np.abs(d[both]-td[both])/td[both]))
                        frame[name+'/depth_delta1']=float(np.mean(ratio<1.25))
                    frame[name+'/depth_coverage']=float(both.sum()/max(tv.sum(),1))
                    frame[name+'/mask_agreement']=float(np.mean((prob[rect]>.5)==(tp[rect]>.5)))
                    frame[name+'/mask_negative_recall']=float(np.mean(prob[rect & (tp<=.5)]<=.5)) if (rect & (tp<=.5)).any() else None
                    normal_valid=tv & nv & np.isfinite(teacher['normal']).all(-1)
                    if normal_valid.any():
                        dot=np.sum(normal[normal_valid]*teacher['normal'][normal_valid],axis=-1)
                        norms=np.linalg.norm(normal[normal_valid],axis=-1)*np.linalg.norm(teacher['normal'][normal_valid],axis=-1)
                        frame[name+'/normal_angle_deg']=float(np.degrees(np.arccos(np.clip(dot/np.maximum(norms,1e-6),-1,1))).mean())
                metrics.append(frame)
                canvas=Image.new('RGB',(2880,960),'#102332');draw=ImageDraw.Draw(canvas)
                draw.text((12,8),f'{session} front_left / {a.split.upper()} / {i+1}/{len(rows)} / Teacher reference, no ground truth',font=font,fill='white')
                for col,title in enumerate(['RGB','Large Teacher',*a.student_titles]):
                    draw.text((col*480+12,43),title,font=font,fill='white')
                image=Image.fromarray(rgb).resize((480,270))
                for j in range(3):canvas.paste(image,(0,76+j*294))
                for col,three in enumerate(panels,1):
                    for j,panel in enumerate(three):canvas.paste(Image.fromarray(panel).resize((480,270)),(col*480,76+j*294))
                for j,label in enumerate(['DEPTH (m) | 0-20','VALID PROBABILITY | 0-1','NORMAL | camera XYZ']):
                    y=76+j*294;draw.rectangle((0,y,480,y+26),fill='#102332');draw.text((10,y+2),label,font=font,fill='white')
                proc.stdin.write(np.asarray(canvas).tobytes())
                if i==len(rows)//2:canvas.save(a.output/'preview_front_left.jpg',quality=93)
                if (i+1)%10==0:print(i+1,'/',len(rows),flush=True)
    finally:
        proc.stdin.close();code=proc.wait()
    if code:raise RuntimeError(f'ffmpeg exited {code}')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_frames,duration,codec_name','-of','json',str(video)]))['streams'][0]
    assert int(probe['nb_frames'])==len(rows)*15
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    keys=sorted({k for row in metrics for k in row if '/' in k})
    means={k:float(np.mean([r[k] for r in metrics if r.get(k) is not None])) for k in keys}
    report=dict(scope=f'{session} front_left RGB {a.split} frames',reference='Large Teacher predictions; not GT',
                samples=len(rows),models=configs,mean=means,per_frame=metrics,video=str(video),probe=probe,
                policy='FP16; teacher 1200/student 400 tokens; no FOV override; mask probability >0.5; depth evaluated at teacher-valid rectified 0.1-100m pixels without scale alignment; normals compared on common valid pixels',
                caveat='Teacher agreement and visual comparison are not absolute accuracy; no camera-aligned depth GT in this evaluation')
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(means,indent=2),flush=True)

if __name__=='__main__':main()
