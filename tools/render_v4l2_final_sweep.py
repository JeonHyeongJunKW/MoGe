"""Render Teacher, LoRA r4/r8 and direct-4 final outputs on identical V4L2 frames."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from moge.model.v2 import MoGeModel
from moge.utils.io import read_depth, read_image
from tools.v4l2_distillation import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs=3, type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    names = ['lora_r4', 'lora_r8', 'direct_4']
    models = []
    configs = {}
    for name, path in zip(names, a.checkpoints):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = MoGeModel(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model'], strict=True)
        model = model.cuda().eval()
        models.append(model)
        configs[name] = dict(checkpoint=str(path.resolve()), sha256=sha256(path),
                             lora=checkpoint['model_config'].get('lora'))
    manifest = json.loads((a.dataset/'dataset.json').read_text())
    assert manifest['format'] == 'moge_v4l2_multitask_v1'
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 19)
    cmap = matplotlib.colormaps['turbo']
    def depth_panel(d, valid):
        c = (cmap(np.clip(np.nan_to_num(d)/20, 0, 1))[...,:3]*255).astype(np.uint8)
        c[~valid] = 255
        return c
    def mask_panel(p, valid):
        x = (np.where(valid, p, 0).clip(0,1)*255).astype(np.uint8)
        return np.repeat(x[...,None],3,-1)
    def normal_panel(n,valid):
        n = np.nan_to_num(n)/np.maximum(np.linalg.norm(np.nan_to_num(n),axis=-1,keepdims=True),1e-6)
        c = ((n*.5+.5).clip(0,1)*255).astype(np.uint8)
        c[~valid] = 0
        return c
    summary = []
    with torch.inference_mode():
        for camera in ['left','front_left','right']:
            rows = sorted([r for r in manifest['samples'] if r['camera']==camera],
                          key=lambda r:int(Path(r['sample']).name))
            video = a.output/f'{camera}_final_models.mp4'
            cmd = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-f','rawvideo',
                   '-pix_fmt','rgb24','-s','2400x960','-r','2','-i','-','-an','-vf','fps=30',
                   '-c:v','libx264','-threads','2','-preset','fast','-crf','19',
                   '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
            proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
            try:
                for i,row in enumerate(rows):
                    folder=a.dataset/row['sample']
                    rgb=read_image(folder/'image.jpg')
                    x=torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255
                    pred=[m.infer(x,num_tokens=400,use_fp16=True,apply_mask=False,
                                  return_mask_probability=True) for m in models]
                    rgbpanel=Image.fromarray(rgb).resize((480,270))
                    panels=[]
                    with np.load(folder/'teacher_labels.npz') as teacher:
                        rect=teacher['mask_valid'];depth=read_depth(folder/'depth.png')
                        panels.append((depth_panel(depth,np.isfinite(depth)),
                                       mask_panel(teacher['mask_probability'],rect),
                                       normal_panel(teacher['normal'],teacher['normal_valid'])))
                        for output in pred:
                            depth=output['depth'].cpu().numpy();mask=output['mask'].cpu().numpy()
                            normal=output['normal'].cpu().numpy();prob=output['mask_probability'].cpu().numpy()
                            valid=rect & mask & np.isfinite(depth) & (depth>=.1) & (depth<=100)
                            panels.append((depth_panel(depth,valid),mask_panel(prob,rect),
                                           normal_panel(normal,rect & mask & np.isfinite(normal).all(-1))))
                    canvas=Image.new('RGB',(2400,960),'#102332');draw=ImageDraw.Draw(canvas)
                    draw.text((15,8),f'V4L2 {camera} {i+1}/84 | final checkpoints | 252 TRAIN frames, Teacher agreement only',font=font,fill='white')
                    titles=['RGB','Teacher','LoRA r4 final','LoRA r8 final','direct last4 final']
                    for col,title in enumerate(titles):draw.text((col*480+12,43),title,font=font,fill='white')
                    canvas.paste(rgbpanel,(0,76));canvas.paste(rgbpanel,(0,370));canvas.paste(rgbpanel,(0,664))
                    for col,three in enumerate(panels,1):
                        for rowidx,panel in enumerate(three):
                            canvas.paste(Image.fromarray(panel).resize((480,270)),(col*480,76+rowidx*294))
                    for rowidx,label in enumerate(['DEPTH (m) | 0-20', 'VALID PROBABILITY | 0-1', 'NORMAL | camera XYZ']):
                        y=76+rowidx*294
                        draw.rectangle((0,y,480,y+26),fill='#102332')
                        draw.text((10,y+2),label,font=font,fill='white')
                    proc.stdin.write(np.asarray(canvas).tobytes())
                    if i==len(rows)//2:canvas.save(a.output/f'preview_{camera}.jpg',quality=93)
                    if (i+1)%20==0:print(camera,i+1,'/',len(rows),flush=True)
            finally:
                proc.stdin.close();code=proc.wait()
            if code:raise RuntimeError(f'ffmpeg exited {code}')
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,nb_frames,duration,codec_name','-of','json',str(video)]))['streams'][0]
            assert int(probe['nb_frames'])==len(rows)*15
            assert abs(float(probe['duration'])-len(rows)/2)<.01
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
            summary.append(dict(camera=camera,frames=len(rows),video=str(video),probe=probe))
            print('Validated',video,flush=True)
    (a.output/'metadata.json').write_text(json.dumps(dict(models=configs,dataset=str(a.dataset),
        teacher_sha256=manifest['settings']['teacher_sha256'],videos=summary,
        display='Depth shared 0-20m scale, positive teacher/student geometry mask and 0.1-100m range; mask raw sigmoid; normal RGB XYZ',
        limitation='252 training frames, no ground truth or independent validation'),indent=2)+'\n')

if __name__=='__main__':main()
