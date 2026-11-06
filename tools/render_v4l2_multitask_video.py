"""Render cached Teacher and a Student on the same V4L2 training frames."""
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
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = MoGeModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.cuda().eval()
    manifest = json.loads((a.dataset / 'dataset.json').read_text())
    assert manifest['format'] == 'moge_v4l2_multitask_v1'
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
    cmap = matplotlib.colormaps['turbo']

    def depth_color(d, valid):
        colored = (cmap(np.clip(np.nan_to_num(d) / 20, 0, 1))[..., :3] * 255).astype(np.uint8)
        colored[~valid] = 255
        return colored

    def normal_color(n, valid):
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        colored = (np.clip(np.nan_to_num(n) * .5 + .5, 0, 1) * 255).astype(np.uint8)
        colored[~valid] = 0
        return colored

    videos = []
    with torch.inference_mode():
        for camera in ('left', 'front_left', 'right'):
            rows = sorted([r for r in manifest['samples'] if r['camera'] == camera], key=lambda r: int(Path(r['sample']).name))
            video = a.output / f'v4l2_{camera}_best_lora_4.mp4'
            command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'rawvideo',
                       '-pix_fmt', 'rgb24', '-s', '1920x700', '-r', '2', '-i', '-', '-an', '-vf', 'fps=30',
                       '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '19',
                       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
            proc = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for i, row in enumerate(rows):
                    folder = a.dataset / row['sample']
                    rgb = read_image(folder / 'image.jpg')
                    pred = model.infer(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda()/255,
                                       num_tokens=400, use_fp16=True, apply_mask=False, return_mask_probability=True)
                    depth, mask, normal, probability = [pred[k].cpu().numpy() for k in ('depth', 'mask', 'normal', 'mask_probability')]
                    with np.load(folder / 'teacher_labels.npz') as teacher:
                        valid = teacher['mask_valid']
                        td = read_depth(folder / 'depth.png')
                        student_valid = valid & mask & np.isfinite(depth) & (depth >= .1) & (depth <= 100)
                        sm = np.repeat((np.where(valid, probability, 0)*255).astype(np.uint8)[..., None], 3, -1)
                        tm = np.repeat((np.where(valid, teacher['mask_probability'], 0)*255).astype(np.uint8)[..., None], 3, -1)
                        panels = [rgb, depth_color(td, np.isfinite(td)), tm,
                                  normal_color(teacher['normal'], teacher['normal_valid']),
                                  rgb, depth_color(depth, student_valid), sm,
                                  normal_color(normal, valid & mask & np.isfinite(normal).all(-1))]
                    canvas = Image.new('RGB', (1920, 700), '#102332')
                    draw = ImageDraw.Draw(canvas)
                    draw.text((14, 8), f'V4L2 / {camera} / {i+1}/{len(rows)} / TRAIN frames - not independent validation', font=font, fill='white')
                    labels = ['Teacher input', 'Teacher depth', 'Teacher valid probability', 'Teacher normal',
                              'Student input', 'LoRA r4 depth', 'LoRA r4 valid probability', 'LoRA r4 normal']
                    for j, (panel, title) in enumerate(zip(panels, labels)):
                        x, y = (j % 4)*480, 45+(j//4)*310
                        draw.text((x+10, y), title, font=font, fill='white')
                        canvas.paste(Image.fromarray(panel).resize((480, 270)), (x, y+30))
                    draw.text((14, 669), 'Depth: common 0-20m scale (saturated above 20m), white=excluded | Mask: black=0, white=1 | Normal: camera XYZ', font=font, fill='white')
                    proc.stdin.write(np.asarray(canvas).tobytes())
                    if i == len(rows)//2:
                        canvas.save(a.output / f'preview_{camera}.jpg', quality=95)
                    if (i+1) % 20 == 0:
                        print(f'{camera}: {i+1}/{len(rows)}', flush=True)
            finally:
                proc.stdin.close()
                code = proc.wait()
            if code:
                raise RuntimeError(f'ffmpeg exited {code}')
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                               '-show_entries', 'stream=width,height,nb_frames,duration,codec_name', '-of', 'json', str(video)]))
            stream = probe['streams'][0]
            assert int(stream['nb_frames']) == len(rows)*15
            assert abs(float(stream['duration'])-len(rows)/2) < .01
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
            videos.append(dict(camera=camera, source_frames=len(rows), video=str(video), probe=probe))
            print(f'Validated: {video}', flush=True)
    metadata = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=sha256(a.checkpoint),
                    checkpoint_validation=checkpoint.get('validation'), model_config=checkpoint['model_config'],
                    dataset=str(a.dataset), teacher_sha256=manifest['settings']['teacher_sha256'],
                    student_tokens=400, teacher_tokens=manifest['settings']['num_tokens'], precision='fp16',
                    source_fps=2, output_fps=30, timing='Sampled frames replayed at fixed 2 fps; not exact source timestamps',
                    scope='252 V4L2 TRAIN frames; no CARLA evaluation or independent accuracy estimate',
                    depth_display='both masked with geometry, rectification, finite, 0.1-100m; shared color 0-20m',
                    videos=videos)
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    main()
