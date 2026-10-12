import {parsePLY} from './ply.js';
import {rotate, sphere, trackball, dot} from './trackball.js';
const $ = id => document.getElementById(id);
const status = (text, error = false) => { $('status').textContent = text; $('status').classList.toggle('error', error); };
const camera = {target: [0, 0, 0], distance: 5, orientation: [0, 0, 0, 1], radius: 1};
let clouds = [], samples = [], requestID = 0;
let localFiles = [null, null];
const tint = [[0, 200 / 255, 1], [1, 120 / 255, 0]];
const vertex = `attribute vec3 position; attribute vec3 color;
uniform vec3 target, right, up, back, tint; uniform float distance, nearPlane, farPlane, aspect, size, solid, maxDepth;
varying vec3 rgb; varying float visible;
void main(){vec3 p=vec3(position.x,-position.y,-position.z)-target; float z=distance-dot(p,back);
 gl_Position=vec4(dot(p,right)*2.41421356/aspect,dot(p,up)*2.41421356,((farPlane+nearPlane)*z-2.0*farPlane*nearPlane)/(farPlane-nearPlane),z);
 gl_PointSize=size; rgb=mix(color,tint,solid); visible=maxDepth<=0.0||position.z<=maxDepth?1.0:0.0;}`;
const fragment = `precision mediump float; varying vec3 rgb; varying float visible;
void main(){if(visible<0.5||length(gl_PointCoord-vec2(0.5))>0.5)discard; gl_FragColor=vec4(rgb,1.0);}`;
function axes() {
  return {right: rotate(camera.orientation,[1,0,0]), up: rotate(camera.orientation,[0,1,0]), back: rotate(camera.orientation,[0,0,1])};
}
function axisLength() { return Math.max(.001, Number($('axisLength').value)||1); }
function drawAxes(layer, rect, dpr) {
  layer.width=Math.round(rect.width*dpr);layer.height=Math.round(rect.height*dpr);
  layer.style.width=`${rect.width}px`;layer.style.height=`${rect.height}px`;
  const ctx=layer.getContext('2d');ctx.scale(dpr,dpr);
  if(!$('showAxes').checked)return;
  const basis=axes(), length=axisLength(), focal=rect.height*1.20710678;
  const view=p=>{const d=p.map((v,i)=>v-camera.target[i]);return [dot(d,basis.right),dot(d,basis.up),camera.distance-dot(d,basis.back)];};
  const project=p=>[rect.width/2+p[0]*focal/p[2],rect.height/2-p[1]*focal/p[2]];
  const origin=view([0,0,0]), near=Math.max(camera.radius*.00001,.000001);
  ctx.font='bold 13px system-ui';ctx.lineWidth=2;ctx.lineCap='round';
  for(const [endpoint,color,label] of [[[length,0,0],'#ff6464','X'],[[0,-length,0],'#63e887','Y'],[[0,0,-length],'#619eff','Z (optical)']]){
    let start=[...origin],end=view(endpoint);
    if(start[2]<=near && end[2]<=near)continue;
    const endVisible=end[2]>near;
    if(start[2]<=near){const t=(near-start[2])/(end[2]-start[2]);start=start.map((v,i)=>v+t*(end[i]-v));}
    if(end[2]<=near){const t=(near-end[2])/(start[2]-end[2]);end=end.map((v,i)=>v+t*(start[i]-v));}
    const a=project(start),b=project(end);
    ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();
    if(endVisible){
      const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);
      ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-9*Math.cos(angle-.4),b[1]-9*Math.sin(angle-.4));ctx.lineTo(b[0]-9*Math.cos(angle+.4),b[1]-9*Math.sin(angle+.4));ctx.closePath();ctx.fill();
      ctx.fillText(label,b[0]+7,b[1]-7);
    }
  }
  if(origin[2]>near){const p=project(origin);ctx.fillStyle='#ffffff';ctx.beginPath();ctx.arc(...p,3,0,Math.PI*2);ctx.fill();ctx.fillText('O · camera',p[0]+7,p[1]+16);}
}
function renderer(canvas) {
  const layer=document.createElement('canvas');layer.className='axis-overlay';layer.setAttribute('aria-hidden','true');canvas.after(layer);
  const gl = canvas.getContext('webgl', {antialias: true});
  if (!gl) throw Error('WebGL을 사용할 수 없습니다. 브라우저의 하드웨어 가속을 확인하세요.');
  function shader(type, source) {
    const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s));
    return s;
  }
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, vertex)); gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
  const uniforms = Object.fromEntries(['target','right','up','back','tint','distance','nearPlane','farPlane','aspect','size','solid','maxDepth'].map(n => [n, gl.getUniformLocation(program,n)]));
  const position = gl.getAttribLocation(program,'position'), color = gl.getAttribLocation(program,'color');
  let buffers = [];
  return {
    upload() { buffers.forEach(b => gl.deleteBuffer(b)); buffers = clouds.map(c => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,b); gl.bufferData(gl.ARRAY_BUFFER,c.data,gl.STATIC_DRAW); return b; }); },
    draw(indices, overlay) {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      canvas.width = Math.round(rect.width*dpr); canvas.height = Math.round(rect.height*dpr);
      gl.viewport(0,0,canvas.width,canvas.height); gl.clearColor(.035,.06,.09,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.useProgram(program);
      const a = axes();
      for (const n of ['right','up','back']) gl.uniform3fv(uniforms[n],a[n]);
      gl.uniform3fv(uniforms.target,camera.target); gl.uniform1f(uniforms.distance,camera.distance);
      gl.uniform1f(uniforms.nearPlane,Math.max(camera.radius*0.00001,camera.distance-camera.radius*1.2));
      gl.uniform1f(uniforms.farPlane,camera.distance+camera.radius*2+1);
      gl.uniform1f(uniforms.aspect,canvas.width/canvas.height); gl.uniform1f(uniforms.size,Number($('size').value)*dpr);
      gl.uniform1f(uniforms.solid,overlay || $('color').value==='solid' ? 1 : 0);
      gl.uniform1f(uniforms.maxDepth,Math.max(0,Number($('depth').value)||0));
      for (const i of indices) {
        if (!buffers[i] || !(i===0 ? $('showGT').checked : $('showPred').checked)) continue;
        gl.bindBuffer(gl.ARRAY_BUFFER,buffers[i]);
        gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position,3,gl.FLOAT,false,24,0);
        gl.enableVertexAttribArray(color); gl.vertexAttribPointer(color,3,gl.FLOAT,false,24,12);
        gl.uniform3fv(uniforms.tint,tint[i]); gl.drawArrays(gl.POINTS,0,clouds[i].count);
      }
      layer.style.top=`${canvas.offsetTop}px`;drawAxes(layer,rect,dpr);
    }
  };
}
let renderers;
function draw() {
  if (!renderers) return;
  const overlay = $('mode').value==='overlay';
  $('views').classList.toggle('overlay',overlay); $('rightPane').hidden=overlay;
  $('leftTitle').firstChild.textContent=overlay?'Overlay · GT 청록 / Prediction 주황 ':'GT ';
  renderers[0].draw(overlay?[0,1]:[0],overlay);
  if (!overlay) renderers[1].draw([1],false);
}
function fit() {
  const lo=[Infinity,Infinity,Infinity], hi=[-Infinity,-Infinity,-Infinity];
  const depth=Number($('depth').value)||Infinity;
  let found=false;
  clouds.forEach((cloud,index) => {
    if (!(index===0 ? $('showGT').checked : $('showPred').checked)) return;
    const d=cloud.data;
    for(let i=0;i<d.length;i+=6){
      if(depth>0 && d[i+2]>depth)continue;
      found=true;
      for(let k=0;k<3;k++){const v=d[i+k]*(k===0?1:-1);lo[k]=Math.min(lo[k],v);hi[k]=Math.max(hi[k],v);}
    }
  });
  if(!found){status('표시할 점이 없습니다. 깊이 제한과 GT / Prediction 표시를 확인하세요.',true);draw();return;}
  if($('showAxes').checked){
    const length=axisLength();
    for(const point of [[0,0,0],[length,0,0],[0,-length,0],[0,0,-length]])
      for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],point[k]);hi[k]=Math.max(hi[k],point[k]);}
  }
  camera.target=lo.map((v,k)=>(v+hi[k])/2);
  camera.radius=Math.max(Math.hypot(...lo.map((v,k)=>hi[k]-v))/2,.001);
  const rect=$('left').getBoundingClientRect();
  const halfAngle=Math.atan(Math.tan(Math.PI/8)*Math.min(1,rect.width/rect.height));
  camera.distance=camera.radius/Math.sin(halfAngle)*1.08;
  draw();
}
function install(data) {
  clouds=data; renderers.forEach(r=>r.upload());
  $('gtCount').textContent=`${data[0].count.toLocaleString()} points`;
  $('predCount').textContent=`${data[1].count.toLocaleString()} points`;
  camera.orientation=[0,0,0,1];fit();
}
function summary(data){return data.map((c,i)=>`${i?'Pred':'GT'} ${c.count.toLocaleString()} / ${c.originalCount.toLocaleString()}점${c.step>1?' (표시용 다운샘플링)':''}${c.omitted?` · 비유한 좌표 ${c.omitted}점 제외`:''}`).join(' · ');}
function clearData() {
  clouds=[]; renderers.forEach(r=>r.upload()); $('rgb').hidden=true; $('rgb').removeAttribute('src'); $('info').textContent='';
  $('gtCount').textContent='';$('predCount').textContent='';draw();
}
const asset=(sample,name)=>`/api/asset?${new URLSearchParams({sample,name})}`;
async function checked(url){const r=await fetch(url);if(!r.ok)throw Error(`파일을 읽을 수 없습니다 (${r.status}). 결과 파일을 확인하세요.`);return r;}
async function loadSample() {
  const sample=$('sample').value;if(!sample)return;
  localFiles=[null,null];updateDropNames();
  const id=++requestID;clearData();status('PLY 불러오는 중…');
  try{
    const data=await Promise.all(['gt.ply','pred.ply'].map(async name=>parsePLY(await (await checked(asset(sample,name))).arrayBuffer())));
    if(id!==requestID)return;install(data);status(`${sample} · ${summary(data)}`);
    $('rgb').src=asset(sample,'image.png');$('rgb').hidden=false;
    const info=await fetch(asset(sample,'info.json'));
    const metadata=info.ok?JSON.stringify(await info.json(),null,2):'info.json 없음 · 원본 좌표와 스케일로 표시합니다.';
    if(id!==requestID)return;
    $('info').textContent=metadata;
  }catch(e){if(id===requestID)status(e.message,true);}
}
function filterSamples() {
  const previous=$('sample').value, q=$('search').value.toLowerCase();
  const filtered=samples.filter(s=>s.toLowerCase().includes(q));
  $('sample').replaceChildren(new Option(`결과 선택 (${filtered.length}개)`,''),...filtered.map(s=>new Option(s,s)));
  if(filtered.includes(previous))$('sample').value=previous;
}
async function refresh(){try{const r=await checked('/api/samples');samples=(await r.json()).samples;filterSamples();status(`${samples.length}개 결과를 찾았습니다. ${samples.length?'결과를 선택하세요.':'--root 경로를 확인하거나 로컬 PLY 파일을 선택하세요.'}`);}catch(e){status(e.message,true);}}
function updateDropNames(){
  ['gtDropName','predDropName'].forEach((id,i)=>$(id).textContent=localFiles[i]?.name??(i?'pred.ply':'gt.ply'));
}
async function loadLocalFiles(){
  const files=[...localFiles],id=++requestID;updateDropNames();clearData();status('로컬 PLY 읽는 중…');
  const empty=()=>({data:new Float32Array(),count:0,originalCount:0,omitted:0,step:1});
  try{
    const data=await Promise.all(files.map(async f=>{
      if(!f)return empty();
      if(f.size>256*1024*1024)throw Error('파일당 최대 256 MiB를 지원합니다.');
      return parsePLY(await f.arrayBuffer());
    }));
    if(id!==requestID)return;
    install(data);status(`${summary(data)}${files.some(f=>!f)?' · 나머지 PLY를 드롭해 비교하세요.':''}`);
    $('info').textContent=`GT: ${files[0]?.name??'없음'}\nPrediction: ${files[1]?.name??'없음'}\n원본 스케일 유지. 카메라 축: X 오른쪽 / Y 아래 / Z 전방.`;
  }catch(e){if(id===requestID)status(e.message,true);}
}
try{
  renderers=[renderer($('left')),renderer($('right'))];
  for(const id of ['left','right']){
    const canvas=$(id);let drag=null;
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.button===1||e.shiftKey};});
    canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
    canvas.addEventListener('pointermove',e=>{
      if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;const previous={x:drag.x,y:drag.y};drag.x=e.clientX;drag.y=e.clientY;
      if(drag.pan||e.shiftKey){const a=axes(),scale=2*camera.distance*Math.tan(Math.PI/8)/canvas.clientHeight;camera.target=camera.target.map((v,k)=>v-dx*scale*a.right[k]+dy*scale*a.up[k]);}
      else{const rect=canvas.getBoundingClientRect();camera.orientation=trackball(camera.orientation,sphere(previous.x,previous.y,rect),sphere(e.clientX,e.clientY,rect));}draw();
    });
    canvas.addEventListener('wheel',e=>{
      e.preventDefault();
      // Normalize wheel units; reduce sensitivity fivefold. Shift allows fine adjustment.
      const unit=e.deltaMode===1?16:e.deltaMode===2?canvas.clientHeight:1;
      const delta=Math.max(-100,Math.min(100,e.deltaY*unit));
      const sensitivity=e.shiftKey?.0001:.0004;
      camera.distance=Math.max(camera.radius*.001,Math.min(camera.radius*1000,camera.distance*Math.exp(delta*sensitivity)));
      draw();
    },{passive:false});
  }
  $('refresh').onclick=refresh;$('load').onclick=loadSample;$('sample').onchange=loadSample;$('search').oninput=filterSamples;
  for(const [id,delta] of [['prev',-1],['next',1]])$(id).onclick=()=>{const s=$('sample');s.selectedIndex=Math.max(1,Math.min(s.options.length-1,s.selectedIndex+delta));loadSample();};
  for(const id of ['mode','color','size','depth','showGT','showPred'])$(id).oninput=draw;
  $('showAxes').onchange=()=>clouds.length?fit():draw();$('axisLength').onchange=()=>clouds.length?fit():draw();
  $('fit').onclick=fit;$('front').onclick=()=>{camera.orientation=[0,0,0,1];draw();};
  $('rgb').onerror=()=>$('rgb').hidden=true;
  $('local').onclick=()=>{
    const files=[$('gtFile').files[0],$('predFile').files[0]];
    if(files.some(f=>!f)){status('GT와 Prediction 파일을 모두 선택하세요.',true);return;}
    localFiles=files;loadLocalFiles();
  };
  let dragDepth=0;
  window.addEventListener('dragenter',e=>{if(e.dataTransfer.types.includes('Files')){e.preventDefault();dragDepth++;document.body.classList.add('dragging');}});
  window.addEventListener('dragleave',()=>{if(--dragDepth<=0){dragDepth=0;document.body.classList.remove('dragging');}});
  window.addEventListener('dragover',e=>{if(e.dataTransfer.types.includes('Files')){e.preventDefault();e.dataTransfer.dropEffect='copy';}});
  window.addEventListener('drop',e=>{
    e.preventDefault();dragDepth=0;document.body.classList.remove('dragging');
    const files=[...e.dataTransfer.files];
    if(!files.length)return;
    if(files.length>2 || files.some(f=>!f.name.toLowerCase().endsWith('.ply'))){status('PLY 파일 한 개 또는 GT / Prediction 두 개를 드롭하세요.',true);return;}
    const role=f=>/(^|[_.-])(gt|ground.?truth)([_.-]|$)/i.test(f.name)?0:/(^|[_.-])(pred|prediction)([_.-]|$)/i.test(f.name)?1:null;
    if(files.length===2){
      const roles=files.map(role);
      if(roles[0]!==null && roles[0]===roles[1]){status('같은 역할의 파일 두 개입니다. 각각의 드롭 영역에 한 개씩 놓으세요.',true);return;}
      const first=roles[0]??(roles[1]===null?0:1-roles[1]);
      localFiles=[null,null];localFiles[first]=files[0];localFiles[1-first]=files[1];
    }else{
      const zone=e.target.closest('[data-slot]');
      const slot=zone?Number(zone.dataset.slot):role(files[0]);
      if(slot===null){status('GT 또는 Prediction 드롭 영역에 파일을 놓으세요.',true);return;}
      localFiles[slot]=files[0];
    }
    loadLocalFiles();
  });
  new ResizeObserver(draw).observe($('views'));draw();refresh();
}catch(e){status(e.message,true);}
