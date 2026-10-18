import {location,validateRules,tagsFor,matches} from './tags.js';
const $=id=>document.getElementById(id);
const colors=['#55d9ea','#ffa45b'];
let reports=[], dataA=null,dataB=null, pathA='',pathB='', paired=[], visible=[], filtered=[];
let page=0,selected=null,bin=null,bars=[],dots=[],generation=0;
const PAGE_SIZE=40;
const localReports=new Map();
let registeredRoots=[];
let cloudGeneration=0;
let distribution=false;
const primaryValue=row=>value(row,'a')??value(row,'b');
function readReport(path){return localReports.has(path)?Promise.resolve(localReports.get(path)):get(`/api/report?${new URLSearchParams({report:path})}`);}
function validateReport(r){
 if(r?.schema_version!==1||!Array.isArray(r.samples)||!r.samples.length)throw Error('이미지별 validation JSON(schema_version=1, samples)이 필요합니다. summary.json이나 CSV는 사용할 수 없습니다.');
 const seen=new Set();for(const s of r.samples){if(typeof s.dataset!=='string'||typeof s.filename!=='string'||!s.metrics||typeof s.metrics!=='object'||Array.isArray(s.metrics)||Object.values(s.metrics).some(v=>!finite(v)))throw Error('이미지 경로 또는 지표 형식이 올바르지 않습니다.');const id=key(s);if(seen.has(id))throw Error('동일 dataset/filename이 중복되어 있습니다.');seen.add(id);}
 return r;
}
const TAG_STORAGE='moge-validation-tags-v1';
let tagRules=[],groupStats=[];
try{const saved=localStorage.getItem(TAG_STORAGE);if(saved)tagRules=validateRules(JSON.parse(saved));}
catch(e){$('tagStatus').textContent=`태그를 불러오지 못했습니다: ${e.message}`;}
const rowTags=row=>row.tags||[];
function updateTags(){
 for(const row of paired)row.tags=tagsFor(row.sample,tagRules);
 populate('tagFilter',[...new Set(paired.flatMap(rowTags))].sort(),'전체');
 renderRules();
}
function saveRules(next){
 validateRules({version:1,rules:next});
 localStorage.setItem(TAG_STORAGE,JSON.stringify({version:1,rules:next}));
 tagRules=next;updateTags();bin=null;page=0;render();
}
function tagMessage(text){$('tagStatus').textContent=text;}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csv(rows){const escape=x=>'"'+String(x??'').replaceAll('"','""')+'"';return '\ufeff'+rows.map(r=>r.map(escape).join(',')).join('\n');}
function renderRules(){
 $('tagRules').replaceChildren();
 for(const r of tagRules){const div=document.createElement('div'),label=document.createElement('span'),button=document.createElement('button');
  const conditions=r.type==='samples'?`고정 이미지 ${r.keys.length}장`:[r.dataset,r.map,r.scene,r.camera,r.prefix,r.start===null&&r.end===null?'':`${r.start??'처음'} ~ ${r.end??'끝'}`].filter(Boolean).join(' · ')||'전체 이미지';
  label.textContent=`${r.name} — ${conditions} (현재 비교 대상 ${paired.filter(row=>matches(row.sample,r)).length}장)`;
  button.textContent='삭제';button.onclick=()=>{try{saveRules(tagRules.filter(rule=>rule.id!==r.id));tagMessage('규칙을 삭제했습니다.');}catch(e){tagMessage(e.message);}};
  div.append(label,button);$('tagRules').append(div);
 }
}
function groupSummary(rows){
 const groups=new Map();
 for(const row of rows){const names=$('groupBy').value==='tags'?rowTags(row):[location(row.sample)[$('groupBy').value]||row.sample[$('groupBy').value]||'unknown'];
  for(const name of names){if(!groups.has(name))groups.set(name,[]);groups.get(name).push(row);}}
 groupStats=[];$('groupRows').replaceChildren();
 const mean=values=>values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
 for(const [name,items] of [...groups].sort(([a],[b])=>a.localeCompare(b))){
  const a=items.map(r=>value(r,'a')).filter(finite),b=dataB?items.map(r=>value(r,'b')).filter(finite):[];
  const row=[name,distribution?`${a.length} / ${b.length}`:items.length,mean(a),quantile(a,.5),quantile(a,.9),mean(b),quantile(b,.9),a.length&&b.length?mean(b)-mean(a):null];groupStats.push(row);
  const tr=document.createElement('tr');for(const [i,v]of row.entries()){const td=document.createElement('td');td.textContent=i<2?String(v):format(v);tr.append(td);}
  if($('groupBy').value==='tags'){tr.tabIndex=0;tr.onclick=()=>{$('tagFilter').value=name;bin=null;page=0;render();};tr.onkeydown=e=>{if(e.key==='Enter')tr.click();};}
  $('groupRows').append(tr);
 }
 if(!groups.size){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=8;td.textContent='조건에 맞는 그룹이 없습니다. 태그를 추가하거나 Map/Scene 그룹을 선택하세요.';tr.append(td);$('groupRows').append(tr);}
}
const value=(row,side)=>row[side]?.metrics[$('metric').value];
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const key=s=>JSON.stringify([s.dataset,s.filename]);
const format=x=>finite(x)?x.toFixed(6):'—';
const deviation=()=>$('metric').value.endsWith('/deviation');
const quality=v=>deviation()?Math.abs(v):v;
const higher=()=>/delta|accuracy|iou|f1/i.test($('metric').value);
function message(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
async function get(url){const r=await fetch(url);if(!r.ok)throw Error(`결과를 읽을 수 없습니다 (${r.status}).`);return r.json();}
function populate(id,values,first){const old=$(id).value;$(id).replaceChildren(...(first?[new Option(first,'')]:[]),...values.map(v=>new Option(v,v)));if(values.includes(old))$(id).value=old;}
async function refresh(){try{reports=[...(await get('/api/reports')).reports,...localReports.keys()];populate('reportA',reports,'결과 선택');populate('reportB',reports,'비교 안 함');message(reports.length?`${reports.length}개 결과를 찾았습니다.`:'이미지별 평가 결과가 없습니다. 새 validation 또는 analyze_validation 명령으로 생성하세요.');}catch(e){message(e.message,true);}}
async function load(){
 const a=$('reportA').value,b=$('reportB').value;if(!a){message('결과 A를 선택하세요.',true);return;}
 const id=++generation;message('평가 결과 불러오는 중…');
 try{
  const [ra,rb]=await Promise.all([readReport(a),b?readReport(b):null]);
  if(id!==generation)return;
  validateReport(ra);if(rb)validateReport(rb);
  cloudGeneration++;$('cloudPanel').hidden=true;$('cloudFrame').removeAttribute('src');$('cloudStatus').textContent='';
  dataA=ra;dataB=rb;pathA=a;pathB=b;bin=null;selected=null;page=0;
  const map=new Map((rb?.samples??[]).map((s,index)=>[key(s),{...s,index}]));
  distribution=$('compareMode').value==='distribution';
  paired=distribution?[
   ...ra.samples.map((s,index)=>({a:{...s,index},sample:{...s,index},id:'a:'+key(s)})),
   ...(rb?.samples??[]).map((s,index)=>({b:{...s,index},sample:{...s,index},id:'b:'+key(s)}))
  ]:ra.samples.map((s,index)=>({a:{...s,index},sample:{...s,index},b:map.get(key(s)),id:key(s)})).filter(r=>!rb||r.b);
  const common=new Set();for(const row of paired)for(const [m,v]of Object.entries(row.sample.metrics))if(finite(v)&&(distribution||!rb||finite(row.b.metrics[m])))common.add(m);
  $('sort').querySelector('[value="change"]').disabled=distribution;
  if(distribution&&$('sort').value==='change')$('sort').value='worst';
  $('sort').querySelector('[value="worst"]').textContent=distribution?'오차가 큰 순서 (A/B 전체)':'A 오차가 큰 순서';
  $('sort').querySelector('[value="best"]').textContent=distribution?'오차가 작은 순서 (A/B 전체)':'A 오차가 작은 순서';
  populate('metric',[...common].sort());if(common.has('depth_metric/rel'))$('metric').value='depth_metric/rel';
  for(const field of ['town','environment','camera'])populate(field,[...new Set(paired.map(r=>r.sample[field]||'unknown'))].sort(),'전체');
  populate('mapFilter',[...new Set(paired.map(r=>location(r.sample).map))].sort(),'전체');
  for(const [id,field]of [['tagDataset','dataset'],['tagCamera','camera']])populate(id,[...new Set(paired.map(r=>r.sample[field]).filter(Boolean))].sort(),'전체');
  populate('tagMap',[...new Set(paired.map(r=>location(r.sample).map))].sort(),'전체');
  populate('tagScene',[...new Set(paired.map(r=>location(r.sample).scene).filter(Boolean))].sort(),'전체');
  updateTags();
  let note=`A ${ra.samples.length}장${rb?` · B ${rb.samples.length}장 · 공통 이미지 ${paired.length}장 · A만 ${ra.samples.length-paired.length}장 / B만 ${rb.samples.length-paired.length}장`:''}`;
  if(distribution)note=`A ${ra.samples.length}장 · B ${rb?.samples.length??0}장 · 독립 분포 비교 (이미지별 대응 없음)`;
  if(rb&&ra.validation_signature!==rb.validation_signature)note+=' · 평가 설정/분할이 다릅니다. 비교 조건을 확인하세요.';
  message(note);render();
 }catch(e){if(id===generation)message(e.message,true);}
}
function quantile(values,q){if(!values.length)return null;const a=[...values].sort((a,b)=>a-b),p=(a.length-1)*q,i=Math.floor(p);return a[i]+(a[Math.min(i+1,a.length-1)]-a[i])*(p-i);}
function summary(){
 $('stats').replaceChildren();
 for(const [side,label]of [['a','A'],['b','B']]){
  if(side==='b'&&!dataB)continue;
  const vals=filtered.map(r=>value(r,side)).filter(finite),mean=vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:null;
  const node=document.createElement('div');node.className='stat';node.textContent=`${label} · ${vals.length}장 / 평균`;
  const strong=document.createElement('strong');strong.textContent=format(mean);node.append(strong);
  const sub=document.createElement('small');sub.textContent=`P50 ${format(quantile(vals,.5))} · P90 ${format(quantile(vals,.9))}`;node.append(document.createElement('br'),sub);$('stats').append(node);
 }
 const note=document.createElement('div');note.className='stat';note.textContent=`${deviation()?'0에 가까울수록 좋음':higher()?'값이 클수록 좋음':'값이 작을수록 좋음'} · 이미지별 동일 가중치`;
 const n=document.createElement('strong');n.textContent=`목록 ${visible.length}장`;note.append(n);$('stats').append(note);
}
function chart(id){const canvas=$(id),rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const c=canvas.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#101b29';c.fillRect(0,0,rect.width,rect.height);c.font='11px system-ui';return {c,w:rect.width,h:rect.height};}
function range(values){let min=Infinity,max=-Infinity;for(const v of values){min=Math.min(min,v);max=Math.max(max,v);}if(!values.length)return [0,1];if(min===max){const pad=Math.max(Math.abs(min)*.05,.01);min-=pad;max+=pad;}return [min,max];}
function histogram(){
 const {c,w,h}=chart('hist');bars=[];const series=[filtered.map(r=>value(r,'a')).filter(finite),dataB?filtered.map(r=>value(r,'b')).filter(finite):[]];
 const [min,max]=range(series.flat()),n=25,step=(max-min)/n,counts=series.map(values=>{const bins=Array(n).fill(0);values.forEach(v=>bins[Math.min(n-1,Math.floor((v-min)/step))]++);return bins;});
 const heights=counts.map((bins,side)=>bins.map(n=>distribution?100*n/Math.max(1,series[side].length):n));
 const top=Math.max(1,...heights.flat()),left=48,bottom=h-30,width=Math.max(1,w-left-12),height=h-58,bw=width/n;
 c.fillStyle='#9ab3cd';c.fillText(distribution?`${top.toFixed(1)}%`:`${top} images`,8,15);c.fillText(format(min),left,h-8);c.fillText(format(max),Math.max(left,w-90),h-8);
 for(let i=0;i<n;i++)for(let side=0;side<(dataB?2:1);side++){
  const count=counts[side][i],x=left+i*bw+side*bw/(dataB?2:1),barW=bw/(dataB?2:1)-1,barH=heights[side][i]/top*height;
  c.fillStyle=colors[side];c.fillRect(x,bottom-barH,barW,barH);
  bars.push({x,y:bottom-barH,w:barW,h:barH,lower:min+i*step,upper:min+(i+1)*step,last:i===n-1,side:side?'b':'a',count});
 }
 $('binInfo').textContent=bin?`${bin.side.toUpperCase()} 구간 ${format(bin.lower)} ~ ${format(bin.upper)} 선택 · ${visible.length}장`:'막대를 클릭하면 해당 구간의 이미지로 목록을 좁힙니다.';
}
function scatter(){
 const {c,w,h}=chart('scatter');dots=[];const values=visible.flatMap(r=>dataB?[value(r,'a'),value(r,'b')]:[value(r,'a')]).filter(finite);const [min,max]=range(values);
 c.fillStyle='#9ab3cd';c.fillText(format(max),5,16);c.fillText(format(min),5,h-25);c.fillText('현재 정렬된 이미지 순서',50,h-6);
 visible.forEach((row,i)=>{
  const x=55+i/Math.max(1,visible.length-1)*(w-70);
  for(const [side,j] of [['a',0],['b',1]]){
   if(!finite(value(row,side)))continue;const y=25+(max-value(row,side))/(max-min)*(h-65);
   c.fillStyle=colors[j];c.beginPath();c.arc(x,y,row.id===selected?5:2,0,Math.PI*2);c.fill();dots.push({x,y,row});
  }
 });
}
function select(row){
 if(selected!==row.id){cloudGeneration++;$('cloudPanel').hidden=true;$('cloudFrame').removeAttribute('src');$('cloudStatus').textContent='';}
 selected=row.id;$('selected').textContent=row.sample.filename;$('figureB').hidden=!dataB;$('imageError').textContent='';
 for(const [side,path,img] of [['a',pathA,'imageA'],['b',pathB,'imageB']]){
  $(img).hidden=true;$(img).removeAttribute('src');const item=row[side];
  const root=$(side==='a'?'dataRootA':'dataRootB').value;
  $('caption'+side.toUpperCase()).textContent=`${side.toUpperCase()} · ${root?'데이터셋 원본 RGB':'평가 입력 RGB'}`;
  if(item&&root){$(img).hidden=false;$(img).src=`/api/dataset-image?${new URLSearchParams({root,filename:item.image_path||item.filename.replace(/\/$/,'')+'/image.jpg'})}`;}
  else if(item?.image&&!localReports.has(path)){$(img).hidden=false;$(img).src=`/api/report-image?${new URLSearchParams({report:path,index:item.index})}`;}
  else if(item)$('imageError').textContent='데이터셋 경로를 선택하세요. JSON을 직접 불러온 경우 원본 이미지 경로가 필요합니다.';
 }
 const meta={dataset:row.sample.dataset,town:row.sample.town,environment:row.sample.environment,camera:row.sample.camera,tags:rowTags(row),data_path:row.sample.data_path,...location(row.sample),'A metrics':row.a?.metrics};
 if(row.b)meta['B metrics']=row.b.metrics;
 $('metadata').textContent=JSON.stringify(meta,null,2);table();scatter();
}
function table(){
 const pages=Math.max(1,Math.ceil(visible.length/PAGE_SIZE));page=Math.min(page,pages-1);$('page').textContent=`${page+1} / ${pages}`;$('previous').disabled=page===0;$('next').disabled=page===pages-1;
 $('rows').replaceChildren();
 for(const row of visible.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE)){
  const tr=document.createElement('tr');tr.classList.toggle('active',row.id===selected);tr.tabIndex=0;
  for(const text of [(distribution?(row.a?'[A] ':'[B] '):'')+row.sample.filename,format(value(row,'a')),format(value(row,'b')),row.a&&row.b?format(value(row,'b')-value(row,'a')):'—']){const td=document.createElement('td');td.textContent=text;tr.append(td);}
  tr.onclick=()=>select(row);tr.onkeydown=e=>{if(e.key==='Enter')select(row);};$('rows').append(tr);
 }
}
function render(){
 if(!dataA)return;
 const base=paired.filter(row=>(distribution?(finite(value(row,'a'))||finite(value(row,'b'))):finite(value(row,'a'))&&(!dataB||finite(value(row,'b'))))&&['town','environment','camera'].every(f=>!$(f).value||$(f).value===(row.sample[f]||'unknown'))&&(!$('mapFilter').value||$('mapFilter').value===location(row.sample).map)&&row.sample.filename.toLowerCase().includes($('search').value.toLowerCase()));
 groupSummary(base);
 filtered=base.filter(row=>!$('tagFilter').value||rowTags(row).includes($('tagFilter').value));
 visible=filtered.filter(r=>!bin||(value(r,bin.side)>=bin.lower&&(bin.last?value(r,bin.side)<=bin.upper:value(r,bin.side)<bin.upper)));
 const sign=higher()?-1:1;
 visible.sort((a,b)=>$('sort').value==='change'&&dataB&&!distribution?sign*((quality(value(b,'b'))-quality(value(b,'a')))-(quality(value(a,'b'))-quality(value(a,'a')))):sign*($('sort').value==='best'?quality(primaryValue(a))-quality(primaryValue(b)):quality(primaryValue(b))-quality(primaryValue(a))));
 summary();histogram();table();scatter();
 if(visible.length)select(visible.find(r=>r.id===selected)||visible[0]);
 if(!visible.length){cloudGeneration++;$('cloudPanel').hidden=true;$('cloudFrame').removeAttribute('src');selected=null;$('selected').textContent='조건에 맞는 이미지가 없습니다.';$('imageA').hidden=true;$('imageB').hidden=true;$('metadata').textContent='';}
}
$('compareMode').onchange=()=>{if(dataA)load();};
$('load').onclick=load;$('refresh').onclick=refresh;
for(const id of ['metric','town','environment','camera','search','sort','mapFilter','tagFilter','groupBy'])$(id).oninput=()=>{bin=null;page=0;render();};
$('resetBin').onclick=()=>{bin=null;page=0;render();};
$('previous').onclick=()=>{page--;table();};$('next').onclick=()=>{page++;table();};
$('hist').onclick=e=>{const rect=$('hist').getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const hit=bars.find(b=>b.count&&x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);if(hit){bin=hit;page=0;render();}};
$('scatter').onclick=e=>{const rect=$('scatter').getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=null,dist=144;for(const p of dots){const d=(p.x-x)**2+(p.y-y)**2;if(d<dist){dist=d;best=p;}}if(best){page=Math.floor(visible.indexOf(best.row)/PAGE_SIZE);select(best.row);}};
for(const id of ['imageA','imageB'])$(id).onerror=()=>{$(id).hidden=true;$('imageError').textContent='이미지 파일을 읽을 수 없습니다. 선택한 데이터셋 경로와 filename/image.jpg 또는 sample_images 폴더를 확인하세요.';};
$('export').onclick=()=>download('validation_comparison.csv',csv([['dataset','filename','tags','metric','A','B','B-A'],...visible.map(r=>[r.sample.dataset,r.sample.filename,rowTags(r).join(';'),$('metric').value,value(r,'a'),value(r,'b'),r.a&&r.b?value(r,'b')-value(r,'a'):''])]),'text/csv;charset=utf-8');
$('groupsExport').onclick=()=>download('validation_groups.csv',csv([['group','images',`${$('metric').value}:A_mean`,'A_P50','A_P90','B_mean','B_P90','B-A_mean'],...groupStats]),'text/csv;charset=utf-8');
$('tagsExport').onclick=()=>download('validation_tags.json',JSON.stringify({version:1,rules:tagRules},null,2),'application/json');
$('tagsImport').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>10*1024*1024)throw Error('태그 파일은 10 MiB 이하로 지정하세요.');saveRules(validateRules(JSON.parse(await file.text())));tagMessage('태그 규칙을 가져왔습니다.');}catch(e){tagMessage(e.message);}finally{$('tagsImport').value='';}};
$('tagFromSelected').onclick=()=>{const row=paired.find(r=>r.id===selected);if(!row){tagMessage('이미지를 먼저 선택하세요.');return;}const l=location(row.sample);$('tagDataset').value=row.sample.dataset;$('tagMap').value=l.map;$('tagScene').value=l.scene;$('tagCamera').value='';$('tagPrefix').value='';$('tagStart').value=l.frame??'';$('tagEnd').value=l.frame??'';tagMessage('선택 이미지의 scene과 번호를 채웠습니다. 범위를 수정하세요.');};
function addTag(snapshot){try{
 if(!dataA)throw Error('결과 A를 먼저 불러오세요.');
 const name=$('tagName').value.trim();if(!name)throw Error('태그 이름을 입력하세요.');
 const rule={id:globalThis.crypto?.randomUUID?.()||`${Date.now()}-${Math.random().toString(36).slice(2)}`,name,type:snapshot?'samples':'rule'};
 if(snapshot){if(!visible.length)throw Error('현재 목록이 비어 있습니다.');rule.keys=[...new Set(visible.map(r=>key(r.sample)))];}
 else{Object.assign(rule,{dataset:$('tagDataset').value,map:$('tagMap').value,scene:$('tagScene').value,camera:$('tagCamera').value,prefix:$('tagPrefix').value.trim(),start:$('tagStart').value===''?null:Number($('tagStart').value),end:$('tagEnd').value===''?null:Number($('tagEnd').value)});}
 validateRules({version:1,rules:[rule]});
 const count=paired.filter(r=>matches(r.sample,rule)).length;if(!count)throw Error('조건에 맞는 이미지가 없습니다. Map/scene/구간 조건을 확인하세요.');
 saveRules([...tagRules,rule]);tagMessage(`${name}: ${count}장에 적용했습니다.`);
 }catch(e){tagMessage(`저장하지 못했습니다: ${e.message}`);}}
$('tagAdd').onclick=()=>addTag(false);$('tagSnapshot').onclick=()=>addTag(true);
renderRules();
new ResizeObserver(()=>{if(dataA){histogram();scatter();}}).observe($('charts'));refresh();

async function initRoots(){try{registeredRoots=(await get('/api/data-roots')).roots;for(const side of ['A','B'])populate('dataRoot'+side,registeredRoots,'저장된 평가 미리보기 사용');}catch(e){message(e.message,true);}}
const rootsReady=initRoots();
for(const side of ['A','B']){
 $('dataRoot'+side).onchange=()=>{const row=paired.find(r=>r.id===selected);if(row)select(row);};
 $('lossFile'+side).onchange=async e=>{try{
  const file=e.target.files[0];if(!file)return;if(file.size>100*1024*1024)throw Error('JSON 파일은 100 MiB 이하로 지정하세요.');
  const report=validateReport(JSON.parse(await file.text()));await rootsReady;
  const id=`local:${side}:${file.name}`;localReports.set(id,report);await refresh();$('report'+side).value=id;
  const roots=[...new Set(report.samples.map(s=>s.dataset_root).filter(Boolean))];
  if(roots.length===1&&registeredRoots.includes(roots[0]))$('dataRoot'+side).value=roots[0];
  else if(registeredRoots.length===1)$('dataRoot'+side).value=registeredRoots[0];
  if($('reportA').value)await load();else message('A 결과도 선택하세요.');
 }catch(e){message(e.message,true);}finally{$('lossFile'+side).value='';}};
}

for(const side of ['A','B'])$('cloud'+side).onclick=async()=>{
 const row=paired.find(r=>r.id===selected),item=row?.[side.toLowerCase()],path=side==='A'?pathA:pathB;
 if(!item){$('cloudStatus').textContent='먼저 해당 결과와 이미지를 선택하세요.';return;}
 if(localReports.has(path)){$('cloudStatus').textContent='3D 생성은 서버 결과를 사용합니다. JSON을 workspace에 복사한 뒤 결과 목록에서 선택하세요.';return;}
 const id=++cloudGeneration;$('cloudPanel').hidden=true;$('cloudStatus').textContent='선택 이미지의 GT / Prediction 생성 중… 첫 실행은 수 초가 걸립니다.';
 $('cloudA').disabled=true;$('cloudB').disabled=true;
 try{
  const response=await fetch('/api/pointcloud',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({report:path,index:item.index,data_root:$('dataRoot'+side).value||item.dataset_root||''})});
  const result=await response.json();if(!response.ok)throw Error(result.error||'3D 생성 실패');if(id!==cloudGeneration)return;
  const query=new URLSearchParams({sample:result.sample,embed:'1'});
  $('cloudFrame').src='/index.html?'+query;$('cloudLink').href='/index.html?'+new URLSearchParams({sample:result.sample});
  $('cloudTitle').textContent=`${side} · ${item.filename} · GT / Prediction`;
  $('cloudPanel').hidden=false;$('cloudStatus').textContent='3D 준비 완료. 아래 뷰어에서 드래그로 트랙볼 회전합니다. 원본 좌표·스케일을 유지하며 선택 이미지 재추론 결과입니다.';
  $('cloudPanel').scrollIntoView({behavior:'smooth',block:'start'});
 }catch(e){if(id===cloudGeneration)$('cloudStatus').textContent=e.message;}
 finally{$('cloudA').disabled=false;$('cloudB').disabled=false;}
};
