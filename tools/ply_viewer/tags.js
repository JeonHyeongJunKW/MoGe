// Tag rules use stable dataset/path identities across reports and checkpoints.
export function location(sample) {
 const path=sample.filename.replaceAll('\\','/');
 const scene=path.match(/^(.*\/scenes\/[^/]+)(?:\/|$)/)?.[1]||'';
 const frame=path.match(/\/samples\/(\d+)(?:\/|$)/)?.[1];
 return {map:String(sample.map_file||sample.map||sample.town||'unknown'),scene,
  frame:frame===undefined?null:Number(frame)};
}
export function validateRules(payload) {
 if(payload?.version!==1||!Array.isArray(payload.rules)||payload.rules.length>1000)throw Error('태그 JSON 형식이 올바르지 않습니다.');
 const ids=new Set();
 for(const r of payload.rules){
  if(!r||typeof r.id!=='string'||!r.id||ids.has(r.id)||typeof r.name!=='string'||!r.name.trim())throw Error('태그 이름 또는 ID가 올바르지 않습니다.');
  ids.add(r.id);
  if(!['rule','samples'].includes(r.type))throw Error('지원하지 않는 태그 규칙입니다.');
  if(r.type==='samples'){
   if(!Array.isArray(r.keys)||r.keys.some(k=>typeof k!=='string'))throw Error('이미지 목록이 올바르지 않습니다.');
  }else{
   for(const k of ['dataset','map','scene','prefix','camera'])if(typeof r[k]!=='string')throw Error(`태그 조건 ${k}가 올바르지 않습니다.`);
   for(const k of ['start','end'])if(r[k]!==null&&(!Number.isSafeInteger(r[k])||r[k]<0))throw Error('구간 번호는 0 이상의 정수여야 합니다.');
   if(r.start!==null&&r.end!==null&&r.start>r.end)throw Error('시작 번호가 끝 번호보다 큽니다.');
   if((r.start!==null||r.end!==null)&&!r.scene)throw Error('샘플 번호 구간에는 scene을 지정하세요.');
  }
 }
 return payload.rules;
}
export function matches(sample,rule) {
 if(rule.type==='samples')return rule.keys.includes(JSON.stringify([sample.dataset,sample.filename]));
 const l=location(sample);
 return (!rule.dataset||rule.dataset===sample.dataset)&&(!rule.map||rule.map===l.map)&&
  (!rule.scene||rule.scene===l.scene)&&(!rule.prefix||sample.filename.startsWith(rule.prefix))&&
  (!rule.camera||rule.camera===sample.camera)&&
  (rule.start===null||(l.frame!==null&&l.frame>=rule.start))&&
  (rule.end===null||(l.frame!==null&&l.frame<=rule.end));
}
export function tagsFor(sample,rules){return [...new Set(rules.filter(r=>matches(sample,r)).map(r=>r.name))].sort();}
