// Quaternion arcball. Camera orientation maps view-space basis vectors to world space.
export const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const normalize = v => { const n=Math.hypot(...v); return v.map(x=>x/n); };
export function multiply(a,b) {
  const v=cross(a,b);
  return [0,1,2].map(i=>a[3]*b[i]+b[3]*a[i]+v[i]).concat(a[3]*b[3]-dot(a.slice(0,3),b.slice(0,3)));
}
export function rotate(q,v) {
  return multiply(multiply(q,[...v,0]),[-q[0],-q[1],-q[2],q[3]]).slice(0,3);
}
export function sphere(x,y,rect) {
  const radius=Math.min(rect.width,rect.height)/2;
  const p=[(x-rect.left-rect.width/2)/radius,(rect.top+rect.height/2-y)/radius,0];
  const r2=p[0]*p[0]+p[1]*p[1];
  p[2]=r2<=.5?Math.sqrt(1-r2):.5/Math.sqrt(r2);
  return normalize(p);
}
export function trackball(q,from,to) {
  // Inverse rotation moves the viewed object with the pointer; no fixed up axis.
  const cosine=Math.max(-1,Math.min(1,dot(to,from)));
  let delta;
  if(cosine < -.999999){
    const axis=normalize(cross(to,Math.abs(to[0])<.9?[1,0,0]:[0,1,0]));
    delta=[...axis,0];
  }else delta=normalize([...cross(to,from),1+cosine]);
  return normalize(multiply(q,delta));
}
