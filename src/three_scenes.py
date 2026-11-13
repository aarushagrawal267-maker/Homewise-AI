"""Interactive WebGL (Three.js) scenes rendered inside Streamlit.

* :func:`house_scene_html` builds procedural 3-D houses whose geometry is driven
  by the property's actual attributes: footprint from first-floor area, a second
  storey when ``2ndFlrSF > 0``, garage bays from ``GarageCars``, an x-ray
  basement sized by ``TotalBsmtSF``, a pool, a chimney for fireplaces, lot
  boundary from ``LotArea`` and edge colour from ``OverallQual``.
* :func:`market_galaxy_html` renders every historical sale as a point in 3-D
  (living area × year built × sale price) with hover details.

Both scenes support drag-to-orbit, scroll/pinch zoom and auto-rotation.
Three.js is served locally from ``static/vendor/three`` (Streamlit static file
serving) with a CDN fallback, so the app works offline.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

LOCAL_THREE = "/app/static/vendor/three/three.module.min.js"
CDN_THREE = "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.min.js"

_BASE_CSS = """
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:transparent;overflow:hidden;
  font-family:Inter,'Segoe UI',system-ui,-apple-system,sans-serif;color:#E6EDF7}
#wrap{position:relative;width:100%;height:100%;border-radius:22px;overflow:hidden;
  background:radial-gradient(120% 90% at 50% 0%,#12233b 0%,#0a1322 45%,#060a13 100%);
  border:1px solid rgba(255,255,255,.08)}
canvas{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
canvas:active{cursor:grabbing}
.tag{position:absolute;transform:translate(-50%,-100%);pointer-events:none;text-align:center;
  padding:10px 16px 11px;border-radius:14px;background:rgba(10,18,32,.72);backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);border:1px solid rgba(126,232,210,.35);
  box-shadow:0 10px 30px rgba(0,0,0,.35),0 0 24px rgba(31,165,146,.18);white-space:nowrap;
  transition:opacity .3s}
.tag .l{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#9FB0C8}
.tag .p{font-family:'Space Grotesk',Inter,sans-serif;font-weight:700;font-size:22px;color:#fff;margin-top:2px}
.tag .s{font-size:11.5px;color:#9FB0C8;margin-top:3px}
.tag:after{content:"";position:absolute;left:50%;bottom:-18px;width:1px;height:18px;
  background:linear-gradient(rgba(126,232,210,.6),transparent)}
.hud{position:absolute;left:14px;bottom:14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.hud button{min-height:36px;padding:0 14px;border-radius:999px;border:1px solid rgba(255,255,255,.14);
  background:rgba(10,18,32,.65);color:#DCE6F5;font:500 12.5px Inter,system-ui,sans-serif;cursor:pointer;
  backdrop-filter:blur(8px)}
.hud button:hover{border-color:rgba(126,232,210,.6);color:#fff}
.hud button[aria-pressed=true]{background:rgba(31,165,146,.28);border-color:rgba(126,232,210,.7)}
.hint{position:absolute;left:16px;top:14px;font-size:11.5px;color:#7D8BA1;letter-spacing:.02em}
.tag.sm{padding:7px 11px 8px}.tag.sm .l{font-size:9.5px}.tag.sm .p{font-size:17px}.tag.sm .s{font-size:10.5px}
.legend{position:absolute;right:12px;bottom:12px;font-size:10.5px;color:#9FB0C8;background:rgba(10,18,32,.6);
  border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:7px 10px;line-height:1.6}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
#tip{position:absolute;pointer-events:none;padding:8px 11px;border-radius:10px;background:rgba(10,18,32,.9);
  border:1px solid rgba(126,232,210,.4);font-size:12px;line-height:1.5;display:none;white-space:nowrap}
#err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;
  padding:30px;color:#9FB0C8;font-size:14px}
"""

_LOADER_JS = """
async function loadThree(){
  const local = new URL("%LOCAL%", document.baseURI).href;
  try { return await import(local); } catch(e) {}
  try { return await import(local.replace(/^about:srcdoc/, "")); } catch(e) {}
  return await import("%CDN%");
}
function showErr(msg){const e=document.getElementById('err');e.style.display='flex';e.textContent=msg;}
function webglOK(){try{const c=document.createElement('canvas');return !!(c.getContext('webgl2')||c.getContext('webgl'));}catch(e){return false;}}

// Minimal orbit controller (drag / touch to rotate, wheel / pinch to zoom, auto-rotate).
function makeOrbit(canvas, cam, target, opts){
  const s={theta:opts.theta??0.8,phi:opts.phi??1.05,r:opts.r??24,auto:opts.auto??true,
    minR:opts.minR??6,maxR:opts.maxR??80,dragging:false,last:null,idle:0,vt:0,vp:0,pinch:null};
  function apply(){
    s.phi=Math.max(0.25,Math.min(1.45,s.phi));s.r=Math.max(s.minR,Math.min(s.maxR,s.r));
    cam.position.set(target.x+s.r*Math.sin(s.phi)*Math.sin(s.theta),target.y+s.r*Math.cos(s.phi),
      target.z+s.r*Math.sin(s.phi)*Math.cos(s.theta));cam.lookAt(target);}
  canvas.addEventListener('pointerdown',e=>{s.dragging=true;s.last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointerup',e=>{s.dragging=false;s.idle=performance.now();});
  canvas.addEventListener('pointermove',e=>{if(!s.dragging||s.pinch)return;const dx=e.clientX-s.last[0],dy=e.clientY-s.last[1];
    s.last=[e.clientX,e.clientY];s.vt=-dx*0.006;s.vp=-dy*0.005;s.theta+=s.vt;s.phi+=s.vp;apply();});
  canvas.addEventListener('wheel',e=>{e.preventDefault();s.r*=1+Math.sign(e.deltaY)*0.08;s.idle=performance.now();apply();},{passive:false});
  canvas.addEventListener('touchmove',e=>{if(e.touches.length===2){const d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,
    e.touches[0].clientY-e.touches[1].clientY);if(s.pinch){s.r*=s.pinch/d;apply();}s.pinch=d;}},{passive:true});
  canvas.addEventListener('touchend',()=>{s.pinch=null;});
  s.update=function(){if(!s.dragging){s.theta+=s.vt;s.phi+=s.vp;s.vt*=0.92;s.vp*=0.92;
      if(s.auto&&performance.now()-s.idle>2500)s.theta+=0.0022;}apply();};
  apply();return s;}
"""

_HOUSE_JS = """
const DATA = %DATA%;
(async()=>{
if(!webglOK()){showErr('3D preview needs WebGL, which is unavailable in this browser.');return;}
let THREE; try{THREE=await loadThree();}catch(e){showErr('Could not load the 3D engine.');return;}
const wrap=document.getElementById('wrap');
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.45;
wrap.prepend(renderer.domElement);
const scene=new THREE.Scene();scene.fog=new THREE.FogExp2(0x070c16,0.018);
const cam=new THREE.PerspectiveCamera(38,1,0.1,400);

scene.add(new THREE.AmbientLight(0x8fa3cc,0.75));
scene.add(new THREE.HemisphereLight(0xbfe6ff,0x0b1220,0.9));
const sun=new THREE.DirectionalLight(0xfff1dd,1.6);sun.position.set(14,22,10);sun.castShadow=true;
sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-30,right:30,top:30,bottom:-30,far:80});
scene.add(sun);
const rim=new THREE.DirectionalLight(0x58d6c4,0.7);rim.position.set(-16,8,-14);scene.add(rim);

let seed=7;const rnd=()=>{seed=(seed*16807)%2147483647;return (seed-1)/2147483646;};
const QCOL=q=>q>=9?0xF5C76B:q>=7?0x2DD4BF:q>=5?0x5EA2F0:0xE08A4A;

function stripes(color){const c=document.createElement('canvas');c.width=64;c.height=64;const g=c.getContext('2d');
  g.fillStyle='#0d1626';g.fillRect(0,0,64,64);g.strokeStyle=color;g.globalAlpha=.55;g.lineWidth=2;
  for(let y=6;y<64;y+=10){g.beginPath();g.moveTo(0,y);g.lineTo(64,y);g.stroke();}
  const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.SRGBColorSpace;return t;}

const wallMat=()=>new THREE.MeshStandardMaterial({color:0x31476b,emissive:0x0c1830,metalness:.15,roughness:.55,transparent:true,opacity:.96});
const roofMat=new THREE.MeshStandardMaterial({color:0x3a4d6e,emissive:0x0a1426,metalness:.35,roughness:.4});
function edges(mesh,color,op=0.95){const l=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry,20),
  new THREE.LineBasicMaterial({color,transparent:true,opacity:op}));l.position.copy(mesh.position);l.rotation.copy(mesh.rotation);return l;}

const walls=[];const growers=[];const pools=[];
function windowsOn(group,cx,cy,cz,w,h,face,count){
  const winMat=new THREE.MeshBasicMaterial({color:0xffc98a});const offMat=new THREE.MeshBasicMaterial({color:0x2a3a55});
  for(let i=0;i<count;i++){const lit=rnd()>.22;const m=new THREE.Mesh(new THREE.PlaneGeometry(.42,.5),lit?winMat:offMat);
    const t=(i+1)/(count+1)-.5;
    if(face==='front'){m.position.set(cx+t*w,cy,cz+.012);} else if(face==='back'){m.position.set(cx+t*w,cy,cz-.012);m.rotation.y=Math.PI;}
    else if(face==='right'){m.position.set(cx+.012,cy,cz+t*w);m.rotation.y=Math.PI/2;} else {m.position.set(cx-.012,cy,cz+t*w);m.rotation.y=-Math.PI/2;}
    group.add(m);
    if(lit&&rnd()>.55){const gl=new THREE.PointLight(0xffb66b,.6,2.2);gl.position.copy(m.position);group.add(gl);}}
}

function dims(p){const A1=Math.max(p.first_sf,300)/100;let W=Math.sqrt(A1*1.35),D=A1/W;W=Math.min(Math.max(W,2.6),10);D=Math.min(Math.max(D,2.2),8);
  const ext=W+(p.garage_cars>0?1.05*p.garage_cars+.35:0)+(p.pool?3.6:0);return {W,D,ext};}
function buildHouse(p,maxL){
  const g=new THREE.Group();const accent=QCOL(p.quality);const FH=1.05;
  const {W,D}=dims(p);
  // lot
  const L=Math.min(Math.max(Math.sqrt(Math.max(p.lot_area,1500))/10,W+5),24,maxL);
  const lotPts=[[-L/2,-L/2],[L/2,-L/2],[L/2,L/2],[-L/2,L/2],[-L/2,-L/2]].map(a=>new THREE.Vector3(a[0]-1,0.02,a[1]));
  const lot=new THREE.Line(new THREE.BufferGeometry().setFromPoints(lotPts),new THREE.LineDashedMaterial({color:0x5EEAD4,dashSize:.5,gapSize:.35,transparent:true,opacity:.8}));
  lot.computeLineDistances();g.add(lot);
  const lawn=new THREE.Mesh(new THREE.PlaneGeometry(L,L),new THREE.MeshStandardMaterial({color:0x14403b,roughness:1,transparent:true,opacity:.8}));
  lawn.rotation.x=-Math.PI/2;lawn.position.set(-1,0.01,0);lawn.receiveShadow=true;g.add(lawn);

  const house=new THREE.Group();g.add(house);growers.push(house);
  // floor 1
  const b1=new THREE.Mesh(new THREE.BoxGeometry(W,FH,D),wallMat());b1.position.set(0,FH/2,0);b1.castShadow=b1.receiveShadow=true;
  house.add(b1,edges(b1,accent));walls.push(b1.material);
  windowsOn(house,0,FH*.55,D/2,W,FH,'front',Math.max(2,Math.round(W/1.3)));
  windowsOn(house,0,FH*.55,-D/2,W,FH,'back',Math.max(2,Math.round(W/1.5)));
  windowsOn(house,W/2,FH*.55,0,D,FH,'right',Math.max(1,Math.round(D/1.6)));
  const door=new THREE.Mesh(new THREE.PlaneGeometry(.55,.8),new THREE.MeshStandardMaterial({color:0x0a1220,emissive:accent,emissiveIntensity:.25}));
  door.position.set(W*.18,.4,D/2+.015);house.add(door);
  let topW=W,topD=D,topY=FH,topX=0;
  if(p.second_sf>0){const A2=p.second_sf/100;let W2=Math.min(W,Math.sqrt(A2*1.35)),D2=Math.min(D,A2/W2);
    const b2=new THREE.Mesh(new THREE.BoxGeometry(W2,FH,D2),wallMat());const x2=(W2-W)/2;b2.position.set(x2,FH*1.5,(D2-D)/2*0);
    b2.castShadow=b2.receiveShadow=true;house.add(b2,edges(b2,accent));walls.push(b2.material);
    windowsOn(house,x2,FH*1.55,D2/2,W2,FH,'front',Math.max(2,Math.round(W2/1.3)));
    windowsOn(house,x2,FH*1.55,-D2/2,W2,FH,'back',Math.max(2,Math.round(W2/1.5)));
    topW=W2;topD=D2;topY=FH*2;topX=x2;}
  // gable roof
  const rh=Math.min(1.4,.38*topD+.2);const sh=new THREE.Shape();sh.moveTo(-topD/2-.18,0);sh.lineTo(topD/2+.18,0);sh.lineTo(0,rh);sh.lineTo(-topD/2-.18,0);
  const rg=new THREE.ExtrudeGeometry(sh,{depth:topW+.3,bevelEnabled:false});rg.translate(0,0,-(topW+.3)/2);rg.rotateY(Math.PI/2);
  const roof=new THREE.Mesh(rg,roofMat);roof.position.set(topX,topY,0);roof.castShadow=true;house.add(roof,edges(roof,accent,.8));
  if(p.fireplaces>0){const ch=new THREE.Mesh(new THREE.BoxGeometry(.38,1.0,.38),new THREE.MeshStandardMaterial({color:0x2a2f3d,roughness:.8}));
    ch.position.set(topX+topW*.3,topY+rh*.6,topD*.18);ch.castShadow=true;house.add(ch,edges(ch,0xE08A4A,.8));}
  // garage
  if(p.garage_cars>0){const gw=1.05*p.garage_cars+.35,gd=Math.min(D*.9,2.6),gh=FH*.85;
    const gm=new THREE.Mesh(new THREE.BoxGeometry(gw,gh,gd),wallMat());gm.position.set(-W/2-gw/2,gh/2,D/2-gd/2);
    gm.castShadow=gm.receiveShadow=true;house.add(gm,edges(gm,accent));walls.push(gm.material);
    const tex=stripes('#2DD4BF');
    for(let i=0;i<p.garage_cars;i++){const dm=new THREE.Mesh(new THREE.PlaneGeometry(.88,gh*.72),new THREE.MeshStandardMaterial({map:tex,emissive:0x0b3b3a,emissiveIntensity:.6}));
      dm.position.set(-W/2-gw+.2+.44+i*1.05,gh*.38,D/2+.015);house.add(dm);}
    const slab=new THREE.Mesh(new THREE.BoxGeometry(gw+.15,.1,gd+.15),roofMat);slab.position.set(gm.position.x,gh+.05,gm.position.z);house.add(slab,edges(slab,accent,.7));
    const drive=new THREE.Mesh(new THREE.PlaneGeometry(gw*.9,L/2-D/2+.5),new THREE.MeshStandardMaterial({color:0x1a2232,roughness:.9}));
    drive.rotation.x=-Math.PI/2;drive.position.set(gm.position.x,.02,D/2+(L/2-D/2)/2+.2);drive.receiveShadow=true;g.add(drive);}
  // basement (x-ray)
  if(p.bsmt_sf>0){const Ab=p.bsmt_sf/100;let Wb=Math.min(W*1.02,Math.sqrt(Ab*1.35)),Db=Math.min(D*1.02,Ab/Wb);
    const bm=new THREE.Mesh(new THREE.BoxGeometry(Wb,.9,Db),new THREE.MeshBasicMaterial({color:0x3987E5,transparent:true,opacity:.13,depthWrite:false}));
    bm.position.set(0,-.47,0);house.add(bm);const be=edges(bm,0x5EA2F0,.75);house.add(be);}
  // pool
  if(p.pool){const pm=new THREE.Mesh(new THREE.BoxGeometry(2.8,.14,1.5),new THREE.MeshStandardMaterial({color:0x1ecbe1,emissive:0x0aa3c2,emissiveIntensity:.9,metalness:.1,roughness:.15}));
    pm.position.set(W/2+2.2,.07,-D/2+.4);g.add(pm,edges(pm,0xbff6ff,.9));pools.push(pm);
    const pl=new THREE.PointLight(0x22d3ee,2.2,6);pl.position.set(pm.position.x,.6,pm.position.z);g.add(pl);}
  // trees
  const nt=Math.max(2,Math.min(8,Math.floor(p.lot_area/2600)));
  for(let i=0;i<nt;i++){let x,z,k=0;do{x=(rnd()-.5)*(L-1.5)-1;z=(rnd()-.5)*(L-1.5);k++;}while(Math.abs(x)<W/2+3.6&&Math.abs(z)<D/2+2.2&&k<30);
    const s=.7+rnd()*.6;const cone=new THREE.Mesh(new THREE.ConeGeometry(.55*s,1.5*s,7),new THREE.MeshStandardMaterial({color:0x14524a,emissive:0x0a2e2a,roughness:.8,flatShading:true}));
    cone.position.set(x,.35*s+.75*s,z);cone.castShadow=true;const tr=new THREE.Mesh(new THREE.CylinderGeometry(.07*s,.09*s,.35*s),new THREE.MeshStandardMaterial({color:0x3b2f2a}));
    tr.position.set(x,.175*s,z);g.add(cone,tr);}
  g.userData={top:topY+rh,W,D,L};
  return g;}

const N=DATA.houses.length;const groups=[];let span=0;
const gap=Math.max(...DATA.houses.map(p=>dims(p).ext))+(N>1?4.5:0);
DATA.houses.forEach((p,i)=>{seed=11+i*97+Math.round(p.first_sf);const g=buildHouse(p,N>1?gap-.4:99);groups.push(g);span=Math.max(span,g.userData.L);});
groups.forEach((g,i)=>{g.position.x=(i-(N-1)/2)*gap;scene.add(g);});
const extentX=(N-1)*gap+span;

// platform, grid and glow
const R=Math.max(extentX*.62,span*.8,9);
const disc=new THREE.Mesh(new THREE.CircleGeometry(R,96),new THREE.MeshStandardMaterial({color:0x0a1424,roughness:.95,transparent:true,opacity:.7}));
disc.rotation.x=-Math.PI/2;disc.position.y=-.01;disc.receiveShadow=true;scene.add(disc);
const pg=new THREE.PolarGridHelper(R,16,8,96,0x1d4e5c,0x12303d);pg.material.transparent=true;pg.material.opacity=.45;scene.add(pg);
const ring=new THREE.Mesh(new THREE.RingGeometry(R-.06,R+.06,128),new THREE.MeshBasicMaterial({color:0x2DD4BF,transparent:true,opacity:.7,side:THREE.DoubleSide}));
ring.rotation.x=-Math.PI/2;scene.add(ring);
const halo=new THREE.Mesh(new THREE.RingGeometry(R+.1,R+1.6,128),new THREE.MeshBasicMaterial({color:0x2DD4BF,transparent:true,opacity:.08,side:THREE.DoubleSide,blending:THREE.AdditiveBlending}));
halo.rotation.x=-Math.PI/2;scene.add(halo);

// scanning rings (one per house)
const scans=groups.map(g=>{const s=new THREE.Mesh(new THREE.RingGeometry(g.userData.W*.75,g.userData.W*.78+.05,96),
  new THREE.MeshBasicMaterial({color:0x7EE8D2,transparent:true,opacity:.5,side:THREE.DoubleSide,blending:THREE.AdditiveBlending,depthWrite:false}));
  s.rotation.x=-Math.PI/2;s.position.x=g.position.x;scene.add(s);return s;});

// particles
const PC=420;const pos=new Float32Array(PC*3);const spd=new Float32Array(PC);
for(let i=0;i<PC;i++){const a=Math.random()*Math.PI*2,r=Math.sqrt(Math.random())*R;pos[i*3]=Math.cos(a)*r;pos[i*3+1]=Math.random()*9;pos[i*3+2]=Math.sin(a)*r;spd[i]=.004+Math.random()*.012;}
const pgeo=new THREE.BufferGeometry();pgeo.setAttribute('position',new THREE.BufferAttribute(pos,3));
const pts=new THREE.Points(pgeo,new THREE.PointsMaterial({color:0x7EE8D2,size:.07,transparent:true,opacity:.7,blending:THREE.AdditiveBlending,depthWrite:false}));scene.add(pts);

// labels
const tags=DATA.houses.map((p,i)=>{const d=document.createElement('div');d.className=N>=3?'tag sm':'tag';
  d.innerHTML=`<div class="l">${p.label}</div><div class="p">${p.price}</div><div class="s">${p.sub||''}</div>`;wrap.appendChild(d);return d;});

const target=new THREE.Vector3(0,1.0,0);
const R0=N>1?extentX*.86+6:Math.max(14,span*1.1+4);const TH=N>1?.18:.72,PH=N>1?1.1:1.06;
const orbit=makeOrbit(renderer.domElement,cam,target,{r:R0,theta:TH,phi:PH,minR:5,maxR:R0*3,auto:DATA.autorotate&&N===1});

// HUD
const btnA=document.getElementById('bAuto'),btnX=document.getElementById('bXray'),btnR=document.getElementById('bReset');
btnA.setAttribute('aria-pressed',String(orbit.auto));btnA.onclick=()=>{orbit.auto=!orbit.auto;btnA.setAttribute('aria-pressed',String(orbit.auto));};
let xray=false;btnX.onclick=()=>{xray=!xray;btnX.setAttribute('aria-pressed',String(xray));walls.forEach(m=>{m.opacity=xray?.22:.94;m.depthWrite=!xray;});};
let fitR=R0;btnR.onclick=()=>{orbit.theta=TH;orbit.phi=PH;orbit.r=fitR;};

function resize(){const w=wrap.clientWidth,h=wrap.clientHeight;renderer.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();
  const nf=R0*Math.max(1,Math.pow(1.7/cam.aspect,.9));if(Math.abs(orbit.r-fitR)<1e-6||orbit.r===R0)orbit.r=nf;fitR=nf;}
new ResizeObserver(resize).observe(wrap);resize();

const t0=performance.now();const v=new THREE.Vector3();
function frame(){const t=(performance.now()-t0)/1000;
  const e=Math.min(1,t/1.3);const k=1-Math.pow(1-e,3);growers.forEach(h=>{h.scale.y=Math.max(.001,k);});
  scans.forEach((s,i)=>{const top=groups[i].userData.top+.4;const y=((t*.45+i*.3)%1);s.position.y=y*top;s.material.opacity=.55*(1-y);});
  pools.forEach(p=>{p.material.emissiveIntensity=.75+.25*Math.sin(t*2.2);});
  const arr=pgeo.attributes.position.array;for(let i=0;i<PC;i++){arr[i*3+1]+=spd[i];if(arr[i*3+1]>9)arr[i*3+1]=0;}pgeo.attributes.position.needsUpdate=true;
  halo.material.opacity=.06+.04*Math.sin(t*1.5);
  orbit.update();renderer.render(scene,cam);
  groups.forEach((g,i)=>{v.set(g.position.x,g.userData.top*k+1.3,0).project(cam);
    const x=(v.x*.5+.5)*wrap.clientWidth,y=(-v.y*.5+.5)*wrap.clientHeight;tags[i].style.left=x+'px';tags[i].style.top=y+'px';
    tags[i].style.opacity=v.z<1&&e>.6?1:0;});
  requestAnimationFrame(frame);}
frame();
})();
"""

_GALAXY_JS = """
const DATA = %DATA%;
(async()=>{
if(!webglOK()){showErr('3D view needs WebGL, which is unavailable in this browser.');return;}
let THREE; try{THREE=await loadThree();}catch(e){showErr('Could not load the 3D engine.');return;}
const wrap=document.getElementById('wrap');const tip=document.getElementById('tip');
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
wrap.prepend(renderer.domElement);
const scene=new THREE.Scene();scene.fog=new THREE.FogExp2(0x070c16,0.012);
const cam=new THREE.PerspectiveCamera(45,1,0.1,500);
const P=DATA.points;const n=P.length;const pos=new Float32Array(n*3),col=new Float32Array(n*3);
const ramp=[[0,[0.06,0.2,0.24]],[.35,[0.08,0.45,0.42]],[.65,[0.12,0.65,0.57]],[.85,[0.49,0.91,0.82]],[1,[1,0.86,0.55]]];
function cramp(t){for(let i=1;i<ramp.length;i++){if(t<=ramp[i][0]){const a=ramp[i-1],b=ramp[i];const u=(t-a[0])/(b[0]-a[0]);
  return a[1].map((c,j)=>c+(b[1][j]-c)*u);}}return ramp[ramp.length-1][1];}
for(let i=0;i<n;i++){const p=P[i];pos[i*3]=p[0];pos[i*3+1]=p[1];pos[i*3+2]=p[2];const c=cramp(p[3]);col.set(c,i*3);}
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setAttribute('color',new THREE.BufferAttribute(col,3));
const sprite=(()=>{const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d');const gr=g.createRadialGradient(32,32,0,32,32,32);
  gr.addColorStop(0,'rgba(255,255,255,1)');gr.addColorStop(.35,'rgba(255,255,255,.85)');gr.addColorStop(1,'rgba(255,255,255,0)');g.fillStyle=gr;g.fillRect(0,0,64,64);
  return new THREE.CanvasTexture(c);})();
const mat=new THREE.PointsMaterial({size:.95,vertexColors:true,map:sprite,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending});
const cloud=new THREE.Points(geo,mat);scene.add(cloud);
// axes box
const B=DATA.bounds;const box=new THREE.Box3(new THREE.Vector3(B[0],0,B[2]),new THREE.Vector3(B[1],B[3],B[4]));
const bh=new THREE.Box3Helper(box,0x1d4e5c);bh.material.transparent=true;bh.material.opacity=.5;scene.add(bh);
const grid=new THREE.GridHelper(Math.max(B[1]-B[0],B[4]-B[2]),20,0x1d4e5c,0x10283a);grid.position.set((B[0]+B[1])/2,0,(B[2]+B[4])/2);
grid.material.transparent=true;grid.material.opacity=.4;scene.add(grid);
const labels=DATA.axes.map(a=>{const d=document.createElement('div');d.className='legend';d.style.position='absolute';d.style.right='auto';d.style.top='auto';
  d.style.bottom='auto';d.style.padding='4px 8px';d.style.transform='translate(-50%,-50%)';d.textContent=a.text;wrap.appendChild(d);return {d,v:new THREE.Vector3(...a.pos)};});
let hl=null;if(DATA.highlight){const h=DATA.highlight;hl=new THREE.Mesh(new THREE.SphereGeometry(.7,24,24),new THREE.MeshBasicMaterial({color:0xF59E42}));
  hl.position.set(h[0],h[1],h[2]);scene.add(hl);const halo=new THREE.Mesh(new THREE.RingGeometry(.8,1.0,48),new THREE.MeshBasicMaterial({color:0xF59E42,side:THREE.DoubleSide,transparent:true,opacity:.8}));
  hl.add(halo);hl.userData.halo=halo;const beam=new THREE.Mesh(new THREE.CylinderGeometry(.03,.03,h[1],8),new THREE.MeshBasicMaterial({color:0xF59E42,transparent:true,opacity:.5}));
  beam.position.set(h[0],h[1]/2,h[2]);scene.add(beam);}
const target=new THREE.Vector3(0,6,0);
const orbit=makeOrbit(renderer.domElement,cam,target,{r:DATA.radius,theta:.75,phi:1.18,minR:8,maxR:DATA.radius*3,auto:true});
const btnA=document.getElementById('bAuto');btnA.setAttribute('aria-pressed','true');btnA.onclick=()=>{orbit.auto=!orbit.auto;btnA.setAttribute('aria-pressed',String(orbit.auto));};
document.getElementById('bReset').onclick=()=>{orbit.theta=.75;orbit.phi=1.18;orbit.r=DATA.radius;};
const ray=new THREE.Raycaster();ray.params.Points.threshold=.6;const mouse=new THREE.Vector2(-9,-9);let mx=0,my=0;
renderer.domElement.addEventListener('pointermove',e=>{const r=renderer.domElement.getBoundingClientRect();mx=e.clientX-r.left;my=e.clientY-r.top;
  mouse.x=mx/r.width*2-1;mouse.y=-(my/r.height)*2+1;});
renderer.domElement.addEventListener('pointerleave',()=>{mouse.set(-9,-9);tip.style.display='none';});
function resize(){const w=wrap.clientWidth,h=wrap.clientHeight;renderer.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}
new ResizeObserver(resize).observe(wrap);resize();
const t0=performance.now();const v=new THREE.Vector3();
function frame(){const t=(performance.now()-t0)/1000;const e=Math.min(1,t/1.6);const k=1-Math.pow(1-e,3);cloud.scale.y=Math.max(.001,k);
  if(hl){const s=1+.25*Math.sin(t*3);hl.userData.halo.scale.set(s,s,s);hl.userData.halo.lookAt(cam.position);}
  orbit.update();renderer.render(scene,cam);
  labels.forEach(l=>{v.copy(l.v).project(cam);l.d.style.left=((v.x*.5+.5)*wrap.clientWidth)+'px';l.d.style.top=((-v.y*.5+.5)*wrap.clientHeight)+'px';l.d.style.opacity=v.z<1?1:0;});
  ray.setFromCamera(mouse,cam);const hit=ray.intersectObject(cloud)[0];
  if(hit){const p=P[hit.index];tip.style.display='block';tip.style.left=(mx+14)+'px';tip.style.top=(my+10)+'px';tip.innerHTML=p[4];}else{tip.style.display='none';}
  requestAnimationFrame(frame);}
frame();
})();
"""


def _page(body_js: str, data: dict, hud: str, extra: str = "") -> str:
    loader = _LOADER_JS.replace("%LOCAL%", LOCAL_THREE).replace("%CDN%", CDN_THREE)
    payload = json.dumps(data).replace("</", "<\\/")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{_BASE_CSS}</style></head><body><div id='wrap'>{hud}{extra}<div id='tip'></div>"
        "<div id='err'></div></div>"
        f"<script type='module'>{loader}\n{body_js.replace('%DATA%', payload)}</script></body></html>"
    )


def _num(v, default=0.0) -> float:
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def house_spec(record: pd.Series | dict, label: str, price: str, sub: str = "") -> dict:
    """Geometry parameters for one house, read from a raw Kaggle-format record."""
    r = record
    return {
        "label": label, "price": price, "sub": sub,
        "first_sf": _num(r.get("1stFlrSF"), 1000), "second_sf": _num(r.get("2ndFlrSF")),
        "bsmt_sf": _num(r.get("TotalBsmtSF")), "garage_cars": int(_num(r.get("GarageCars"))),
        "pool": _num(r.get("PoolArea")) > 0, "fireplaces": int(_num(r.get("Fireplaces"))),
        "quality": int(_num(r.get("OverallQual"), 5)), "lot_area": _num(r.get("LotArea"), 9000),
    }


def house_scene_html(houses: list[dict], autorotate: bool = True) -> str:
    hud = ("<div class='hud'><button id='bAuto' type='button'>Auto-rotate</button>"
           "<button id='bXray' type='button' aria-pressed='false'>X-ray walls</button>"
           "<button id='bReset' type='button'>Reset view</button></div>"
           "<div class='hint'>Drag to orbit · scroll to zoom</div>")
    legend = ("<div class='legend'><i style='background:#F5C76B'></i>Quality 9–10<br>"
              "<i style='background:#2DD4BF'></i>Quality 7–8<br><i style='background:#5EA2F0'></i>Quality 5–6<br>"
              "<i style='background:#E08A4A'></i>Quality 1–4<br><i style='background:#3987E5;opacity:.6'></i>Basement</div>")
    return _page(_HOUSE_JS, {"houses": houses, "autorotate": autorotate}, hud, legend)


def _scale(v: np.ndarray, lo: float, hi: float, a: float, b: float) -> np.ndarray:
    return a + (np.clip(v, lo, hi) - lo) / (hi - lo + 1e-9) * (b - a)


def market_galaxy_html(df: pd.DataFrame, highlight: dict | None = None) -> str:
    """3-D point cloud: x = living area, z = year built, y = sale price (each axis scaled to a cube)."""
    gla, yb, sp = (df["GrLivArea"].to_numpy(float), df["YearBuilt"].to_numpy(float), df["SalePrice"].to_numpy(float))
    ref = {"gla": (334.0, 3500.0), "yb": (1880.0, 2010.0), "sp": (34900.0, 500000.0)}
    x = _scale(gla, *ref["gla"], -12, 12)
    z = _scale(yb, *ref["yb"], -12, 12)
    y = _scale(sp, *ref["sp"], 0, 16)
    lp = np.log(sp)
    t = (lp - np.log(ref["sp"][0])) / (np.log(ref["sp"][1]) - np.log(ref["sp"][0]))
    t = np.clip(t, 0, 1)
    tips = [f"<b>${p:,.0f}</b><br>{nb} · Q{q}<br>{int(a):,} sq ft · built {int(b)}<br><span style='color:#7D8BA1'>Id {i}</span>"
            for p, nb, q, a, b, i in zip(sp, df["Neighborhood"], df["OverallQual"], gla, yb, df["Id"])]
    pts = [[round(a, 3), round(b, 3), round(c, 3), round(d, 4), s_] for a, b, c, d, s_ in zip(x, y, z, t, tips)]
    B = [-13.0, 13.0, -13.0, 17.0, 13.0]
    axes = [{"text": "Living area →", "pos": [13, 0, -13]}, {"text": "Newer →", "pos": [-13, 0, 13]},
            {"text": "↑ Sale price", "pos": [-13, 17, -13]}]
    hl = None
    if highlight:
        hl = [float(_scale(np.array([highlight["GrLivArea"]]), *ref["gla"], -12, 12)[0]),
              float(_scale(np.array([highlight["price"]]), *ref["sp"], 0, 16)[0]),
              float(_scale(np.array([highlight["YearBuilt"]]), *ref["yb"], -12, 12)[0])]
    hud = ("<div class='hud'><button id='bAuto' type='button'>Auto-rotate</button>"
           "<button id='bReset' type='button'>Reset view</button></div>"
           "<div class='hint'>Drag to orbit · scroll to zoom · hover a point</div>")
    legend = ("<div class='legend'>Each point = one real sale<br>"
              "<i style='background:linear-gradient(90deg,#0f3338,#1FA592,#7EE8D2,#FFDB8C)'></i>Price low → high"
              + ("<br><i style='background:#F59E42'></i>Your property" if hl else "") + "</div>")
    return _page(_GALAXY_JS, {"points": pts, "bounds": B, "axes": axes, "highlight": hl, "radius": 46}, hud, legend)
