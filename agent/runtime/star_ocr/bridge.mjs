// Packaged synchronous YuanStar adapter. No network or OCR model.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {pathToFileURL} from 'node:url';
const runtime=process.argv[2];
const load=n=>import(pathToFileURL(path.join(runtime,n+'.js')).href);
const grid=await load('main-grid'),profiles=await load('profiles'),bounds=await load('star-content-bounds'),quality=await load('quality-postprocess'),post=await load('star-postprocess');
const catalog=JSON.parse(fs.readFileSync(path.join(runtime,'star_catalog.json'),'utf8')).stars;
const aliases=JSON.parse(fs.readFileSync(path.join(runtime,'ocr_aliases.json'),'utf8'));
for await(const line of readline.createInterface({input:process.stdin})) {
 try {
  const q=JSON.parse(line);let result;
  if(q.op==='geometry') {
   const start=performance.now(),im={width:q.width,height:q.height,data:new Uint8ClampedArray(Buffer.from(q.rgba,'base64'))};
   const base=profiles.createScreenshotProfile(im),proposals=grid.findCircleProposals(im,base),profile=bounds.refineStarContentBounds(base,null,'maayuan_mumu');
   const cards=grid.buildCardCandidates(proposals,profile);const geometryMs=performance.now()-start,qt=performance.now();
   for(const c of cards)c.qualityEvidence={...quality.recognizeQuality(im,c),source_rect:quality.qualityRectForCard(c)};
   const rows=new Map();for(const c of cards){const r=rows.get(c.rowIndex)??[];r.push(c);rows.set(c.rowIndex,r);}
   const centers=[...rows.values()].map(r=>r.reduce((s,c)=>s+c.circle.centerY,0)/r.length);
   result={profile,cards,geometryMs,qualityMs:performance.now()-qt,rowCenters:centers,rowPitches:centers.slice(1).map((y,i)=>y-centers[i]),completenessCounts:Object.fromEntries(['complete','partial_top','partial_bottom','invalid'].map(k=>[k,cards.filter(c=>c.completeness===k).length]))};
  } else if(q.op==='experienceGeometry') {
   const start=performance.now(),im={width:q.width,height:q.height,data:new Uint8ClampedArray(Buffer.from(q.rgba,'base64'))};
   const eg=await load('experience-geometry'),ep=await load('experience-postprocess');
   const profile=profiles.createScreenshotProfile(im),tab=eg.selectedExperienceTabEvidence(im,profile.viewport);
   const found=eg.findExperienceCircles(im,profile,tab);
   const candidates=eg.buildExperienceCandidates(found.row,found.bounds).map(c=>{const quality=ep.classifyExperienceKind(im,c.iconRect);return {...c,quality,canonicalName:quality.kind?ep.EXPERIENCE_CANONICAL[quality.kind]:null};});
   result={profile,tab,bounds:found.bounds,coverage:found.coverage,candidates,geometryMs:performance.now()-start};
  } else if(q.op==='experienceAggregate') {
   const ep=await load('experience-postprocess');result=ep.aggregateExperience(q.results,q.options);
  } else if(q.op==='experienceCount') {
   const ep=await load('experience-postprocess');result=ep.resolveExperienceCount(q.candidates);
  } else if(q.op==='resolve') {
   const allowed=catalog.filter(s=>s.kind===(q.section==='main'?'主星':'辅星')).map(s=>s.name);
   result=q.kind==='name'?post.resolveCatalogName(q.candidates,allowed,aliases):post.resolveLevel(q.candidates);
  } else throw Error('unknown operation');
  console.log(JSON.stringify({ok:true,result}));
 }catch(e){console.log(JSON.stringify({ok:false,error:String(e),stack:e.stack}));}
}
