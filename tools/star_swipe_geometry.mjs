// No OCR/browser. Reuse the previous audit's frozen authoritative pure functions.
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const root=process.argv[2];
if(!root) throw new Error('Pass the previous audit _runtime directory explicitly');
const load=name=>import(pathToFileURL(path.join(root,name+'.js')).href);
const {createScreenshotProfile}=await load('profiles');
const {findCircleProposals,buildCardCandidates}=await load('main-grid');
const {refineStarContentBounds}=await load('star-content-bounds');
let input='';for await(const part of process.stdin)input+=part;
const results=JSON.parse(input).map(record=>{
 const image={width:record.width,height:record.height,data:new Uint8ClampedArray(Buffer.from(record.rgba,'base64'))};
 const base=createScreenshotProfile(image);
 const proposals=findCircleProposals(image,base);
 const profile=refineStarContentBounds(base,null,'maayuan_mumu');
 const cards=buildCardCandidates(proposals,profile);
 const rows=new Map();for(const card of cards){const row=rows.get(card.rowIndex)??[];row.push(card);rows.set(card.rowIndex,row)}
 const centers=[...rows.values()].map(row=>row.reduce((s,c)=>s+c.circle.centerY,0)/row.length);
 const pitches=centers.slice(1).map((v,i)=>v-centers[i]);
 return {fileName:record.fileName,profile,cards,
  fourColumnRows:[...rows.values()].filter(row=>new Set(row.map(c=>c.columnIndex)).size===4).length,
  rowCenters:centers,rowPitches:pitches,
  completenessCounts:Object.fromEntries(['complete','partial_top','partial_bottom','invalid'].map(k=>[k,cards.filter(c=>c.completeness===k).length])),
  limit:'geometry evidence only; excludes cannot be repaired here; no OCR or full-inventory claim'};
});
process.stdout.write(JSON.stringify(results));
