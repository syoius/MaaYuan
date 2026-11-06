// Shadow-only synchronous adapter to unchanged current YuanStar Web pure functions.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {pathToFileURL} from 'node:url';
if(process.argv[2]==='prepare') {
 const {stripTypeScriptTypes}=await import('node:module');
 const source=process.argv[3],out=process.argv[4]; fs.mkdirSync(out,{recursive:true});
 const names=['main-grid','profiles','card-completeness','quality-postprocess','experience-geometry','experience-postprocess','star-content-bounds','star-postprocess'];
 for(const name of names)fs.writeFileSync(path.join(out,name+'.js'),stripTypeScriptTypes(fs.readFileSync(path.join(source,'web/src/structured',name+'.ts'),'utf8')).replace(/[ \t]+$/gm,'').trimEnd()+'\n','utf8');
 for(const name of ['star_catalog.json','ocr_aliases.json'])fs.copyFileSync(path.join(source,'data',name),path.join(out,name));
 fs.writeFileSync(path.join(out,'package.json'),JSON.stringify({type:'module'}));
 console.log(JSON.stringify({prepared:names,source,out}));process.exit(0);
}
await import(pathToFileURL(path.resolve(import.meta.dirname,'../agent/runtime/star_ocr/bridge.mjs')).href);
