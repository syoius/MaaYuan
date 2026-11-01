"""Build-time only: pin and bundle the portable Node used by local star OCR.

Desktop archives and hashes are from nodejs.org/dist/v24.19.0/SHASUMS256.txt.
No downloads, PATH lookup or installation occur in the running task.
"""
from pathlib import Path
import hashlib,io,json,os,platform,sys,tarfile,urllib.request,zipfile

VERSION='v24.19.0'
ARCHIVES={
 ('win','x64'):('zip','57f71ab3652e797d84acddc79c81cc9ff1c6ddb2a1974cdb83f00fee9bff4c73'),
 ('win','arm64'):('zip','8502f4a50b458d4cc38ed8f2001556c2cd239d464920f74017926ccb1e1c157f'),
 ('linux','x64'):('tar.gz','f625d97cd707df4ff96254916fbc5ff014f09c09effe5a1e0ca8f6d41a8789d4'),
 ('linux','arm64'):('tar.gz','d28c8a5bf0a808f0ed434a1dce8c54ae98f0371c0bd86ac58abc613f73e6643f'),
 ('darwin','x64'):('tar.gz','d1b5e999db158c62fe8f7267a4476b035d8bd93b1a605bac24a3f0dd166e3316'),
 ('darwin','arm64'):('tar.gz','8294b7aa9b03997481c06babf1e8b270c859358f27da57a11509afe537ac381d'),
}


def target(system=None,arch=None):
 system=system or ('win' if sys.platform=='win32' else sys.platform)
 raw=arch or os.environ.get('STAR_OCR_NODE_ARCH') or platform.machine().lower()
 arch={'amd64':'x64','x86_64':'x64','aarch64':'arm64'}.get(raw,raw)
 if (system,arch) not in ARCHIVES:raise RuntimeError(f'Local star OCR Node not packaged for {system}/{arch}')
 return system,arch


def bundle(install_path,*,system=None,arch=None,fetch=None):
 system,arch=target(system,arch);extension,digest=ARCHIVES[system,arch]
 archive=f'node-{VERSION}-{system}-{arch}.{extension}';url=f'https://nodejs.org/dist/{VERSION}/{archive}'
 out=Path(install_path)/'runtimes/star-ocr-node';out.mkdir(parents=True,exist_ok=True)
 binary='node.exe' if system=='win' else 'node';destination=out/binary
 manifest=out/'manifest.json'
 if manifest.exists() and destination.exists():
  prior=json.loads(manifest.read_text(encoding='utf-8'))
  if prior.get('archive')==archive and prior.get('binarySha256')==hashlib.sha256(destination.read_bytes()).hexdigest() and (out/'LICENSE').is_file():return destination
 def download():
  with urllib.request.urlopen(url,timeout=120) as response:return response.read()
 content=fetch(url) if fetch else download()
 if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Node archive SHA256 mismatch')
 prefix=f'node-{VERSION}-{system}-{arch}/';member=prefix+('node.exe' if system=='win' else 'bin/node')
 # Read only these exact members; never extract archive paths into the workspace.
 if extension=='zip':
  with zipfile.ZipFile(io.BytesIO(content)) as z:executable=z.read(member);license_bytes=z.read(prefix+'LICENSE')
 else:
  with tarfile.open(fileobj=io.BytesIO(content),mode='r:gz') as z:
   executable=z.extractfile(member).read();license_bytes=z.extractfile(prefix+'LICENSE').read()
 destination.write_bytes(executable);destination.chmod(0o755);(out/'LICENSE').write_bytes(license_bytes)
 manifest.write_text(json.dumps({'version':VERSION,'platform':system,'arch':arch,'archive':archive,'url':url,'archiveSha256':digest,'binarySha256':hashlib.sha256(executable).hexdigest()},indent=2),encoding='utf-8')
 return destination


if __name__=='__main__':print(bundle(Path(sys.argv[1])))
