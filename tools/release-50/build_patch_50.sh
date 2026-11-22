nice python3 - <<'PY'
from pathlib import Path
import subprocess
import zipfile

workspace = Path('/Users/snake/Desktop/MaaYuan-dev')
build = Path('/private/tmp/claude-501/-Users-snake-Desktop-MaaYuan-dev/173c51f9-c67e-40ed-9f84-8ccde2b5734b/scratchpad/build-rel')
version = 'v2.3.2-yuanhub.20261111.50'
previous = build / 'MaaYuan-v2.3.2-yuanhub.20261111.49-win-x64'
current = build / f'MaaYuan-{version}-win-x64'
output = workspace / 'dist' / f'MaaYuan-{version}-patch.zip'
with zipfile.ZipFile(workspace / 'dist/MaaYuan-v2.3.2-yuanhub.20261111.49-patch.zip') as archive:
    inherited = {name for name in archive.namelist() if not name.endswith('/') and name != '补丁说明.txt'}
    old_note = archive.read('补丁说明.txt').decode('utf-8-sig').replace('\\r\\n', '\n').replace('\r\n', '\n')
changes = subprocess.check_output(['rsync', '-rcn', '--out-format=%n', str(current) + '/', str(previous) + '/'], text=True)
changed = {name for name in changes.splitlines() if (current / name).is_file()}
files = inherited | changed | {'interface.json'}
missing = sorted(name for name in files if not (current / name).is_file())
if missing:
    raise RuntimeError('累积补丁存在已删除文件：' + repr(missing))
previous_files = {p.relative_to(previous).as_posix() for p in previous.rglob('*') if p.is_file()}
current_files = {p.relative_to(current).as_posix() for p in current.rglob('*') if p.is_file()}
deleted = sorted(previous_files - current_files - {'补丁说明.txt'})
if deleted:
    raise RuntimeError('本版有删除文件，需处理删除说明：' + repr(deleted))
history = '\n'.join(old_note.splitlines()[3:]).replace('本次（.49）', '此前（.49）')
note = '\n'.join([
    f'MaaYuan 补丁 {version}',
    f'适用于已安装 v2.3.2-yuanhub.20261111.x 的 MaaYuan，覆盖后升级至 {version}。',
    '用法：关闭 MaaYuan，把压缩包内的文件和文件夹解压到 MaaYuan 安装目录覆盖，再重新打开。',
    '本次（.50）：背包列表识别为陈纪、张仲景的心纸时，自动点开详情核对名称和拥有数量；未确认时保留原库存。优化背包提示，不再把目录中没确认到的条目总数显示为警告，仍提示具体未确认的物品名称。',
    history,
]).rstrip('\n') + '\n'
with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name in sorted(files):
        archive.write(current / name, name)
    archive.writestr('补丁说明.txt', note.replace('\n', '\r\n').encode('utf-8-sig'))
print('本版差异文件:', '\n'.join(sorted(changed)), sep='\n')
print('继承文件:', len(inherited), '累积补丁文件:', len(files) + 1, '删除程序文件:', len(deleted))
print('补丁字节数:', output.stat().st_size)
PY
