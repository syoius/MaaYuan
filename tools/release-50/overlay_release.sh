#!/usr/bin/env bash
# Official MaaYuan Windows release + current local MaaYuan (install4release output) overlaid on top.
set -euo pipefail
S=/private/tmp/claude-501/-Users-snake-Desktop-MaaYuan-dev/173c51f9-c67e-40ed-9f84-8ccde2b5734b/scratchpad
REPO=/Users/snake/Desktop/MaaYuan-dev/MaaYuan
VERSION="${1:?version}"
B=$S/build-rel
OUT=/Users/snake/Desktop/MaaYuan-dev/dist
NAME="MaaYuan-${VERSION}-win-x64"
PY=$S/venv/bin/python
mkdir -p "$OUT"
rm -rf "$B/src" "$B/$NAME"

echo "[1/5] 解压官方发布包 v2.3.2-beta.1 (win x86_64)"
"$PY" - "$B/dl/rel.zip" "$B/$NAME" <<'EOF'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
EOF
cp "$B/$NAME/interface.json" "$B/interface.release.json"

echo "[2/5] 用本地代码执行 install4release.py"
rsync -a --exclude .git --exclude install --exclude debug --exclude 'agent/agent' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude config --exclude '*.sync.meta' "$REPO/" "$B/src/"
(cd "$B/src" && "$PY" install4release.py "$VERSION" >/dev/null)
I="$B/src/install"

echo "[3/5] 覆盖 agent / resource / interface.json / 说明文件"
rm -rf "$B/$NAME/agent"
cp -R "$I/agent" "$B/$NAME/agent"
cp -R "$I/resource/." "$B/$NAME/resource/"
for f in README.md LICENSE requirements.txt 自定义派遣脚本修改说明.md; do cp "$I/$f" "$B/$NAME/$f"; done
"$PY" - "$I/interface.json" "$B/interface.release.json" "$B/$NAME/interface.json" <<'EOF'
import json, sys
ours = json.load(open(sys.argv[1], encoding="utf-8"))
rel = json.load(open(sys.argv[2], encoding="utf-8"))
ours["agent"] = rel["agent"]  # keep the release's Windows embed-python launcher
json.dump(ours, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=4)
print("agent:", ours["agent"]["child_exec"], "version:", ours["version"])
print("keys only in release:", sorted(set(rel) - set(ours)), "only in ours:", sorted(set(ours) - set(rel)))
EOF

echo "[4/5] 自检"
find "$B/$NAME" -name '__pycache__' -type d -prune -exec rm -rf {} +
test -f "$B/$NAME/MaaYuan.exe" && test -f "$B/$NAME/python/python.exe"
test -f "$B/$NAME/agent/custom/action/toolbox_diagnostics_transport.py"
test ! -e "$B/$NAME/agent/tests"
ls "$B/$NAME/resource/base/model/ocr" | head -3

echo "[5/5] 打包"
rm -f "$OUT/$NAME.zip"
nice "$PY" - "$B" "$NAME" "$OUT/$NAME.zip" <<'EOF'
import os, sys, zipfile
base, name, out = sys.argv[1:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk(os.path.join(base, name)):
        dirs.sort()
        for f in sorted(files):
            if f != ".DS_Store":
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, os.path.join(base, name)))
EOF
ls -lh "$OUT/$NAME.zip"
