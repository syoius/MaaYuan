"""构建 MaaYuan Windows 完整包和累积补丁包。

用法（在 MaaYuan-dev 工作区内任意目录执行）：
  python3 MaaYuan/tools/release/build_win_release.py \
      --version v2.3.2-yuanhub.20261111.56 \
      --prev-version v2.3.2-yuanhub.20261111.55 \
      --note "本次改动的玩家说明"

内测版使用 N-beta.M 版本号，并传入 --gui-zip 和 --gui-sha256。
--source-ref 可指定已提交的源码版本，保留工作区中的 OCR 子模块和模型。

流程与约定见工作区 CLAUDE.md 第 2 节。脚本只读主仓，产物写入 dist/，构建目录默认在 .release-build/。
任一检查失败即中止，不会覆盖 dist/ 中已有的文件。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
BASE_ZIP = WORKSPACE / ".release-cache" / "MaaYuan-win-x86_64-v2.3.2-beta.1.zip"
BASE_SHA256 = "f1e4d34576bb89c0dab30d6c88f7309c2c44bc2be4812037cc777f093469bf8e"
SERIES = "v2.3.2-yuanhub.20261111."
# install4release.py 会读取或打进包里的路径，必须与已提交版本一致
PACKAGED_PATHS = ["agent", "assets", "README.md", "LICENSE", "requirements.txt",
                  "自定义派遣脚本修改说明.md", "install4release.py", "configure.py"]
RSYNC_EXCLUDES = [".git", "/install", "debug/", "agent/agent", "__pycache__", ".pytest_cache",
                  "/config", "*.sync.meta", "xinzhi-cache", "operators.remote.json", "/tools"]
FORBIDDEN = re.compile(r"(^|/)(debug|xinzhi-cache|__pycache__)/|\.sync\.meta$|operators\.remote\.json$")
CHORES = ["README.md", "LICENSE", "requirements.txt", "自定义派遣脚本修改说明.md"]


def fail(message: str) -> None:
    sys.exit(f"中止：{message}")


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout


def short_version(version: str) -> str:
    if not version.startswith(SERIES):
        fail(f"版本 {version} 不属于 {SERIES}N 系列，变更系列需要用户决定")
    suffix = version[len(SERIES):]
    if not re.fullmatch(r"[0-9]+(?:-beta\.[0-9]+)?", suffix):
        fail(f"版本 {version} 应为 {SERIES}N 或 {SERIES}N-beta.M")
    number, _, beta = suffix.partition("-")
    return beta or f".{number}"


def check_repo_clean() -> str:
    dirty = run("git", "status", "--porcelain", "--", *PACKAGED_PATHS, cwd=REPO).strip()
    if dirty:
        fail("以下打包路径有未提交改动，先提交或与用户确认后再打包，避免漏带或误带：\n" + dirty)
    return run("git", "rev-parse", "--short", "HEAD", cwd=REPO).strip()


def check_base_zip() -> None:
    digest = hashlib.sha256(BASE_ZIP.read_bytes()).hexdigest()
    if digest != BASE_SHA256:
        fail(f"底包 SHA-256 不符：{digest}")


def overlay_source(src: Path, commit: str, excludes: list[str]) -> None:
    head_files = run("git", "ls-tree", "-rz", "--name-only", "HEAD", "--", *PACKAGED_PATHS, cwd=REPO)
    ref_files = run("git", "ls-tree", "-rz", "--name-only", commit, "--", *PACKAGED_PATHS, cwd=REPO)
    for name in set(head_files.split("\0")) - set(ref_files.split("\0")):
        path = src / name
        if path.is_symlink() or path.is_file():
            path.unlink()

    with tempfile.TemporaryDirectory(dir=src.parent) as staging:
        archive = Path(staging) / "source.tar"
        tree = Path(staging) / "tree"
        tree.mkdir()
        run("git", "archive", f"--output={archive}", commit, cwd=REPO)
        shutil.unpack_archive(archive, tree)
        run("rsync", "-aI", *excludes, f"{tree}/", f"{src}/")


def overlay_gui(gui_zip: Path, target: Path) -> None:
    replaced, added = [], []
    with zipfile.ZipFile(gui_zip) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = entry.filename
            path = PurePosixPath(name)
            root_gui = name in {"MaaYuan.exe", "libloader.dll"} or name.startswith("MFAAvalonia.")
            if not (len(path.parts) == 1 and root_gui or name.startswith(("libs/", "runtimes/"))):
                continue
            if name != path.as_posix() or ".." in path.parts or "\\" in name or ":" in name:
                fail(f"GUI ZIP 中的文件路径无效：{name}")
            destination = target / name
            content = archive.read(entry)
            if destination.is_file():
                if destination.read_bytes() == content:
                    continue
                replaced.append(name)
            else:
                added.append(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    print(f"替换 GUI 文件（{len(replaced)}）：", *sorted(replaced), sep="\n  ")
    print(f"新增 GUI 文件（{len(added)}）：", *sorted(added), sep="\n  ")


def duplicate_nodes(pipelines: dict[str, dict]) -> dict[str, list[str]]:
    seen = collections.defaultdict(list)
    for rel, nodes in pipelines.items():
        for key in nodes:
            seen[key].append(rel)
    return {k: v for k, v in seen.items() if len(v) > 1}


def check_build_duplicates(build: Path) -> None:
    bad = {}
    for res in sorted((build / "resource").iterdir()):
        pipe = res / "pipeline"
        if pipe.is_dir():
            dup = duplicate_nodes({p.relative_to(pipe).as_posix(): json.loads(p.read_text(encoding="utf-8-sig"))
                                   for p in sorted(pipe.rglob("*.json"))})
            if dup:
                bad[res.name] = dup
    if bad:
        fail(f"合并底包后 pipeline 节点重名（底包旧文件需用 {{}} 占位覆盖）：{bad}")


def build_full(version: str, build_dir: Path, out: Path, source_commit: str | None,
               gui_zip: Path | None) -> Path:
    name = f"MaaYuan-{version}-win-x64"
    target, src = build_dir / name, build_dir / "src"
    shutil.rmtree(target, ignore_errors=True)
    shutil.rmtree(src, ignore_errors=True)
    zipfile.ZipFile(BASE_ZIP).extractall(target)
    release_agent = json.loads((target / "interface.json").read_text(encoding="utf-8"))["agent"]

    excludes = [arg for pattern in RSYNC_EXCLUDES for arg in ("--exclude", pattern)]
    run("rsync", "-a", *excludes, f"{REPO}/", f"{src}/")
    if source_commit:
        overlay_source(src, source_commit, excludes)
    run(sys.executable, "install4release.py", version, cwd=src)
    install = src / "install"

    shutil.rmtree(target / "agent")
    shutil.copytree(install / "agent", target / "agent")
    shutil.copytree(install / "resource", target / "resource", dirs_exist_ok=True)
    for chore in CHORES:
        shutil.copy2(install / chore, target / chore)
    interface = json.loads((install / "interface.json").read_text(encoding="utf-8"))
    interface["agent"] = release_agent  # 保留底包的 Windows 内置 Python 启动方式
    (target / "interface.json").write_text(json.dumps(interface, ensure_ascii=False, indent=4), encoding="utf-8")
    if gui_zip:
        overlay_gui(gui_zip, target)

    for cache in target.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    for junk in target.rglob(".DS_Store"):
        junk.unlink()
    for required in ("MaaYuan.exe", "python/python.exe"):
        if not (target / required).is_file():
            fail(f"完整包缺少 {required}")
    if (target / "agent" / "tests").exists():
        fail("完整包含 agent/tests")
    leaked = [p.relative_to(target).as_posix() for p in target.rglob("*")
              if not p.relative_to(target).as_posix().startswith("python/")
              and FORBIDDEN.search(p.relative_to(target).as_posix() + ("/" if p.is_dir() else ""))]
    if leaked:
        fail(f"完整包含运行时缓存或调试数据：{leaked[:10]}")
    check_build_duplicates(target)

    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in target.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(target).as_posix())
    return target


def build_patch(version: str, prev_version: str, note_text: str, current: Path, build_dir: Path, out: Path,
                dist: Path) -> list[str]:
    prev_full = dist / f"MaaYuan-{prev_version}-win-x64.zip"
    prev_patch = dist / f"MaaYuan-{prev_version}-patch.zip"
    for path in (prev_full, prev_patch):
        if not path.is_file():
            fail(f"缺少上一版产物 {path.name}，无法确定补丁基线")
    previous = build_dir / f"MaaYuan-{prev_version}-win-x64"
    shutil.rmtree(previous, ignore_errors=True)
    zipfile.ZipFile(prev_full).extractall(previous)

    with zipfile.ZipFile(prev_patch) as archive:
        inherited = {n for n in archive.namelist() if not n.endswith("/") and n != "补丁说明.txt"}
        old_note = archive.read("补丁说明.txt").decode("utf-8-sig").replace("\r\n", "\n").splitlines()
    diff = run("rsync", "-rcn", "--out-format=%n", f"{current}/", f"{previous}/")
    changed = {n for n in diff.splitlines() if (current / n).is_file()}
    files = inherited | changed | {"interface.json"}
    missing = sorted(n for n in files if not (current / n).is_file())
    if missing:
        fail(f"累积补丁中的文件在新版已删除，补丁无法表达删除，需要用户决定：{missing}")
    current_files = {p.relative_to(current).as_posix() for p in current.rglob("*") if p.is_file()}
    deleted = sorted({p.relative_to(previous).as_posix() for p in previous.rglob("*") if p.is_file()} - current_files)

    current_short, prev_short = short_version(version), short_version(prev_version)
    beta = current_short.startswith("beta.")
    if beta and not prev_short.startswith("beta."):
        history = [f"已包含正式版 {prev_short} 及之前的全部改动。"]
    else:
        if len(old_note) < 4 or not old_note[3].startswith(f"本次（{prev_short}）"):
            fail(f"上一版补丁说明第 4 行不是「本次（{prev_short}）」，请人工核对说明格式")
        history = [old_note[3].replace(f"本次（{prev_short}）", f"此前（{prev_short}）", 1)] + old_note[4:]
    scope = f"{SERIES}x（正式版或内测版）的" if beta else f"{SERIES}x 的"
    note = "\n".join([
        f"MaaYuan 补丁 {version}",
        f"适用于已安装 {scope} MaaYuan，覆盖后升级至 {version}。",
        "用法：关闭 MaaYuan，把压缩包内的文件和文件夹解压到 MaaYuan 安装目录覆盖，再重新打开。",
        f"本次（{current_short}）：{note_text}",
        *history,
    ]).rstrip("\n") + "\n"

    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(files):
            archive.write(current / name, name)
        archive.writestr("补丁说明.txt", note.replace("\n", "\r\n").encode("utf-8-sig"))

    # 模拟玩家在上一版完整包上覆盖补丁后的 pipeline，检查节点重名
    with zipfile.ZipFile(out) as archive:
        patched = set(archive.namelist())
        for res in sorted({n.split("/")[1] for n in patched if re.match(r"resource/[^/]+/pipeline/.+\.json$", n)}
                          | {p.name for p in (previous / "resource").iterdir() if (p / "pipeline").is_dir()}):
            pipelines = {}
            prefix = f"resource/{res}/pipeline/"
            for path in sorted((previous / prefix).rglob("*.json")) if (previous / prefix).is_dir() else []:
                rel = path.relative_to(previous).as_posix()
                source = archive.read(rel).decode("utf-8-sig") if rel in patched else path.read_text(encoding="utf-8-sig")
                pipelines[rel] = json.loads(source)
            for rel in patched:
                if rel.startswith(prefix) and rel.endswith(".json") and rel not in pipelines:
                    pipelines[rel] = json.loads(archive.read(rel).decode("utf-8-sig"))
            dup = duplicate_nodes(pipelines)
            if dup:
                out.unlink()
                fail(f"上一版完整包覆盖补丁后 {res} 节点重名：{dup}")
    print("本版差异文件：", *sorted(changed), sep="\n  ")
    print(f"继承 {len(inherited)} 个，补丁共 {len(files) + 1} 个文件")
    return deleted


def verify(version: str, full_zip: Path, patch_zip: Path) -> None:
    for label, path in (("完整包", full_zip), ("补丁包", patch_zip)):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            packaged = json.loads(archive.read("interface.json"))["version"]
            if packaged != version:
                fail(f"{label} interface.json 版本为 {packaged}")
            leaked = [n for n in names if not n.startswith("python/") and FORBIDDEN.search(n)]
            if leaked:
                fail(f"{label}含运行时缓存或调试数据：{leaked[:10]}")
            print(f"{label} {path.name}：{len(names)} 个文件，SHA-256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--prev-version", required=True)
    parser.add_argument("--note", required=True, help="本次改动的玩家说明，不含「本次（.N 或 beta.M）：」前缀")
    parser.add_argument("--source-ref", help="覆盖源码副本中已跟踪文件的 Git ref")
    parser.add_argument("--gui-zip", type=Path, help="内测 GUI 的 Windows x64 ZIP")
    parser.add_argument("--gui-sha256", help="内测 GUI ZIP 的 SHA-256")
    parser.add_argument("--build-dir", type=Path, default=WORKSPACE / ".release-build")
    parser.add_argument("--dist-dir", type=Path, default=WORKSPACE / "dist", help="读取上一版产物并写入新产物的目录")
    args = parser.parse_args()

    beta = short_version(args.version).startswith("beta.")
    if beta != (args.gui_zip is not None):
        parser.error("内测版本必须传入 --gui-zip，正式版本使用官方底包 GUI")
    if (args.gui_zip is None) != (args.gui_sha256 is None):
        parser.error("--gui-zip 和 --gui-sha256 必须一起传入")
    short_version(args.prev_version)
    full_zip = args.dist_dir / f"MaaYuan-{args.version}-win-x64.zip"
    patch_zip = args.dist_dir / f"MaaYuan-{args.version}-patch.zip"
    existing = [p.name for p in (full_zip, patch_zip) if p.exists()]
    if existing:
        fail(f"目标文件已存在，先与用户确认如何处理：{existing}")
    head = check_repo_clean()
    check_base_zip()
    source_commit = None
    if args.source_ref:
        source_commit = run("git", "rev-parse", "--verify", "--end-of-options",
                            f"{args.source_ref}^{{commit}}", cwd=REPO).strip()
    if args.gui_zip:
        digest = hashlib.sha256(args.gui_zip.read_bytes()).hexdigest()
        if digest != args.gui_sha256.lower():
            fail(f"GUI ZIP SHA-256 不符：{digest}")
    args.build_dir.mkdir(parents=True, exist_ok=True)

    print(f"主仓 HEAD {head}，构建 {args.version}")
    if source_commit:
        print(f"源码 ref {args.source_ref}：{source_commit}")
    current = build_full(args.version, args.build_dir, full_zip, source_commit, args.gui_zip)
    try:
        deleted = build_patch(args.version, args.prev_version, args.note, current, args.build_dir, patch_zip,
                              args.dist_dir)
    except BaseException:
        full_zip.unlink(missing_ok=True)
        raise
    verify(args.version, full_zip, patch_zip)
    if deleted:
        print("提醒用户：以下文件在新版已删除，补丁无法删除玩家本地的旧文件：", *deleted, sep="\n  ")


if __name__ == "__main__":
    main()
