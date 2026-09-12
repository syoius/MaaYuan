[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$RuntimePath
)

$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtimeRoot = (Resolve-Path -LiteralPath $RuntimePath).Path
$backupTag = Get-Date -Format 'yyyyMMdd-HHmmss'

function Backup-TargetFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Copy-Item -LiteralPath $Path -Destination "$Path.yuanstar-probe-backup-$backupTag" -Force
    }
}

function Copy-ProbeFile {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRelativePath,
        [Parameter(Mandatory = $true)][string]$RuntimeRelativePath
    )
    $source = Join-Path $sourceRoot $SourceRelativePath
    $target = Join-Path $runtimeRoot $RuntimeRelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "开发文件不存在: $source"
    }
    $targetDirectory = Split-Path -Parent $target
    if (-not (Test-Path -LiteralPath $targetDirectory -PathType Container)) {
        throw "runtime 目录不存在: $targetDirectory"
    }
    Backup-TargetFile -Path $target
    Copy-Item -LiteralPath $source -Destination $target -Force
    Write-Host "已同步: $SourceRelativePath -> $RuntimeRelativePath"
}

# 只同步正式星石背包采集与完整批次上传所需文件；不触碰 config、日志、MaaYuan.exe、MFAAvalonia 或 Python runtime。
Copy-ProbeFile -SourceRelativePath 'agent/custom/action/star_backpack_capture_probe.py' -RuntimeRelativePath 'agent/custom/action/star_backpack_capture_probe.py'
Copy-ProbeFile -SourceRelativePath 'agent/custom/action/star_backpack_capture_orchestration.py' -RuntimeRelativePath 'agent/custom/action/star_backpack_capture_orchestration.py'
Copy-ProbeFile -SourceRelativePath 'agent/custom/action/star_capture_transport.py' -RuntimeRelativePath 'agent/custom/action/star_capture_transport.py'
Copy-ProbeFile -SourceRelativePath 'agent/custom/action/__init__.py' -RuntimeRelativePath 'agent/custom/action/__init__.py'
Copy-ProbeFile -SourceRelativePath 'assets/resource/base/pipeline/star_backpack_capture_orchestration.json' -RuntimeRelativePath 'resource/base/pipeline/star_backpack_capture_orchestration.json'

$sourceInterfacePath = Join-Path $sourceRoot 'assets/interface.json'
$runtimeInterfacePath = Join-Path $runtimeRoot 'interface.json'
if (-not (Test-Path -LiteralPath $runtimeInterfacePath -PathType Leaf)) {
    throw "runtime interface.json 不存在: $runtimeInterfacePath"
}

$sourceInterface = Get-Content -LiteralPath $sourceInterfacePath -Encoding UTF8 -Raw | ConvertFrom-Json
$runtimeInterface = Get-Content -LiteralPath $runtimeInterfacePath -Encoding UTF8 -Raw | ConvertFrom-Json
if ($null -eq $sourceInterface.task -or $null -eq $runtimeInterface.task) {
    throw 'interface schema 缺少 task，拒绝覆盖 runtime interface.json'
}

$debugTaskNames = @(
    '开发调试｜星石背包截图探针',
    '开发调试｜星石背包单次滑动探针',
    '开发调试｜星石背包反馈滑动探针',
    '开发调试｜星石背包连续采集',
    '开发调试｜星石背包正式三段采集'
)

$sourceToolboxMode = $sourceInterface.option.PSObject.Properties['百宝箱-模式'].Value
$runtimeToolboxMode = $runtimeInterface.option.PSObject.Properties['百宝箱-模式'].Value
if ($null -eq $sourceToolboxMode -or $null -eq $runtimeToolboxMode) {
    throw 'interface schema 缺少 百宝箱-模式，拒绝覆盖 runtime interface.json'
}
$sourceStarCase = @($sourceToolboxMode.cases | Where-Object { $_.name -eq '星石背包自动采集' })
if ($sourceStarCase.Count -ne 1) {
    throw '开发 interface 中未找到唯一的 星石背包自动采集 case'
}

# 只移除历史开发 task，并在既有百宝箱模式里替换正式星石 case。
$runtimeInterface.task = @($runtimeInterface.task | Where-Object { $debugTaskNames -notcontains [string]$_.name })
$patchedCases = New-Object System.Collections.Generic.List[object]
$inserted = $false
foreach ($case in @($runtimeToolboxMode.cases)) {
    if ($case.name -eq '星石背包自动采集') {
        continue
    }
    $patchedCases.Add($case)
    if ($case.name -eq '自动识别背包') {
        $patchedCases.Add($sourceStarCase[0])
        $inserted = $true
    }
}
if (-not $inserted) {
    $patchedCases.Add($sourceStarCase[0])
}
$runtimeToolboxMode.cases = $patchedCases.ToArray()

$sourceSyncOption = $sourceInterface.option.PSObject.Properties['同步星石至YuanHub'].Value
if ($null -eq $sourceSyncOption) {
    throw '开发 interface 中缺少 同步星石至YuanHub option'
}
[void]$runtimeInterface.option.PSObject.Properties.Remove('同步星石至YuanHub')
$runtimeInterface.option | Add-Member -NotePropertyName '同步星石至YuanHub' -NotePropertyValue $sourceSyncOption

Backup-TargetFile -Path $runtimeInterfacePath
$runtimeInterface | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $runtimeInterfacePath -Encoding UTF8
Write-Host "已安全 patch runtime interface.json（百宝箱正式 case 与同步 option）"
Write-Host "完成。每个改写目标已有同目录 .yuanstar-probe-backup-$backupTag 备份。"
