function Elevate {
    param
    (
        [parameter(Position = 0)]
        [string]$ScriptPath = $script:MyInvocation.MyCommand.Path,
        [parameter(Position = 1, Mandatory = $true)]
        [hashtable]$Params
    )
    # This will hold our argument string that gets passed to the new powershell instance.
    $arg = ""
    # Only iterate over the params object if we need to
    if (-not [string]::IsNullOrEmpty($Params)) {
        # Iterate over the parameters the parent script got and turn them into a string of arguments
        # to pass to the new session
        Foreach ($key in $params.Keys) {
            $value = $params[$key]
            $arg += "-$key $value"
        }
    }
    # Provide Feedback
    Write-Host("Relaunching script as Administrator!")
    # Only run if we aren't running as Administrator
    If (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
    [Security.Principal.WindowsBuiltInRole] "Administrator")) {
        Write-Verbose("Restarting script with Administrator rights")
        # Run script in a new session as Administrator.
        Start-Process -FilePath powershell.exe -ArgumentList @("-File `"$ScriptPath`" $arg") `
                      -Verb runas #-WindowStyle 'Hidden'
        Write-Verbose("Ending current Session")
        # Return non zero exit code that can be used to check if script was relaunched
        $host.SetShouldExit(42)
        # End current session and let the new one take over
        Exit
    }
}
# Elevate this script, passing any arguments to the new session
Elevate -Params $PSBoundParameters

# 检查并安装 .NET 8 和 Visual C++ 可再发行程序包

# 检查 .NET Runtime 是否已安装的函数
function Test-DotNetRuntime {
<# 暂时注释掉，需要始终安装最新版本
    $dotnetVersions = dotnet --list-runtimes
    return $dotnetVersions -match "Microsoft.WindowsDesktop.App 8.0"
#>
    return $false
}

# 检查 VC++ Redistributable 是否已安装的函数
function Test-VCRedist {
<# 暂时注释掉，vcredist需要始终安装最新版本
$vcRedist = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
                                     "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*" |
                Where-Object { $_.DisplayName -like "*Microsoft Visual C++ 2015-2022*" -and $_.DisplayName -like "*x64*" }
    return $null -ne $vcRedist
#>
    return $false
}

# 下载并安装程序的函数
function Install-FromUrl {
    param (
        [string]$url,
        [string]$outputFile,
        [string]$arguments
    )

    Write-Host "正在下载安装程序..."
    $tempPath = [System.IO.Path]::GetTempPath()
    $downloadPath = Join-Path $tempPath $outputFile

    try {
        Invoke-WebRequest -Uri $url -OutFile $downloadPath
        Write-Host "正在安装..."
        Start-Process -FilePath $downloadPath -ArgumentList $arguments -Wait
        Remove-Item $downloadPath
    }
    catch {
        Write-Host "安装过程中出现错误: $_"
    }
}

# 检查并安装 .NET 8 Desktop Runtime
if (-not (Test-DotNetRuntime)) {
    Write-Host "正在安装 .NET 8 Desktop Runtime..."
    $tempPath = [System.IO.Path]::GetTempPath()
    $installerPath = Join-Path $tempPath "dotnet-install.ps1"

    try {
        # 下载安装脚本
        Invoke-WebRequest -Uri "https://dot.net/v1/dotnet-install.ps1" -OutFile $installerPath

        # 执行安装脚本
        & $installerPath -Channel 8.0 -Runtime windowsdesktop -InstallDir "C:\Program Files\dotnet" -Architecture x64

        # 清理安装脚本
        Remove-Item $installerPath

        # 刷新环境变量
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    }
    catch {
        Write-Host "安装 .NET Runtime 时出现错误: $_"
    }
} else {
    Write-Host ".NET 8 Desktop Runtime 已安装"
}

# 检查并安装 Visual C++ 可再发行程序包
if (-not (Test-VCRedist)) {
    Write-Host "正在安装 Visual C++ 可再发行程序包..."
    $vcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Install-FromUrl -url $vcRedistUrl -outputFile "vc_redist.x64.exe" -arguments "/quiet /norestart"
} else {
    Write-Host "Visual C++ 可再发行程序包已安装"
}

# 检查系统是否为 64 位
if ([Environment]::Is64BitOperatingSystem) {
    Write-Host "系统为 64 位，继续执行"
} else {
    Write-Host "系统不支持，MAA 不支持 32 位操作系统。"
    exit
}

# 检查 Windows 7 的补丁更新（如果适用）
if ([System.Environment]::OSVersion.Version.Major -eq 6 -and [System.Environment]::OSVersion.Version.Minor -eq 1) {
    Write-Host "检测到 Windows 7，检查必需的补丁..."

    # 检查 Service Pack 1 是否安装
    if (-not (Get-WmiObject -Query "SELECT * FROM Win32_OperatingSystem WHERE ServicePackMajorVersion >= 1")) {
        Write-Host "需要安装 Windows 7 Service Pack 1。"
        Write-Host "请访问以下链接下载并安装 Windows 7 SP1："
        Write-Host "https://www.microsoft.com/download/details.aspx?id=5842"
        Write-Host "安装 SP1 后请重新运行此脚本。"
        exit
    }

    # 检查并安装 SHA-2 更新补丁 (KB4474419)
    $sha2Update = Get-HotFix -Id KB4474419 -ErrorAction SilentlyContinue
    if (-not $sha2Update) {
        Write-Host "正在安装 SHA-2 更新补丁 (KB4474419)..."
        $kb4474419Url = "http://download.windowsupdate.com/d/msdownload/update/software/secu/2019/09/windows6.1-kb4474419-v3-x64_b5614c6cea5cb4e198717789633dca16308ef79c.msu"
        Install-FromUrl -url $kb4474419Url -outputFile "KB4474419.msu" -arguments "/quiet /norestart"
    }

    # 检查并安装 Platform Update (KB2670838)
    $platformUpdate = Get-HotFix -Id KB2670838 -ErrorAction SilentlyContinue
    if (-not $platformUpdate) {
        Write-Host "正在安装 Platform Update (KB2670838)..."
        $kb2670838Url = "http://download.windowsupdate.com/msdownload/update/software/ftpk/2013/02/windows6.1-kb2670838-x64_9f667ff60e80b64cbed2774681302baeaf0fc6a6.msu"
        Install-FromUrl -url $kb2670838Url -outputFile "KB2670838.msu" -arguments "/quiet /norestart"
    }
}

Write-Host "所有检查通过，MAA 应该可以正常运行。"

# 提示用户重启计算机以完成安装
Read-Host -Prompt "请重启计算机以完成所有更改。按回车键退出"
