#!/bin/bash

# 启用颜色输出（
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # 重置颜色

# 核心配置
DOWNLOAD_DIR="$HOME/Downloads"  # 脚本下载目录（用户熟悉的下载文件夹）
DOTNET_INSTALL_SCRIPT="dotnet-install.sh"
DOTNET_INSTALL_PATH="$HOME/.dotnet"  # 非系统级目录，无需管理员权限
DOTNET_CHANNEL="10.0"  # 强制安装 .NET 10，不支持其他版本
INSTALL_TYPE="运行时"  # 固定为运行时，给用户明确提示

# 极简帮助说明
show_help() {
    echo -e "${BOLD}${BLUE}======================================= .NET 10 运行时专属安装工具（macOS）=======================================${NC}"
    echo -e "📌 唯一功能：安装软件必需的 .NET 10 运行时（普通用户直接运行即可）"
    echo -e "📌 适用场景：运行需要 .NET 10 环境的软件（无需开发功能）"
    echo -e "\n${YELLOW}进阶选项（仅开发者使用）：${NC}"
    echo -e "  ./脚本名.sh sdk   → 安装 .NET 10 SDK（普通用户无需使用）"
    echo -e "${NC}"
}

# 解析参数
parse_args() {
    if [ $# -eq 1 ]; then
        if [ "$1" = "sdk" ]; then
            INSTALL_TYPE="SDK"
            echo -e "${YELLOW}🔧 已切换为：安装 .NET 10 SDK（开发者专用）${NC}"
        else
            echo -e "${RED}❌ 无效参数！普通用户直接运行脚本即可（仅支持 'sdk' 开发者选项）${NC}"
            exit 1
        fi
    elif [ $# -gt 1 ]; then
        echo -e "${RED}❌ 无需输入任何参数！直接运行脚本即可安装 .NET 10 运行时${NC}"
        exit 1
    fi

    # 显示最终安装信息（明确告知是 .NET 10）
    echo -e "${YELLOW}📋 安装信息：${NC}"
    echo -e "  - 安装版本：.NET 10（${INSTALL_TYPE}）"
    echo -e "  - 安装目录：${DOTNET_INSTALL_PATH}"
    echo -e "  - 无需管理员权限，安装后即可运行目标软件"
    echo -e "${NC}"
}

# 检查并自动安装 Homebrew（普通用户无需手动操作）
check_brew() {
    echo -e "${YELLOW}🔍 检查必备工具 Homebrew...${NC}"
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}⚠️  未找到 Homebrew，正在自动安装（需输入 macOS 登录密码）...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Homebrew 安装失败，请检查网络连接后重试${NC}"
            exit 1
        fi
        # 自动加载 Homebrew（用户无需手动配置）
        if [ -x /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"  # Apple Silicon 芯片
        else
            eval "$(/usr/local/bin/brew shellenv)"    # Intel 芯片
        fi
        echo -e "${GREEN}✅ Homebrew 安装成功${NC}"
    else
        echo -e "${GREEN}✅ Homebrew 已安装${NC}"
    fi
}

# 自动安装 wget（下载官方脚本必需）
install_wget() {
    echo -e "${YELLOW}🔍 检查下载工具 wget...${NC}"
    if ! command -v wget &> /dev/null; then
        echo -e "${YELLOW}⚠️  未找到 wget，正在自动安装...${NC}"
        brew install wget -q  # 静默安装，不打扰用户
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ wget 安装失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ wget 安装成功${NC}"
    else
        echo -e "${GREEN}✅ wget 已安装${NC}"
    fi
}

# 下载官方安装脚本（普通用户无需手动下载）
download_dotnet_script() {
    echo -e "${YELLOW}📥 正在下载 .NET 10 官方安装脚本...${NC}"
    mkdir -p "${DOWNLOAD_DIR}"  # 确保下载目录存在
    cd "${DOWNLOAD_DIR}" || {
        echo -e "${RED}❌ 无法访问下载文件夹，请检查权限${NC}"
        exit 1
    }

    # 静默下载，避免用户看到复杂日志
    wget https://dot.net/v1/dotnet-install.sh -O "${DOTNET_INSTALL_SCRIPT}" -q
    if [ $? -ne 0 ] || [ ! -f "${DOTNET_INSTALL_SCRIPT}" ]; then
        echo -e "${RED}❌ 官方脚本下载失败，请检查网络（推荐科学上网）${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ 官方脚本下载成功${NC}"
}

# 自动授予执行权限（用户无需手动输入 chmod）
add_exec_permission() {
    echo -e "${YELLOW}🔑 正在准备安装脚本...${NC}"
    chmod +x "${DOTNET_INSTALL_SCRIPT}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 脚本准备失败，请检查权限${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ 脚本准备完成${NC}"
}

# 核心安装步骤（仅安装 .NET 10，无其他版本）
install_dotnet() {
    echo -e "${YELLOW}🚀 开始安装 .NET 10 ${INSTALL_TYPE}...${NC}"
    echo -e "${YELLOW}⌛ 安装过程约 1-3 分钟（取决于网络速度），请耐心等待...${NC}"

    # 构造安装参数（固定 .NET 10 通道）
    install_args=(
        --channel "${DOTNET_CHANNEL}"
        --install-dir "${DOTNET_INSTALL_PATH}"
        #--quiet 不能静默 # 静默安装，只显示关键结果
    )
    # 区分运行时和 SDK（默认运行时）
    if [ "${INSTALL_TYPE}" = "运行时" ]; then
        install_args+=(--runtime dotnet)
    fi

    # 执行官方安装脚本
    ./"${DOTNET_INSTALL_SCRIPT}" "${install_args[@]}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ .NET 10 ${INSTALL_TYPE} 安装失败，请重试或检查网络${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ .NET 10 ${INSTALL_TYPE} 安装完成！${NC}"
}

# 自动配置环境变量（普通用户无需手动输入命令）
config_env() {
    echo -e "${YELLOW}⚙️  正在配置环境变量...${NC}"
    # 覆盖常见 shell（zsh/login/bash），新终端和 GUI 都能拿到
    env_files=()
    [ -f "$HOME/.zshrc" ] && env_files+=("$HOME/.zshrc")
    [ -f "$HOME/.zprofile" ] && env_files+=("$HOME/.zprofile")
    [ -f "$HOME/.bash_profile" ] && env_files+=("$HOME/.bash_profile")
    # 若都不存在，默认写入 .zshrc
    if [ ${#env_files[@]} -eq 0 ]; then
        env_files+=("$HOME/.zshrc")
    fi

    for env_file in "${env_files[@]}"; do
        if ! grep -q "DOTNET_ROOT=$DOTNET_INSTALL_PATH" "$env_file" 2>/dev/null; then
            echo "export DOTNET_ROOT=$DOTNET_INSTALL_PATH" >> "$env_file"
        fi
        if ! grep -q 'PATH=.*DOTNET_ROOT' "$env_file" 2>/dev/null; then
            echo "export PATH=\$PATH:\$DOTNET_ROOT" >> "$env_file"
        fi
    done

    # 立即生效（当前终端会话）
    export DOTNET_ROOT="$DOTNET_INSTALL_PATH"
    export PATH="$PATH:$DOTNET_ROOT"
    hash -r 2>/dev/null

    echo -e "${GREEN}✅ 环境变量配置成功！${NC}"
    echo -e "${YELLOW}💡 说明：终端/GUI 均会使用 $DOTNET_INSTALL_PATH 下的运行时${NC}"
}

# 简单验证安装结果（普通用户能看懂）
verify_install() {
    echo -e "\n${YELLOW}🔍 正在验证 .NET 10 安装结果...${NC}"
    local dotnet_bin="$DOTNET_INSTALL_PATH/dotnet"
    if [ ! -x "$dotnet_bin" ]; then
        echo -e "${RED}❌ 未找到 $dotnet_bin，可尝试重新运行本脚本${NC}"
        exit 1
    fi

    local runtime_line
    runtime_line=$("$dotnet_bin" --list-runtimes 2>/dev/null | grep -E "Microsoft\.NETCore\.App 10\.0")
    if [ -z "$runtime_line" ]; then
        echo -e "${RED}❌ 未检测到 Microsoft.NETCore.App 10.0.x 运行时，请检查网络后重试${NC}"
        exit 1
    fi

    local dotnet_version=$("$dotnet_bin" --version 2>/dev/null)
    echo -e "${GREEN}🎉 安装成功！当前 .NET 版本：${dotnet_version}${NC}"
    echo -e "${GREEN}🎉 现在可以正常运行你的软件了！${NC}"
}

# 为 GUI/双击场景写入用户级环境变量
config_launchctl_env() {
    if command -v launchctl &> /dev/null; then
        launchctl setenv DOTNET_ROOT "$DOTNET_INSTALL_PATH"
        launchctl setenv PATH "$PATH:$DOTNET_INSTALL_PATH"
        echo -e "${GREEN}✅ 已为 GUI 进程配置 DOTNET_ROOT（launchctl）${NC}"
    fi
}

# 主逻辑（普通用户无需干预，一键完成 .NET 10 运行时安装）
main() {
    show_help
    parse_args "$@"
    check_brew
    install_wget
    download_dotnet_script
    add_exec_permission
    install_dotnet
    config_env
    config_launchctl_env
    verify_install

    echo -e "\n${BOLD}${GREEN}======================================= 安装完成！=======================================${NC}"
    echo -e "${YELLOW}📌 后续操作：直接运行你的软件即可（终端/双击均已配置运行时）${NC}"
    read -p "按 Enter 键退出..."
}

# 执行主逻辑（普通用户直接运行）
main "$@"
