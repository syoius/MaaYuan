#!/bin/bash

# 启用颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # 重置颜色

# 初始化错误标志和架构变量
error_occurred=0
arch="x64"

# 检测系统架构（x64/arm64）
detect_arch() {
    local uname_arch=$(uname -m)
    case $uname_arch in
        x86_64) arch="x64" ;;
        aarch64) arch="arm64" ;;
        *) 
            echo -e "${RED}不支持的架构: $uname_arch${NC}"
            exit 1 
            ;;
    esac
    echo -e "${BOLD}${BLUE}检测到系统架构: $arch${NC}"
}

# 检查并获取管理员权限
check_admin() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${YELLOW}需要管理员权限，即将请求 sudo 密码...${NC}"
        sudo "$0" "$@"
        exit $?
    fi
}

# 安装 .NET Runtime 10（基于官方 backports PPA 仓库）
install_dotnet() {
    echo -e "\n${BLUE}===================================================================================================="
    echo -e "${BOLD}${CYAN}正在安装 .NET Runtime 10 ($arch)${NC}"
    echo -e "${BOLD}${CYAN}Installing .NET Runtime 10 ($arch)${NC}"
    echo -e "${BLUE}===================================================================================================="${NC}

    # 步骤1：清理冲突的旧版本 dotnet-host（解决依赖冲突问题）
    echo -e "${YELLOW}0/4 清理冲突的 dotnet-host 旧版本...${NC}"
    if dpkg -l | grep -q "dotnet-host"; then
        echo -e "${YELLOW}  发现已安装的 dotnet-host 旧版本，正在删除...${NC}"
        apt-get remove -y dotnet-host > /dev/null 2>&1
        apt-get autoremove -y > /dev/null 2>&1
        
        if [ $? -ne 0 ]; then
            error_occurred=1
            echo -e "${RED}❌ 删除旧版本 dotnet-host 失败，请手动执行: sudo apt-get remove -y dotnet-host${NC}"
            return
        else
            echo -e "${GREEN}  ✅ 成功删除冲突的 dotnet-host 旧版本${NC}"
        fi
    else
        echo -e "${GREEN}  ✅ 未发现冲突的 dotnet-host 旧版本${NC}"
    fi

    # 步骤2：添加官方 .NET backports PPA 仓库（官方推荐）
    echo -e "${YELLOW}1/4 添加 .NET backports PPA 仓库...${NC}"
    add-apt-repository -y ppa:dotnet/backports > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        error_occurred=1
        echo -e "${RED}❌ 添加 .NET backports PPA 仓库失败${NC}"
        return
    fi

    # 步骤3：更新包列表
    echo -e "${YELLOW}2/4 更新系统包列表...${NC}"
    apt-get update > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        error_occurred=1
        echo -e "${RED}❌ 更新包列表失败，请检查网络连接${NC}"
        return
    fi

    # 步骤4：安装 .NET Runtime 10（根据架构选择对应包）
    echo -e "${YELLOW}3/4 安装 .NET Runtime 10...${NC}"
    if [ "$arch" = "x64" ]; then
        apt-get install -y dotnet-runtime-10.0
    else
        apt-get install -y dotnet-runtime-10.0:arm64
    fi

    if [ $? -ne 0 ]; then
        error_occurred=1
        echo -e "${RED}❌ .NET Runtime 10 安装失败${NC}"
    else
        echo -e "${GREEN}✅ .NET Runtime 10 安装成功${NC}"
    fi
}

# 输出手动下载链接（区分 SDK 和 Runtime，补充官方正确链接）
print_manual_links() {
    echo -e "\n${YELLOW}🔗 您可以手动下载以下组件安装：${NC}"
    echo -e "${YELLOW}🔗 You can manually download and install the following components:${NC}\n"

    # 官方 SDK 链接（安装 SDK 后无需单独安装 Runtime）
    echo -e "${WHITE}• .NET SDK 10 ($arch)（推荐，包含 Runtime）:${NC}"
    echo -e "  ${CYAN}https://builds.dotnet.microsoft.com/dotnet/Sdk/10.0.100/dotnet-sdk-10.0.100-linux-$arch.tar.gz${NC}"
    
    # 官方 Runtime 链接
    echo -e "\n${WHITE}• .NET Runtime 10 ($arch)（仅运行时）:${NC}"
    echo -e "  ${CYAN}https://builds.dotnet.microsoft.com/dotnet/Runtime/10.0.0/dotnet-runtime-10.0.0-linux-$arch.tar.gz${NC}"
    
    echo -e "\n${YELLOW}📝 手动安装说明：下载后解压到 /usr/share/dotnet，然后执行：${NC}"
    echo -e "${CYAN}  export PATH=\$PATH:/usr/share/dotnet${NC}"
}

# 主逻辑
main() {
    detect_arch
    check_admin
    install_dotnet

    # 输出最终结果
    echo -e "\n"
    if [ $error_occurred -eq 0 ]; then
        echo -e "${BOLD}${GREEN}===================================================================================================="
        echo -e "${BOLD}${GREEN}依赖安装完成！建议重启后再运行应用。${NC}"
        echo -e "${BOLD}${GREEN}Dependencies installed successfully! Please restart your system before running the application.${NC}"
        echo -e "${BOLD}${GREEN}===================================================================================================="${NC}
    else
        echo -e "${RED}===================================================================================================="
        echo -e "${BOLD}${RED}依赖安装过程中出现错误${NC}"
        echo -e "${BOLD}${RED}Errors occurred during dependency installation${NC}"
        echo -e "\n${YELLOW}💡 解决方案：${NC}"
        echo -e "${YELLOW}1. 若使用非 Debian/Ubuntu 系统，请手动安装或调整包管理器命令（如 yum/dnf）${NC}"
        echo -e "${YELLOW}   For non-Debian/Ubuntu systems, use yum/dnf or other package managers${NC}"
        echo -e "${YELLOW}2. 若包源同步未完成，可等待一段时间后重试，或使用手动下载方式${NC}"
        echo -e "${YELLOW}   If repo sync is incomplete, wait and retry or use manual download${NC}"
        echo -e "${YELLOW}3. 若删除旧版本失败，请手动执行：sudo apt-get remove -y dotnet-host && sudo apt-get autoremove -y${NC}"
        print_manual_links
        echo -e "${RED}===================================================================================================="${NC}
    fi

    read -p "按 Enter 键退出..."
}

main