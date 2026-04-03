# VeighNa (vn.py) 零基础保姆级使用指南

本指南专为初次接触 VeighNa 的用户设计，涵盖从下载、安装到运行 KHAOS 策略的全过程。

---

## 第一部分：获取 VeighNa Studio

如果您觉得命令行安装太复杂，**强烈建议**直接下载 VeighNa Studio（全功能集成版）。它像安装普通软件一样简单。

### 1. 下载
*   **官方下载地址**：[https://www.vnpy.com/](https://www.vnpy.com/)
*   在首页点击大大的 **"下载 VeighNa Studio"** 按钮。
*   选择 **Windows** 版本下载（通常是一个 `.exe` 安装包，约 800MB）。

### 2. 安装
1.  双击下载好的 `VeighNa-Studio-x.x.x.exe`。
2.  **安装路径**：建议保持默认（例如 `C:\veighna_studio`），**不要**安装在包含中文或空格的路径下。
3.  一路点击 "Next" 直到安装完成。

### 3. 启动
*   安装完成后，您的桌面上会出现一个 **"VeighNa Station"** 的图标。
*   双击图标，即可启动 VeighNa 的图形化管理界面。

---

## 第二部分：如果您已经运行了 `install.bat` (源码安装)

如果您是通过 GitHub 下载源码并运行了 `install.bat`，那么您使用的是“开发者模式”。这种模式下没有桌面图标，需要通过命令行启动。

### 如何启动
1.  打开命令提示符 (CMD) 或 PowerShell。
2.  进入您运行 `install.bat` 的目录（或者激活对应的 Python 虚拟环境）。
3.  输入以下命令并回车：
    ```bash
    python -m vnstation
    ```
4.  如果安装成功，VeighNa Station 的登录界面就会弹出。

---

## 第三部分：配置 KHAOS 策略

无论您是用哪种方式启动的，接下来的步骤是一样的。

### 1. 登录 VeighNa Station
*   启动后，如果是第一次使用，可能需要注册一个 vn.py 社区账号（免费），或者直接点击“微信登录”。
*   登录后，您会看到一个包含各种图标（如“CTP连接”、“CTA回测”等）的主界面。

### 2. 放置策略文件
VeighNa 需要将策略代码放在特定的文件夹里才能识别。

1.  **找到策略文件夹**：
    *   通常位于：`C:\Users\您的用户名\vnpy_run\strategies`
    *   如果找不到，可以运行本项目提供的辅助脚本 `scripts/setup/locate_strategy_folder.py` 来自动查找。

2.  **复制文件**：
    *   将本项目中的 **`src/vnpy_integration/khaos_strategy_bundled.py`** 文件复制到上述文件夹中。
    *   *注意：一定要用这个 `bundled` 版本，因为它包含了所有需要的算法库，不会报错。*

### 3. 导入数据
在进行回测前，我们需要把数据塞进 VeighNa 的数据库。

1.  在 VeighNa Station 界面上，点击 **"数据管理" (Data Manager)** 图标。
2.  在弹出的窗口右上角，点击 **"导入数据"**。
3.  **填写配置**：
    *   **文件路径**：选择本项目下的 `data/raw/Commodity/xauusd/DAT_ASCII_XAUUSD_M1_2023.csv`
    *   **交易所**：选择 `LOCAL`
    *   **代码**：输入 `XAUUSD`
    *   **数据格式**：选择 `HistData` (如果您的数据是分号分隔) 或 `CSV` (如果是逗号分隔)。
        *   *提示：如果不确定，请先使用本项目提供的 `src/vnpy_integration/import_data.py` 脚本自动导入。*
    *   点击 **"确定"**。
4.  等待进度条走完，数据就导入成功了。

### 4. 开始回测
1.  回到 VeighNa Station 主界面，点击 **"CTA回测" (CTA Backtester)**。
2.  **设置回测选项**（左侧面板）：
    *   **策略类**：点击下拉框。如果策略文件放置正确，您应该能看到 `KhaosStrategy`。选择它。
    *   **本地代码**：输入 `XAUUSD.LOCAL`。
    *   **K线周期**：选择 `1m` (1分钟)。
    *   **开始日期**：`2023-01-01`
    *   **结束日期**：`2023-12-31`
    *   **手续费率**：`0.0001` (万一)
    *   **交易滑点**：`0.2`
    *   **合约乘数**：`1`
    *   **价格跳动**：`0.01`
    *   **回测资金**：`1000000`
3.  **编辑策略参数**（中间上方）：
    *   您会看到 `hurst_window`, `fixed_size` 等参数。保持默认即可。
4.  点击左侧的 **"开始回测"** 按钮。
5.  **查看结果**：
    *   回测完成后，右侧会显示资金曲线图。
    *   下方会显示详细的统计指标（总收益、夏普比率、最大回撤等）。
    *   点击“成交记录”标签页，可以看到每一笔具体的买卖操作。

---

## 常见问题排查

**Q: 下拉框里找不到 `KhaosStrategy`？**
A: 说明策略文件没有放对位置，或者文件名/类名有误。
1. 确保复制的是 `khaos_strategy_bundled.py`。
2. 确保它在 `C:\Users\...\vnpy_run\strategies` 目录下。
3. 点击 CTA 回测界面右上角的 **"刷新策略"** 按钮。

**Q: 点击开始回测后，日志提示 "History data is empty"？**
A: 说明数据没导入进去，或者代码/交易所填错了。
1. 检查数据管理模块里是否真的有数据。
2. 确保回测界面的“本地代码”是 `XAUUSD.LOCAL`（中间有个点，且全大写）。

**Q: 报错 "No module named torch"？**
A: VeighNa Studio 自带的 Python 环境可能没有安装 PyTorch。
1. 打开 VeighNa Studio 的命令行（在安装目录下有个 `cmd.exe` 或快捷方式）。
2. 运行 `pip install torch numpy scipy`。
