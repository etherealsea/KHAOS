# 基于物理信息增强神经网络 (PI-KAN) 的金融时间序列相变探测研究

## 项目简介

**PI-KAN Phase Detection** 是一个集成了物理信息增强与神经网络技术的量化研究系统。通过结合 **扩展卡尔曼滤波 (EKF)**、**分形几何 (Hurst 指数)** 与最新的 **Kolmogorov-Arnold Network (KAN)**，系统能够精准探测市场的机制切换（相变），如趋势反转、动能衰竭及极端拐点。

### 🔥 核心创新
- **PI-KAN 架构**: 将物理先验知识以损失函数约束的形式注入 KAN 网络，大幅提升模型的泛化能力。
- **相变探测引擎**: 利用 KHAOS 物理引擎捕捉市场从层流到湍流的相变过程。
- **可解释性 AI**: 样条函数激活机制使得模型决策过程具备物理意义。

## 核心特性

### 🤖 AI Agent架构
- **数据收集Agent**: 自动化多源数据获取和质量监控
- **技术分析Agent**: 专业的技术指标计算和趋势识别
- **KHAOS物理引擎**: 基于熵(Entropy)和卡尔曼滤波(EKF)的市场状态监测系统
- **情绪分析Agent**: 社交媒体和新闻情绪智能分析
- **风险评估Agent**: 全面的风险指标计算和预警

### 📊 多维度分析
- **KHAOS物理分析**: 
  - **Instability (不稳定性)**: 识别市场层流/湍流状态，规避震荡磨损
  - **Anomaly (能量异常)**: 捕捉物理惯性失效时刻，预警剧烈波动
- **技术面分析**: 20+技术指标，多时间框架分析
- **基本面分析**: 项目质量、生态发展、代币经济学
- **市场情绪**: 社交媒体热度、新闻情感、搜索趋势
- **风险评估**: 波动率、VaR、流动性、相关性分析

### 🎯 智能评估
- **综合评分**: 多因子加权评分模型
- **动态权重**: 基于市场条件的权重自适应调整
- **可解释AI**: 提供详细的分析依据和决策逻辑
- **实时更新**: 支持实时数据更新和分析

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户交互层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Web界面    │  │  API接口    │  │  报告生成    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                 Trae AI Agent层                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ 数据分析Agent│  │ 技术分析Agent│  │ 情绪分析Agent│    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   数据处理层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  数据收集    │  │  数据清洗    │  │  特征工程    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求
- Python 3.11+
- Docker & Docker Compose
- 8GB+ RAM
- 网络连接（用于API调用）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd altcoin_analyzer
```

2. **环境配置**
```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件，填入必要的API密钥
nano .env
```

3. **安装依赖**
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

4. **启动服务**
```bash
# 开发模式启动
python main.py

# 或使用Docker
docker-compose up -d
```

5. **访问应用**
- Web界面: http://localhost:8000
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## API使用示例

### 分析单个币种
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "bitcoin",
    "workflow_type": "full_analysis",
    "data_types": ["price", "social", "news"],
    "timeframe": "24h"
  }'
```

### 批量分析
```bash
curl -X POST "http://localhost:8000/api/v1/batch-analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["bitcoin", "ethereum", "cardano"],
    "workflow_type": "quick_scan",
    "max_concurrent": 3
  }'
```

### 生成报告
```bash
curl -X POST "http://localhost:8000/api/v1/reports/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["bitcoin", "ethereum"],
    "report_type": "comprehensive",
    "timeframe": "7d",
    "include_charts": true,
    "format": "json"
  }'
```

## 配置说明

### 必需的API密钥
```env
# Trae AI配置
TRAE_API_KEY=your_trae_api_key_here

# 数据源API
COINGECKO_API_KEY=your_coingecko_api_key
BINANCE_API_KEY=your_binance_api_key
TWITTER_API_KEY=your_twitter_api_key
```

### Agent配置
```python
AGENT_CONFIG = {
    "data_collector": {
        "enabled": True,
        "timeout": 60,
        "max_retries": 3
    },
    "technical_analyzer": {
        "enabled": True,
        "timeout": 120,
        "indicators": ["SMA", "EMA", "MACD", "RSI"]
    }
}
```

## 开发指南

### 项目结构
```
altcoin_analyzer/
├── agents/              # AI Agent实现
├── api/                 # API端点
├── config/              # 配置管理
├── core/                # 核心模块
├── services/            # 业务服务
├── utils/               # 工具函数
├── web/                 # 前端资源
├── tests/               # 测试用例
└── docs/                # 文档
```

### 添加新的Agent
1. 继承`BaseAgent`类
2. 实现`initialize()`和`process()`方法
3. 在配置中注册Agent
4. 编写单元测试

### 代码规范
- 遵循PEP8标准
- 使用类型注解
- 编写文档字符串
- 单元测试覆盖率>80%

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_agents.py

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

## 部署

### Docker部署
```bash
# 构建镜像
docker build -t altcoin-analyzer .

# 运行容器
docker run -p 8000:8000 --env-file .env altcoin-analyzer
```

### 生产环境
```bash
# 使用Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# 使用Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

## 监控和日志

- **日志文件**: `logs/app.log`
- **健康检查**: `/health`
- **指标监控**: `/metrics` (如果启用)
- **性能监控**: 内置执行时间统计

## 注意事项

### 学术研究声明
- 本系统仅用于学术研究和技术验证
- 分析结果不构成投资建议
- 请理性对待分析结果，注意风险管理

### API限制
- 请遵守各数据源的API使用限制
- 建议配置适当的请求频率限制
- 注意API密钥的安全管理

### 数据质量
- 系统会自动进行数据质量检查
- 建议定期验证数据源的可用性
- 注意处理网络异常和API错误

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 编写测试
5. 提交Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

- 项目作者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目地址: [GitHub链接]

## 致谢

感谢以下开源项目和服务：
- FastAPI - 现代化的Python Web框架
- Trae AI - 强大的AI Agent平台
- CoinGecko - 加密货币数据API
- TA-Lib - 技术分析库
- 其他依赖项目的贡献者

---

**免责声明**: 本系统仅供学术研究使用，不构成任何投资建议。加密货币投资存在高风险，请谨慎决策。