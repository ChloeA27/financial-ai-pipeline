# WORKFLOW_LOG — Financial AI Pipeline

> 项目心跳 / 架构变更审计追踪 / Architectural Changelog
>
> Established: 2026-05-31
> Rule: 每次完成 significant action 后，必须在此追加条目。英文带时间戳和类型，中文另起一行用 tab 缩进。
> Rule: Every significant action MUST be logged below. EN line with timestamp & type, ZH line indented below.

---

- [2026-05-31 15:10] **[FEATURE]**: Scaffolded project directory tree and requirements.txt with LangGraph/Pydantic/aiosqlite deps
     搭建项目骨架 + requirements.txt，引入 LangGraph/Pydantic/aiosqlite 核心依赖

- [2026-05-31 15:25] **[FEATURE]**: Created BaseDoc/Metadata models (UUID v4 + timestamp + source_path traceability) and PipelineState with CorrectionLog for self-correction loop
     创建 BaseDoc/Metadata 溯源模型 + PipelineState + CorrectionLog 自纠错数据结构

- [2026-05-31 15:40] **[FEATURE]**: Built async Reader Node (asyncio.to_thread) + LLM Client (tenacity retry + ChatOpenAI.with_structured_output)
     实现异步 Reader Node + LLM Client（tenacity 重试 + 结构化输出）

- [2026-05-31 15:55] **[FEATURE]**: Implemented Classifier Node with Pydantic DocumentClassification; LLM-powered doc_type routing ("M&A", "Dividend", "Other")
     实现 Classifier Node，通过 Pydantic + LLM 完成文档类型路由

- [2026-05-31 16:10] **[FEATURE]**: Built Extractor Node with two-model protocol (LLM model vs pipeline model + Metadata) and Registry Pattern for extensibility
     设计 Extractor Node 双模型协议 + 注册表模式，支持多类型扩展

- [2026-05-31 16:25] **[FEATURE]**: Built Validator Node with rule-based quality gate, self-correction loop (max 3 retries), and Dead Letter Queue on exhaustion
     构建 Validator 规则引擎 + 自纠错循环（最多 3 次），超限路由到 Dead Letter Queue

- [2026-05-31 16:40] **[FEATURE]**: Assembled LangGraph pipeline with conditional_edges for self-correction cycle and DLQ routing
     用 LangGraph 组装完整管线，通过 conditional_edges 实现自纠错 + DLQ 分流

- [2026-05-31 16:55] **[FEATURE]**: Implemented persistent storage — repository.py (SCD Type 2 upsert), json_writer.py, db.py (aiosqlite async wrapper)
     实现持久化存储层——SCD Type 2 版本溯源 upsert + JSON 输出 + aiosqlite 异步封装

- [2026-05-31 17:10] **[FEATURE]**: Created main.py with asyncio.gather concurrency + CLI args + loguru logging; first M&A end-to-end pass test
     创建 main.py 入口（asyncio.gather 并发 + CLI + loguru），首条 M&A 端到端测试通过

- [2026-05-31 17:25] **[FEATURE]**: Added Dividend extraction schema (dividend.py) with two-model protocol and validators for currency/type/frequency/date; registered in Extractor Registry
     新增 Dividend 提取 Schema（双模型协议 + 币种/类型/频率/日期校验），注册到 Extractor Registry

- [2026-05-31 17:40] **[REFACTOR]**: Upgraded validator.py to doc_type-aware _VALIDATION_RULESETS Registry Pattern — M&A & Dividend rulesets decoupled, zero-code-change for new types
     重构 Validator 为 doc_type-aware 注册表模式，M&A 与 Dividend 规则分离，新增类型无需改 Validator 代码

- [2026-05-31 17:55] **[TEST]**: Ran 4-file concurrent batch (3 M&A + 1 Dividend): 3 passed (incl. first-ever Dividend), 1 routed to DLQ after 3 retries — verified via SQLite + JSON
     4 文件并发批处理（3 M&A + 1 Dividend）：3 通过（含首笔 Dividend），1 个 3 次纠错后路由到 DLQ，SQLite + JSON 双重验证

- [2026-05-31 18:05] **[CHORE]**: Created WORKFLOW_LOG.md bilingual audit trail (EN/ZH interleaved) documenting full project evolution from zero to Dividend data flow production
     创建 WORKFLOW_LOG.md 中英双语审计日志，记录项目从零到 Dividend 数据流落地的完整架构演进

- [2026-05-31 18:10] **[CHORE]**: Fixed Streamlit dashboard 502 error — restarted on port 8501, verified HTTP 200, accessible at http://localhost:8501
     修复 Streamlit Dashboard 502 错误——重启用 8501 端口启动，HTTP 200 验证通过

- [2026-05-31 17:42] **[BUGFIX]**: Fixed 3 business-critical Dividend schema issues — added declaration_date field, clarified dividend_cash_amount as "Gross (pre-tax)", aligned frequency Description with _allowed_frequencies (added Semi-Annual). Updated Model A + Model B + Extractor prompt + Validator ruleset + test sample. All validated via end-to-end batch (retries=0).
     修复 3 个业务级 Dividend Schema 问题——增加宣告日 declaration_date、金额明确为税前 Gross (pre-tax)、统一 frequency 描述与白名单（补全 Semi-Annual）。更新 Model A/B + Extractor prompt + Validator + 测试样本，端到端压测通过。

- [2026-06-01 11:47] **[REFACTOR]**: Implemented Validation Profile Registry — profiles.py (single-source-of-truth), base_model.py (ProfileValidatorMixin + _make_allowed_validator factory + closure-safe __init_subclass__), refactored DividendExtraction + DividendExtractionResult to use shared DividendValidationProfile. 8 manually-duplicated @field_validator blocks eliminated. Runtime dynamic lookup supports Profile inheritance (e.g. HKProfile adding CNH). E2E: 3/4 passed, Dividend verified.
     实现 Validation Profile Registry 架构——profiles.py（合法值唯一数据源）、base_model.py（ProfileValidatorMixin + 闭包安全的 factory + __init_subclass__ 自动注入），Dividend Model A/B 统一指向 DividendValidationProfile。消除 8 个重复 @field_validator。运行时动态查找天然支持 Profile 继承（如港股加 CNH）。端到端 3/4 通过，Dividend 字段验证正确。

- [2026-06-01 12:04] **[FEATURE]**: Added ``_inject_date_validator`` to base_model.py — any field ending in ``_date`` automatically gets YYYY-MM-DD format validation via convention. Removed all manual ``validate_date_format`` from Dividend and M&A schemas.
     base_model.py 新增 ``_inject_date_validator``——任何以 ``_date`` 结尾的字段自动获得 YYYY-MM-DD 格式校验。Dividend + M&A 所有手动 ``validate_date_format`` 全部删除。

- [2026-06-01 12:06] **[REFACTOR]**: Refactored MandaExtraction + MandaExtractionResult to use BaseExtractionModel / ProfileValidatorMixin with MandaValidationProfile. Zero manual @field_validator in manda.py — same convention-based architecture as Dividend. E2E: 3/4 passed, M&A fields verified.
     重构 MandaExtraction + MandaExtractionResult 使用 BaseExtractionModel/ProfileValidatorMixin + MandaValidationProfile。manda.py 零手动 @field_validator——与 Dividend 同架构。端到端 3/4 通过，M&A 字段验证正确。
