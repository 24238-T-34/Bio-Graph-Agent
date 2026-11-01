import streamlit as st
import tempfile
import os
import shutil
from datetime import datetime
from pypdf import PdfReader
from back_logic import (
    BioGraphPipeline, GraphVisualizer,
    consolidate_homogeneous_relations, purify_mechanism_relations,
    merge_coarse_relation_into_target, is_mechanism_relation,
    merge_hierarchy_relation, fold_symmetric_relations
)
from IO_SYS import (
    extract_paper_title_and_pmid, sanitize_paper_filename,
    compute_file_content_hash, compute_source_identity_hash,
    get_clean_display_title, get_physical_pdf_filename,
    load_project_paper_registry, save_project_paper_registry,
    register_or_update_paper, resolve_paper_identity,
    canonicalize_project_papers,
    extract_parent_doc_source, is_figure_doc_source
)
import streamlit.components.v1 as components
import gc
import fitz  # 🚀 优化：把 PyMuPDF 移到文件顶部，规范代码结构
import json
import pandas as pd
import uuid
import re
from typing import Optional, List

CONFIG_FILE = ".bio_graph_config.json"


@st.dialog("📖 User Guide / 操作指南", width="large")
def show_help_dialog():
    # 🌐 动态获取当前系统的 UI 语言
    current_ui_lang = st.session_state.get("language", "zh")
    is_en = "en" in current_ui_lang.lower()

    if is_en:
        st.markdown(""" 
    📖 **Bio-Graph Agent Core Functionality Guide**
    Welcome to Bio-Graph Agent! A specialized LLM-powered scientific counselor crafted specifically for biomedical researchers.

    🧬 **Graph Base Logic & Academic Mission:**
    This system is far more than a simple "summarization tool". It leverages Large Language Models to deeply read literature and extract structured **"Entities (Nodes)"** and **"Regulatory Relationships (Edges)"**. When analyzing multiple papers, the system automatically performs semantic alignment (e.g., merging "p53" from Paper A and "TP53" from Paper B into a single node), weaving isolated papers into a globally verifiable, dynamic knowledge network to uncover multi-step cascade pathways ($A \to B \to C$) for scientific grant proposals!

    ⚙️ **I. Initial Environment Setup (Sidebar "More Settings")**
    Before starting, expand the ⚙️ **More Settings** panel in the left sidebar for basic configuration:

    * 🌐 **LLM Engine:** Select your API provider (OpenRouter, Tencent Hunyuan, SiliconFlow, etc.) and enter the API Key. For offline privacy mode, select **Ollama** to process classified biomedical data key-free without an internet connection.
    * 📚 **Smart Retrieval Source:** Choose PubMed (authoritative biomedical literature) or Semantic Scholar (all-discipline AI-driven high-speed retrieval with automatic rate-limit fallback).
    * ⚡ **Concurrency Workers (Parallel Pipeline):** Set concurrent threads (1-8) to execute chunk analysis and multimodal pathway figure transcriptions in parallel, drastically reducing literature processing time.
    * 🛡️ **Relation Node Literature Grounding:** When enabled, automatically regex-verifies unregistered relation endpoints against the original paper text. Real entities mentioned in the paper are recovered, while hallucinated edges are strictly dropped.
    * 👁️ **Dedicated Vision Model:** Choose auto-follow main model, or specify dedicated high-performance multimodal models (such as Gemini 2.5 Flash-Lite, Gemini 3.8 Flash, or Qwen 2.5 VL 72B) with optional suffix overrides.
    * 🎛️ **Interface Feature Toggles:** Enable the 📝 **Manual Editing Panel** and 🤖 **AI Intelligent Graph Tools** as needed.

    📂 **II. Knowledge Graph Construction (Left Control Panel)**
    In the left "1. Literature Input & Configuration" panel, the system offers two distinct paths to build graphs:

    **Option A: 💡 Zero-Based Cold Start**
    Have no papers on hand? Simply tell the AI your research direction in the text box (e.g., *"Role of macrophage polarization in tumor microenvironment"*). Click **"One-click Retrieve & Generate Initial Graph"**, and the AI will formulate professional PubMed queries, retrieve cutting-edge literature, verify relevance, and construct your initial graph!

    **Option B: 📄 Precise Local Literature Analysis**
    * **Upload Literature:** Drag a PDF into the upload zone.
    * **🎯 Parsing Range:** Specify starting and ending pages (e.g., parsing pages 1-5 with key mechanism figures) to save token budgets and time.
    * **⚙️ Advanced Extraction Settings:**
        * ⚡ **Abstract-Only Mode:** High-speed regex extraction of abstract sections for rapid exploration.
        * 🔍 **Self-Reflection & Error Correction:** The AI acts as an independent quality auditor, cross-examining initial extraction against the raw text to eliminate hallucinated nodes.
        * 🔗 **Append Mode (Continuous Growth):** When enabled, new literature fuses and grows into the existing graph. Turning this off clears the canvas to start fresh.
        * 🌐 **Entity Naming Convention:** Automatically standardizes graph nodes into pure Chinese, pure English, or keeps original phrasing.
    * 📷 **Pathway & Mechanism Figure Strategy (Multimodal Vision):**
        * 🚫 *Off (Pure Text):* Traditional text-only parsing.
        * 🎯 *Keyword Match:* Targeted sniffing of a specific figure (e.g., `Fig 4`, `pathway`, `Figure 2`).
        * ⚡ *Extract All in Pages:* Automatically runs **Spatial Clustering** to merge overlapping, sliced raster layers (such as multi-panel Graphical Abstracts and composite Figure subpanels) into pristine high-res composites, plus balanced vector diagram detection for posters and pathways. Transcriptions are generated in parallel and fused into downstream entity/relation modeling.

    🕸️ **III. Visual Empowerment Engine (Top Console of Main Interface)**
    * 👁️ **Mechanism Shortcuts:** Reveal "express shortcuts" bypassing intermediate nodes to inspect macro upstream-to-downstream regulatory outcomes.
    * 🧬 **Ontology Empowerment (Containment Feedback):** Feedback the weight of micro-entities (e.g., individual proteins) to macro parent categories (e.g., Kinase Family), making core pathways visually prominent.
    * 🔥 **Node Empowerment (Degree Centrality):** Hub genes with higher connectivity dynamically scale up in node size.
    * 🔗 **Edge Empowerment (Consensus Boldening):** Regulatory edges validated by multiple papers and rich textual evidence chains are automatically bolded.

    🎨 **IV. Topological Visual Standards**
    * 🟢 **Green Solid Line + Arrow:** Positive regulation / Activation / Promotion ($A \to B$).
    * 🔴 **Red Solid Line + Arrow:** Negative regulation / Inhibition / Downregulation ($A \dashv B$).
    * ⚪ **Gray Solid Line:** General correlation / Undirected association.
    * ⚠️ **Gray Dashed Line (---):** Mechanism shortcuts (condensed macro pathways).
    * 🟣 **Purple Solid Line:** Hierarchical containment / Subunit / Family membership.

    ✂️ **V. Expert Manual Intervention (Data Management Center)**
    * 🧹 **Global Operations:** Scan and eliminate isolated orphan nodes; batch-rename messy literature source tags across all connected nodes and edges; manually add missing key proteins.
    * 🎯 **Node Operations:** Rename nodes; **Force Merge** (transfer all edges of Node A to Node B and remove A); **Node Split** (reallocate edges to two newly split sub-nodes); dangerous permanent deletion.
    * 🔗 **Relationship Operations:** Reverse edge upstream/downstream directions; switch relation categories; edit original textual evidence; sever false-positive edges or add newly confirmed experimental links.

    💾 **VI. Project Workspaces & Central Memory Vault (Self-Healing Architecture)**
    * 🗂️ **Project Isolation (`projects/<project_id>/`):** Every research topic has an independent workspace directory, storing its private `graph_state.json` and attached PDFs. Freely create blank projects, switch between historical projects, or permanently delete old topics to reclaim disk space.
    * 🏛️ **Central Memory Vault (`vault/`):** All imported literature is registered in `literature_registry.json` via SHA-256 physical cryptographic hashes, preventing duplicate file storage across different projects.
    * 🌐 **Automatic Standard Literature Renaming & Online PMID Alignment:** One-click online alignment with PubMed/DOI. Cryptic or temporary PDF filenames are automatically reconstructed into academic standards (e.g., `PMID: 35123456 | Article_Title.pdf`), and all node/edge provenance tags are updated seamlessly.
    * 🩹 **Missing Literature Detection & Self-Healing Replenishment:** When moving projects across machines or relocating PDFs, the system automatically detects missing physical files, searches the Central Vault by file hash, and restores physical links with a single click.
    * 📷 **Multimodal Figure Source Exemption:** Nodes and edges originating from mechanism figures carry tags such as `[Figure: Page1_Fig1 (P.1)]` and are permanently tied to their parent paper, enjoying auto-healing exemption.

    🤖 **VII. AI Intelligent Graph Tools (Right Dedicated Panel)**
    * 1. 🧹 **Intelligent Tree Cleaning (Pruning):**
        * 🔍 *Topological Diagnosis:* LLM scans whole-graph logic to recommend **Merge** (synonyms), **Hierarchy** (families), or **Downgrade** (shortcuts).
        * 🧲 *Symmetric Relation Folding:* Automatically merges bidirectional $A \leftrightarrow B$ correlation lines into a single undirected edge, summing popularity and merging evidence chains without loss.
        * 🔄 *Mutual Containment Deadlock Resolution:* Detects and resolves reciprocal containment loops ($A \subset B$ and $B \subset A$).
        * 🧪 *Coarse Relation Purification (Selective Pruning):* Placed at the bottom of the checklist to let researchers decide whether to consolidate or strip vague correlation lines without early information loss.
        * ✂️ *Intent-Based Node Cleaning (Intent Pruning):* Type your focus in plain language (e.g., *"Focus exclusively on tau proteostasis in neurons"*); AI identifies off-topic organs, compounds, or bystander molecules for one-click removal.
    * 2. 🔍 **Intelligent Expansion (Smart Expansion):** Select 1-3 core nodes; AI formulates advanced background queries to pull the latest PubMed papers and enqueues them for **Autopilot Continuous Assimilation**.
    * 3. 🔗 **Intelligent Bridging (Bridging):** Select any two seemingly unconnected molecules A and B; AI navigates the network to find hidden multi-step cascades or searches external literature to uncover unverified regulatory hypotheses.
    * 4. 🤖 **Ask AI (Micro-Context Explanation):** Click any node or edge on the canvas to pass its micro-context (raw extraction evidence and literature origin) to the LLM, producing an in-depth biological mechanism essay directly suitable for grant proposals.
    """)
    else:
        st.markdown(""" 
    📖 **Bio-Graph Agent 核心功能操作指南**
    欢迎使用 Bio-Graph Agent！这是一款专为生物医学研究者打造的“大模型科研军师”。

    🧬 **基础图谱逻辑与学术使命 (Graph Base Logic)：**
    本系统并非简单的“文本总结工具”，而是通过大语言模型深度研读文献，抽取出结构化的“生物实体（节点）”与“调控关系（连线）”。当你导入多篇文献时，系统会自动进行跨文献实体语义对齐（例如将文献 A 中的 p53 和文献 B 中的 TP53 完美融合为同一个节点），将孤立论文编织成一张具备交叉印证能力的全局动态知识网络，帮助学者一目了然洞察潜在的 $A \to B \to C$ 级联通路，激发基金课题选题灵感！

    ⚙️ **一、 初始环境与高级设置 (侧边栏「更多设置」)**
    在使用前，请展开左侧边栏的 ⚙️ **更多设置** 进行环境配置：

    * 🌐 **大模型引擎：** 选择 API 提供商（OpenRouter、腾讯混元、硅基流动等）并填入 Key；支持离线隐私模式（选择 **Ollama** 即可在断网环境下处理极密医学数据，免密运行）。
    * 📚 **智能检索源：** 选择 PubMed (医学绝对权威) 或 Semantic Scholar (全科 AI 智能极速检索，自带智能降级防封机制)。
    * ⚡ **并发线程数 (多切块与插图并行解析)：** 设置 1~8 并发线程，大幅压缩长篇文献切块抽取与机制图转译的等待耗时。
    * 🛡️ **关系节点文献正则补漏 (Literature Grounding)：** 开启后，切块分析完成后若发现端点未在实体字典中登记，自动使用正则比对原文。若原文真实存在则回填为合法节点；若纯属大模型幻觉虚构，则坚决剔除虚假连线。
    * 👁️ **专职视觉模型与后缀：** 支持自由跟随主模型，或指定独立的多模态视觉模型（如 Gemini 2.5 Flash-Lite、Gemini 3.8 Flash、千问 VL 72B），并支持后缀覆盖。
    * 🎛️ **界面功能开关：** 根据需要开启 📝 **手动编辑面板** 与 🤖 **AI 智能图谱分析**。

    📂 **二、 从零到一：构建知识图谱 (左侧控制栏)**
    在左侧「1. 文献输入与配置」面板，系统提供两种截然不同的起步方式：

    **方式 A：💡 零基础冷启动 (Cold Start)**
    手头没有任何文献？只需在输入框用大白话输入研究方向（例如：“巨噬细胞极化在肿瘤微环境中的作用”）。点击“一键检索并生成初始图谱”，AI 将自动生成专业检索式，潜入数据库抓取最新文献并构建出第一张初始知识图谱！

    **方式 B：📄 本地文献精准解析**
    * **上传文献：** 将单篇 PDF 拖入上传区。
    * **🎯 划定解析范围：** 通过起始页与终点页精准定位（如仅解析含核心机制图的第 1-5 页），极大节省时间与计算开销。
    * **⚙️ 高级提取模式设置：**
        * ⚡ **仅提取摘要模式：** 极速模式，智能匹配摘要章节并提取核心脉络。
        * 🔍 **自我反思纠错：** 启用独立质检员 Agent，二次比对剔除无原文依据的幻觉实体。
        * 🔗 **追加模式 (核心灵魂)：** 开启后，新解析文献与现有图谱无缝融合生长！关闭此项将清空画布从零开始。
        * 🌐 **实体命名规范：** 统一将节点翻译规范为纯中文、纯英文或保持文献原文。
    * 📷 **机制图/通路图解析策略 (前置多模态视觉转译)：**
        * 🚫 *关闭 (纯文本)：* 传统纯文本解析流。
        * 🎯 *精准匹配 (输入图号)：* 依据关键词（如 `Fig 4`、`pathway`、`Figure 2`）定向狙击目标机制图。
        * ⚡ *选定页全开 (提取全部插图)：* 启用 **空间邻近聚类合并算法**，自动将 PDF 内部重叠、分层的光栅切片（如 Graphical Abstract 及多子图复合 Figure）合并为完整高清机制图，结合均衡适度的矢量机制图检测，并发调用多模态大模型转译为分子拓扑机制文本并注入图谱生成流！

    🕸️ **三、 视觉赋权引擎 (主界面上方控制台)**
    * 👁️ **机制捷径边：** 开启后展现跨越中间节点的“直达捷径”，宏观审视上游到下游的最终调控结果。
    * 🧬 **本体赋权 (包含反馈)：** 将微观实体（如具体蛋白）权重回馈给宏观分类父级（如激酶家族），使关键通路家族在图中自动壮大。
    * 🔥 **节点赋权 (连接度中心性)：** 基于图论度中心性算法，核心枢纽基因随分子连线增多自动放大。
    * 🔗 **连线赋权 (主干通路加粗)：** 被多篇文献反复验证且证据链最坚实的共识连线自动加粗显示。

    🎨 **四、 拓扑网络可视化视觉规范**
    * 🟢 **绿色实线 + 箭头：** 正作用 / 促进 / 激活 ($A \to B$)。
    * 🔴 **红色实线 + 箭头：** 负作用 / 抑制 / 下调 ($A \dashv B$)。
    * ⚪ **灰色实线：** 一般关联 / 相互作用但方向未明。
    * ⚠️ **灰色虚线 (---)：** 机制捷径边（跨级宏观粗放结论）。
    * 🟣 **紫色实线：** 包含 / 归属 / 亚基组成关系。

    ✂️ **五、 专家手动干预 (数据管理中心)**
    * 🧹 **全局操作：** 一键清理游离孤岛节点；批量修改文献来源标签（如统一替换为 *Nature_2026_Tau*）；手动增补关键分子节点。
    * 🎯 **节点手术：** 节点重命名；**强制合并**（将节点 A 的所有连线转移给节点 B 并销毁 A）；**节点拆分**（将宽泛节点拆解为两个新节点并重新分配连线）；彻底危险删除。
    * 🔗 **关系操作：** 逆转上下游方向；切换作用类型；编辑原文文本证据；彻底斩断错误连线或手动搭建已知新桥梁。

    💾 **六、 专属课题工程库与 Memory Vault 集中文献自愈库**
    * 🗂️ **专属课题工程隔离 (`projects/<project_id>/`)：** 每个研究课题享有独立的工作区目录，持久化保存各自的 `graph_state.json` 与专属文献文件。支持随时一键新建空白工程、快速切换历史课题，或彻底删除旧工程释放磁盘空间。
    * 🏛️ **Memory Vault 集中式文献库与 SHA-256 注册表：** 所有导入文献由集中库 `vault/` 通过 SHA-256 物理哈希指纹统一归档于 `literature_registry.json`，不同课题引用同一篇论文时自动复用，杜绝磁盘冗余。
    * 🌐 **文献自动规范重命名与在线权威 PMID 对齐：** 一键联网检索 PubMed/DOI，将随机或哈希命名的 PDF 文件自动重构为学术标准命名（如 `PMID: 35123456 | Article_Title.pdf`），并无缝同步所有实体与关系的溯源标签。
    * 🩹 **缺失文献探测与智能自愈回填：** 跨设备迁移工程或不小心移动文件后，系统能够自动探测缺失的文献原件，基于 SHA-256 物理扫描并在 Central Vault 中一键找回并回填绑定。
    * 📷 **机制图学术溯源豁免：** 来自插图转译的证据标有 `[Figure: Page1_Fig1 (P.1)]` 标签，深度锚定原文且在文献自愈排查中享有自动豁免，不会被误判为缺失论文。

    🤖 **七、 AI 智能图谱分析 (右侧专属面板)**
    * 1. 🧹 **智能洗树 (Pruning)：**
        * 🔍 *拓扑结构智能诊断：* 大模型通读全图连线逻辑，生成清单建议 **Merge** 同义词、**Hierarchy** 建立分类层级、**Downgrade** 宽泛捷径。
        * 🧲 *对称关系无向折叠：* 自动将 $A \leftrightarrow B$ 的相互相关关系折叠为单条无向连线，热度与证据无损合并累加。
        * 🔄 *循环包含死锁化解：* 自动排查双向包含死锁（$A$ 包含 $B$ 且 $B$ 包含 $A$），智能纠偏层次从属关系。
        * 🧪 *粗糙关系高阶提纯 (选配清洗)：* 默认置于审查清单最下方，供学者自主选择是否合并或剔除粗糙的宽泛“相关”连线，防止信息过早丢失。
        * ✂️ *意图节点清洗 (Intent-Based Pruning)：* 输入研究焦点（如“我只关注神经元 tau 蛋白稳态”），AI 将作为严苛审稿人，自动揪出偏题的器官、药物或无关分子，一键物理抹除。
    * 2. 🔍 **智能拓展 (Smart Expansion)：** 选定 1~3 个核心锚点，结合图谱已知背景自动反向生成高级检索式，从 PubMed 抓取最新文献送入队列，开启**无人值守自动化融合模式**，让图谱自我生长！
    * 3. 🔗 **智能桥接 (Bridging)：** 选中任意两个看似无关的分子 A 和 B，AI 自动遍历寻找隐秘的多步传导路径，或呼叫外网 API 强行探勘未被注意的调控桥梁，直接启发全新基金标书构思！
    * 4. 🤖 **问 AI (微观机制解读报告)：** 点击画布上任意节点或连线，提取其微上下文（原文提取证据与文献来源），大模型现场推演并生成详实的生物学机制解读报告，直出标书素材！
    """)


API_PROVIDERS_CONF = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1",
        "tkey": "api_openrouter"
    },
    "tencent": {
        "url": "https://api.hunyuan.cloud.tencent.com/v1",
        "tkey": "api_tencent"
    },
    "siliconflow": {
        "url": "https://api.siliconflow.cn/v1",
        "tkey": "api_siliconflow"
    },
    "ollama": {
        "url": "http://localhost:11434/v1",
        "tkey": "api_ollama"
    },
    "custom": {
        "url": "custom",
        "tkey": "api_custom"
    }
}

CLOUD_MODEL_OPTIONS = [
    ("qwen/qwen-2.5-72b-instruct", "model_qwen_2_5_72b"),
    ("qwen/qwen-3.8-27b", "model_qwen_3_8_27b"),
    ("deepseek/deepseek-v4.1-flash", "model_deepseek_4_1_flash"),
    ("google/gemini-3.8-flash", "model_gemini_3_8_flash"),
    ("google/gemini-3.5-flash-lite", "model_gemini_3_5_flash_lite"),
    ("google/gemini-2.5-flash-lite", "model_gemini_2_5_flash_lite"),
    ("z-ai/glm-5.3-flash", "model_glm_5_3_f"),
    ("tencent/hy4-preview", "model_hy4_preview"),
    ("tencent/hy3-preview", "model_hy3_preview"),
    ("tencent/hy3", "model_hy3"),
]
MODEL_ID_TO_TRANSLATION_KEY = dict(CLOUD_MODEL_OPTIONS)


def normalize_api_provider(prov: str) -> str:
    """将历史或本地化 provider 转换为语言中立 key"""
    if not prov:
        return "openrouter"
    if prov in API_PROVIDERS_CONF:
        return prov
    prov_str = str(prov).lower()
    if "openrouter" in prov_str:
        return "openrouter"
    if "tencent" in prov_str or "混元" in prov_str:
        return "tencent"
    if "siliconflow" in prov_str or "硅基流动" in prov_str:
        return "siliconflow"
    if "ollama" in prov_str:
        return "ollama"
    if "custom" in prov_str or "自定义" in prov_str:
        return "custom"
    return "openrouter"


def normalize_model_id(val: str) -> str:
    """将历史或本地化模型展示名转换为语言中立 model ID"""
    if not val:
        return "qwen/qwen-2.5-72b-instruct"
    if val in MODEL_ID_TO_TRANSLATION_KEY:
        return val
    # 动态匹配 UI_TEXT 中存在的翻译（若 UI_TEXT 已加载）
    ui_dict = globals().get("UI_TEXT", {})
    for mid, tkey in CLOUD_MODEL_OPTIONS:
        if tkey in ui_dict:
            zh_val = ui_dict[tkey].get("zh")
            en_val = ui_dict[tkey].get("en")
            if val == zh_val or val == en_val:
                return mid
    val_l = str(val).lower()
    if "gemini-3.5" in val_l or "gemini 3.5" in val_l:
        return "google/gemini-3.5-flash-lite"
    if "gemini-2.5" in val_l or "gemini 2.5" in val_l:
        return "google/gemini-2.5-flash-lite"
    if "gemini-3.8" in val_l or "gemini 3.8" in val_l:
        return "google/gemini-3.8-flash"
    if "deepseek" in val_l:
        return "deepseek/deepseek-v4.1-flash"
    if "glm" in val_l:
        return "z-ai/glm-5.3-flash"
    if "hy4" in val_l or "hunyuan 4" in val_l:
        return "tencent/hy4-preview"
    if "hy3-preview" in val_l or "hunyuan 3 preview" in val_l:
        return "tencent/hy3-preview"
    if "hy3" in val_l or "hunyuan 3" in val_l:
        return "tencent/hy3"
    if "3.8" in val_l and "qwen" in val_l:
        return "qwen/qwen-3.8-27b"
    if "qwen" in val_l:
        return "qwen/qwen-2.5-72b-instruct"
    return "qwen/qwen-2.5-72b-instruct"


def load_config():
    """在应用初次加载或视图切换重跑时，持久化确保本地配置及默认状态无缝注入 session_state"""
    # 1. 设定需要持久化的默认基准值（去掉了所有绘图相关的选项）
    default_config = {
        "api_provider": "openrouter",
        "custom_base_url": "https://api.openai.com/v1",
        "selected_model_id": "google/gemini-3.5-flash-lite",
        "selected_model_name": "Gemini 3.5 Flash-Lite (谷歌性价比)",  # 保持兼容
        "custom_model_id": "",  # 用于 Ollama 等自定义手填的模型
        "model_suffix": "",  # 记录云端模型的后缀或保底覆盖
        "search_database": "PubMed",
        "is_summary_only": False,
        "use_reflection": True,
        "append_mode": True,
        "enable_relation_grounding": True,
        "concurrency_workers": 4,
        "ENABLE_EDITOR": True,  # ✨ 恢复的开关 1
        "ENABLE_AI_CLEANER": True,  # ✨ 恢复的开关 2
        "ui_language": "zh",
        "vision_strategy": "opt_off",
        "vision_match_keyword": "",
        "vision_model_choice": "auto",
        "vision_model_suffix": "",
        "custom_vision_model_id": "",
        "active_project_id": ""
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                local_config = json.load(f)
                local_config.pop("entity_language", None)
                if "selected_model_id" not in local_config and "selected_model_name" in local_config:
                    local_config["selected_model_id"] = normalize_model_id(local_config["selected_model_name"])
                if "api_provider" in local_config:
                    local_config["api_provider"] = normalize_api_provider(local_config["api_provider"])
                if "search_database" in local_config:
                    cur_db = local_config["search_database"]
                    local_config["search_database"] = "PubMed" if "PubMed" in str(cur_db) else "Semantic Scholar"
                if "vision_strategy" in local_config:
                    vs = str(local_config["vision_strategy"])
                    if vs not in ["opt_off", "opt_match", "opt_all"]:
                        if "关" in vs or "Off" in vs:
                            local_config["vision_strategy"] = "opt_off"
                        elif "匹配" in vs or "Match" in vs:
                            local_config["vision_strategy"] = "opt_match"
                        elif "全" in vs or "All" in vs:
                            local_config["vision_strategy"] = "opt_all"
                        else:
                            local_config["vision_strategy"] = "opt_off"
                default_config.update(local_config)
        except Exception as e:
            print(f"读取本地配置失败: {e}")

    # 无论初次加载还是从工作台切回主图谱，确保每个配置键都在 session_state 中稳态驻留
    for k, v in default_config.items():
        if k not in st.session_state or st.session_state[k] is None:
            st.session_state[k] = v

    if "language" not in st.session_state:
        st.session_state.language = st.session_state.ui_language

    st.session_state.config_loaded = True


def save_config():
    """仅保存核心业务设置，合并现有磁盘配置以防非活动视图擦除其他视图的配置"""
    config_keys = [
        "api_provider", "custom_base_url", "selected_model_id", "selected_model_name", "custom_model_id", "model_suffix",
        "search_database", "is_summary_only", "use_reflection", "append_mode", "enable_relation_grounding",
        "concurrency_workers", "ENABLE_EDITOR", "ENABLE_AI_CLEANER", "ui_language",
        "vision_strategy", "vision_match_keyword", "vision_model_choice", "vision_model_suffix", "custom_vision_model_id",
        "active_project_id"
    ]
    disk_config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                disk_config = json.load(f)
        except Exception:
            disk_config = {}

    disk_config.pop("entity_language", None)

    for k in config_keys:
        if k in st.session_state and st.session_state[k] is not None:
            disk_config[k] = st.session_state[k]

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(disk_config, f, indent=4)
    except Exception as e:
        print(f"保存本地配置失败: {e}")


def resolve_vision_model_id(main_model_id, vision_choice, vision_suffix):
    """
    智能解析专职视觉模型 ID：
    - 若指定预设或自定义模型，则直接使用；
    - 若选择自由跟随，极简匹配已有模型（gemini, glm, deepseek）则跟随主模型，否则安全分流到 gemini-2.5-flash-lite；
    - 拼接视觉后缀（若有）。
    """
    preset_map = {
        "gemini_2_5_flash_lite": "google/gemini-2.5-flash-lite",
        "gemini_3_8_flash": "google/gemini-3.8-flash",
        "qwen_2_5_vl_72b": "qwen/qwen-2.5-vl-72b-instruct"
    }

    if vision_choice in preset_map:
        base_id = preset_map[vision_choice]
    elif vision_choice != "auto" and vision_choice:
        base_id = vision_choice
    else:
        # 自由跟随主模型模式：极简匹配已知多模态模型
        mid = (main_model_id or "").lower()
        if any(k in mid for k in ["gemini",  "deepseek"]):
            base_id = main_model_id
        else:
            base_id = "google/gemini-2.5-flash-lite"

    if vision_suffix and vision_suffix.strip():
        s = vision_suffix.strip()
        if s.startswith(":"):
            return f"{base_id}{s}"
        elif "/" in s:
            return s
        else:
            return f"{base_id}:{s}"
    return base_id


PROJECTS_ROOT_DIR = "projects"
os.makedirs(PROJECTS_ROOT_DIR, exist_ok=True)


def generate_project_id() -> str:
    """生成唯一且格式规范的工程内部唯一标识 (ID)"""
    return f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"


def get_current_project_id() -> str:
    """获取当前活动工程的内部唯一 ID（若无则按 active_project_id 或最近修改的工程恢复，否则自动生成）"""
    if "current_project_id" not in st.session_state or not st.session_state.current_project_id:
        active_pid = st.session_state.get("active_project_id", "")
        if active_pid and os.path.exists(os.path.join(PROJECTS_ROOT_DIR, active_pid)):
            st.session_state.current_project_id = active_pid
        elif os.path.exists(PROJECTS_ROOT_DIR):
            dirs = [
                d for d in os.listdir(PROJECTS_ROOT_DIR)
                if os.path.isdir(os.path.join(PROJECTS_ROOT_DIR, d)) and os.path.exists(os.path.join(PROJECTS_ROOT_DIR, d, "project.biokg"))
            ]
            if dirs:
                dirs.sort(key=lambda d: os.path.getmtime(os.path.join(PROJECTS_ROOT_DIR, d)), reverse=True)
                st.session_state.current_project_id = dirs[0]
            else:
                st.session_state.current_project_id = generate_project_id()
        else:
            st.session_state.current_project_id = generate_project_id()
    return st.session_state.current_project_id


def get_current_project_name() -> str:
    """获取当前活动工程的用户可读名称（作为属性值存储，不作为物理路径或文件名）"""
    if "current_project_name" not in st.session_state or not st.session_state.current_project_name:
        st.session_state.current_project_name = "默认课题"
    return str(st.session_state.current_project_name).strip() or "默认课题"


def get_project_dir_by_id(project_id: str) -> str:
    """根据工程唯一 ID 获取对应的物理目录 projects/{project_id}/"""
    clean_id = re.sub(r'[\\/*?:"<>|]', "_", str(project_id).strip())
    pdir = os.path.join(PROJECTS_ROOT_DIR, clean_id)
    os.makedirs(pdir, exist_ok=True)
    return pdir


def get_project_papers_dir_by_id(project_id: str) -> str:
    """根据工程唯一 ID 获取文献存放目录"""
    pdir = get_project_dir_by_id(project_id)
    papers_dir = os.path.join(pdir, "papers")
    os.makedirs(papers_dir, exist_ok=True)
    return papers_dir


def get_current_papers_dir() -> str:
    """获取当前活动工程的文献存放目录"""
    return get_project_papers_dir_by_id(get_current_project_id())


def find_project_paper(fname: str, proj_id: str = None) -> Optional[str]:
    """在当前工程及所有本地工程的 papers/ 目录中查找文献，支持哈希、纯净展示名、物理名与别名多级检索及自动自愈拷贝"""
    if not fname:
        return None
    fname = extract_parent_doc_source(fname)
    if not fname:
        return None
    pid = proj_id if proj_id else get_current_project_id()
    cur_papers_dir = get_project_papers_dir_by_id(pid)

    # 1. 尝试直接按原字符串在当前工程查找
    cand_path = os.path.join(cur_papers_dir, fname)
    if os.path.exists(cand_path) and os.path.isfile(cand_path):
        return cand_path

    # 2. 尝试追加 .pdf 后缀查找
    pdf_name = get_physical_pdf_filename(fname)
    cand_path_pdf = os.path.join(cur_papers_dir, pdf_name)
    if os.path.exists(cand_path_pdf) and os.path.isfile(cand_path_pdf):
        return cand_path_pdf

    # 3. 通过注册表解析 (命中 doc_hash, display_name, physical_file, 或 aliases)
    ident = resolve_paper_identity(pid, fname)
    if ident:
        phys = ident.get("physical_file")
        if phys:
            reg_phys_path = os.path.join(cur_papers_dir, phys)
            if os.path.exists(reg_phys_path) and os.path.isfile(reg_phys_path):
                return reg_phys_path

    # 4. 跨工程检索与自动自愈补齐 (避免同内容文件重复拷贝)
    if os.path.exists(PROJECTS_ROOT_DIR):
        for other_pid in os.listdir(PROJECTS_ROOT_DIR):
            if other_pid == pid:
                continue
            alt_papers = os.path.join(PROJECTS_ROOT_DIR, other_pid, "papers")
            search_candidates = [fname, pdf_name]
            if ident and ident.get("physical_file"):
                search_candidates.append(ident.get("physical_file"))
            for cand_f in search_candidates:
                alt_path = os.path.join(alt_papers, cand_f)
                if os.path.exists(alt_path) and os.path.isfile(alt_path):
                    # 若当前工程已存在相同哈希文件，优先返回本地文件路径，严禁生成冗余副本
                    alt_hash = compute_file_content_hash(alt_path)
                    if alt_hash:
                        local_ident = resolve_paper_identity(pid, alt_hash)
                        if local_ident and local_ident.get("physical_file"):
                            existing_local = os.path.join(cur_papers_dir, local_ident["physical_file"])
                            if os.path.exists(existing_local) and os.path.isfile(existing_local):
                                return existing_local
                    target_local = os.path.join(cur_papers_dir, cand_f if cand_f.lower().endswith(".pdf") else f"{cand_f}.pdf")
                    if os.path.exists(target_local):
                        return target_local
                    try:
                        shutil.copy2(alt_path, target_local)
                        return target_local
                    except Exception:
                        return alt_path
    return None


def is_pmid_or_virtual_source(source: str) -> bool:
    """利用正则精准识别是否属于线上 PubMed/PMID 摘要来源或系统虚拟标记，完全放行不报缺失"""
    if not source:
        return True
    s = str(source).strip()
    # 1. 匹配标准 PubMed/PMID 前缀及变体 (如 PubMed:39218274, PMID: 12345678, pmid:12345678, PubMed_xxx)
    if re.search(r'^(?:pubmed|pmid)\s*[:#\-_\s]?\s*\d+', s, re.IGNORECASE):
        return True
    # 2. 单词边界包含 pubmed 或 pmid 的标识
    if re.search(r'\b(?:pubmed|pmid)\b', s, re.IGNORECASE):
        return True
    # 3. 纯 6-9 位数字（常用于直接填写的 PMID）
    if re.match(r'^\d{6,9}$', s):
        return True
    # 4. 系统内置或手动虚拟标记
    virtual_tags = {
        "AI 分析", "未知", "手动添加", "手动修改", "手动拆分",
        "AI Analysis", "Unknown", "Manual Add", "Manual Edit", "Manual Split",
        "无", "None", ""
    }
    if s in virtual_tags:
        return True
    if not extract_parent_doc_source(s):
        return True
    return False


def get_expected_papers_for_project(proj_id: str, data: Optional[dict] = None) -> List[str]:
    """获取指定工程期望包含的物理文献文件名列表（已排除 PMID 和系统虚拟来源）"""
    if data is None:
        pkg_file = os.path.join(get_project_dir_by_id(proj_id), "project.biokg")
        if not os.path.exists(pkg_file):
            return []
        try:
            with open(pkg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

    expected_files = set()

    # 1. 来自 analyzed_files 列表
    for f in data.get("analyzed_files", []):
        f_clean = extract_parent_doc_source(str(f).strip())
        if f_clean and not is_pmid_or_virtual_source(f_clean):
            expected_files.add(f_clean)

    # 2. 来自 entities 与 relations 中的 doc_source 引用
    for item in data.get("entities", []) + data.get("relations", []):
        ds = item.get("doc_source", "")
        if ds:
            for part in str(ds).split("|"):
                part_clean = extract_parent_doc_source(part.strip())
                if part_clean and not is_pmid_or_virtual_source(part_clean):
                    if "." in part_clean or part_clean in expected_files:
                        expected_files.add(part_clean)

    return sorted(list(expected_files))


def get_project_missing_papers(proj_id: str, data: Optional[dict] = None) -> List[str]:
    """检测指定工程中缺失的实体文献（通过哈希、别名链与多级检索，全面豁免已更名文献）"""
    expected = get_expected_papers_for_project(proj_id, data)
    if not expected:
        return []

    missing = []
    for fname in expected:
        # 使用增强版检索（涵盖别名映射、.pdf 后缀对齐与跨工程自愈）
        found_path = find_project_paper(fname, proj_id=proj_id)
        if not found_path:
            # 再使用 resolve_paper_identity 终极核对
            ident = resolve_paper_identity(proj_id, fname)
            if ident and ident.get("physical_file"):
                p_papers_dir = get_project_papers_dir_by_id(proj_id)
                if os.path.exists(os.path.join(p_papers_dir, ident["physical_file"])):
                    continue
            missing.append(fname)

    return missing


def save_replenished_paper(target_fname: str, file_bytes: bytes) -> str:
    """补齐缺失文献：将上传的数据写入当前工程 papers 目录并同步更新记忆库与哈希注册表（哈希防重）"""
    target_dir = get_current_papers_dir()
    os.makedirs(target_dir, exist_ok=True)
    fhash = compute_file_content_hash(file_bytes)
    disp_name = get_clean_display_title(target_fname)
    phys_name = get_physical_pdf_filename(target_fname)
    cur_pid = get_current_project_id()

    registry = load_project_paper_registry(cur_pid)
    papers = registry.get("papers", {})

    if fhash and fhash in papers:
        existing = papers[fhash]
        canon_disp = existing.get("display_name") or disp_name
        existing_phys = existing.get("physical_file")
        if existing_phys and os.path.exists(os.path.join(target_dir, existing_phys)):
            register_or_update_paper(
                cur_pid,
                doc_hash=fhash,
                display_name=canon_disp,
                physical_file=existing_phys,
                aliases=[target_fname, disp_name, phys_name],
                file_size=len(file_bytes),
                source_type="pdf"
            )
            cur_list = st.session_state.get("analyzed_files", [])
            st.session_state.analyzed_files = canonicalize_project_papers(cur_pid, cur_list + [canon_disp])
            save_local_vault()
            return canon_disp

    fpath = os.path.join(target_dir, phys_name)
    with open(fpath, "wb") as f:
        f.write(file_bytes)

    register_or_update_paper(
        cur_pid,
        doc_hash=fhash,
        display_name=disp_name,
        physical_file=phys_name,
        aliases=[target_fname, disp_name, phys_name],
        file_size=len(file_bytes),
        source_type="pdf"
    )

    cur_list = st.session_state.get("analyzed_files", [])
    st.session_state.analyzed_files = canonicalize_project_papers(cur_pid, cur_list + [disp_name])
    save_local_vault()
    return disp_name


def find_project_by_id_or_name(proj_id: str, proj_name: str) -> str:
    """根据 project_id 或 project_name 在本地 projects/ 目录中检索已有的工程目录 ID"""
    if proj_id and os.path.exists(os.path.join(PROJECTS_ROOT_DIR, proj_id)):
        return proj_id
    if proj_name and os.path.exists(PROJECTS_ROOT_DIR):
        for pid in os.listdir(PROJECTS_ROOT_DIR):
            pkg_file = os.path.join(PROJECTS_ROOT_DIR, pid, "project.biokg")
            if os.path.exists(pkg_file):
                try:
                    with open(pkg_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("project_name") == proj_name:
                        return pid
                except Exception:
                    pass
    return proj_id if proj_id else generate_project_id()


def save_analyzed_paper(file_name: str, file_bytes: bytes) -> str:
    """保存解析的论文到当前工程的 papers 目录并同步注册不可变哈希索引（防同内容多重副本，返回规范展示名）"""
    target_dir = get_current_papers_dir()
    os.makedirs(target_dir, exist_ok=True)
    fhash = compute_file_content_hash(file_bytes)
    disp_name = get_clean_display_title(file_name)
    phys_name = get_physical_pdf_filename(file_name)
    cur_pid = get_current_project_id()

    registry = load_project_paper_registry(cur_pid)
    papers = registry.get("papers", {})

    # 检查是否已存在相同内容哈希的文献
    if fhash and fhash in papers:
        existing = papers[fhash]
        canon_disp = existing.get("display_name") or disp_name
        existing_phys = existing.get("physical_file")
        if existing_phys and os.path.exists(os.path.join(target_dir, existing_phys)):
            # 物理文件已完好存在，仅将新文件名作为别名登记，不产生多余磁盘副本
            register_or_update_paper(
                cur_pid,
                doc_hash=fhash,
                display_name=canon_disp,
                physical_file=existing_phys,
                aliases=[file_name, disp_name, phys_name],
                file_size=len(file_bytes),
                source_type="pdf"
            )
            return canon_disp

    fpath = os.path.join(target_dir, phys_name)
    with open(fpath, "wb") as f:
        f.write(file_bytes)

    register_or_update_paper(
        cur_pid,
        doc_hash=fhash,
        display_name=disp_name,
        physical_file=phys_name,
        aliases=[file_name, disp_name, phys_name],
        file_size=len(file_bytes),
        source_type="pdf"
    )
    return disp_name


def rename_project_paper(old_fname: str, new_fname: str, proj_id: str = None) -> bool:
    """
    原子更新：将工程文献物理重命名（严格确保 .pdf 后缀），同步更新注册表别名链、纯净展示名（纯净无 .pdf 后缀）、
    会话文献列表、实体与关系 doc_source 及 doc_hash 双锚定，并原子落盘 project.biokg 与 paper_registry.json
    """
    if not old_fname or not new_fname or old_fname == new_fname:
        return True

    target_pid = proj_id if proj_id else get_current_project_id()
    papers_dir = get_project_papers_dir_by_id(target_pid)

    clean_disp_new = get_clean_display_title(new_fname)
    clean_disp_old = get_clean_display_title(old_fname)
    phys_new_name = get_physical_pdf_filename(clean_disp_new)

    # 1. 查找旧文件对应的物理文件路径与注册表身份
    old_phys_path = None
    for cand in [old_fname, get_physical_pdf_filename(old_fname), clean_disp_old]:
        cand_p = os.path.join(papers_dir, cand)
        if os.path.exists(cand_p) and os.path.isfile(cand_p):
            old_phys_path = cand_p
            break

    doc_hash = ""
    ident = resolve_paper_identity(target_pid, old_fname)
    if ident:
        doc_hash = ident.get("doc_hash", "")
        if not old_phys_path and ident.get("physical_file"):
            reg_phys_p = os.path.join(papers_dir, ident["physical_file"])
            if os.path.exists(reg_phys_p) and os.path.isfile(reg_phys_p):
                old_phys_path = reg_phys_p

    # 构建该文献的全部已知别名集合（包含历史别名链、展示名及.pdf物理名变体）
    all_old_aliases = set()
    if ident:
        for a in ident.get("aliases", []):
            if a:
                all_old_aliases.add(str(a).strip())
                all_old_aliases.add(get_clean_display_title(str(a)))
                all_old_aliases.add(get_physical_pdf_filename(str(a)))
        if ident.get("display_name"):
            all_old_aliases.add(ident["display_name"])
            all_old_aliases.add(get_physical_pdf_filename(ident["display_name"]))
        if ident.get("physical_file"):
            all_old_aliases.add(ident["physical_file"])
            all_old_aliases.add(get_clean_display_title(ident["physical_file"]))
    all_old_aliases.update({old_fname, clean_disp_old, get_physical_pdf_filename(old_fname)})
    all_old_aliases = {a for a in all_old_aliases if a}

    # 2. 物理文件重命名并清理多余历史残留副本
    new_phys_path = os.path.join(papers_dir, phys_new_name)
    is_physical = False
    if old_phys_path and os.path.exists(old_phys_path):
        is_physical = True
        try:
            if not doc_hash:
                doc_hash = compute_file_content_hash(old_phys_path)
            if old_phys_path != new_phys_path:
                if os.path.exists(new_phys_path):
                    os.remove(new_phys_path)
                os.rename(old_phys_path, new_phys_path)
            # 扫描 papers_dir，清理属于该文献别名的历史冗余物理副本（防止旧文件自克隆残留）
            for f in os.listdir(papers_dir):
                if f != phys_new_name and (f in all_old_aliases or get_clean_display_title(f) in all_old_aliases):
                    dup_p = os.path.join(papers_dir, f)
                    try:
                        if os.path.isfile(dup_p) and dup_p != new_phys_path:
                            os.remove(dup_p)
                    except Exception:
                        pass
        except Exception as e:
            print(f"⚠️ [rename_project_paper] 物理文件更名失败: {e}")
            return False
    else:
        # 线上/虚拟来源生成规范不可变哈希
        if not doc_hash:
            doc_hash = compute_source_identity_hash(clean_disp_new if "pmid" in clean_disp_new.lower() else old_fname)

    # 3. 更新文献注册表 (Paper Registry)
    pmid_val = None
    pmid_m = re.search(r"(?:^|[\W_])(?:pmid|pubmed)?[:\s#_-]*(\d{6,9})\b", clean_disp_new, re.I)
    if pmid_m:
        pmid_val = pmid_m.group(1)
    elif ident and ident.get("pmid"):
        pmid_val = ident.get("pmid")

    register_or_update_paper(
        target_pid,
        doc_hash=doc_hash,
        display_name=clean_disp_new,
        physical_file=phys_new_name if is_physical else None,
        aliases=list(all_old_aliases) + [clean_disp_new, phys_new_name],
        pmid=pmid_val,
        source_type="pdf" if is_physical else "online_pmid"
    )

    # 4. 会话中的 analyzed_files 列表替换与哈希全局规范化
    cur_files = st.session_state.get("analyzed_files", [])
    replaced = False
    updated_files = []
    for f in cur_files:
        f_str = str(f).strip()
        if f_str in all_old_aliases or get_clean_display_title(f_str) in all_old_aliases:
            updated_files.append(clean_disp_new)
            replaced = True
        else:
            updated_files.append(f)
    if not replaced:
        updated_files.append(clean_disp_new)
    # 强制通过注册表身份折叠与去重，杜绝曾用名残余
    st.session_state.analyzed_files = canonicalize_project_papers(target_pid, updated_files)

    # 5. 实体 master_entities 中 doc_source 与 doc_hash 的双锚定更新（按长度降序安全替换）
    sorted_aliases = sorted(list(all_old_aliases), key=len, reverse=True)
    for ent in st.session_state.get("master_entities", []):
        ds = ent.get("doc_source", "")
        if ds:
            matched = False
            for target_old in sorted_aliases:
                if target_old and target_old in ds:
                    ds = ds.replace(target_old, clean_disp_new)
                    matched = True
            # 清洗 ds，按管道符拆解并去重
            parts = [p.strip() for p in ds.split("|") if p.strip()]
            cleaned_parts = []
            for p in parts:
                cp = get_clean_display_title(p) if not is_pmid_or_virtual_source(p) else p
                if cp not in cleaned_parts:
                    cleaned_parts.append(cp)
            sanitized_ds = " | ".join(cleaned_parts) if cleaned_parts else clean_disp_new
            if matched:
                ent["doc_source"] = sanitized_ds
                ent["doc_hash"] = doc_hash
            elif ent.get("doc_hash") == doc_hash:
                ent["doc_source"] = clean_disp_new

    # 6. 关系 master_relations 中 doc_source、doc_hash 与 reason 证据的全局替换
    for rel in st.session_state.get("master_relations", []):
        ds = rel.get("doc_source", "")
        if ds:
            matched = False
            for target_old in sorted_aliases:
                if target_old and target_old in ds:
                    ds = ds.replace(target_old, clean_disp_new)
                    matched = True
            parts = [p.strip() for p in ds.split("|") if p.strip()]
            cleaned_parts = []
            for p in parts:
                cp = get_clean_display_title(p) if not is_pmid_or_virtual_source(p) else p
                if cp not in cleaned_parts:
                    cleaned_parts.append(cp)
            sanitized_ds = " | ".join(cleaned_parts) if cleaned_parts else clean_disp_new
            if matched:
                rel["doc_source"] = sanitized_ds
                rel["doc_hash"] = doc_hash
            elif rel.get("doc_hash") == doc_hash:
                rel["doc_source"] = clean_disp_new

        if "reason" in rel:
            r = rel.get("reason", "")
            for target_old in sorted_aliases:
                if target_old and target_old in r:
                    r = r.replace(f"[源自: {target_old}]", f"[源自: {clean_disp_new}]").replace(target_old, clean_disp_new)
            rel["reason"] = r

    # 7. 持久化落盘到 project.biokg
    save_local_vault()
    return True


def reconcile_loaded_vault(target_pid: str, loaded_data: dict) -> dict:
    """
    对载入的记忆库数据执行文献身份自愈对齐：
    1. 导入/合并外部记忆库自带的 paper_registry；
    2. 遍历 analyzed_files、entities、relations，将曾用名/旧文件名自动解析升级为现行展示名（无.pdf），并锚定 doc_hash；
    3. 同步补全物理文献。
    """
    registry = load_project_paper_registry(target_pid)
    incoming_reg = loaded_data.get("paper_registry", {})
    if isinstance(incoming_reg, dict) and "papers" in incoming_reg:
        for h, item in incoming_reg.get("papers", {}).items():
            if h not in registry["papers"]:
                registry["papers"][h] = item
            else:
                combined_aliases = list(dict.fromkeys(registry["papers"][h].get("aliases", []) + item.get("aliases", [])))
                registry["papers"][h]["aliases"] = combined_aliases
        save_project_paper_registry(target_pid, registry)

    alias_to_info = {}
    for h, item in registry.get("papers", {}).items():
        disp = item.get("display_name", "")
        for a in item.get("aliases", []):
            if a:
                alias_to_info[a] = (disp, h)
                alias_to_info[get_clean_display_title(a)] = (disp, h)
        if disp:
            alias_to_info[disp] = (disp, h)
        phys = item.get("physical_file")
        if phys:
            alias_to_info[phys] = (disp, h)

    # 更新 analyzed_files
    new_analyzed = []
    for f in loaded_data.get("analyzed_files", []):
        f_clean = str(f).strip()
        if f_clean in alias_to_info:
            new_analyzed.append(alias_to_info[f_clean][0])
        elif get_clean_display_title(f_clean) in alias_to_info:
            new_analyzed.append(alias_to_info[get_clean_display_title(f_clean)][0])
        else:
            new_analyzed.append(get_clean_display_title(f_clean) if not is_pmid_or_virtual_source(f_clean) else f_clean)
    loaded_data["analyzed_files"] = canonicalize_project_papers(target_pid, new_analyzed)

    sorted_alias_items = sorted(list(alias_to_info.items()), key=lambda x: len(x[0]), reverse=True)

    # 更新 entities
    for ent in loaded_data.get("entities", []):
        ds = ent.get("doc_source", "")
        if ds:
            matched_disp, matched_hash = None, None
            if ds in alias_to_info:
                matched_disp, matched_hash = alias_to_info[ds]
            elif get_clean_display_title(ds) in alias_to_info:
                matched_disp, matched_hash = alias_to_info[get_clean_display_title(ds)]
            else:
                for a, (d, h) in sorted_alias_items:
                    if a and a in ds:
                        ds = ds.replace(a, d)
                        matched_hash = h
            if matched_disp:
                ent["doc_source"] = matched_disp
            else:
                parts = [p.strip() for p in ds.split("|") if p.strip()]
                cleaned_parts = []
                for p in parts:
                    cp = get_clean_display_title(p) if not is_pmid_or_virtual_source(p) else p
                    if cp not in cleaned_parts:
                        cleaned_parts.append(cp)
                ent["doc_source"] = " | ".join(cleaned_parts) if cleaned_parts else ds
            if matched_hash:
                ent["doc_hash"] = matched_hash

    # 更新 relations
    for rel in loaded_data.get("relations", []):
        ds = rel.get("doc_source", "")
        if ds:
            matched_disp, matched_hash = None, None
            if ds in alias_to_info:
                matched_disp, matched_hash = alias_to_info[ds]
            elif get_clean_display_title(ds) in alias_to_info:
                matched_disp, matched_hash = alias_to_info[get_clean_display_title(ds)]
            else:
                for a, (d, h) in sorted_alias_items:
                    if a and a in ds:
                        ds = ds.replace(a, d)
                        matched_hash = h
            if matched_disp:
                rel["doc_source"] = matched_disp
            else:
                parts = [p.strip() for p in ds.split("|") if p.strip()]
                cleaned_parts = []
                for p in parts:
                    cp = get_clean_display_title(p) if not is_pmid_or_virtual_source(p) else p
                    if cp not in cleaned_parts:
                        cleaned_parts.append(cp)
                rel["doc_source"] = " | ".join(cleaned_parts) if cleaned_parts else ds
            if matched_hash:
                rel["doc_hash"] = matched_hash

        if "reason" in rel:
            r = rel.get("reason", "")
            for a, (d, h) in sorted_alias_items:
                if a and a in r:
                    r = r.replace(f"[源自: {a}]", f"[源自: {d}]")
            rel["reason"] = r

    return loaded_data


def save_local_vault():
    """将当前记忆库（实体、关系、历史文献、文献注册表、工程 ID 与名称）原地保存到专属工程归档 projects/{project_id}/project.biokg（单一真实源）"""
    proj_id = get_current_project_id()
    proj_name = get_current_project_name()
    registry = load_project_paper_registry(proj_id)
    data = {
        "version": "2.0",
        "project_id": proj_id,
        "project_name": proj_name,
        "entities": st.session_state.get("master_entities", []),
        "relations": st.session_state.get("master_relations", []),
        "analyzed_files": st.session_state.get("analyzed_files", []),
        "paper_registry": registry,
        "figure_transcriptions": st.session_state.get("figure_transcriptions", {}),
        "workbench_state": st.session_state.get("workbench_state", {})
    }
    # 1. 原地归档到当前工程专属目录 projects/{project_id}/project.biokg（唯一真实数据源）
    try:
        proj_dir = get_project_dir_by_id(proj_id)
        proj_biokg = os.path.join(proj_dir, "project.biokg")
        with open(proj_biokg, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"工程专属记忆库写入失败: {e}")

    # 2. 将当前活动工程 ID 记录到本地配置中，作为 F5 刷新或冷启动的轻量快速指针
    st.session_state.active_project_id = proj_id
    save_config()


def redraw_and_update():
    """重新渲染网络拓扑交互图谱，并实时自动落盘同步工程专属记忆库"""
    from back_logic import GraphVisualizer
    import os
    visualizer = GraphVisualizer()
    html_file = ".bio_knowledge_graph.html"
    if os.path.exists(html_file):
        os.remove(html_file)

    current_show_shortcuts = st.session_state.get("show_shortcuts_toggle", False)

    # 🧮 抓取三大赋权引擎的开关状态
    empower_ontology = st.session_state.get("empower_ontology", False)
    alpha_ontology = st.session_state.get("alpha_ontology", 0.5)

    # ✨ 新增：节点互相辐射开关
    empower_node = st.session_state.get("empower_node", False)
    beta_node = st.session_state.get("beta_node", 0.2)

    # ✨ 新增：主干连线加粗开关
    empower_edge = st.session_state.get("empower_edge", False)
    gamma_edge = st.session_state.get("gamma_edge", 0.1)

    visualizer.generate_html(
        st.session_state.master_entities,
        st.session_state.master_relations,
        output_file=html_file,
        show_shortcuts=current_show_shortcuts,
        empower_ontology=empower_ontology,
        alpha_ontology=alpha_ontology,
        empower_node=empower_node,
        beta_node=beta_node,
        empower_edge=empower_edge,
        gamma_edge=gamma_edge,
        output_lang=st.session_state.get("ui_language", "zh")
    )

    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            st.session_state.html_data = f.read()

        # 同步将生成的图谱 HTML 备份到工程专属目录
        try:
            cur_pid = get_current_project_id()
            proj_dir = get_project_dir_by_id(cur_pid)
            proj_html = os.path.join(proj_dir, "graph.html")
            with open(proj_html, "w", encoding="utf-8") as f:
                f.write(st.session_state.html_data)
        except Exception:
            pass

    # 💾 核心闭环：每次图谱重绘（编辑节点/增删关系/赋权/剪枝/扩展），自动将最新图谱原子落盘到专属工程与根缓存
    save_local_vault()


def switch_to_project(proj_id: str) -> bool:
    """切换当前活动工程至指定的历史工程（支持原子落盘与画布完全隔离切换）"""
    # 🛡️ 步骤 1：切换前首先自动保存当前活动工程，确保未保存的改动落盘
    save_local_vault()

    proj_dir = get_project_dir_by_id(proj_id)
    pkg_file = os.path.join(proj_dir, "project.biokg")
    if not os.path.exists(pkg_file):
        return False
    try:
        with open(pkg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = reconcile_loaded_vault(proj_id, data)
        pname = data.get("project_name", proj_id)
        st.session_state.current_project_id = proj_id
        st.session_state.current_project_name = pname
        st.session_state.master_entities = data.get("entities", [])
        st.session_state.master_relations = data.get("relations", [])
        st.session_state.figure_transcriptions = data.get("figure_transcriptions", {})
        st.session_state.workbench_state = data.get("workbench_state", {})
        loaded_files = data.get("analyzed_files", [])

        # 检查专属 papers 目录中已有文献并补充
        papers_dir = os.path.join(proj_dir, "papers")
        existing_in_target = [get_clean_display_title(f) for f in os.listdir(papers_dir) if os.path.isfile(os.path.join(papers_dir, f))] if os.path.exists(papers_dir) else []
        st.session_state.analyzed_files = canonicalize_project_papers(proj_id, loaded_files + existing_in_target)

        # 🛡️ 步骤 2：严格重置画布，目标工程有数据则重新生成，若为空白工程则彻底销毁画布与残留
        html_file = ".bio_knowledge_graph.html"
        if st.session_state.master_entities:
            from back_logic import GraphVisualizer
            visualizer = GraphVisualizer()
            if os.path.exists(html_file):
                os.remove(html_file)
            visualizer.generate_html(
                st.session_state.master_entities,
                st.session_state.master_relations,
                output_file=html_file,
                output_lang=st.session_state.get("ui_language", "zh")
            )
            if os.path.exists(html_file):
                with open(html_file, "r", encoding="utf-8") as f:
                    st.session_state.html_data = f.read()
            st.session_state.show_results = True
        else:
            st.session_state.show_results = False
            st.session_state.html_data = ""
            if os.path.exists(html_file):
                os.remove(html_file)

        # 🛡️ 步骤 3：保存目标工程并更新 active_project_id 配置指针
        save_local_vault()
        return True
    except Exception as e:
        print(f"切换工程失败: {e}")
        return False


def load_local_vault():
    """应用启动或页面刷新时，根据 active_project_id 直接载入专属工程记忆库以恢复图谱与工作区元数据"""
    proj_id = get_current_project_id()
    proj_dir = get_project_dir_by_id(proj_id)
    pkg_file = os.path.join(proj_dir, "project.biokg")
    if os.path.exists(pkg_file):
        try:
            with open(pkg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data = reconcile_loaded_vault(proj_id, data)
            # 无论实体是否为空，优先恢复持久化的工程 ID 与课题名称，防止意外生成新工程
            st.session_state.workbench_state = data.get("workbench_state", {})
            st.session_state.current_project_id = data.get("project_id", proj_id)
            pname = data.get("project_name", "默认课题")
            st.session_state.current_project_name = pname
            if "project_name_input_widget" not in st.session_state:
                st.session_state["project_name_input_widget"] = pname

            entities = data.get("entities", [])
            # 仅在会话未初始化实体时（冷启动或浏览器真 F5 刷新）恢复图谱
            if entities and not st.session_state.get("master_entities"):
                st.session_state.master_entities = entities
                st.session_state.master_relations = data.get("relations", [])
                st.session_state.analyzed_files = data.get("analyzed_files", [])
                st.session_state.show_results = True
                redraw_and_update()
            elif not entities and not st.session_state.get("master_entities"):
                # 若磁盘持久化为空白工程且当前也是空，确保关闭图谱渲染
                st.session_state.master_entities = []
                st.session_state.master_relations = []
                st.session_state.analyzed_files = data.get("analyzed_files", [])
                st.session_state.show_results = False
                st.session_state.html_data = ""
        except Exception as e:
            print(f"专属工程记忆库自动加载失败: {e}")


def clear_local_vault():
    """兼容旧接口：当前工程已完全交由 projects/ 目录管理，无根目录临时文件需清理"""
    pass


# ==========================================
# 页面全局配置
# ==========================================
st.set_page_config(page_title="Bio-Graph Agent", page_icon="🧬", layout="wide")
# ==========================================
# 🌍 国际化 (i18n) 双语引擎
# ==========================================
UI_TEXT = {
    # 全局 & 标题
    "page_title": {"zh": "🧬 Bio-Graph Agent 生物知识图谱自动生成引擎", "en": "🧬 Bio-Graph Agent Biomedical Knowledge Graph Engine"},
    "btn_help": {"zh": "❓ 帮助与说明", "en": "❓ Help & Docs"},

    # 侧边栏
    "sidebar_title": {"zh": "## 🧬 Bio-Graph Agent", "en": "## 🧬 Bio-Graph Agent"},
    "sidebar_api_placeholder_local": {"zh": "Ollama 免密运行，可留空", "en": "Ollama local mode, can be empty"},
    "sidebar_api_placeholder_cloud": {"zh": "🔑 请输入您的 API Key", "en": "🔑 Enter your API Key"},
    "sidebar_api_help": {"zh": "刷新即焚，不会本地留存。", "en": "Cleared on refresh, not saved locally."},
    "sidebar_more_settings": {"zh": "⚙️ 更多设置", "en": "⚙️ More Settings"},
    "sidebar_engine_title": {"zh": "#### 🌐 大模型引擎", "en": "#### 🌐 LLM Engine"},
    "sidebar_api_provider": {"zh": "API 提供商", "en": "API Provider"},
    "sidebar_base_url": {"zh": "🔗 Base URL", "en": "🔗 Base URL"},
    "sidebar_model_id_custom": {"zh": "🧠 模型 ID (如 qwen2.5:7b)", "en": "🧠 Model ID (e.g. qwen2.5:7b)"},
    "sidebar_model_select": {"zh": "🧠 选择底层驱动模型", "en": "🧠 Select Base Model"},
    "sidebar_model_suffix": {"zh": "🔧 模型后缀 / 自定义保底 (可选)", "en": "🔧 Model Suffix / Custom Fallback (Optional)"},
    "sidebar_model_suffix_placeholder": {"zh": "如 :free 或完整模型ID", "en": "e.g. :free or full model ID"},
    "sidebar_model_suffix_help": {
        "zh": "可输入后缀（如 :free）自动拼接在所选模型后；亦可输入完整模型ID（如 meta-llama/llama-3.3-70b-instruct）进行全量覆盖。",
        "en": "Enter a suffix (e.g. :free) to append to the model, or enter a full model ID to override."
    },
    "sidebar_model_active_preview": {"zh": "当前生效模型: ", "en": "Active Model: "},
    "sidebar_vision_model_active_preview": {"zh": "生效视觉模型: ", "en": "Active Vision Model: "},
    "sidebar_db_title": {"zh": "#### 📚 智能检索源", "en": "#### 📚 Smart Search Source"},
    "sidebar_db_select": {"zh": "数据库", "en": "Database"},
    "sidebar_features_title": {"zh": "#### 🎛️ 界面功能开关", "en": "#### 🎛️ Feature Toggles"},
    "sidebar_toggle_editor": {"zh": "📝 开启手动编辑面板", "en": "📝 Enable Manual Editor"},
    "sidebar_toggle_ai": {"zh": "🤖 开启 AI 智能图谱分析", "en": "🤖 Enable AI Graph Analysis"},
    "sidebar_contact": {"zh": "🤝 联系我们 / 了解更多", "en": "🤝 Contact / Learn More"},
    "contact_desc": {
        "zh": "**Bio-Graph Agent** 致力于将大语言模型与生物知识图谱无缝融合。\n\n- 🐛 发现 Bug？\n- 💡 有绝妙的新功能想法？\n- 🌟 觉得好用的话，来给我们点个 Star 吧！",
        "en": "**Bio-Graph Agent** is dedicated to seamlessly integrating LLMs with biological knowledge graphs.\n\n- 🐛 Found a bug?\n- 💡 Have a brilliant feature idea?\n- 🌟 If you like it, please give us a Star!"
    },
    "contact_github": {"zh": "⭐ 访问 Github 仓库", "en": "⭐ Visit Github Repo"},
    "contact_email": {"zh": "📧 合作邮箱: zoyuhandd@126.com", "en": "📧 Contact Email: zoyuhandd@126.com"},
    # ==========================
    # 选项下拉框 (Dropdown Options)
    # ==========================
    # 1. API 提供商
    "api_openrouter": {"zh": "OpenRouter (国际节点)", "en": "OpenRouter (Global)"},
    "api_tencent": {"zh": "腾讯云 (官方兼容接口)", "en": "Tencent Cloud (Official API)"},
    "api_siliconflow": {"zh": "硅基流动 (国内极速)", "en": "SiliconFlow (Fast CN)"},
    "api_ollama": {"zh": "Ollama (本地断网运行)", "en": "Ollama (Local Offline)"},
    "api_custom": {"zh": "自定义 (Custom)", "en": "Custom URL"},

    # 2. 模型选项
    "model_gemini_2_5_flash_lite": {"zh": "Gemini 2.5 Flash-Lite (谷歌极致低价版)", "en": "Gemini 2.5 Flash-Lite (Ultra Low Cost)"},
    "model_qwen_2_5_72b": {"zh": "Qwen 2.5 72B (千问学术双语旗舰版)", "en": "Qwen 2.5 72B (Bilingual Flagship)"},
    "model_qwen_3_8_27b": {"zh": "Qwen 3.8 27B (千问学术双语旗舰版)", "en": "Qwen 3.8 27B (Bilingual Flagship)"},
    "model_hy4_preview": {"zh": "Hunyuan 4 Preview (混元4预览版)", "en": "Hunyuan 4 Preview (Cost-effective)"},
    "model_hy3_preview": {"zh": "Hunyuan 3 Preview (预览版 - 高性价比)", "en": "Hunyuan 3 Preview (Cost-effective)"},
    "model_hy3": {"zh": "Hunyuan 3 (正式版 - 最强推理)", "en": "Hunyuan 3 (Official - Max Reasoning)"},
    "model_deepseek_4_1_flash": {"zh": "DeepSeek 4.1 Flash (深度求索极速版)", "en": "DeepSeek 4.1 Flash"},
    "model_glm_5_3": {"zh": "GLM 5.3 (智谱大模型)", "en": "GLM 5.3 (Zhipu AI)"},
    "model_glm_5_3_f": {"zh": "GLM 5.3 flash(智谱大模型)", "en": "GLM 5.3 flash(Zhipu AI)"},
    "model_gemini_3_8_flash": {"zh": "Gemini 3.8 Flash (谷歌高性价比极速版)", "en": "Gemini 3.8 Flash"},
    "model_gemini_3_5_flash_lite": {"zh": "Gemini 3.5 Flash-Lite (谷歌性价比)", "en": "Gemini 3.5 Flash-Lite "},



    # 3. 数据库选项
    "db_pubmed": {"zh": "PubMed (生物医学权威)", "en": "PubMed (Biomedical Auth)"},
    "db_semanticscholar": {"zh": "Semantic Scholar (全科AI智能)", "en": "Semantic Scholar (AI Smart Search)"},
    # 预留给主界面的 Key（后续慢慢加）
    # 主界面：左侧文献输入与配置区
    "col1_title": {"zh": "📂 1. 文献输入与配置", "en": "📂 1. Document Input & Config"},
    "upload_pdf": {"zh": "上传生物学文献 (PDF)", "en": "Upload Biological Document (PDF)"},
    "pdf_read_success": {"zh": "📄 成功读取 PDF 文件，总计 **{pages}** 页。", "en": "📄 Successfully loaded PDF, **{pages}** pages in total."},

    # 零基础冷启动区
    "cold_start_title": {"zh": "### 💡 没有文献？对 AI 说说你要研究什么吧", "en": "### 💡 No PDF? Tell AI what you want to study"},
    "cold_start_caption": {"zh": "只需输入一句话，AI 将自动检索、分批验证并为你搭建初始知识网络。", "en": "Just type a sentence, AI will automatically search, verify, and build your initial knowledge network."},
    "cold_start_placeholder": {"zh": "你的研究方向 (例如: 巨噬细胞极化在肿瘤微环境中的作用)", "en": "Your research topic (e.g. Role of macrophage polarization in tumor microenvironment)"},
    "btn_ai_search": {"zh": "🚀 AI 一键检索并生成初始图谱", "en": "🚀 AI: Search & Generate Initial Graph"},
    "err_no_api_key": {"zh": "❌ 请先在上方配置 API Key！", "en": "❌ Please configure API Key above first!"},
    "warn_no_topic": {"zh": "⚠️ 请输入你想研究的内容！", "en": "⚠️ Please enter your research topic!"},
    "msg_generating_query": {"zh": "🤖 AI 正在生成专业检索式...", "en": "🤖 AI is generating professional query..."},
    "err_invalid_topic": {"zh": "🛑 输入内容似乎不是有效的科研方向，请换个说法试试！", "en": "🛑 The input doesn't seem to be a valid research topic, please try another phrase!"},
    "msg_generated_query": {"zh": "🔎 **生成检索式**: `{query}`", "en": "🔎 **Generated Query**: `{query}`"},
    "msg_fetching_papers": {"zh": "🌐 正在 {db} 抓取前 10 篇高相关文献准备排查...", "en": "🌐 Fetching top 10 relevant papers from {db} for screening..."},
    "warn_no_papers": {"zh": "⚠️ 抱歉，未能检索到任何相关文献，请尝试调整研究方向词汇。", "en": "⚠️ Sorry, no relevant papers found. Please try adjusting your keywords."},
    "msg_fetch_success": {"zh": "✅ 成功获取 {count} 篇候选文献，启动滑动窗口排查...", "en": "✅ Successfully fetched {count} candidate papers. Starting sliding window screening..."},
    "msg_validating_batch": {"zh": "🤔 正在验证第 {start} 到 {end} 篇文献的匹配度...", "en": "🤔 Validating relevance of papers {start} to {end}..."},
    "msg_batch_pass": {"zh": "✅ **第 {start}-{end} 篇文献验证通过**！与你的意图高度匹配。", "en": "✅ **Papers {start}-{end} passed validation**! Highly matches your intent."},
    "msg_ai_review": {"zh": "**AI 评审意见**：{reason}", "en": "**AI Review Opinion**: {reason}"},
    "msg_batch_fail": {"zh": "⏭️ 第 {start}-{end} 篇文献跑偏了 ({reason})。继续排查下一批...", "en": "⏭️ Papers {start}-{end} are off-topic ({reason}). Checking next batch..."},
    "msg_extracting_graph": {"zh": "🧠 正在精读过审文献，深度抽取初始图谱网络...", "en": "🧠 Deeply reading approved papers, extracting initial graph network..."},
    "msg_parsing_paper_progress": {"zh": "正在解析第 {idx}/{total} 篇过审文献 (PMID: {pmid})...", "en": "Parsing approved paper {idx}/{total} (PMID: {pmid})..."},
    "msg_graph_done": {"zh": "✨ 初始图谱构建完成！", "en": "✨ Initial graph construction completed!"},
    "toast_start_success": {"zh": "✅ 一键起步成功！", "en": "✅ One-click start successful!"},
    "err_all_validation_fail": {"zh": "❌ 检索到的 {count} 篇候选文献全部未能通过相关性验证。可能是研究方向过于前沿或关键词歧义，请尝试修改研究意图后重试！", "en": "❌ All {count} candidate papers failed relevance validation. Your topic might be too cutting-edge or ambiguous. Please revise and try again!"},

    # 提取模式设置区
    "extract_settings_title": {"zh": "⚙️ 2. 提取模式设置", "en": "⚙️ 2. Extraction Mode Settings"},
    "toggle_summary_only": {"zh": "⚡ 仅提取摘要模式 (速度极快)", "en": "⚡ Summary Only Mode (Extremely Fast)"},
    "toggle_reflection": {"zh": "🔍 启用自我反思纠错", "en": "🔍 Enable Self-Reflection Check"},
    "toggle_append_mode": {"zh": "🔗 追加模式 (叠加到现有图谱)", "en": "🔗 Append Mode (Merge to Existing Graph)"},
    "toggle_relation_grounding": {"zh": "🛡️ 关系节点文献正则补漏 (自动落实漏检实体)", "en": "🛡️ Relation Node Literature Grounding (Recover Missed Entities)"},
    "toggle_relation_grounding_help": {"zh": "切块分析完成后，若发现关系端点未在实体字典中登记，自动使用正则比对原文。若原文确实存在该实体则自动回填落实为真实节点，若完全不存在则坚决剔除虚构关系。", "en": "After chunk analysis, regex-verify unregistered relation endpoints against original paper text. Real entities are recovered; hallucinated edges are strictly dropped."},
    "slider_concurrency": {"zh": "⚡ 并发线程数 (多切块并行解析)", "en": "⚡ Concurrency Workers (Parallel Chunks)"},
    "slider_concurrency_help": {"zh": "多线程并行抽取切块并并发转译机制插图，可成倍压缩文献解析耗时。若遇到频繁 API 限流报错可调小此数值。", "en": "Extract chunks and transcribe pathway figures concurrently to drastically reduce parsing time. Reduce this if you hit API rate limits."},
    "entity_lang_title": {"zh": "#### 🌐 实体命名规范", "en": "#### 🌐 Entity Naming Convention"},
    "entity_lang_label": {"zh": "强制统一图谱主节点（实体）的输出语言：", "en": "Force unify output language of main graph nodes (entities):"},
    "entity_lang_opt_original": {"zh": "关闭 (保持原文语言)", "en": "Off (Keep Original Language)"},
    "entity_lang_opt_zh": {"zh": "中文 (强制翻译为中文)", "en": "Chinese (Force Translate to Chinese)"},
    "entity_lang_opt_en": {"zh": "English (强制翻译为英文)", "en": "English (Force Translate to English)"},

    # 机制图视觉解析策略
    "vision_strategy_title": {"zh": "📷 机制图/通路图解析策略", "en": "📷 Pathway Figure Strategy"},
    "vision_strategy_opt_off": {"zh": "🚫 关闭 (纯文本)", "en": "🚫 Off (Text Only)"},
    "vision_strategy_opt_match": {"zh": "🎯 精准匹配 (输入图号)", "en": "🎯 Keyword Match"},
    "vision_strategy_opt_all": {"zh": "⚡ 选定页全开 (提取全部插图)", "en": "⚡ Extract All in Pages"},
    "vision_keyword_label": {"zh": "🔍 匹配图号/特征词", "en": "🔍 Figure Keyword"},
    "vision_keyword_placeholder": {"zh": "如: Fig 4, pathway, Figure 2", "en": "e.g. Fig 4, pathway, Figure 2"},
    "sidebar_expander_advanced_models": {"zh": "🛠️ 模型微调与视觉高级设置", "en": "🛠️ Model Tuning & Vision Settings"},
    "sidebar_vision_model_select": {"zh": "👁️ 专职视觉模型", "en": "👁️ Dedicated Vision Model"},
    "sidebar_vision_model_suffix": {"zh": "🏷️ 视觉模型后缀 (可选)", "en": "🏷️ Vision Model Suffix (Optional)"},
    "sidebar_vision_model_suffix_placeholder": {"zh": "如 :free 或完整模型名", "en": "e.g. :free or model ID"},
    "vision_model_auto": {"zh": "🔄 自由跟随主模型 (智能自适应)", "en": "🔄 Auto-follow Main Model"},
    "vision_model_gemini_2_5_flash_lite": {"zh": "Gemini 2.5 Flash-Lite (极速推荐)", "en": "Gemini 2.5 Flash-Lite (Recommended)"},
    "vision_model_gemini_3_8_flash": {"zh": "Gemini 3.8 Flash (高智商旗舰)", "en": "Gemini 3.8 Flash (Flagship)"},
    "vision_model_qwen_2_5_vl_72b": {"zh": "Qwen 2.5 VL 72B (千问视觉旗舰)", "en": "Qwen 2.5 VL 72B (Vision Flagship)"},
    "vision_model_custom": {"zh": "自定义视觉模型...", "en": "Custom Vision Model..."},

    # 追加模式提示
    "append_info_ready": {"zh": "📦 记忆库就绪：当前已有 {count} 个节点。新知识将与之融合！", "en": "📦 Memory Bank Ready: Currently {count} nodes. New knowledge will be merged!"},
    "append_warn_off": {"zh": "⚠️ 注意：关闭追加模式，将会【清空】现有图谱，从零开始！历史文献也会清空", "en": "⚠️ Warning: Disabling Append Mode will [CLEAR] the existing graph and start from scratch! History will be cleared too."},

    # 解析范围设置区
    "parse_range_title": {"zh": "#### 🎯 设定解析范围", "en": "#### 🎯 Set Parsing Range"},
    "start_page": {"zh": "起始页码", "en": "Start Page"},
    "end_page": {"zh": "结束页码", "en": "End Page"},
    "warn_summary_range": {"zh": "💡 提示：摘要通常在文献前两页。为了保证极速提取，建议缩小页码范围！", "en": "💡 Tip: Abstracts are usually in the first two pages. To ensure speed, suggest narrowing the page range!"},
    "btn_start_parsing": {"zh": "🚀 开始解析 & 生成图谱", "en": "🚀 Start Parsing & Generate Graph"},

    # 历史文献侧边栏
    "history_sidebar_title": {"zh": "📚 历史分析文献", "en": "📚 Analyzed History Papers"},
    "history_no_docs": {"zh": "暂无已分析的文献", "en": "No analyzed papers yet"},

    # 右侧边界预览区
    "preview_col_title": {"zh": "👀 3. 边界预览", "en": "👀 3. Boundary Preview"},
    "preview_expander": {"zh": "🔍 点击查看选定范围的起止页 (用于确认边界)", "en": "🔍 Click to view start/end pages of selected range (Confirm boundaries)"},
    "preview_start_page": {"zh": "**🟢 起始边界 (第 {page} 页)**", "en": "**🟢 Start Boundary (Page {page})**"},
    "preview_end_page": {"zh": "**🔴 结束边界 (第 {page} 页)**", "en": "**🔴 End Boundary (Page {page})**"},
    "preview_render_fail": {"zh": "渲染失败: {e}", "en": "Render Failed: {e}"},
    "preview_same_page": {"zh": "起止页相同，无需重复展示。", "en": "Start and end pages are the same. No need to show again."},
    "preview_no_pdf": {"zh": "👈 请先在左侧上传 PDF 文献以激活边界预览。", "en": "👈 Please upload a PDF document on the left to activate preview."},

    # 记忆库与工程管理区
    "memory_mgr_title": {"zh": "💾 记忆库与工程管理", "en": "💾 Memory Bank & Project Management"},
    "btn_new_project": {"zh": "✨ 新建空白工程", "en": "✨ New Blank Project"},
    "btn_export_project": {"zh": "📤 导出工程 (.biokg)", "en": "📤 Export Project (.biokg)"},
    "toast_export_empty": {"zh": "⚠️ 当前记忆库为空，请先上传并解析文献！", "en": "⚠️ Memory bank is empty. Please upload and parse documents first!"},
    "info_new_project_confirm": {
        "zh": "💡 **提示**：当前课题图谱已自动归档保存。新建后，您仍可在下方的「本地历史工程库」中随时切回访问当前工程；若您希望完全清除，请在历史工程库管理中操作。\n\n确认新建空白课题工程吗？",
        "en": "💡 **Notice**: The current project graph has been automatically archived. After creating a new project, you can still switch back to it anytime in 'Historical Projects' below. To delete it permanently, please manage it in Historical Projects.\n\nConfirm creating a new blank project?"
    },
    "btn_confirm_new_project": {"zh": "✨ 确认新建", "en": "✨ Confirm Create"},
    "btn_cancel": {"zh": "取消", "en": "Cancel"},
    "doc_file_missing": {"zh": "未在本地文件库中找到源文件", "en": "Source file not found in local vault"},
    "upload_project": {"zh": "📥 载入历史工程文件 (.biokg / .json)", "en": "📥 Load History Project (.biokg / .json)"},
    "btn_confirm_load": {"zh": "🚀 确认载入该工程", "en": "🚀 Confirm Load Project"},
    "help_confirm_load": {"zh": "载入该工程档案；若工程ID不同将作为独立工程载入，绝不影响当前工程", "en": "Load project archive; loads as an independent project if ID differs, safeguarding current project"},
    "toast_load_success": {"zh": "工程【{name}】已成功载入！", "en": "Project [{name}] successfully loaded!"},
    "missing_docs_section_title": {"zh": "🛠️ 缺失文献快捷补全", "en": "🛠️ Missing Literature Replenishment"},
    "err_load_project": {"zh": "解析工程文件失败，请检查文件格式是否正确。报错信息: {e}", "en": "Failed to parse project file. Check format. Error: {e}"},
    "project_name_label": {"zh": "🏷️ 项目课题名称", "en": "🏷️ Project Topic Name"},
    "project_name_help": {"zh": "当前研究课题名称，导出、归档目录与 Central Vault 文献库均将以此锚定", "en": "Current topic name; used for export filename, project folder, and Central Vault literature synchronization."},
    "archive_mgr_title": {"zh": "🗄️ 磁盘空间与工程归档管理", "en": "🗄️ Disk Space & Project Archive Management"},
    "btn_clean_orphans": {"zh": "🧹 清理当前工程孤儿文献", "en": "🧹 Clean Orphaned Literature"},
    "help_clean_orphans": {"zh": "清除当前工程文献库中没有任何实体或关系引用的无用文献（自动识别并保护插图 [Figure: ...] 关联文献）。", "en": "Remove unused papers in current project with no node/edge references (automatically recognizes and protects figure [Figure: ...] citations)."},
    "toast_orphans_cleaned": {"zh": "✅ 已安全清理 {count} 篇孤儿文献！", "en": "✅ Safely cleaned {count} orphaned papers!"},
    "toast_no_orphans": {"zh": "✨ 当前工程文献均有图谱引用，无孤儿文献。", "en": "✨ All papers are referenced by graph nodes, no orphans."},
    "orphan_found_count": {"zh": "发现 {count} 篇无引用文献", "en": "Found {count} unreferenced papers"},
    "orphan_none": {"zh": "无孤儿文献", "en": "No orphans"},
    "manage_projects_expander": {"zh": "🗂️ 本地历史工程库清理", "en": "🗂️ Manage & Clean Historical Projects"},
    "no_saved_projects": {"zh": "暂无已归档的历史工程。", "en": "No archived historical projects."},
    "badge_current": {"zh": "当前活动", "en": "Active"},
    "unit_papers": {"zh": "篇文献", "en": "papers"},
    "btn_del_proj_help": {"zh": "彻底删除历史工程文件夹 {name}，释放磁盘空间", "en": "Delete historical project folder {name} and free disk space"},
    "btn_switch_proj_help": {"zh": "载入并切换到历史工程【{name}】", "en": "Load and switch to project [{name}]"},
    "warn_del_proj_confirm": {"zh": "⚠️ 确认彻底删除历史工程【{name}】及其所有文献吗？此操作无法撤销。", "en": "⚠️ Confirm deleting project [{name}] and all its papers? This cannot be undone."},
    "btn_confirm_del": {"zh": "确认彻底删除", "en": "Confirm Delete"},
    "toast_proj_deleted": {"zh": "工程 {name} 已彻底删除", "en": "Project {name} permanently deleted"},
    "toast_proj_switched": {"zh": "已切换到工程: {name}", "en": "Switched to project: {name}"},
    "badge_missing_docs": {"zh": "缺失 {count} 篇文献", "en": "Missing {count} papers"},
    "missing_docs_alert": {"zh": "⚠️ 当前课题【{name}】检测到缺失 {count} 篇本地文献，可在此便捷补齐：", "en": "⚠️ Project [{name}] is missing {count} local paper(s). You can replenish them here:"},
    "lbl_select_missing_doc": {"zh": "📑 选择需要补全的缺失文献", "en": "📑 Select Missing Paper to Replenish"},
    "lbl_upload_replenish_doc": {"zh": "📤 上传对应文献文件 (PDF)", "en": "📤 Upload Corresponding Paper (PDF)"},
    "toast_replenish_success": {"zh": "✅ 成功补齐文献【{fname}】！文件已归档至当前工程。", "en": "✅ Successfully replenished paper [{fname}]! Archived to project."},
    # 核心引擎触发区 (图谱生成、洗树、融合、渲染)
    "err_missing_api_key": {"zh": "请先在左侧输入 API Key！", "en": "Please enter API Key on the left first!"},
    "msg_parsing_pages": {"zh": "🧠 智能体正在解析第 {start} 到 {end} 页...", "en": "🧠 Agent is parsing pages {start} to {end}..."},
    "msg_init_engine": {"zh": "🚀 正在启动知识图谱自动化构建引擎...", "en": "🚀 Starting Knowledge Graph Automated Build Engine..."},
    "msg_aligning_entities": {"zh": "🔗 正在进行跨文献 AI 语义对齐与图谱融合...", "en": "🔗 Performing cross-document AI semantic alignment and graph fusion..."},
    "msg_found_synonyms": {"zh": "🧬 AI 识别到 {count} 个跨文献同义实体！", "en": "🧬 AI identified {count} cross-document synonymous entities!"},
    "msg_rendering_graph": {"zh": "🎨 正在生成网络拓扑交互图谱...", "en": "🎨 Generating network topology interactive graph..."},
    "msg_gen_success": {"zh": "🎉 图谱生成与数据融合成功！", "en": "🎉 Graph generation and data fusion successful!"},
    "err_gen_fail": {"zh": "❌ 图谱文件未生成。", "en": "❌ Graph file not generated."},
    "err_process_fail": {"zh": "处理过程中发生错误：{e}", "en": "Error occurred during processing: {e}"},
    # 图谱渲染、视觉引擎与导出区
    "export_section_title": {"zh": "### 📥 4. 知识图谱结果导出", "en": "### 📥 4. Export Knowledge Graph Results"},
    "btn_export_ent_json": {"zh": "📄 导出实体 (JSON)", "en": "📄 Export Entities (JSON)"},
    "btn_export_ent_csv": {"zh": "📊 导出实体 (CSV)", "en": "📊 Export Entities (CSV)"},
    "btn_export_rel_csv": {"zh": "🔗 导出关系 (CSV)", "en": "🔗 Export Relations (CSV)"},
    "btn_export_graph_html": {"zh": "🕸️ 导出交互图谱 (HTML)", "en": "🕸️ Export Interactive Graph (HTML)"},

    "toggle_show_shortcuts": {"zh": "👁️ 显示机制捷径边", "en": "👁️ Show Mechanism Shortcuts"},
    "toggle_empower_ontology": {"zh": "🧬 包含关系反哺\n(父节点变大)", "en": "🧬 Ontology Empower\n(Parent Node Enlarge)"},
    "slider_alpha_ontology": {"zh": "反哺转移比例 (α)", "en": "Transfer Ratio (α)"},
    "toggle_empower_node": {"zh": "🌟 节点互相辐射\n(强关联变大)", "en": "🌟 Node Empower\n(Hub Nodes Enlarge)"},
    "slider_beta_node": {"zh": "互相辐射加成率 (β)", "en": "Radiation Bonus Rate (β)"},
    "toggle_empower_edge": {"zh": "🔗 核心主干加粗\n(高频连线变粗)", "en": "🔗 Edge Empower\n(High-freq Edges Thicken)"},
    "slider_gamma_edge": {"zh": "连线增粗系数 (γ)", "en": "Edge Thickening Factor (γ)"},

    "default_biokg_filename": {"zh": "我的图谱工程.biokg", "en": "My_Graph_Project.biokg"},
    # 图谱数据管理中心：全局与导航
    "editor_title": {"zh": "🎛️ 图谱数据管理中心", "en": "🎛️ Graph Data Management Center"},
    "nav_select_module": {"zh": "选择管理模块：", "en": "Select Management Module:"},
    "nav_quick": {"zh": "⚡ 快捷功能", "en": "⚡ Quick Functions"},
    "nav_node": {"zh": "🎯 节点操作", "en": "🎯 Node Operations"},
    "nav_rel": {"zh": "🔗 关系操作", "en": "🔗 Relation Operations"},

    # 快捷功能区
    "quick_clear_title": {"zh": "#### 🧹 一键清理", "en": "#### 🧹 One-Click Cleanup"},
    "quick_clear_desc": {"zh": "清理掉那些既不是起点、也不是终点的“孤立无援”的节点。", "en": "Clean up isolated nodes that are neither source nor target."},
    "btn_clear_isolated": {"zh": "🗑️ 清除所有孤立节点", "en": "🗑️ Clear All Isolated Nodes"},
    "toast_clear_success": {"zh": "✅ 成功清理了 {count} 个孤立节点！", "en": "✅ Successfully cleared {count} isolated nodes!"},
    "toast_no_isolated": {"zh": "当前图谱很健康，没有发现孤立节点。", "en": "The graph is healthy. No isolated nodes found."},
    "quick_rename_title": {"zh": "#### 📝 批量修改文献/来源名", "en": "#### 📝 Batch Rename Source"},
    "lbl_no_source": {"zh": "当前图谱没有来源记录。", "en": "No source records in current graph."},
    "lbl_select_source": {"zh": "选择要修改的文献/来源名", "en": "Select source to rename"},
    "lbl_new_source": {"zh": "输入新的名称 (如：Nature_2023_p53)", "en": "Enter new name (e.g. Nature_2023_p53)"},
    "btn_replace_source": {"zh": "🔄 一键替换全图谱来源", "en": "🔄 Replace Source Globally"},
    "err_empty_name": {"zh": "❌ 新名称不能为空！", "en": "❌ New name cannot be empty!"},
    "toast_rename_success": {"zh": "✅ 成功替换 '{old}'", "en": "✅ Successfully replaced '{old}'"},
    "quick_add_title": {"zh": "#### ➕ 手动添加节点", "en": "#### ➕ Manually Add Node"},
    "lbl_new_node_name": {"zh": "输入新节点标准名 (必填)", "en": "New Node Standard Name (Required)"},
    "lbl_new_node_aliases": {"zh": "输入别名 (选填，逗号分隔)", "en": "Aliases (Optional, comma separated)"},
    "lbl_new_node_source": {"zh": "输入文献来源 (为空记为'手动添加')", "en": "Doc Source (Leave blank for 'Manual Add')"},
    "btn_add_node": {"zh": "💾 保存并添加节点", "en": "💾 Save & Add Node"},
    "err_node_exists": {"zh": "⚠️ 节点已经存在！", "en": "⚠️ Node already exists!"},
    "toast_add_node_success": {"zh": "✅ 节点添加成功！", "en": "✅ Node added successfully!"},
    "default_manual_add": {"zh": "手动添加", "en": "Manual Add"},

    # 节点操作区
    "err_empty_graph": {"zh": "当前图谱为空，无法操作。", "en": "Graph is empty, cannot operate."},
    "lbl_search_node": {"zh": "🔍 搜索并选择要操作的节点", "en": "🔍 Search & select node to operate"},
    "opt_select": {"zh": "-- 请选择 --", "en": "-- Please Select --"},
    "lbl_current_selected": {"zh": "### 当前选中: `{node}`", "en": "### Currently Selected: `{node}`"},
    "lbl_select_action": {"zh": "选择操作：", "en": "Select Action:"},
    "act_edit": {"zh": "✏️ 修改信息", "en": "✏️ Edit Info"},
    "act_merge": {"zh": "🧲 合并到...", "en": "🧲 Merge to..."},
    "act_split": {"zh": "🔀 节点拆分", "en": "🔀 Split Node"},
    "act_delete": {"zh": "🗑️ 危险删除", "en": "🗑️ Danger Delete"},
    "lbl_new_std_name": {"zh": "新标准名", "en": "New Standard Name"},
    "lbl_new_aliases": {"zh": "别名 (逗号分隔)", "en": "Aliases (comma separated)"},
    "btn_save_edit": {"zh": "💾 保存修改", "en": "💾 Save Changes"},
    "toast_edit_success": {"zh": "✅ 节点修改成功！", "en": "✅ Node edited successfully!"},
    "desc_merge": {"zh": "将 `{node}` 的所有关系转移给另一个节点，随后销毁本体。", "en": "Transfer all relations of `{node}` to another node, then destroy the original."},
    "lbl_select_merge_target": {"zh": "选择目标节点", "en": "Select Target Node"},
    "btn_confirm_merge": {"zh": "🧲 确认合并", "en": "🧲 Confirm Merge"},
    "toast_merge_success": {"zh": "✅ 成功合并入 {target}！", "en": "✅ Successfully merged into {target}!"},
    "desc_split": {"zh": "将 `{node}` 拆分为两个独立节点，并重新分配它的关系连线。", "en": "Split `{node}` into two independent nodes and reallocate its relations."},
    "split_step1": {"zh": "##### 1. 定义拆分后的新节点", "en": "##### 1. Define Splitted New Nodes"},
    "lbl_new_node_a": {"zh": "新节点 A (如：mutant p53)", "en": "New Node A (e.g. mutant p53)"},
    "lbl_new_aliases_a": {"zh": "节点 A 别名 (逗号分隔)", "en": "Node A Aliases (comma separated)"},
    "lbl_new_node_b": {"zh": "新节点 B (如：wild-type p53)", "en": "New Node B (e.g. wild-type p53)"},
    "lbl_new_aliases_b": {"zh": "节点 B 别名 (逗号分隔)", "en": "Node B Aliases (comma separated)"},
    "split_step2": {"zh": "##### 2. 分配原有关系连线", "en": "##### 2. Allocate Original Relations"},
    "desc_no_rel_to_split": {"zh": "（该节点目前没有任何连线，无需分配）", "en": "(This node has no relations to allocate)"},
    "split_a": {"zh": "给 A", "en": "To A"},
    "split_b": {"zh": "给 B", "en": "To B"},
    "split_both": {"zh": "A 和 B 都要 (复制)", "en": "To Both (Copy)"},
    "split_discard": {"zh": "丢弃该线", "en": "Discard Edge"},
    "btn_confirm_split": {"zh": "🔀 确认执行拆分", "en": "🔀 Confirm Split"},
    "err_split_names_empty": {"zh": "❌ 两个新节点的名称都不能为空！", "en": "❌ Both new node names cannot be empty!"},
    "err_split_names_same": {"zh": "❌ 新节点 A 和 B 不能同名！", "en": "❌ Node A and B cannot have the same name!"},
    "toast_split_success": {"zh": "✅ 节点已成功拆分为 {a} 和 {b}！", "en": "✅ Node successfully split into {a} and {b}!"},
    "default_manual_split": {"zh": "手动拆分", "en": "Manual Split"},
    "warn_delete_node": {"zh": "⚠️ 警告：删除 `{node}` 将同时拔除所有相关连线！", "en": "⚠️ Warning: Deleting `{node}` will remove all its relations!"},
    "btn_confirm_delete": {"zh": "🚨 确认彻底删除", "en": "🚨 Confirm Delete"},
    "toast_delete_success": {"zh": "✅ 彻底清除！", "en": "✅ Completely Cleared!"},

    # 关系操作区
    "lbl_search_rel_title": {"zh": "### 🔍 搜索两个节点间的关联", "en": "### 🔍 Search Association Between Nodes"},
    "lbl_select_node_a": {"zh": "📍 选择节点 A", "en": "📍 Select Node A"},
    "lbl_select_node_b": {"zh": "🎯 选择节点 B", "en": "🎯 Select Node B"},
    "warn_same_node": {"zh": "⚠️ 节点 A 和节点 B 不能相同，暂不支持操作自环关系。", "en": "⚠️ Node A and B cannot be the same. Self-loop not supported yet."},
    "lbl_rel_op_mode": {"zh": "针对 `{a}` 与 `{b}` 之间的操作：", "en": "Operations between `{a}` and `{b}`:"},
    "rel_mode_edit": {"zh": "✏️ 管理两者间已有关系", "en": "✏️ Manage Existing Relations"},
    "rel_mode_add": {"zh": "➕ 新增一条关系连线", "en": "➕ Add New Relation Edge"},
    "info_no_direct_rel": {"zh": "💡 `{a}` 和 `{b}` 之间目前没有任何直接连线。", "en": "💡 No direct relation between `{a}` and `{b}` currently."},
    "lbl_select_specific_rel": {"zh": "二级菜单：选择要修改的具体连线", "en": "Sub-menu: Select specific edge to edit"},
    "lbl_change_dir": {"zh": "🔄 修改连线方向", "en": "🔄 Change Direction"},
    "dir_x_to_y": {"zh": "从 {x} ─▶ 指向 {y}", "en": "From {x} ─▶ To {y}"},
    "lbl_change_rel_type": {"zh": "🔖 修改关系类型", "en": "🔖 Change Relation Type"},
    "lbl_evidence": {"zh": "原文证据 (Evidence)", "en": "Evidence"},
    "lbl_reason": {"zh": "推理原因 (Reason)", "en": "Reason"},
    "lbl_doc_source": {"zh": "文献来源 (Doc Source)", "en": "Doc Source"},
    "btn_save_rel": {"zh": "💾 保存修改", "en": "💾 Save Changes"},
    "default_manual_edit": {"zh": "手动修改", "en": "Manual Edit"},
    "toast_rel_mod_success": {"zh": "✅ 关系及方向修改成功！", "en": "✅ Relation and direction modified successfully!"},
    "btn_delete_rel": {"zh": "🚨 斩断 / 删除此连线", "en": "🚨 Delete this Edge"},
    "toast_rel_del_success": {"zh": "✅ 连线已彻底删除！", "en": "✅ Edge completely deleted!"},
    "lbl_confirm_dir": {"zh": "确认新连线的指向：", "en": "Confirm New Edge Direction:"},
    "lbl_select_rel_type": {"zh": "🔖 选择关系类型", "en": "🔖 Select Relation Type"},
    "lbl_rel_evidence_opt": {"zh": "原文证据 / 备注 (选填)", "en": "Evidence / Notes (Optional)"},
    "lbl_rel_source_opt": {"zh": "文献来源 (选填，为空则记为'手动添加')", "en": "Doc Source (Leave blank for 'Manual Add')"},
    "btn_add_rel": {"zh": "💾 保存并添加连线", "en": "💾 Save & Add Edge"},
    "warn_rel_exists": {"zh": "⚠️ `{src}` 到 `{tgt}` 已经存在 `{type}` 类型的连线了！", "en": "⚠️ `{type}` relation already exists from `{src}` to `{tgt}`!"},
    "toast_rel_add_success": {"zh": "✅ 新关系建立成功！", "en": "✅ New relation established successfully!"},

    # 三大核心关系词典 (极其重要：用于UI显示，内部依然保持中文硬编码以匹配算法)
    "rel_positive": {"zh": "正作用", "en": "Positive"},
    "rel_negative": {"zh": "负作用", "en": "Negative"},
    "rel_correlation": {"zh": "相关", "en": "Correlation"},
    # 🤖 AI 智能图谱工具区
    "ai_tools_title": {"zh": "🤖 AI 智能图谱工具", "en": "🤖 AI Smart Graph Tools"},
    "ai_nav_select": {"zh": "选择 AI 模块：", "en": "Select AI Module:"},
    "ai_nav_pruning": {"zh": "🧹 智能洗树 (Pruning)", "en": "🧹 Smart Pruning"},
    "ai_nav_expansion": {"zh": "🔍 智能拓展 (Smart Expansion)", "en": "🔍 Smart Expansion"},
    "ai_nav_bridging": {"zh": "🔗 智能桥接 (Bridging)", "en": "🔗 Smart Bridging"},
    "ai_nav_ask": {"zh": "🤖 问 AI", "en": "🤖 Ask AI"},

    # 🧹 智能洗树 (Pruning) 模块
    "pruning_info": {"zh": "💡 拓扑清洗：基于大模型上下文与图论算法，自动处理同义词冗余、层级从属、机制捷径边降级、对称关系双向折叠与循环包含死锁化解。", "en": "💡 Topology Pruning: Leverages LLM reasoning and graph theory to resolve synonym redundancies, hierarchical containment, shortcut downgrading, symmetric relation folding, and mutual containment cycles."},
    "pruning_desc": {"zh": "点击下方按钮，将当前图谱的拓扑结构与文献证据发送给大模型进行诊断。",
                     "en": "Click the button below to send the current graph topology and evidence to LLM for diagnosis."},
    "btn_start_pruning": {"zh": "🚀 开始 AI 智能诊断图谱", "en": "🚀 Start AI Smart Graph Diagnosis"},
    "err_pruning_no_key": {"zh": "❌ 拦截：系统检测到您未填写 API Key！AI 图谱清洗需要调用大模型，请先在侧边栏配置密钥。",
                           "en": "❌ Blocked: No API Key detected! AI Pruning requires LLM. Please configure the key in the sidebar first."},
    "msg_pruning_diagnosing": {"zh": "🧠 大模型正在深度阅读文献证据并梳理图谱拓扑... (请耐心等待)",
                               "en": "🧠 LLM is deeply reading evidence and analyzing topology... (Please wait)"},
    "toast_pruning_done": {"zh": "✅ 大模型诊断完成！", "en": "✅ LLM diagnosis completed!"},
    "pruning_tip_shortcut": {"zh": "💡 诊断完成后，请直接在上方【图谱显示控制面板】灵活开启或关闭捷径边的显示。",
                             "en": "💡 After diagnosis, you can flexibly toggle the display of shortcut edges in the [Display Control Panel] above."},

    # 📋 审查清单
    "review_list_title": {"zh": "### 📋 智能修改审查清单", "en": "### 📋 Smart Modification Review List"},
    "rev_merge": {"zh": "🧲 **合并同义词**: 将 `{removes}` 合并入 `{target}` (原因: {reason})",
                  "en": "🧲 **Merge Synonyms**: Merge `{removes}` into `{target}` (Reason: {reason})"},
    "rev_hierarchy": {"zh": "🌳 **建立层级**: 建立 `{parent}` ─[包含]▶ `{child}` (原因: {reason})",
                      "en": "🌳 **Build Hierarchy**: Build `{parent}` ─[Contain]▶ `{child}` (Reason: {reason})"},
    "rev_resolve_cycle": {"zh": "🔄 **破除循环包含**: 确立 `{parent}` ─[包含]▶ `{child}`（无损合并反向证据并消除逆向包含）(原因: {reason})",
                          "en": "🔄 **Resolve Containment Cycle**: Establish `{parent}` ─[Contain]▶ `{child}` (merge reverse evidence and remove reverse edge) (Reason: {reason})"},
    "rev_fold_symmetric": {"zh": "🔀 **折叠双向相关**: 将 `{node_a}` 与 `{node_b}` 之间的双向“相关”连线折叠为单向规范线（无损合并双方证据与权重）(原因: {reason})",
                           "en": "🔀 **Fold Symmetric Relation**: Fold bidirectional 'related' edges between `{node_a}` and `{node_b}` into a canonical single edge (merge evidence & heat) (Reason: {reason})"},
    "rev_downgrade": {"zh": "📉 **降级捷径边**: 将连线 `{src} ─[{rel}]▶ {tgt}` 降级为推导虚线 (原因: {reason})",
                      "en": "📉 **Downgrade Shortcut**: Downgrade `{src} ─[{rel}]▶ {tgt}` to inferred dashed line (Reason: {reason})"},
    "rev_remove": {"zh": "✂️ **删除越级连线**: 彻底移除 `{src} ─[{rel}]▶ {tgt}` (原因: {reason})",
                   "en": "✂️ **Remove Edge**: Completely remove `{src} ─[{rel}]▶ {tgt}` (Reason: {reason})"},
    "rev_purify": {"zh": "🧪 **机制提纯合并**: 将粗糙连线 `{src} ─[{rel}]▶ {tgt}` 提纯合并入明确机制（无损转移证据并累加热度）(原因: {reason})",
                   "en": "🧪 **Purify & Merge**: Purify and merge coarse edge `{src} ─[{rel}]▶ {tgt}` into explicit mechanism (transfer evidence & heat) (Reason: {reason})"},
    "rev_purify_with_target": {"zh": "🧪 **机制提纯合并**: 将 `{src} ─[{rel}]▶ {tgt}` 提纯并入 `[{target_rel}]`（无损转移证据并累加热度）(原因: {reason})",
                               "en": "🧪 **Purify & Merge**: Purify `{src} ─[{rel}]▶ {tgt}` into `[{target_rel}]` (transfer evidence & heat) (Reason: {reason})"},
    "no_reason": {"zh": "未提供原因", "en": "No reason provided"},
    "btn_confirm_pruning": {"zh": "💾 确认执行选中的优化", "en": "💾 Confirm Executing Selected Optimizations"},
    "toast_pruning_exec_done": {"zh": "✅ 优化已完美执行！", "en": "✅ Optimizations executed perfectly!"},

    # ✂️ 智能节点清洗 (Intent-Based Pruning)
    "intent_pruning_title": {"zh": "✂️ 智能节点清洗 (Intent-Based Pruning)", "en": "✂️ Intent-Based Pruning"},
    "intent_pruning_info": {"zh": "💡 意图剪枝：大模型将根据你输入的研究兴趣方向，揪出图谱中偏题或无关的对象（如误抓的其他器官/疾病），一键抹去它们及其连线。",
                            "en": "💡 Intent Pruning: LLM will identify off-topic or irrelevant nodes (e.g., wrong organs/diseases) based on your research interest, erasing them and their edges with one click."},
    "lbl_user_interest": {"zh": "请描述你当前聚焦的核心研究方向（例如：我只关心肠道上皮细胞的衰老机制，不关心其他器官）",
                          "en": "Describe your core research focus (e.g., I only care about intestinal epithelial cell aging, ignore other organs)"},
    "btn_find_irrelevant": {"zh": "🔍 根据我的方向，找出偏题节点", "en": "🔍 Find Off-topic Nodes Based on My Focus"},
    "warn_empty_interest": {"zh": "⚠️ 请输入您的研究兴趣方向！", "en": "⚠️ Please enter your research interest focus!"},
    "msg_finding_irrelevant": {"zh": "🧠 大模型正在逐一审视节点及邻居，挖掘偏题内容...",
                               "en": "🧠 LLM is reviewing nodes and neighbors to find off-topic content..."},
    "toast_find_irrelevant_done": {"zh": "✅ 诊断完成，请在下方确认清洗名单！",
                                   "en": "✅ Diagnosis done, please confirm the purge list below!"},

    # 🧹 偏题节点处决名单
    "purge_list_title": {"zh": "### 🧹 偏题节点处决名单", "en": "### 🧹 Off-topic Nodes Purge List"},
    "purge_list_caption": {"zh": "请确认以下清洗建议。勾选【删除】的节点及其**所有相关连线**将被彻底从图谱中抹去。",
                           "en": "Please confirm the purge suggestions. Checked nodes and **ALL their relations** will be completely erased from the graph."},
    "col_delete": {"zh": "🗑️ 删除", "en": "🗑️ Delete"},
    "col_node_name": {"zh": "节点名称", "en": "Node Name"},
    "col_neighbors": {"zh": "其连接的邻居", "en": "Connected Neighbors"},
    "col_risk_level": {"zh": "风险等级", "en": "Risk Level"},
    "col_ai_reason": {"zh": "AI 判定依据", "en": "AI Judgement Basis"},
    "btn_execute_purge": {"zh": "🚨 确认处决勾选节点", "en": "🚨 Execute Purge on Checked Nodes"},
    "toast_purge_success": {"zh": "✅ 成功清理了 {count} 个偏题节点及其连线！图谱瞬间纯净！",
                            "en": "✅ Successfully purged {count} off-topic nodes & edges! Graph is now pure!"},
    "btn_cancel_purge": {"zh": "放弃本次剪枝", "en": "Cancel This Pruning"},

    # 🏷️ 智能文献名称识别与规范化 (Smart Literature Renaming) 模块
    "paper_rename_title": {"zh": "🏷️ 智能文献名称识别与规范化", "en": "🏷️ Smart Paper Title Identification & Renaming"},
    "paper_rename_info": {
        "zh": "💡 全盘文献与出处规范化：同时支持本地物理文献（.pdf）与自动拓展/探索获得的线上文献（如 PubMed:xxxx），统一检索官方学术大标题并经轻量大模型清洗版权杂质，无感同步图谱与记忆库。",
        "en": "💡 Global literature & source standardization: supports both local PDF papers and online sources (e.g. PubMed:xxxx) from auto-expansion, retrieving official academic titles and sanitizing copyright with LLM."
    },
    "no_papers_to_rename": {"zh": "当前工程暂无可供规范化的文献或出处来源。", "en": "No papers or sources available to standardize in current project."},
    "rename_opt_all": {"zh": "全部文献与来源", "en": "All Papers & Sources in Current Project"},
    "lbl_select_rename_target": {"zh": "选择重命名目标：", "en": "Select Renaming Target:"},
    "btn_start_rename": {"zh": "🏷️ 开始识别并重命名", "en": "🏷️ Start Identification & Renaming"},
    "msg_renaming_papers": {"zh": "🧠 正在检索官方元数据并进行大模型统一清洗审校...", "en": "🧠 Retrieving metadata and performing unified LLM sanitization..."},
    "toast_rename_success": {"zh": "✅ 文献【{old}】已规范重命名为【{new}】", "en": "✅ Paper [{old}] renamed to [{new}]"},
    "info_rename_skipped": {"zh": "ℹ️ 共 {count} 篇文献名称已符合学术规范，跳过更新。", "en": "ℹ️ {count} paper(s) already standardized, skipped."},
    "warn_rename_failed": {"zh": "⚠️ 文献【{fname}】未能成功识别出标题（{reason}），已保留原文件名。", "en": "⚠️ Failed to identify title for [{fname}] ({reason}), kept original name."},
    "err_file_not_found": {"zh": "未在本地文件库中找到物理文件", "en": "Physical file not found in local vault"},
    "err_title_unrecognized": {"zh": "所有识别策略（PubMed检索、大模型快提与审校）均未成功识别出真实学术标题", "en": "All strategies (PubMed search, fast extraction, LLM review) failed to identify academic title"},

    # 🔍 智能拓展 (Smart Expansion) 模块
    "expansion_info": {"zh": "💡 选中图谱中的关键节点，系统将自动在 {db} 中检索前沿文献，并由 AI 提取摘要知识，自动将其延伸至当前图谱。",
                       "en": "💡 Select key nodes, the system will automatically search {db} for cutting-edge papers. AI will extract abstract knowledge and extend the current graph."},
    "lbl_select_nodes_exp": {"zh": "🎯 请选择要拓展的核心节点 (建议 1-3 个):",
                             "en": "🎯 Select core nodes to expand (Recommend 1-3):"},
    "lbl_search_strategy": {"zh": "🧠 选择检索与拓展策略：", "en": "🧠 Select Search & Expand Strategy:"},
    "strategy_direct": {"zh": "⚡ 直接搜索 (速度极快，精准字符匹配)", "en": "⚡ Direct Search (Extremely fast, exact match)"},
    "strategy_smart": {"zh": "🧠 智能搜索 (AI 扩充同义词与 Mesh 主题词)",
                       "en": "🧠 Smart Search (AI expands synonyms & Mesh terms)"},
    "strategy_bg": {"zh": "🌐 背景搜索 (最强：AI 阅读图谱已知连线，智能发散机制)",
                    "en": "🌐 Background Search (Strongest: AI reads known edges, smart divergence)"},
    "lbl_max_results": {"zh": "📄 检索返回文献数量上限", "en": "📄 Max Papers to Fetch"},
    "btn_web_search": {"zh": "🔍 联网搜索相关文献", "en": "🔍 Web Search Related Papers"},
    "warn_select_nodes_first": {"zh": "⚠️ 请先在上方选择至少一个节点！", "en": "⚠️ Please select at least one node above!"},
    "msg_fetching_api": {"zh": "🌐 正在生成策略并呼叫 {db} API 获取最新文献...",
                         "en": "🌐 Generating strategy & calling {db} API for latest papers..."},
    "err_need_key_for_ai_search": {"zh": "❌ 拦截：请先在侧边栏配置 API Key 以启用 AI 搜索功能。",
                                   "en": "❌ Blocked: Please configure API Key in sidebar to enable AI search."},
    "toast_search_success": {"zh": "✅ 成功检索到 {count} 篇相关文献！", "en": "✅ Successfully fetched {count} related papers!"},

    # 检索结果表格
    "search_results_title": {"zh": "### 📑 检索结果 (请勾选需要深度分析的文献)",
                             "en": "### 📑 Search Results (Check papers for deep analysis)"},
    "col_include_graph": {"zh": "✅ 引入图谱", "en": "✅ Include to Graph"},
    "col_paper_title": {"zh": "文献标题", "en": "Paper Title"},
    "col_year": {"zh": "年份", "en": "Year"},
    "btn_add_to_queue": {"zh": "📥 将勾选文献加入解析队列", "en": "📥 Add Checked Papers to Queue"},
    "warn_check_one_paper": {"zh": "⚠️ 请至少在表格中勾选一篇文献！", "en": "⚠️ Please check at least one paper in the table!"},
    "toast_added_to_queue": {"zh": "✅ 已加入队列！请在下方控制台进行单步处理。",
                             "en": "✅ Added to queue! Please process step-by-step in the console below."},

    # 深度融合任务队列 (状态机面板)
    "queue_title": {"zh": "### ⚙️ 深度融合任务队列 (进度: {idx}/{total})",
                    "en": "### ⚙️ Deep Fusion Task Queue (Progress: {idx}/{total})"},
    "queue_current_doc": {"zh": "👉 **当前准备解析:**\n\n📄 {title}\n*(PMID: {pmid})*",
                          "en": "👉 **Currently Parsing:**\n\n📄 {title}\n*(PMID: {pmid})*"},
    "toggle_auto_run": {"zh": "🔁 开启连续自动解析 (无人值守模式)", "en": "🔁 Auto-Run Mode (Unattended Execution)"},
    "warn_auto_running": {"zh": "⚠️ 自动巡航模式运行中... (若需中途强行中止，请点击页面右上角自带的 'Stop' 按钮)",
                          "en": "⚠️ Auto-cruise running... (To stop forcefully, click the 'Stop' button at the top right of the page)"},
    "btn_run_this": {"zh": "▶️ 解析并融合本篇", "en": "▶️ Parse & Fuse This Paper"},
    "btn_skip_this": {"zh": "⏭️ 跳过这篇", "en": "⏭️ Skip This Paper"},
    "btn_stop_queue": {"zh": "🛑 终止任务并清空", "en": "🛑 Abort Task & Clear Queue"},
    "msg_deep_parsing": {"zh": "⏳ 正在深度解析并提取 {pmid} 的知识...",
                         "en": "⏳ Deep parsing and extracting knowledge from {pmid}..."},
    "toast_skip_short_abstract": {"zh": "文献 {pmid} 摘要为空或过短，已自动跳过。",
                                  "en": "Paper {pmid} abstract is empty or too short, auto-skipped."},
    "toast_fused_success": {"zh": "✅ 文献 {pmid} 融合成功！", "en": "✅ Paper {pmid} fused successfully!"},
    "msg_queue_all_done": {"zh": "✅ 队列中的 {total} 篇文献已全部处理完毕！图谱已是最新形态。", "en": "✅ All {total} papers in queue processed! Graph is up to date."},
    "btn_clear_queue_and_results": {"zh": "🧹 清理队列与搜索结果", "en": "🧹 Clear Queue & Search Results"},
# 🔗 智能桥接 (Intelligent Bridging)
    "bridge_info": {"zh": "💡 选中两个节点，AI 将在图谱中寻找它们之间的所有多步通路，并推导深层分子机制。如果是孤岛，可一键呼叫 AI 联网挖掘！", "en": "💡 Select two nodes. AI will find all multi-step paths between them and deduce deep mechanisms. If they are isolated, one-click to call AI for web mining!"},
    "lbl_bridge_node_a": {"zh": "🎯 起点实体 (Node A):", "en": "🎯 Source Entity (Node A):"},
    "lbl_bridge_node_b": {"zh": "🎯 终点实体 (Node B):", "en": "🎯 Target Entity (Node B):"},
    "lbl_max_depth": {"zh": "🛤️ 最大探索深度 (Max Hops限制，防噪音)", "en": "🛤️ Max Exploration Depth (Max Hops limit to prevent noise)"},
    "btn_start_bridge": {"zh": "🚀 启动跨文献机制桥接与推导", "en": "🚀 Start Cross-Literature Mechanism Bridging & Deduction"},
    "warn_same_nodes_bridge": {"zh": "⚠️ 起点和终点不能是同一个节点！", "en": "⚠️ Source and target cannot be the same node!"},
    "warn_select_nodes_bridge": {"zh": "⚠️ 请先在上方选择节点！", "en": "⚠️ Please select nodes above first!"},
    "msg_scanning_paths": {"zh": "🔍 正在执行图论算法，扫描全图拓扑路径...", "en": "🔍 Executing graph algorithm to scan full topology paths..."},
    "msg_island": {"zh": "🕳️ **图谱孤岛**：在当前最高 {depth} 步的限制下，【{a}】和【{b}】之间毫无关联。", "en": "🕳️ **Graph Island**: Under the limit of {depth} hops, there is no connection between [{a}] and [{b}]."},
    "prompt_island_search": {"zh": "呼叫 AI 联网挖掘【{a}】与【{b}】的潜在机制", "en": "Call AI to web mine potential mechanism between [{a}] and [{b}]"},
    "msg_direct_only": {"zh": "⚡ **直接相关**：当前图谱仅发现【{a}】和【{b}】的表面直接联系。似乎缺乏中间介导物（如通路蛋白）的解释。", "en": "⚡ **Direct Only**: Only surface direct links found between [{a}] and [{b}]. Seems to lack explanations of intermediate mediators (e.g. pathway proteins)."},
    "prompt_direct_search": {"zh": "呼叫 AI 联网挖掘【{a}】与【{b}】的深层中间通路", "en": "Call AI to web mine deep intermediate pathways between [{a}] and [{b}]"},
    "msg_complex_paths": {"zh": "🧬 发现 {count} 条跨越节点的复杂机制路径！", "en": "🧬 Found {count} complex mechanism paths across nodes!"},
    "err_need_key_for_summary": {"zh": "❌ 请先在侧边栏配置 API Key 以启用 AI 总结。", "en": "❌ Please configure API Key in the sidebar to enable AI summary."},
    "msg_writing_report": {"zh": "🧠 资深 AI 学者正在阅读证据碎片，撰写跨文献机制报告...", "en": "🧠 Senior AI scholar is reading evidence fragments to write cross-literature mechanism report..."},
    "report_title": {"zh": "### 📑 AI 跨文献机制推导报告", "en": "### 📑 AI Cross-Literature Mechanism Deduction Report"},
    "report_subtitle": {"zh": "> *起点: {a} | 终点: {b} | 基于本地图谱 {count} 条拓扑路径*", "en": "> *Source: {a} | Target: {b} | Based on {count} topology paths in local graph*"},
    "msg_evaluating_strategy": {"zh": "🧠 AI 正在评估并生成专属检索策略...", "en": "🧠 AI is evaluating and generating exclusive search strategy..."},
    "err_ai_rejected": {"zh": "🛑 **AI 驳回请求**：根据现有生命科学常识，【{a}】和【{b}】之间没有合理的生物学机制交集。", "en": "🛑 **AI Rejected**: Based on common life science knowledge, there is no reasonable biological mechanism intersection between [{a}] and [{b}]."},
    "msg_search_strategy": {"zh": "💡 **检索策略**: `{query}`", "en": "💡 **Search Strategy**: `{query}`"},
    "warn_no_mechanism_papers": {"zh": "⚠️ PubMed 中未检索到相关机制文献。", "en": "⚠️ No relevant mechanism papers found in PubMed."},
    "search_res_title_bridge": {"zh": "### 🔎 联网检索结果：关于【{a}】与【{b}】的关系文献", "en": "### 🔎 Web Search Results: Papers on the relation between [{a}] and [{b}]"},
    "search_res_caption_bridge": {"zh": "💡 请勾选你认为真正相关的文献，AI 将只下载并精读这些文献的摘要来拼接机制。", "en": "💡 Please check papers you think are truly relevant. AI will only download and deep-read their abstracts to splice the mechanism."},
    "col_deep_read": {"zh": "精读", "en": "Deep Read"},
    "col_pub_date": {"zh": "发表日期", "en": "Pub Date"},
    "btn_start_sniper_extract": {"zh": "🧬 对已勾选文献启动【狙击式机制提取】", "en": "🧬 Start [Sniper Mechanism Extraction] on checked papers"},
    "warn_check_one_paper_bridge": {"zh": "⚠️ 请至少勾选一篇文献供 AI 阅读！", "en": "⚠️ Please check at least one paper for AI to read!"},
    "msg_sniper_extracting": {"zh": "📚 正在后台秒传摘要并进行跨文献机制缝合...", "en": "📚 Sending abstracts in background for cross-literature mechanism suturing..."},
    "fmt_paper_title": {"zh": "--- 文献 PMID: {pmid} (Title: {title}) ---", "en": "--- Paper PMID: {pmid} (Title: {title}) ---"},
    "ai_bridge_report_title": {"zh": "### 💡 AI 联网桥接机制报告 (等待审批)", "en": "### 💡 AI Web Bridging Mechanism Report (Pending Approval)"},
    "no_text_explanation": {"zh": "无文本解释", "en": "No text explanation"},
    "warn_no_clear_chain": {"zh": "🕵️‍♂️ 虽然阅读了你指定的文献，但 AI 未能提取到明确连接这俩靶点的可靠微观机制链。", "en": "🕵️‍♂️ Although read the specified papers, AI failed to extract a reliable micro-mechanism chain clearly connecting these two targets."},
    "btn_clear_report": {"zh": "清空报告", "en": "Clear Report"},
    "msg_extracted_links": {"zh": "🧩 成功提取到 {count} 条专属机制链路。是否将其并入主图谱？", "en": "🧩 Successfully extracted {count} exclusive mechanism links. Merge them into the main graph?"},
    "btn_accept_merge": {"zh": "✅ 认可并完美融入主图谱", "en": "✅ Accept & Perfectly Merge into Main Graph"},
    "msg_bulletproof_merging": {"zh": "正在执行防弹级融合...", "en": "Executing bulletproof fusion..."},
    "fmt_downgrade_reason": {"zh": "[智能桥接自动降级：已挖掘出深层通路] {old}", "en": "[Smart Bridging Auto-downgrade: Deep pathways mined] {old}"},
    "toast_bridge_merged": {"zh": "✅ 桥接机制已成功融合！图谱已生长！", "en": "✅ Bridge mechanism successfully merged! Graph has grown!"},
    "btn_reject_clear": {"zh": "❌ 感觉不对，放弃并清空", "en": "❌ Feels wrong, reject and clear"},

    # 🤖 问 AI (Ask AI)
    "ask_ai_info": {"zh": "💡 选中图谱中的任何节点或连线，AI 将结合它的【文献来源依据】和【网络拓扑】，为您提供独家解读。", "en": "💡 Select any node or edge in the graph. AI will provide an exclusive interpretation combining its [Literature Evidence] and [Network Topology]."},
    "lbl_what_to_explain": {"zh": "你想让 AI 解读什么？", "en": "What do you want AI to explain?"},
    "explain_node": {"zh": "解读节点 (Node)", "en": "Explain Node"},
    "explain_edge": {"zh": "解读关系连线 (Edge)", "en": "Explain Relation (Edge)"},
    "warn_no_nodes_in_graph": {"zh": "图谱中还没有节点。", "en": "No nodes in the graph yet."},
    "lbl_select_node_to_explain": {"zh": "请选择要解读的节点：", "en": "Please select the node to explain:"},
    "btn_explain_node": {"zh": "✨ 让 AI 深度解读该节点", "en": "✨ Let AI deeply explain this node"},
    "msg_extracting_context": {"zh": "🧠 正在提取 {node} 的微上下文并生成解读...", "en": "🧠 Extracting micro-context of {node} and generating explanation..."},
    "msg_no_connections": {"zh": "该节点目前没有与其他节点相连。", "en": "This node currently has no connections to other nodes."},
    "report_node_title": {"zh": "### 📝 AI 解读报告", "en": "### 📝 AI Explanation Report"},
    "lbl_select_two_nodes": {"zh": "### 🔍 选择需要解读关系的两个节点", "en": "### 🔍 Select two nodes to explain their relation"},
    "warn_same_node_ask": {"zh": "⚠️ 节点 A 和节点 B 不能相同，暂不支持解读自环关系。", "en": "⚠️ Node A and B cannot be the same. Self-loop not supported yet."},
    "info_no_rel_between": {"zh": "ℹ️ 暂无 `{a}` 与 `{b}` 之间的关系连线。", "en": "ℹ️ No relation edge between `{a}` and `{b}` yet."},
    "lbl_select_rel_to_explain": {"zh": "🔗 发现可用连线，请选择具体要解读的那一条：", "en": "🔗 Found available edges, please select the specific one to explain:"},
    "btn_explain_edge": {"zh": "✨ 让 AI 深度解读该连线", "en": "✨ Let AI deeply explain this edge"},
    "msg_analyzing_edge": {"zh": "🧠 正在分析连线机制并生成解读...", "en": "🧠 Analyzing edge mechanism and generating explanation..."},
    "report_edge_title": {"zh": "### 📝 AI 连线机制解读报告", "en": "### 📝 AI Edge Mechanism Explanation Report"},
    # 📊 图谱分析工作台 (Workbench i18n)
    "wb_nav_bottom_btn": {"zh": "📊 前往图谱分析工作台 (实体与关系机制深度解析) ➔", "en": "📊 Open Graph Analysis Workbench (Entities & Mechanisms) ➔"},
    "wb_btn_top_return": {"zh": "⬅️ 返回主图谱工作区", "en": "⬅️ Return to Main Workspace"},
    "wb_title": {"zh": "📊 图谱分析工作台 (Graph Analysis Workbench)", "en": "📊 Graph Analysis Workbench"},
    "wb_caption": {"zh": "🔬 支持基于权威生物医学本体体系的层级递进研判（命名规范识别 ➔ 拓扑上下文推理 ➔ 原文精准溯源 ➔ 领域先验研判）、实体机制分布看板与专家协同校准。", "en": "🔬 Multi-stage biological entity and mechanism curation (Nomenclature Rules ➔ Topological Context ➔ Literature Grounding ➔ Domain Prior Knowledge) with biomedical ontology presets and expert review."},
    "wb_project_info": {"zh": "🗂️ 当前工程: **{name}** (`{id}`) | 实体总数: **{ents}** | 关系总数: **{rels}**", "en": "🗂️ Current Project: **{name}** (`{id}`) | Total Entities: **{ents}** | Total Relations: **{rels}**"},
    "wb_warn_no_data": {"zh": "⚠️ 当前工程暂无图谱数据。请先返回主工作区上传解析文献或导入已有工程。", "en": "⚠️ No graph data in current project. Please return to the main workspace to upload papers or import a project first."},
    "wb_base_graph_preview": {"zh": "🗺️ 查看基础图谱拓扑全景 (Base Graph Preview)", "en": "🗺️ Base Knowledge Graph Topology Preview"},
    "wb_no_base_graph": {"zh": "暂无基础图谱拓扑数据", "en": "No base graph topology data available yet"},
    "wb_config_title": {"zh": "⚙️ 分析预设与流水线控制", "en": "⚙️ Ontology Presets & Pipeline Control"},
    "wb_preset_select": {"zh": "📚 选择分析本体预设体系", "en": "📚 Select Ontology Preset"},
    "wb_custom_preset_title": {"zh": "自定义规则管理", "en": "Custom Schema Management"},
    "wb_btn_open_presets_dir": {"zh": "📁 打开规则文件夹", "en": "📁 Open Rules Folder"},
    "wb_open_presets_dir_desc": {"zh": "将符合三要素规范的 .json 规则文件直接放入该目录即可自动加载；文件夹内附详细中英指南。", "en": "Place 3-element .json schema files directly into this folder to load them automatically. Detailed guides are included inside."},
    "wb_btn_refresh_presets": {"zh": "🔄 刷新预设列表", "en": "🔄 Refresh Presets"},
    "wb_toast_folder_opened": {"zh": "已尝试在文件资源管理器中打开规则目录: {path}", "en": "Attempted to open rules folder in file manager: {path}"},
    "wb_chk_entities": {"zh": "🧬 开启实体分级分类分析", "en": "🧬 Enable Entity Categorization"},
    "wb_chk_relations": {"zh": "🔗 开启关系机制动词归纳", "en": "🔗 Enable Relation Mechanism Distillation"},
    "wb_chk_force": {"zh": "🔄 忽略已有缓存强制全量重算", "en": "🔄 Force Full Re-analysis (Bypass Cache)"},
    "wb_esc_title": {"zh": "⚙️ 层级分析策略高级配置 (Multi-Stage Analysis Settings)", "en": "⚙️ Multi-Stage Analysis Configuration"},
    "wb_esc_r0": {"zh": "⚡ 阶段一: 权威生物命名规范与词缀识别 (基于命名规则高速解析)", "en": "⚡ Stage 1: Nomenclature & Affix Recognition (Rule-based fast parsing)"},
    "wb_esc_r1": {"zh": "🧠 阶段二: 局部网络拓扑推理 (结合相邻三元组与证据)", "en": "🧠 Stage 2: Local Network Topology Reasoning (Neighboring triples & context)"},
    "wb_esc_r2": {"zh": "📖 阶段三: 文献全文精准溯源 (提取文献原句深度研判)", "en": "📖 Stage 3: Full-Text Literature Grounding (Source passage retrieval & analysis)"},
    "wb_esc_r3": {"zh": "🌐 阶段四: 生物医学全局先验知识 (领域专家背景知识综合研判)", "en": "🌐 Stage 4: Biomedical Domain Prior Knowledge (Global biomedical knowledge reasoning)"},
    "wb_btn_start_run": {"zh": "🚀 启动/更新图谱分析工作台", "en": "🚀 Run / Update Graph Analysis Workbench"},
    "wb_btn_rerender_canvas": {"zh": "🎨 仅重新渲染工作台画布", "en": "🎨 Re-render Workbench Canvas Only"},
    "wb_metrics_title": {"zh": "### 📈 工作台科学指标大盘", "en": "### 📈 Workbench Scientific Metrics Dashboard"},
    "wb_metric_total": {"zh": "📊 实体总数", "en": "📊 Total Entities"},
    "wb_metric_resolved": {"zh": "✅ 已解析实体", "en": "✅ Resolved Entities"},
    "wb_metric_uncertain": {"zh": "❓ 待定/未分类", "en": "❓ Uncertain / Pending"},
    "wb_metric_relations": {"zh": "🔗 机制关系数", "en": "🔗 Mechanism Relations"},
    "wb_sec_ent_dist": {"zh": "#### 🧬 实体分类分布", "en": "#### 🧬 Entity Category Distribution"},
    "wb_sec_rel_dist": {"zh": "#### 🔗 关系机制分布", "en": "#### 🔗 Relation Mechanism Distribution"},
    "wb_canvas_title": {"zh": "### 🌐 工作台科学拓扑图 (科学层级多形态机制图谱)", "en": "### 🌐 Workbench Scientific Multi-Shape Topology Graph"},
    "wb_not_analyzed_hint": {"zh": "💡 尚未执行工作台分级深度解析。请在上方选择本体预设并点击【🚀 启动/更新图谱分析工作台】开始解析。", "en": "💡 Deep workbench analysis has not been executed yet. Select an ontology preset above and click [🚀 Run / Update Graph Analysis Workbench] to begin."},
    "wb_override_title": {"zh": "### 🛠️ 专家人工分类与机制校准工作台", "en": "### 🛠️ Expert Manual Override & Curation Console"},
    "wb_override_caption": {"zh": "支持学者对自动推理的实体分类与关系机制动词进行精细修正，修改后即时更新缓存与拓扑图。", "en": "Allows researchers to curate and override entity classifications and relation mechanisms with immediate cache & canvas updates."},
    "wb_tab_ent_override": {"zh": "🧬 实体分类专家校准", "en": "🧬 Entity Classification Curation"},
    "wb_tab_rel_override": {"zh": "🔗 关系机制专家校准", "en": "🔗 Relation Mechanism Curation"},
    "wb_sel_ent_lbl": {"zh": "选择待调整的实体", "en": "Select Entity to Adjust"},
    "wb_target_cat_lbl": {"zh": "调整目标类别", "en": "Target Category"},
    "wb_sub_type_lbl": {"zh": "细分子类 / 具体功能 (可选)", "en": "Sub-type / Function (Optional)"},
    "wb_reason_lbl": {"zh": "专家校准说明", "en": "Curation Note / Reason"},
    "wb_btn_save_ent": {"zh": "💾 保存实体分类修正并即时更新", "en": "💾 Save & Reclassify Entity"},
    "wb_sel_rel_lbl": {"zh": "选择待调整的关系连线", "en": "Select Relation Edge to Adjust"},
    "wb_target_relcat_lbl": {"zh": "调整机制类别", "en": "Target Mechanism Category"},
    "wb_verb_lbl": {"zh": "提炼机制谓词动词 (如: 催化磷酸化, 竞争性抑制)", "en": "Mechanism Verb (e.g. Phosphorylates, Inhibits)"},
    "wb_btn_save_rel": {"zh": "💾 保存关系机制修正并即时更新", "en": "💾 Save & Override Mechanism"},
    "wb_btn_bottom_return": {"zh": "⬅️ 返回主图谱构建工作区", "en": "⬅️ Return to Main Graph Workspace"},
    "wb_log_expander_title": {"zh": "📝 实时执行分析日志 (Real-time Analysis Logs)", "en": "📝 Real-time Analysis Logs"},
    "wb_toast_analysis_done": {"zh": "🎉 工作台全量分析完成并生成拓扑图谱！", "en": "🎉 Workbench analysis completed and topology graph generated!"},
    "wb_toast_rerender_done": {"zh": "工作台画布重新渲染完成！", "en": "Workbench canvas re-rendered successfully!"},
    "wb_toast_override_done": {"zh": "实体分类修正已保存！", "en": "Entity classification updated successfully!"},
    "wb_toast_rel_override_done": {"zh": "关系机制修正已保存！", "en": "Relation mechanism updated successfully!"},
    "wb_no_rel_edges": {"zh": "当前图谱尚无关系连线。", "en": "No relation edges in current graph."},
    "wb_cur_cat_lbl": {"zh": "当前类别", "en": "Category"},
    "wb_round_lbl": {"zh": "研判来源", "en": "Resolution Source"},
    "wb_conf_lbl": {"zh": "置信度", "en": "Confidence"},
    "wb_reason_trace_lbl": {"zh": "推断依据", "en": "Reasoning"},
    "wb_cur_mech_lbl": {"zh": "机制类别", "en": "Mechanism"},
    "wb_cur_verb_lbl": {"zh": "机制动词", "en": "Verb"},
    "wb_orig_rel_prefix": {"zh": "原", "en": "orig"},
    "wb_unit_entities": {"zh": "个实体", "en": "entities"},
    "wb_unit_relations": {"zh": "条关系", "en": "relations"},

    # 🧬 工作台第二阶段：信号/关系通路拆解与建模 (Stage 2: Pathway Decomposition)
    "wb_sec2_title": {"zh": "### 🧬 第二阶段：核心信号通路拆解与自调节回路建模", "en": "### 🧬 Stage 2: Signaling Pathway Decomposition & Autoregulation Modeling"},
    "wb_sec2_caption": {"zh": "融合本体分类、拓扑特征与文献溯源，自由组合挖掘核心因果传导轴、自促进/自抑制闭环反馈回路及跨文献全新假说。", "en": "Mine core causal axes, self-promotion/inhibition feedback loops, and cross-literature hypotheses leveraging ontology, topology, and citations."},
    "wb_pw_input_label": {"zh": "🎯 通路探索焦点（输入关键科学问题/疾病/表型/药物；留空则触发全局自主发掘）：", "en": "🎯 Pathway Exploration Focus (Enter biological question/disease/phenotype/drug; leave blank for autonomous discovery):"},
    "wb_pw_input_placeholder": {"zh": "例如：探讨索拉非尼耐药机制与铁死亡调控轴；或留空", "en": "e.g. Explore sorafenib resistance mechanism and ferroptosis regulatory axis; or leave blank"},
    "wb_pw_chk_diversity": {"zh": "✨ 通路探索广度强化 (MMR 累积惩罚，向外发散发掘多条独立主线)", "en": "✨ Exploration Breadth (MMR Visited Penalty for multi-axis diversity)"},
    "wb_pw_sel_layout": {"zh": "📐 通路拓扑排版模式", "en": "📐 Pathway Topology Layout"},
    "wb_pw_layout_hierarchical": {"zh": "有向级联流向 (从左到右 LR)", "en": "Directed Cascade Flow (Left-to-Right LR)"},
    "wb_pw_layout_force": {"zh": "经典弹性网络 (力导向)", "en": "Classic Force-Directed Network"},
    "wb_pw_chk_loops": {"zh": "🔄 闭环与自反馈调节挖掘 (检测自抑制/自促进)", "en": "🔄 Feedback Loop Mining (Detect self-inhibition / self-promotion)"},
    "wb_pw_chk_bridges": {"zh": "🌟 跨文献合成通路识别 (发现新假说)", "en": "🌟 Cross-Paper Bridge Detection (Discover novel hypotheses)"},
    "wb_pw_chk_subgraphs": {"zh": "🎨 精简独立子图渲染 (生成轻量交互画布)", "en": "🎨 Mini-Subgraph Rendering (Generate lightweight interactive canvas)"},
    "wb_pw_slider_count": {"zh": "🔢 输出通路数量上限", "en": "🔢 Max Pathway Count"},
    "wb_pw_btn_start": {"zh": "🚀 开始信号通路拆解与建模", "en": "🚀 Run Pathway Decomposition & Modeling"},
    "wb_pw_btn_rerun": {"zh": "🔄 重新拆解与生成通路", "en": "🔄 Re-generate Signaling Pathways"},
    "wb_pw_btn_clear": {"zh": "🗑️ 清空通路结果", "en": "🗑️ Clear Results"},
    "wb_pw_clear_done": {"zh": "已清空通路拆解结果与历史快照", "en": "Cleared pathway results and snapshots"},
    "wb_pw_status_start": {"zh": "正在启动信号通路拆解与因果建模流水线...", "en": "Starting pathway decomposition and causal modeling pipeline..."},
    "wb_pw_toast_done": {"zh": "🎉 信号通路拆解与建模完成！", "en": "🎉 Pathway decomposition and modeling completed!"},
    "wb_pw_no_pathways": {"zh": "未能在当前图谱中检测到有效通路，请先完善文献关系抽取。", "en": "No valid pathway subgraphs detected. Please upload papers or extract relations first."},
    "wb_pw_card_title": {"zh": "通路 {idx}：{title}", "en": "Pathway {idx}: {title}"},
    "wb_pw_lbl_subgraph": {"zh": "查看/收起独立子图谱", "en": "View / Collapse Subgraph"},
    "wb_pw_lbl_axis": {"zh": "🚩 核心因果传导主轴", "en": "🚩 Core Causal Axis"},
    "wb_pw_lbl_summary": {"zh": "📝 机制精炼阐述", "en": "📝 Mechanistic Summary"},
    "wb_pw_lbl_loops": {"zh": "🔄 自调节反馈回路分析", "en": "🔄 Autoregulatory Feedback Loop Analysis"},
    "wb_pw_lbl_bridges": {"zh": "🌟 跨文献融合与创新假说", "en": "🌟 Cross-Paper Synthesis & Novel Insights"},
    "wb_pw_lbl_canvas": {"zh": "🎨 独立精简拓扑画布", "en": "🎨 Dedicated Mini-Topology Canvas"},
    "wb_pw_lbl_evidence": {"zh": "📚 步骤文献证据溯源 (展开查看各步原文依据)", "en": "📚 Step-by-Step Evidence Traceability"},
    "wb_pw_btn_dl_html": {"zh": "📥 导出本通路交互子图 (HTML)", "en": "📥 Export Pathway Canvas (HTML)"},
    "wb_pw_btn_dl_json": {"zh": "📥 导出本通路数据 (JSON)", "en": "📥 Export Pathway Data (JSON)"},
    "wb_pw_btn_dl_all_json": {"zh": "📦 导出全量通路数据包 (JSON)", "en": "📦 Export All Pathways Package (JSON)"},
    "wb_pw_badge_loop": {"zh": "🔄 反馈循环", "en": "🔄 Feedback Loop"},
    "wb_pw_badge_pos_loop": {"zh": "🟢 自促进闭环", "en": "🟢 Self-Promotion Loop"},
    "wb_pw_badge_neg_loop": {"zh": "🔴 自抑制闭环", "en": "🔴 Self-Inhibition Loop"},
    "wb_pw_badge_cross_paper": {"zh": "🌟 跨文献通路", "en": "🌟 Cross-Paper Pathway"},
    "wb_pw_no_loops": {"zh": "本通路呈单向线性级联传导，未检测到闭合反馈回路。", "en": "Linear cascade; no closed feedback loop detected."},
    "wb_pw_single_source": {"zh": "本通路所有机制证据均源自单篇文献内部。", "en": "All mechanism evidence in this pathway originates within a single literature source."},
    "wb_pw_history_title": {"zh": "📂 已保存的历史通路拆解记录 (自动从工程目录加载)", "en": "📂 Saved Pathway Decomposition Records (Auto-loaded from project directory)"},
    "wb_pw_snapshot_note": {"zh": "📂 已自动恢复保存的历史通路快照", "en": "📂 Restored saved pathway decomposition snapshot"},
    "wb_pw_fallback_warn": {"zh": "大模型调用未成功，当前采用拓扑保底命名与阐述", "en": "LLM call was unsuccessful; using topological fallback naming."},
}


def t(key, default="", lang=None):
    """
    智能翻译函数：根据当前 session_state 中的 ui_language 选项返回对应文本。
    如果没找到 key，则返回 default，若无 default 则直接返回 key 本身。
    """
    if lang is None:
        lang = st.session_state.get("ui_language", "zh")  # 默认中文
    if key in UI_TEXT:
        return UI_TEXT[key].get(lang, default if default else key)
    return default if default else key

title_col, btn_col = st.columns([5, 1])
with title_col:
    st.title(t("page_title"))
with btn_col:
    # 稍微往下挤一点点（用一行空行），让按钮在垂直方向上和标题的文字居中对齐
    st.markdown("<br>", unsafe_allow_html=True)

    # 渲染帮助按钮，点击即触发上面的弹窗函数
    if st.button(t("btn_help"), use_container_width=True, type="secondary"):
        show_help_dialog()
st.markdown("---")
# 📁 创建一个持久化文件夹，用来保存所有分析过的原文
# 初始化全局图谱记忆中枢
if "master_entities" not in st.session_state:
    st.session_state.master_entities = []
if "current_view" not in st.session_state:
    st.session_state.current_view = "main"
if "workbench_state" not in st.session_state:
    st.session_state.workbench_state = {}
if "master_relations" not in st.session_state:
    st.session_state.master_relations = []
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "html_data" not in st.session_state:
    st.session_state.html_data = ""
if "pubmed_search_results" not in st.session_state:
    st.session_state.pubmed_search_results = []

# 📝 初始化历史文献列表与课题工程 ID / 名称
if "analyzed_files" not in st.session_state:
    st.session_state.analyzed_files = []
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = ""
if "current_project_name" not in st.session_state:
    st.session_state.current_project_name = "默认课题"

load_config()
load_local_vault()

if not st.session_state.get("current_project_id"):
    st.session_state.current_project_id = generate_project_id()
    save_local_vault()

# append_mode = True
# ==========================================

def render_sidebar_history_docs():
    """在侧边栏持久化展示当前工程的历史分析文献列表与下载按钮 (无论处于哪个视图均稳定渲染)"""
    # 📚 历史文献始终稳定呈现，绝不因 append_mode 开关而隐匿
    # 保证在无论何种运行模式与视图下，文献档案与下载入口均始终可见
    pass
    cur_files = st.session_state.get("analyzed_files", [])
    st.markdown("---")
    st.header(t("history_sidebar_title"))
    if not cur_files:
        st.info(t("history_no_docs"))
    else:
        for fname in cur_files:
            fpath = find_project_paper(fname)
            if fpath and os.path.exists(fpath):
                try:
                    with open(fpath, "rb") as f:
                        f_bytes = f.read()
                    st.download_button(
                        label=f"📄 {fname}",
                        data=f_bytes,
                        file_name=os.path.basename(fpath),
                        mime="application/pdf",
                        key=f"history_sb_{fname}_{st.session_state.get('current_view', 'main')}",
                        width="stretch"
                    )
                except Exception as e:
                    st.caption(f"📄 {fname}")
            else:
                if is_pmid_or_virtual_source(fname):
                    st.caption(f"🌐 {fname}")
                else:
                    st.caption(f"📄 {fname} ({t('doc_file_missing')})")

# 侧边栏配置：核心 API 和模型选择
# ==========================================


with st.sidebar:
    st.markdown("## 🧬 Bio-Graph Agent")
    st.markdown("---")
    is_local = "ollama" in str(st.session_state.get("api_provider", "")).lower()
    api_key_placeholder = t("sidebar_api_placeholder_local") if is_local else t("sidebar_api_placeholder_cloud")

    api_key = st.text_input("API Key", type="password",
                            label_visibility="collapsed",
                            placeholder=api_key_placeholder,
                            help=t("sidebar_api_help"))
    if is_local and not api_key:
        api_key = "dummy-key-for-ollama"

    st.markdown("---")


    with st.popover(t("sidebar_more_settings"), use_container_width=True):
        # ==========================================
        # 🌍 全局语言切换开关 (无缝接入原生配置中心版)
        # ==========================================
        def on_ui_language_change():
            choice = st.session_state.get("lang_radio_choice", "")
            new_l = "en" if "English" in choice else "zh"
            st.session_state.ui_language = new_l
            st.session_state.language = new_l
            save_config()

        current_index = 0 if st.session_state.get("ui_language", "zh") == "zh" else 1

        lang_choice = st.radio(
            "🌐 Language / 语言",
            ["中文 (Chinese)", "英文（English）"],
            index=current_index,
            key="lang_radio_choice",
            horizontal=True,
            on_change=on_ui_language_change
        )

        st.markdown("---")

        st.markdown(t("sidebar_engine_title"))

        cur_ui_lang = st.session_state.get("ui_language", "zh")
        provider_keys = list(API_PROVIDERS_CONF.keys())
        cur_provider = normalize_api_provider(st.session_state.get("api_provider", "openrouter"))
        st.session_state["api_provider"] = cur_provider

        selected_provider = st.selectbox(
            t("sidebar_api_provider"),
            options=provider_keys,
            format_func=lambda k: UI_TEXT[API_PROVIDERS_CONF[k]["tkey"]].get(cur_ui_lang, k),
            key="api_provider",
            on_change=save_config
        )

        if selected_provider == "custom":
            base_url = st.text_input("🔗 Base URL", key="custom_base_url", on_change=save_config)
        else:
            base_url = API_PROVIDERS_CONF[selected_provider]["url"]

        is_local_or_custom = selected_provider in ["ollama", "custom"]

        if is_local_or_custom:
            # 对于本地或自定义源，只能手填模型 ID
            selected_model_id = st.text_input(t("sidebar_model_id_custom"), key="custom_model_id", on_change=save_config)
            resolved_vision_model_id = selected_model_id
        else:
            # 对于云端源，使用语言中立的模型 ID 列表
            cloud_model_ids = [m[0] for m in CLOUD_MODEL_OPTIONS]
            cur_raw_model = st.session_state.get("selected_model_id") or st.session_state.get("selected_model_name", "")
            cur_model_id = normalize_model_id(cur_raw_model)
            st.session_state["selected_model_id"] = cur_model_id

            def format_model_choice(mid):
                tkey = MODEL_ID_TO_TRANSLATION_KEY.get(mid)
                if tkey and tkey in UI_TEXT:
                    return UI_TEXT[tkey].get(cur_ui_lang, mid)
                return mid

            base_model_id = st.selectbox(
                t("sidebar_model_select"),
                options=cloud_model_ids,
                format_func=format_model_choice,
                key="selected_model_id",
                on_change=save_config
            )
            # 同步更新展示名，以便任何依赖该字段的代码保持兼容
            st.session_state["selected_model_name"] = format_model_choice(base_model_id)

            # 🛠️ 二级高级设置：收纳视觉模型选择、视觉模型后缀与主模型后缀
            with st.expander(t("sidebar_expander_advanced_models"), expanded=False):
                # 1. 专职视觉模型选择
                vision_model_options = {
                    "auto": t("vision_model_auto"),
                    "gemini_2_5_flash_lite": t("vision_model_gemini_2_5_flash_lite"),
                    "gemini_3_8_flash": t("vision_model_gemini_3_8_flash"),
                    "qwen_2_5_vl_72b": t("vision_model_qwen_2_5_vl_72b"),
                    "custom": t("vision_model_custom")
                }
                vision_choice_keys = list(vision_model_options.keys())
                cur_v_choice = st.session_state.get("vision_model_choice", "auto")
                if cur_v_choice not in vision_choice_keys:
                    st.session_state["vision_model_choice"] = "auto"

                selected_v_key = st.selectbox(
                    t("sidebar_vision_model_select"),
                    vision_choice_keys,
                    format_func=lambda k: vision_model_options.get(k, k),
                    key="vision_model_choice",
                    on_change=save_config
                )

                if selected_v_key == "custom":
                    st.text_input("🔗 自定义视觉模型 ID", key="custom_vision_model_id", on_change=save_config)

                # 2. 视觉模型自定义后缀
                st.text_input(
                    t("sidebar_vision_model_suffix"),
                    key="vision_model_suffix",
                    placeholder=t("sidebar_vision_model_suffix_placeholder"),
                    on_change=save_config
                )

                # 3. 普通主模型后缀 (收拢至此)
                st.text_input(
                    t("sidebar_model_suffix"),
                    key="model_suffix",
                    placeholder=t("sidebar_model_suffix_placeholder"),
                    help=t("sidebar_model_suffix_help"),
                    on_change=save_config
                )

                # 智能合成最终使用的模型 ID
                model_suffix = st.session_state.get("model_suffix", "")
                if model_suffix and model_suffix.strip():
                    s = model_suffix.strip()
                    if s.startswith(":"):
                        selected_model_id = f"{base_model_id}{s}"
                    elif "/" in s:
                        # 包含斜杠则判定为全量自定义模型 ID（如 meta-llama/llama-3.3-70b-instruct）
                        selected_model_id = s
                    else:
                        # 未输入冒号但输入了后缀（如 free）自动补齐冒号后缀
                        selected_model_id = f"{base_model_id}:{s}"
                else:
                    selected_model_id = base_model_id

                # 智能解析当前生效的专职视觉模型 ID
                v_choice = st.session_state.get("custom_vision_model_id") if st.session_state.get("vision_model_choice") == "custom" else st.session_state.get("vision_model_choice", "auto")
                v_suffix = st.session_state.get("vision_model_suffix", "")
                resolved_vision_model_id = resolve_vision_model_id(selected_model_id, v_choice, v_suffix)

                # 实时回显生效的 ID (统一收拢在折叠菜单内部)
                st.markdown("---")
                st.caption(f"🚀 {t('sidebar_model_active_preview')} `{selected_model_id}`")
                st.caption(f"👁️ {t('sidebar_vision_model_active_preview')} `{resolved_vision_model_id}`")

        st.markdown("---")
        st.markdown(t("sidebar_db_title"))
        db_sources = ["PubMed", "Semantic Scholar"]
        cur_db = st.session_state.get("search_database", "PubMed")
        cur_db_clean = "PubMed" if "PubMed" in str(cur_db) else "Semantic Scholar"
        st.session_state["search_database"] = cur_db_clean
        db_idx = 0 if cur_db_clean == "PubMed" else 1

        is_en_ui = "en" in str(st.session_state.get("ui_language", "zh")).lower()
        def get_db_display(k):
            if k == "PubMed":
                return "PubMed (Biomedical Authority)" if is_en_ui else "PubMed (生物医学权威)"
            return "Semantic Scholar (Multidisciplinary AI)" if is_en_ui else "Semantic Scholar (全科AI智能)"
        st.selectbox(t("sidebar_db_select"), db_sources, format_func=get_db_display, key="search_database", on_change=save_config)

        st.markdown("---")

        # ✨ 恢复：功能模块开关 (防注销保底)
        st.markdown(t("sidebar_features_title"))
        if "ENABLE_EDITOR" not in st.session_state:
            st.session_state["ENABLE_EDITOR"] = True
        if "ENABLE_AI_CLEANER" not in st.session_state:
            st.session_state["ENABLE_AI_CLEANER"] = True

        ENABLE_EDITOR = st.toggle(t("sidebar_toggle_editor"), value=st.session_state.get("ENABLE_EDITOR", True), key="ENABLE_EDITOR", persist_state="session", on_change=save_config)
        ENABLE_AI_CLEANER = st.toggle(t("sidebar_toggle_ai"), value=st.session_state.get("ENABLE_AI_CLEANER", True), key="ENABLE_AI_CLEANER", persist_state="session", on_change=save_config)

        st.markdown("---")
        with st.expander(t("sidebar_contact")):
            st.markdown(t("contact_desc"))
            st.link_button(t("contact_github"), "https://github.com/24238-T-34/Bio-Graph-Agent", width="stretch")
            st.caption(t("contact_email"))
        st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n...")
        st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nit's a bug\nif you see this")

    # 📚 历史分析文献 (直接渲染在主侧边栏顶层，脱离设置弹窗，双视图均稳定渲染)
    render_sidebar_history_docs()

#    st.markdown("---")



def get_project_paper_full_texts(project_id: str) -> dict:
    """提取当前工程专属 papers/ 目录下所有已存入 PDF 的文本，供 Round 2 精准正则回溯"""
    proj_dir = get_project_dir_by_id(project_id)
    papers_dir = os.path.join(proj_dir, "papers")
    texts = {}
    if not os.path.exists(papers_dir):
        return texts
    from IO_SYS import PDFProcessor
    proc = PDFProcessor()
    for fname in os.listdir(papers_dir):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(papers_dir, fname)
            try:
                chunks = proc.extract_chunks(fpath, start_page=0, end_page=None)
                texts[fname] = "\n".join(chunks)
            except Exception as e:
                print(f"⚠️ [Workbench] 提取文献文本失败 {fname}: {e}")
    return texts


def open_directory_in_file_manager(dir_path: str) -> bool:
    """
    跨平台尝试在系统文件管理器中打开指定目录：
    - Windows: os.startfile
    - macOS: open
    - Linux: xdg-open
    若运行于无图形界面的远程/Docker环境，平滑捕获异常并返回 False。
    """
    import sys
    import subprocess
    import os
    abs_path = os.path.abspath(dir_path)
    try:
        if sys.platform == "win32":
            os.startfile(abs_path)
            return True
        elif sys.platform == "darwin":
            subprocess.Popen(["open", abs_path])
            return True
        else:
            subprocess.Popen(["xdg-open", abs_path])
            return True
    except Exception as e:
        print(f"⚠️ [open_directory_in_file_manager] 无法打开图形文件管理器: {e}")
        return False


def ingest_verified_hypothesis_via_pipeline(card: dict, api_key: str, selected_model_id: str, base_url: str, searcher):
    """
    完全参照【智能拓展 (Smart Expansion)】的标准流水线：
    提取确证假说对应的 PubMed 真实文献摘要，执行：
    1. 实体抽取与自我反思 (extract_entities_with_reflection)
    2. 文献注册与哈希溯源 (register_or_update_paper & canonicalize_project_papers)
    3. 机制关系抽取 (extract_relations)
    4. 全局实体对齐映射 (align_global_entities)
    5. 主图谱融合与同质边规整 (consolidate_homogeneous_relations)
    6. 拓展历史日志记录与本地持久化
    7. 刷新主画布并平滑返回主界面
    """
    adj = card.get("adjudication", {}) or {}
    pmid = str(adj.get("direct_evidence_pmid", "")).strip()
    articles = card.get("retrieved_articles", [])
    target_art = None
    if pmid:
        for a in articles:
            if str(a.get("pmid", "")).strip() == pmid:
                target_art = a
                break
    if not target_art and articles:
        target_art = articles[0]
        if not pmid:
            pmid = str(target_art.get("pmid", "")).strip()

    title = target_art.get("title", f"PubMed:{pmid}") if target_art else f"PubMed:{pmid}"
    abstract = target_art.get("abstract", "") if target_art else ""

    # 若摘要缺失或过短，尝试向数据库现场补抓
    if (not abstract or len(abstract) < 50) and pmid and hasattr(searcher, "fetch_abstract"):
        try:
            fetched = searcher.fetch_abstract(pmid)
            if fetched and len(fetched) >= 50:
                abstract = fetched
        except Exception as e:
            print(f"⚠️ [Pipeline] 现场获取摘要异常: {e}")

    # 若极端情况下无摘要或无文献，以假说陈述与依据作为保底证据文本
    if not abstract or len(abstract) < 30:
        quote = adj.get("evidence_quote", "")
        rationale = adj.get("adjudication_rationale", "")
        stmt = card.get("hypothesis_statement", "")
        abstract = f"{title}. {quote} {rationale} {stmt}".strip()

    current_api_key = api_key.strip() if api_key else ""
    from LLM_SYS import BioBrainAgent
    agent = BioBrainAgent(api_key=current_api_key, model=selected_model_id, base_url=base_url)

    use_reflection = st.session_state.get("use_reflection", True)
    entity_language = st.session_state.get("entity_language", "关闭 (保持原文语言)")

    # 1. 实体抽取与自我反思
    new_entities = agent.extract_entities_with_reflection(
        abstract,
        use_reflection=use_reflection,
        entity_lang=entity_language
    )

    # 2. 文献注册与溯源标签 (PMID)
    pmid_tag = f"PubMed:{pmid}" if pmid else f"PubMed:Hypo_{card.get('id', '')}"
    v_hash = f"pmid:{pmid}" if pmid else f"hypo:{card.get('id', '')}"
    proj_id = get_current_project_id()

    register_or_update_paper(
        proj_id,
        doc_hash=v_hash,
        display_name=pmid_tag,
        aliases=[pmid_tag, str(pmid)] if pmid else [pmid_tag],
        pmid=str(pmid) if pmid else "",
        source_type="online_pmid"
    )

    for ent in new_entities:
        ent["doc_source"] = pmid_tag
        ent["doc_hash"] = v_hash

    if pmid_tag not in st.session_state.analyzed_files:
        st.session_state.analyzed_files.append(pmid_tag)
    st.session_state.analyzed_files = canonicalize_project_papers(proj_id, st.session_state.analyzed_files)

    # 3. 机制关系抽取
    combined_entities_dict = st.session_state.master_entities + new_entities
    new_relations = agent.extract_relations(abstract, combined_entities_dict)
    for rel in new_relations:
        rel["doc_source"] = pmid_tag
        rel["doc_hash"] = v_hash
        rel["weight"] = 1

    # 4. 全局实体对齐映射
    alignment_map = agent.align_global_entities(st.session_state.master_entities, new_entities)

    aligned_new_entities = []
    master_names = {ent["standard_name"]: ent for ent in st.session_state.master_entities}

    for new_ent in new_entities:
        new_name = new_ent.get("standard_name")
        new_source = new_ent.get("doc_source", "")

        if new_name in master_names:
            master_ent = master_names[new_name]
            aliases = master_ent.setdefault("aliases", [])
            for new_alias in new_ent.get("aliases", []):
                if new_alias not in aliases and new_alias != new_name:
                    aliases.append(new_alias)
            old_source = master_ent.get("doc_source", "")
            if new_source and new_source not in old_source:
                master_ent["doc_source"] = f"{old_source} | {new_source}"
        elif alignment_map and new_name in alignment_map:
            master_name = alignment_map[new_name]
            if master_name in master_names:
                master_ent = master_names[master_name]
                aliases = master_ent.setdefault("aliases", [])
                if new_name not in aliases:
                    aliases.append(new_name)
                for new_alias in new_ent.get("aliases", []):
                    if new_alias not in aliases and new_alias != master_name:
                        aliases.append(new_alias)
                old_source = master_ent.get("doc_source", "")
                if new_source and new_source not in old_source:
                    master_ent["doc_source"] = f"{old_source} | {new_source}"
            else:
                aligned_new_entities.append(new_ent)
        else:
            aligned_new_entities.append(new_ent)

    if alignment_map:
        for rel in new_relations:
            if rel.get("source") in alignment_map:
                rel["source"] = alignment_map[rel["source"]]
            if rel.get("target") in alignment_map:
                rel["target"] = alignment_map[rel["target"]]

    # 5. 确保学者检验的核心机制边必定入库
    node_a = card.get("node_a", "")
    node_b = card.get("node_b", "")
    aligned_a = alignment_map.get(node_a, node_a) if alignment_map else node_a
    aligned_b = alignment_map.get(node_b, node_b) if alignment_map else node_b
    has_direct_rel = any(
        (r.get("source") == aligned_a and r.get("target") == aligned_b) or
        (r.get("source") == aligned_b and r.get("target") == aligned_a)
        for r in new_relations
    )
    if not has_direct_rel and aligned_a and aligned_b:
        new_relations.append({
            "source": aligned_a,
            "target": aligned_b,
            "relation": card.get("proposed_relation", "Positive_Regulation"),
            "doc_source": pmid_tag,
            "doc_hash": v_hash,
            "evidence": adj.get("evidence_quote") or card.get("hypothesis_statement", ""),
            "weight": 1
        })

    # 6. 主图谱合并与同质边融合
    all_rels = st.session_state.master_relations + new_relations
    st.session_state.master_relations = consolidate_homogeneous_relations(all_rels)
    st.session_state.master_entities.extend(aligned_new_entities)

    # 7. 拓展历史日志记录
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] PMID: {pmid} | Title: {title} (Verified Hypothesis: {node_a} ⟷ {node_b})\n"
    try:
        log_dir = ".expanded_logs"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "expansion_history.txt"), "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass

    try:
        proj_dir = get_project_dir_by_id(proj_id)
        with open(os.path.join(proj_dir, "expansion_history.txt"), "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass

    # 8. 保存 Vault + 重绘主画布 + 设为主视图
    save_local_vault()
    redraw_and_update()
    st.session_state.current_view = "main"


def render_analysis_workbench(api_key: str, selected_model_id: str, base_url: str):
    """图谱分析工作台独立视图渲染 (Graph Analysis Workbench) - 完整双语国际化与条件渲染"""
    import importlib
    import workbench_engine
    import IO_SYS
    import LLM_SYS
    importlib.reload(workbench_engine)
    importlib.reload(IO_SYS)
    importlib.reload(LLM_SYS)
    from workbench_engine import OntologyManager, WorkbenchAnalysisEngine, PathwayDecompositionEngine
    from IO_SYS import GraphVisualizer
    from collections import Counter
    import streamlit.components.v1 as components

    is_en = "en" in str(st.session_state.get("ui_language", "zh")).lower()
    UI_TEXT.setdefault("wb_pw_btn_dl_biokg", {"zh": "🧬 降级导出为工程 (.biokg)", "en": "🧬 Export as Project (.biokg)"})
    UI_TEXT.setdefault("wb_pw_help_dl_biokg", {
        "zh": "将当前调控通路/回路打包为标准 .biokg 独立工程文件，可直接载入系统进行智能拓展、桥接等二次深度分析",
        "en": "Package this pathway/circuit as a standard .biokg project archive, ready for secondary smart expansion, bridging, and deep analysis"
    })
    UI_TEXT.update({
        "wb_sec3_title": {
            "zh": "### 💡 第三阶段：智能假说推演与科研探勘 (Stage 3: Hypothesis Deduction & Discovery Copilot)",
            "en": "### 💡 Stage 3: Hypothesis Deduction & Discovery Copilot"
        },
        "wb_sec3_caption": {
            "zh": "结合阶段一本体分类与图论拓扑结构洞，自主挖掘网络断裂带；生成高价值学术猜想，并由裁判大模型并发通读 PubMed 检验科研创新性与空白填补。",
            "en": "Combines Stage 1 ontology with graph-theoretic structural holes to autonomously mine network gaps, formulate high-value hypotheses, and concurrently verify novelty via literature referee."
        },
        "wb_hypo_mode_select": {"zh": "🧭 选择探勘策略模式", "en": "🧭 Select Discovery Strategy Mode"},
        "wb_hypo_mode_auto": {"zh": "🤖 自动感知智能推荐 (Auto Mode)", "en": "🤖 Auto-Perception Smart Recommendation"},
        "wb_hypo_mode_shortcut": {"zh": "⚡ 高维语义隐性捷径 (Detour Shortcuts)", "en": "⚡ High-Dimensional Detour Shortcuts"},
        "wb_hypo_mode_bridge": {"zh": "🔗 跨簇孤岛自主桥接 (Cluster Bridging)", "en": "🔗 Autonomous Cluster Bridging"},
        "wb_hypo_mode_homeostasis": {"zh": "⚖️ 逆向稳态闭环探索 (Homeostatic Feedback)", "en": "⚖️ Homeostatic Feedback Loops"},
        "wb_hypo_mode_growth": {"zh": "🌱 周界边缘薄层生长 (Peripheral Growth)", "en": "🌱 Peripheral Thin-Layer Growth"},
        "wb_hypo_mode_intent": {"zh": "🎯 学者科研意向定向探究 (Intent Exploration)", "en": "🎯 Intent-Driven Directional Exploration"},
        "wb_hypo_intent_label": {"zh": "🎯 输入学者的科研关注点 / 探索意向 Prompt", "en": "🎯 Enter Research Intent / Focus Prompt"},
        "wb_hypo_intent_placeholder": {"zh": "例如：探索该通路在克服耐药性或介导自噬中的关键交互靶点", "en": "e.g., Explore potential targets mediating drug resistance or autophagy crosstalk"},
        "wb_hypo_btn_mine": {"zh": "🚀 启动拓扑寻孔并推演候选假说 (~5项)", "en": "🚀 Mine Structural Holes & Formulate Hypotheses (~5 items)"},
        "wb_hypo_list_title": {"zh": "### 💡 探勘候选学术假说 (共 {count} 项)", "en": "### 💡 Candidate Academic Hypotheses ({count} items)"},
        "wb_hypo_btn_verify_single": {"zh": "🔬 启动文献检索与裁判终审", "en": "🔬 Launch Literature Search & Referee Review"},
        "wb_hypo_btn_verify_batch": {"zh": "⚡ 并发核验所有待审假说 (多线程加速)", "en": "⚡ Concurrently Verify All Pending Hypotheses (Multi-threaded)"},
        "wb_hypo_btn_clear": {"zh": "🗑️ 清空假说卡片", "en": "🗑️ Clear Hypotheses"},
        "wb_hypo_card_rationale": {"zh": "🧬 生物学生化合理性依据：", "en": "🧬 Biochemical Rationale:"},
        "wb_hypo_card_direction": {"zh": "🧭 建议科研探索方向与留白描述：", "en": "🧭 Suggested Research Direction & Gap Description:"},
        "wb_hypo_verdict_known": {"zh": "💡 文献已确证的事实补全 (Known Evidence)", "en": "💡 Fact Gap Filled by Literature Evidence (Known Fact)"},
        "wb_hypo_verdict_novel": {"zh": "🌟 真正的科研空白 (原始创新假说 / Novelty High)", "en": "🌟 Genuine Scientific White Space (Novelty High)"},
        "wb_hypo_verdict_cascade": {"zh": "🔗 多步级联中介通路 (Cascade Pathway)", "en": "🔗 Multi-Step Cascade Pathway"},
        "wb_hypo_verdict_inconclusive": {"zh": "❓ 证据不足或存在争议 (Inconclusive)", "en": "❓ Inconclusive or Disputed Evidence"},
        "wb_hypo_btn_merge_return": {"zh": "📥 采纳文献证据并触发抽取流水线 (入库主图谱并返回主界面)", "en": "📥 Ingest via Pipeline & Merge to Master Graph (Return)"},
        "wb_hypo_btn_merge_cascade": {"zh": "📥 采纳级联文献证据并智能桥接入库 (抽取中介实体与两段式调控关系)", "en": "📥 Ingest Cascade Literature Evidence & Bridge to Master Graph (Extract Mediators)"},
        "wb_hypo_btn_dl_biokg": {"zh": "📦 导出为专题研究子工程 (.biokg)", "en": "📦 Export as Research Subproject (.biokg)"},
        "wb_hypo_toast_merged_return": {"zh": "🎉 确证文献已通过流水线深度解析！新实体与调控关系已成功吸收入主图谱。", "en": "🎉 Confirmed literature successfully parsed via pipeline and merged into master graph!"},
        "wb_hypo_toast_merged_cascade": {"zh": "🎉 级联文献已通过流水线深度解析！中介实体与两段式调控关系已智能桥接并合并入主图谱。", "en": "🎉 Cascade literature parsed via pipeline! Intermediate mediators and relations bridged into master graph."},
        "wb_hypo_novel_notice": {"zh": "ℹ️ 依据科研严谨性规范，未经验证的科学空白假说留作专享选题灵感，不污染主图谱事实库；学者可导出为独立子工程深挖。", "en": "ℹ️ By scientific standards, unverified gaps remain as research inspiration without polluting factual graph; export as subproject to investigate."},
        "wb_hypo_articles_title": {"zh": "📚 裁判检索审阅的 PubMed 文献摘要 ({count} 篇)", "en": "📚 Reviewed PubMed Abstracts ({count} items)"},
        "wb_hypo_badge_auto": {"zh": "🤖 自动感知模式", "en": "🤖 Auto Mode"},
        "wb_hypo_badge_shortcut": {"zh": "⚡ 捷径短路模体", "en": "⚡ Detour Shortcut"},
        "wb_hypo_badge_bottleneck": {"zh": "🛡️ 瓶颈加固模体", "en": "🛡️ Bottleneck Reinforce"},
        "wb_hypo_badge_homeostasis": {"zh": "⚖️ 稳态闭环模体", "en": "⚖️ Homeostasis Loop"},
        "wb_hypo_badge_frontier": {"zh": "🌿 周界前沿级联", "en": "🌿 Frontier Cascade"},
        "wb_hypo_badge_bridge": {"zh": "🔗 跨簇桥接", "en": "🔗 Cluster Bridge"},
        "wb_hypo_badge_growth": {"zh": "🌱 周界生长", "en": "🌱 Peripheral Growth"},
        "wb_hypo_badge_intent": {"zh": "🎯 意向探究", "en": "🎯 Intent Exploration"},
    })

    # 1. 顶部返回导航栏与标题
    col_back1, col_back2 = st.columns([1, 4])
    with col_back1:
        if st.button(t("wb_btn_top_return"), key="btn_wb_top_return", width="stretch"):
            st.session_state.current_view = "main"
            st.rerun()
    with col_back2:
        st.markdown(f"## {t('wb_title')}")

    st.caption(t("wb_caption"))

    proj_id = get_current_project_id()
    proj_name = get_current_project_name()
    proj_dir = get_project_dir_by_id(proj_id)
    ents = st.session_state.get("master_entities", [])
    rels = st.session_state.get("master_relations", [])

    st.info(t("wb_project_info").format(name=proj_name, id=proj_id, ents=len(ents), rels=len(rels)))

    if len(ents) == 0:
        st.warning(t("wb_warn_no_data"))
        return

    # 检查工作台是否已具备分析结果
    wb_cached = st.session_state.get("workbench_state", {})
    ent_annots = wb_cached.get("entity_annotations", {})
    rel_annots = wb_cached.get("relation_annotations", {})
    wb_html_file = os.path.join(proj_dir, "workbench_graph.html")
    has_analysis = bool(ent_annots and os.path.exists(wb_html_file))

    # 2. 顶部嵌入基础图谱全景折叠预览 (Point 6)
    with st.expander(t("wb_base_graph_preview"), expanded=False if has_analysis else True):
        base_html = st.session_state.get("html_data", "")
        if not base_html:
            base_html_path = os.path.join(proj_dir, "graph.html")
            if os.path.exists(base_html_path):
                try:
                    with open(base_html_path, "r", encoding="utf-8") as f:
                        base_html = f.read()
                except Exception:
                    pass
        if base_html:
            components.html(base_html, height=480, scrolling=True)
        else:
            st.caption(t("wb_no_base_graph"))

    # 3. 本体预设与分析控制面板
    mgr = OntologyManager()
    presets = mgr.list_presets()
    preset_id_map = {p["id"]: p for p in presets}
    preset_ids = list(preset_id_map.keys())
    default_idx = preset_ids.index("universal_biored") if "universal_biored" in preset_ids else 0

    with st.container(border=True):
        st.markdown(f"#### {t('wb_config_title')}")
        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            def get_preset_fmt(pid):
                p = preset_id_map.get(pid, {})
                return p.get("en_name", p.get("name", pid)) if is_en else p.get("name", pid)

            selected_pid = st.selectbox(
                t("wb_preset_select"),
                options=preset_ids,
                format_func=get_preset_fmt,
                index=default_idx,
                key="wb_preset_select"
            )
            selected_preset = preset_id_map[selected_pid]
            desc = selected_preset.get('en_description', selected_preset.get('description', '')) if is_en else selected_preset.get('description', '')
            st.caption(f"ℹ️ {desc}")
        with col_p2:
            st.markdown(f"**{t('wb_custom_preset_title')}**")
            st.caption(t("wb_open_presets_dir_desc"))
            btn_col1, btn_col2 = st.columns([3, 2])
            with btn_col1:
                if st.button(t("wb_btn_open_presets_dir"), key="wb_btn_open_presets_dir", width="stretch"):
                    abs_dir = os.path.abspath(mgr.presets_dir)
                    opened = open_directory_in_file_manager(abs_dir)
                    if opened:
                        st.toast(t("wb_toast_folder_opened").format(path=abs_dir), icon="📁")
                    else:
                        st.info(f"📁 `{abs_dir}`")
            with btn_col2:
                if st.button(t("wb_btn_refresh_presets"), key="wb_btn_refresh_presets", width="stretch"):
                    st.rerun()
            st.caption(f"📁 `{os.path.abspath(mgr.presets_dir)}`")

        c_t1, c_t2, c_t3 = st.columns(3)
        with c_t1:
            enable_entities = st.checkbox(t("wb_chk_entities"), value=True, key="wb_chk_entities")
        with c_t2:
            enable_relations = st.checkbox(t("wb_chk_relations"), value=True, key="wb_chk_relations")
        with c_t3:
            force_reanalyze = st.checkbox(t("wb_chk_force"), value=False, key="wb_chk_force")

        with st.expander(t("wb_esc_title"), expanded=False):
            c_r0, c_r1 = st.columns(2)
            c_r2, c_r3 = st.columns(2)
            with c_r0:
                enable_r0 = st.checkbox(t("wb_esc_r0"), value=True, key="wb_esc_r0")
            with c_r1:
                enable_r1 = st.checkbox(t("wb_esc_r1"), value=True, key="wb_esc_r1")
            with c_r2:
                enable_r2 = st.checkbox(t("wb_esc_r2"), value=True, key="wb_esc_r2")
            with c_r3:
                enable_r3 = st.checkbox(t("wb_esc_r3"), value=True, key="wb_esc_r3")

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            btn_run = st.button(t("wb_btn_start_run"), type="primary", key="wb_btn_start_run", width="stretch")
        with col_btn2:
            btn_rerender = st.button(t("wb_btn_rerender_canvas"), key="wb_btn_rerender_canvas", width="stretch")

        if btn_run:
            needs_llm = (enable_r1 or enable_r2 or enable_r3 or enable_relations)
            current_key = api_key.strip() if api_key else ""
            if needs_llm and not current_key and "ollama" not in str(st.session_state.get("api_provider", "")).lower():
                st.error(t("err_missing_api_key"))
                st.stop()

            from LLM_SYS import BioBrainAgent
            llm_agent = BioBrainAgent(api_key=current_key, model=selected_model_id, base_url=base_url) if needs_llm else None
            paper_texts = get_project_paper_full_texts(proj_id) if enable_r2 else {}

            concurrency_val = int(st.session_state.get("concurrency_workers", 4))
            engine = WorkbenchAnalysisEngine(
                llm_agent=llm_agent,
                ontology=selected_preset,
                paper_full_texts=paper_texts,
                cached_entities=wb_cached.get("entity_annotations", {}),
                cached_relations=wb_cached.get("relation_annotations", {})
            )

            p_bar = st.progress(0.0)
            status_elem = st.empty()
            log_expander = st.expander(t("wb_log_expander_title"), expanded=True)
            log_placeholder = log_expander.empty()
            captured_logs = []

            def on_prog(frac, msg):
                p_bar.progress(frac)
                status_elem.info(f"⏳ {msg}")

            def on_log(msg):
                captured_logs.append(msg)
                log_placeholder.code("\n".join(captured_logs[-15:]), language="text")

            analysis_kwargs = {
                "entities": ents,
                "relations": rels,
                "enable_entities": enable_entities,
                "enable_relations": enable_relations,
                "enable_round0_rules": enable_r0,
                "enable_round1_context": enable_r1,
                "enable_round2_regex": enable_r2,
                "enable_round3_world": enable_r3,
                "force_reanalyze": force_reanalyze,
                "max_workers": concurrency_val,
                "language": st.session_state.get("ui_language", "zh"),
                "progress_callback": on_prog,
                "log_callback": on_log
            }
            # 兼容性防御：如果极端情况下引擎为早期缓存模块且无 max_workers 参数，安全自适应
            import inspect
            sig = inspect.signature(engine.run_analysis)
            if "max_workers" not in sig.parameters and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                analysis_kwargs.pop("max_workers", None)

            new_state = engine.run_analysis(**analysis_kwargs)

            st.session_state.workbench_state = new_state

            # 生成工作台拓扑 HTML
            vis = GraphVisualizer()
            wb_html_path = os.path.join(proj_dir, "workbench_graph.html")
            vis.generate_workbench_html(
                global_entities=ents,
                global_relations=rels,
                entity_annotations=new_state.get("entity_annotations", {}),
                relation_annotations=new_state.get("relation_annotations", {}),
                ontology=selected_preset,
                output_file=wb_html_path,
                output_lang=st.session_state.get("ui_language", "zh")
            )

            save_local_vault()
            st.toast(t("wb_toast_analysis_done"), icon="✨")
            st.rerun()

        if btn_rerender:
            vis = GraphVisualizer()
            wb_html_path = os.path.join(proj_dir, "workbench_graph.html")
            vis.generate_workbench_html(
                global_entities=ents,
                global_relations=rels,
                entity_annotations=wb_cached.get("entity_annotations", {}),
                relation_annotations=wb_cached.get("relation_annotations", {}),
                ontology=selected_preset,
                output_file=wb_html_path,
                output_lang=st.session_state.get("ui_language", "zh")
            )
            st.toast(t("wb_toast_rerender_done"), icon="🎨")
            st.rerun()

    # 4. 条件渲染：仅在存在分析结果时展示大盘、画布与调换操作台 (Point 4)
    if not has_analysis:
        st.markdown("---")
        st.info(t("wb_not_analyzed_hint"))
    else:
        cur_ontology = wb_cached.get("ontology", selected_preset)
        ent_cat_defs = cur_ontology.get("entity_categories", {})
        rel_cat_defs = cur_ontology.get("relation_categories", {})

        resolved_cnt = sum(1 for a in ent_annots.values() if a.get("status") == "resolved")
        uncertain_cnt = sum(1 for a in ent_annots.values() if a.get("status") == "uncertain" or a.get("primary_type") == "Uncertain")
        analyzed_rel_cnt = len(rel_annots)

        st.markdown(t("wb_metrics_title"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("wb_metric_total"), len(ents))
        m2.metric(t("wb_metric_resolved"), resolved_cnt)
        m3.metric(t("wb_metric_uncertain"), uncertain_cnt)
        m4.metric(t("wb_metric_relations"), analyzed_rel_cnt)

        col_ent_view, col_rel_view = st.columns([1, 1], gap="medium")

        with col_ent_view:
            st.markdown(t("wb_sec_ent_dist"))
            cat_to_nodes = {}
            for n in ents:
                sname = n.get("standard_name")
                cat = ent_annots.get(sname, {}).get("primary_type", "Uncertain")
                if cat not in ent_cat_defs or cat == "Uncertain":
                    cat = "Uncertain"
                cat_to_nodes.setdefault(cat, []).append(sname)

            sorted_cats = sorted(cat_to_nodes.items(), key=lambda x: len(x[1]), reverse=True)
            for cat_id, nodes in sorted_cats:
                c_def = ent_cat_defs.get(cat_id, {})
                c_lbl = ((c_def.get("en_name") or c_def.get("en_label", cat_id)) if is_en else (c_def.get("name") or c_def.get("label", cat_id))) or cat_id
                pct = (len(nodes) / max(1, len(ents))) * 100
                unit_txt = t("wb_unit_entities")
                with st.expander(f"{c_lbl} — {len(nodes)} {unit_txt} ({pct:.1f}%)", expanded=False):
                    for nodename in nodes:
                        a = ent_annots.get(nodename, {})
                        sub = a.get("sub_type", "")
                        rnd = a.get("round", "")
                        conf = a.get("confidence", 0.0)
                        rsn = a.get("reasoning", "")
                        st.markdown(f"**{nodename}**" + (f" `[{sub}]`" if sub else "") + f" *(Round {rnd}, {int(conf*100)}%)*")
                        if rsn:
                            st.caption(f"💡 {rsn}")

        with col_rel_view:
            st.markdown(t("wb_sec_rel_dist"))
            mech_to_edges = {}
            for r in rels:
                src = r.get("source")
                tgt = r.get("target")
                rel = r.get("relation", "相关")
                k = f"{src}|{tgt}|{rel}"
                mech_cat = rel_annots.get(k, {}).get("mechanism_category", "未分析" if not is_en else "Unanalyzed")
                if mech_cat not in rel_cat_defs and mech_cat not in ["未分析", "Unanalyzed"]:
                    mech_cat = "Association_Correlation" if "Association_Correlation" in rel_cat_defs else ("未分析" if not is_en else "Unanalyzed")
                mech_to_edges.setdefault(mech_cat, []).append((src, tgt, rel, k))

            sorted_mechs = sorted(mech_to_edges.items(), key=lambda x: len(x[1]), reverse=True)
            for mech_id, edge_tuples in sorted_mechs:
                r_def = rel_cat_defs.get(mech_id, {})
                r_lbl = ((r_def.get("en_name") or r_def.get("en_label", mech_id)) if is_en else (r_def.get("name") or r_def.get("label", mech_id))) or mech_id
                unit_rel = t("wb_unit_relations")
                with st.expander(f"{r_lbl} — {len(edge_tuples)} {unit_rel}", expanded=False):
                    for src, tgt, rel, k in edge_tuples:
                        a = rel_annots.get(k, {})
                        verb = a.get("mechanism_verb", rel)
                        conf = a.get("confidence", 0.0)
                        rsn = a.get("reasoning", "")
                        orig_prefix = t("wb_orig_rel_prefix")
                        st.markdown(f"**{src}** ➔ **{tgt}** : `{verb}` *({orig_prefix}: {rel})*")
                        if rsn:
                            st.caption(f"💡 {rsn}")

        # 5. 拓扑画布展示与结果导出
        st.markdown("---")
        st.markdown(t("wb_canvas_title"))

        wb_html_data = ""
        if os.path.exists(wb_html_file):
            try:
                with open(wb_html_file, "r", encoding="utf-8") as f:
                    wb_html_data = f.read()
            except Exception as e:
                st.error(f"Error loading workbench html: {e}")

        # 准备工作台导出数据 (包含本体分类与深层机制注解)
        wb_ents_list = []
        for e in ents:
            item = dict(e)
            sname = e.get("standard_name")
            annot = ent_annots.get(sname, {})
            item["primary_type"] = annot.get("primary_type", "Uncertain")
            item["sub_type"] = annot.get("sub_type", "")
            item["tags"] = annot.get("tags", [])
            item["confidence"] = annot.get("confidence", 0.0)
            item["reasoning"] = annot.get("reasoning", "")
            item["round"] = annot.get("round", "")
            wb_ents_list.append(item)

        wb_entities_json = json.dumps(wb_ents_list, ensure_ascii=False, indent=2)

        df_wb_entities = pd.DataFrame(wb_ents_list)
        if 'aliases' in df_wb_entities.columns:
            df_wb_entities['aliases'] = df_wb_entities['aliases'].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x)
        if 'tags' in df_wb_entities.columns:
            df_wb_entities['tags'] = df_wb_entities['tags'].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x)
        csv_wb_entities = df_wb_entities.to_csv(index=False).encode('utf-8')

        wb_rels_list = []
        for r in rels:
            item = dict(r)
            src = r.get("source")
            tgt = r.get("target")
            rel = r.get("relation", "相关")
            k = f"{src}|{tgt}|{rel}"
            annot = rel_annots.get(k, {})
            item["mechanism_category"] = annot.get("mechanism_category", "")
            item["mechanism_verb"] = annot.get("mechanism_verb", rel)
            item["confidence"] = annot.get("confidence", 0.0)
            item["reasoning"] = annot.get("reasoning", "")
            item["round"] = annot.get("round", "")
            wb_rels_list.append(item)

        df_wb_relations = pd.DataFrame(wb_rels_list)
        csv_wb_relations = df_wb_relations.to_csv(index=False).encode('utf-8') if not df_wb_relations.empty else b""

        wb_exp_col1, wb_exp_col2, wb_exp_col3, wb_exp_col4 = st.columns(4)
        with wb_exp_col1:
            st.download_button(t("btn_export_ent_json"), wb_entities_json, "Bio_Analysis_entities.json", "application/json",
                               width="stretch", key="wb_dl_ent_json")
        with wb_exp_col2:
            st.download_button(t("btn_export_ent_csv"), csv_wb_entities, "Bio_Analysis_entities.csv", "text/csv",
                               width="stretch", key="wb_dl_ent_csv")
        with wb_exp_col3:
            st.download_button(t("btn_export_rel_csv"), csv_wb_relations, "Bio_Analysis_relations.csv", "text/csv",
                               width="stretch", key="wb_dl_rel_csv")
        with wb_exp_col4:
            st.download_button(t("btn_export_graph_html"), wb_html_data, "Bio_Analysis_graph.html", "text/html",
                               width="stretch", key="wb_dl_graph_html")

        if wb_html_data:
            components.html(wb_html_data, height=800, scrolling=True)

        # 6. 人工专家调换工作台 (Human-in-the-Loop)
        st.markdown("---")
        with st.container(border=True):
            st.markdown(t("wb_override_title"))
            st.caption(t("wb_override_caption"))
            tab_ent, tab_rel = st.tabs([t("wb_tab_ent_override"), t("wb_tab_rel_override")])

            with tab_ent:
                degrees = Counter()
                for r in rels:
                    degrees[r.get("source")] += r.get("weight", 1)
                    degrees[r.get("target")] += r.get("weight", 1)
                all_node_names = sorted([e.get("standard_name") for e in ents if e.get("standard_name")],
                                        key=lambda n: degrees.get(n, 0), reverse=True)

                col_e1, col_e2 = st.columns([1, 1])
                with col_e1:
                    sel_node = st.selectbox(t("wb_sel_ent_lbl"), all_node_names, key="wb_select_override_node")
                    cur_a = ent_annots.get(sel_node, {})
                    cur_type = cur_a.get("primary_type", "Uncertain")
                    cur_cat_lbl = ((ent_cat_defs.get(cur_type, {}).get("en_name") or ent_cat_defs.get(cur_type, {}).get("en_label", cur_type)) if is_en else (ent_cat_defs.get(cur_type, {}).get("name") or ent_cat_defs.get(cur_type, {}).get("label", cur_type))) or cur_type
                    st.caption(f"{t('wb_cur_cat_lbl')}: **{cur_cat_lbl}** | {t('wb_round_lbl')}: `{cur_a.get('round', 'N/A')}` | {t('wb_conf_lbl')}: `{int(cur_a.get('confidence', 0)*100)}%`")
                    if cur_a.get("reasoning"):
                        st.caption(f"{t('wb_reason_trace_lbl')}: {cur_a.get('reasoning')}")

                with col_e2:
                    cat_keys = list(ent_cat_defs.keys())
                    def_cat_idx = cat_keys.index(cur_type) if cur_type in cat_keys else 0
                    def get_cat_fmt(k):
                        cd = ent_cat_defs.get(k, {})
                        return ((cd.get("en_name") or cd.get("en_label", k)) if is_en else (cd.get("name") or cd.get("label", k))) or k
                    target_cat = st.selectbox(t("wb_target_cat_lbl"), cat_keys, index=def_cat_idx,
                                              format_func=get_cat_fmt,
                                              key="wb_select_target_cat")
                    target_sub = st.text_input(t("wb_sub_type_lbl"), value=cur_a.get("sub_type", ""), key="wb_input_target_sub")
                    target_reason = st.text_input(t("wb_reason_lbl"), value="Expert Curation" if is_en else "领域专家手动订正", key="wb_input_target_reason")

                if st.button(t("wb_btn_save_ent"), key="wb_btn_save_ent_override", width="stretch"):
                    cur_state = st.session_state.get("workbench_state", {})
                    if "entity_annotations" not in cur_state:
                        cur_state["entity_annotations"] = {}
                    cur_state["entity_annotations"][sel_node] = {
                        "primary_type": target_cat,
                        "sub_type": target_sub,
                        "tags": cur_a.get("tags", []),
                        "confidence": 1.0,
                        "reasoning": target_reason,
                        "round": "manual",
                        "status": "resolved"
                    }
                    st.session_state.workbench_state = cur_state
                    vis = GraphVisualizer()
                    wb_html_path = os.path.join(proj_dir, "workbench_graph.html")
                    vis.generate_workbench_html(
                        global_entities=ents,
                        global_relations=rels,
                        entity_annotations=cur_state.get("entity_annotations", {}),
                        relation_annotations=cur_state.get("relation_annotations", {}),
                        ontology=cur_ontology,
                        output_file=wb_html_path,
                        output_lang=st.session_state.get("ui_language", "zh")
                    )
                    save_local_vault()
                    st.toast(f"✅ {sel_node} ➔ {target_cat} ({t('wb_toast_override_done')})", icon="✅")
                    st.rerun()

            with tab_rel:
                edge_display_list = []
                edge_key_map = {}
                for r in rels:
                    s = r.get("source")
                    t_node = r.get("target")
                    rel = r.get("relation", "相关")
                    k = f"{s}|{t_node}|{rel}"
                    label = f"{s} ➔ {t_node} ({rel})"
                    edge_display_list.append(label)
                    edge_key_map[label] = (s, t_node, rel, k)

                if not edge_display_list:
                    st.info(t("wb_no_rel_edges"))
                else:
                    col_r1, col_r2 = st.columns([1, 1])
                    with col_r1:
                        sel_edge_lbl = st.selectbox(t("wb_sel_rel_lbl"), edge_display_list, key="wb_select_override_edge")
                        s, t_node, orig_rel, k = edge_key_map[sel_edge_lbl]
                        cur_ra = rel_annots.get(k, {})
                        cur_mech_cat = cur_ra.get("mechanism_category", "Association_Correlation")
                        cur_mech_lbl = ((rel_cat_defs.get(cur_mech_cat, {}).get("en_name") or rel_cat_defs.get(cur_mech_cat, {}).get("en_label", cur_mech_cat)) if is_en else (rel_cat_defs.get(cur_mech_cat, {}).get("name") or rel_cat_defs.get(cur_mech_cat, {}).get("label", cur_mech_cat))) or cur_mech_cat
                        st.caption(f"{t('wb_cur_mech_lbl')}: **{cur_mech_lbl}** | {t('wb_cur_verb_lbl')}: `{cur_ra.get('mechanism_verb', orig_rel)}`")
                        if cur_ra.get("reasoning"):
                            st.caption(f"{t('wb_reason_trace_lbl')}: {cur_ra.get('reasoning')}")

                    with col_r2:
                        rel_keys = list(rel_cat_defs.keys())
                        def_rel_idx = rel_keys.index(cur_mech_cat) if cur_mech_cat in rel_keys else 0
                        def get_rel_fmt(k):
                            rd = rel_cat_defs.get(k, {})
                            return ((rd.get("en_name") or rd.get("en_label", k)) if is_en else (rd.get("name") or rd.get("label", k))) or k
                        target_rel_cat = st.selectbox(t("wb_target_relcat_lbl"), rel_keys, index=def_rel_idx,
                                                      format_func=get_rel_fmt,
                                                      key="wb_select_target_relcat")
                        target_verb = st.text_input(t("wb_verb_lbl"), value=cur_ra.get("mechanism_verb", orig_rel), key="wb_input_target_verb")
                        target_rel_reason = st.text_input(t("wb_reason_lbl"), value="Expert Curation" if is_en else "领域专家手动订正机制", key="wb_input_target_relreason")

                    if st.button(t("wb_btn_save_rel"), key="wb_btn_save_rel_override", width="stretch"):
                        cur_state = st.session_state.get("workbench_state", {})
                        if "relation_annotations" not in cur_state:
                            cur_state["relation_annotations"] = {}
                        cur_state["relation_annotations"][k] = {
                            "mechanism_category": target_rel_cat,
                            "mechanism_verb": target_verb,
                            "confidence": 1.0,
                            "reasoning": target_rel_reason,
                            "round": "manual",
                            "status": "resolved"
                        }
                        st.session_state.workbench_state = cur_state
                        vis = GraphVisualizer()
                        wb_html_path = os.path.join(proj_dir, "workbench_graph.html")
                        vis.generate_workbench_html(
                            global_entities=ents,
                            global_relations=rels,
                            entity_annotations=cur_state.get("entity_annotations", {}),
                            relation_annotations=cur_state.get("relation_annotations", {}),
                            ontology=cur_ontology,
                            output_file=wb_html_path,
                            output_lang=st.session_state.get("ui_language", "zh")
                        )
                        save_local_vault()
                        st.toast(f"✅ {s} ➔ {t_node} : {target_verb} ({t('wb_toast_rel_override_done')})", icon="✅")
                        st.rerun()

    # ==========================================
    # 7. 第二阶段：信号/关系通路拆解与建模 (Stage 2: Pathway Decomposition & Modeling)
    # ==========================================
    st.markdown("---")
    st.markdown(t("wb_sec2_title"))
    st.caption(t("wb_sec2_caption"))

    pathways_dir = os.path.join(proj_dir, "pathways")
    manifest_path = os.path.join(pathways_dir, "pathways_manifest.json")

    # 自动从项目专属目录载入已有通路记录
    saved_pathways_data = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                saved_pathways_data = json.load(f)
        except Exception as e:
            print(f"⚠️ [Workbench] 载入已有通路快照失败: {e}")

    display_pathways = st.session_state.get("decomposed_pathways")
    if display_pathways is None and saved_pathways_data:
        display_pathways = saved_pathways_data.get("pathways", [])

    # 1. 引导输入与操作面板
    with st.container(border=True):
        st.markdown(f"**{t('wb_pw_input_label')}**")
        default_query = saved_pathways_data.get("user_query", "") if saved_pathways_data else ""
        pw_user_query = st.text_input(
            t("wb_pw_input_label"),
            value=st.session_state.get("pw_user_query_input", default_query),
            placeholder=t("wb_pw_input_placeholder"),
            key="pw_user_query_input",
            label_visibility="collapsed"
        )

        st.caption("⚙️ " + ("子功能独立控制与配置" if not is_en else "Sub-feature Configuration"))
        c_sw1, c_sw2, c_sw3, c_sw4, c_sw5 = st.columns([4, 3, 3, 3, 2])
        with c_sw1:
            pw_chk_diversity = st.checkbox(t("wb_pw_chk_diversity"), value=True, key="wb_pw_chk_diversity")
        with c_sw2:
            pw_chk_loops = st.checkbox(t("wb_pw_chk_loops"), value=True, key="wb_pw_chk_loops")
        with c_sw3:
            pw_chk_bridges = st.checkbox(t("wb_pw_chk_bridges"), value=True, key="wb_pw_chk_bridges")
        with c_sw4:
            pw_chk_subgraphs = st.checkbox(t("wb_pw_chk_subgraphs"), value=True, key="wb_pw_chk_subgraphs")
        with c_sw5:
            pw_max_count = st.slider(t("wb_pw_slider_count"), min_value=1, max_value=5, value=3, key="wb_pw_slider_count")

        btn_run_label = t("wb_pw_btn_rerun") if display_pathways else t("wb_pw_btn_start")
        btn_run_pathway = st.button(btn_run_label, type="primary", key="wb_pw_btn_start", width="stretch")

    if btn_run_pathway:
        current_key = api_key.strip() if api_key else ""
        is_local = "ollama" in str(st.session_state.get("api_provider", "")).lower()
        if not current_key and not is_local:
            st.error(t("err_missing_api_key"))
            st.stop()

        from LLM_SYS import BioBrainAgent
        llm_agent = BioBrainAgent(api_key=current_key, model=selected_model_id, base_url=base_url)

        pw_engine = PathwayDecompositionEngine(
            llm_agent=llm_agent,
            ontology=selected_preset,
            entity_annotations=ent_annots,
            relation_annotations=rel_annots,
            project_dir=proj_dir,
            canonical_papers=st.session_state.get("analyzed_files", [])
        )

        pw_p_bar = st.progress(0.0)
        pw_status = st.empty()

        def on_pw_prog(frac, msg):
            pw_p_bar.progress(frac)
            pw_status.info(f"⏳ {msg}")

        decomposed_pathways = pw_engine.run_decomposition(
            entities=ents,
            relations=rels,
            user_query=pw_user_query.strip(),
            enable_loops=pw_chk_loops,
            enable_bridges=pw_chk_bridges,
            enable_subgraphs=pw_chk_subgraphs,
            max_pathways=pw_max_count,
            enable_diversity_penalty=pw_chk_diversity,
            layout_mode="force",
            language=st.session_state.get("ui_language", "zh"),
            progress_callback=on_pw_prog
        )
        st.session_state.decomposed_pathways = decomposed_pathways
        st.toast(t("wb_pw_toast_done"), icon="🎉")
        st.rerun()

    # 2. 通路结果展示 (学术极简排版 + 可收纳子图谱)
    if display_pathways:
        sec_header = f"已解构的核心信号通路清单 ({len(display_pathways)} 条)" if not is_en else f"Decomposed Signaling Pathways ({len(display_pathways)})"
        col_hdr1, col_hdr2, col_hdr3 = st.columns([3, 1, 1])
        with col_hdr1:
            st.markdown(f"#### 📜 {sec_header}")
        with col_hdr2:
            all_pw_json = json.dumps(display_pathways, ensure_ascii=False, indent=2)
            st.download_button(
                t("wb_pw_btn_dl_all_json"),
                data=all_pw_json,
                file_name=f"Bio_Pathways_{proj_id}.json",
                mime="application/json",
                key="dl_all_pathways_json",
                width="stretch"
            )
        with col_hdr3:
            if st.button(t("wb_pw_btn_clear"), key="btn_clear_pathways_results", width="stretch"):
                st.session_state.decomposed_pathways = None
                if os.path.exists(pathways_dir):
                    import shutil
                    shutil.rmtree(pathways_dir, ignore_errors=True)
                st.toast(t("wb_pw_clear_done"), icon="🗑️")
                st.rerun()

        for idx, pw in enumerate(display_pathways):
            cid = pw.get("cluster_id", idx + 1)
            title = pw.get("title", f"通路 {cid}")
            summary = pw.get("summary", pw.get("executive_summary", ""))

            with st.container(border=True):
                # 1. 标题 (学术命名，支持 LaTeX)
                st.markdown(f"##### 🧬 {t('wb_pw_card_title').format(idx=cid, title=title)}")

                # 2. 通路机制简单介绍 (如果涉及反馈循环适当分析)
                if summary:
                    st.markdown(summary)

                if pw.get("is_fallback"):
                    st.caption(f"⚠️ {t('wb_pw_fallback_warn')}")

                # 3. 标记：反馈循环 / 跨文献通路 (只做标记，不做冗余分析)
                badges = []
                if pw.get("has_feedback_loop") or pw.get("feedback_loops"):
                    badges.append(t("wb_pw_badge_loop"))
                if pw.get("is_cross_paper"):
                    badges.append(t("wb_pw_badge_cross_paper"))
                if badges:
                    st.markdown(" ".join([f"`{b}`" for b in badges]))

                # 4. 可以收起的子图谱
                pw_nodes = pw.get("nodes", [])
                pw_rels = pw.get("relations", [])
                sub_label = f"🎨 {t('wb_pw_lbl_subgraph')} ({len(pw_nodes)} {t('lbl_nodes') if not is_en else 'Nodes'}, {len(pw_rels)} {t('lbl_relations') if not is_en else 'Edges'})"
                with st.expander(sub_label, expanded=False):
                    html_rel = pw.get("html_file", "")
                    html_abs = os.path.join(proj_dir, html_rel) if html_rel else ""
                    pw_html_data = ""
                    if html_abs and os.path.exists(html_abs):
                        try:
                            with open(html_abs, "r", encoding="utf-8") as hf:
                                pw_html_data = hf.read()
                        except Exception as e:
                            st.caption(f"⚠️ 读取子图遇到异常: {e}")

                    if pw_html_data:
                        import streamlit.components.v1 as components
                        components.html(pw_html_data, height=480, scrolling=False)

                    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
                    with col_dl1:
                        if pw_html_data:
                            st.download_button(
                                t("wb_pw_btn_dl_html"),
                                data=pw_html_data,
                                file_name=f"Pathway_{cid}_{proj_id}.html",
                                mime="text/html",
                                key=f"dl_pw_html_{cid}",
                                width="stretch"
                            )
                    with col_dl2:
                        pw_single_json = json.dumps(pw, ensure_ascii=False, indent=2)
                        st.download_button(
                            t("wb_pw_btn_dl_json"),
                            data=pw_single_json,
                            file_name=f"Pathway_{cid}_{proj_id}.json",
                            mime="application/json",
                            key=f"dl_pw_json_{cid}",
                            width="stretch"
                        )
                    with col_dl3:
                        # 🧬 降级打包为独立 .biokg 工程数据，支持载入主系统二次分析
                        from workbench_engine import PathwayDecompositionEngine
                        biokg_data = PathwayDecompositionEngine.export_pathway_as_biokg(
                            pathway_dict=pw,
                            master_entities=st.session_state.get("master_entities", []),
                            project_id=proj_id,
                            project_name=proj_name,
                            paper_registry=st.session_state.get("paper_registry", {}),
                            figure_transcriptions=st.session_state.get("figure_transcriptions", {}),
                            workbench_state=st.session_state.get("workbench_state", {})
                        )
                        biokg_json_bytes = json.dumps(biokg_data, ensure_ascii=False, indent=2).encode('utf-8')
                        clean_p_title = re.sub(r'[\\/*?:"<>|$\s]', "_", str(title).strip())[:25]
                        biokg_filename = f"BioKG_Pathway_{cid}_{clean_p_title}.biokg"
                        st.download_button(
                            t("wb_pw_btn_dl_biokg"),
                            data=biokg_json_bytes,
                            file_name=biokg_filename,
                            mime="application/json",
                            key=f"dl_pw_biokg_{cid}",
                            help=t("wb_pw_help_dl_biokg"),
                            width="stretch"
                        )

    # ==========================================
    # 8. 第三阶段：智能假说推演与科研探勘 (Stage 3: Hypothesis Deduction & Discovery Copilot)
    # ==========================================
    st.markdown("---")
    st.markdown(t("wb_sec3_title"))
    st.caption(t("wb_sec3_caption"))

    import hypothesis_engine
    importlib.reload(hypothesis_engine)
    from hypothesis_engine import HypothesisEngine
    from WebSearcher import get_searcher

    cur_db = st.session_state.get("search_database", "PubMed")
    searcher = get_searcher(cur_db)

    # 显式采用顶层 Master 全局图谱数据，彻底防御局部变量混淆
    master_ents = st.session_state.get("master_entities", [])
    master_rels = st.session_state.get("master_relations", [])

    # 1. 探勘策略控制面板
    with st.container(border=True):
        col_m1, col_m2 = st.columns([2, 3])
        with col_m1:
            mode_options = [
                ("auto", t("wb_hypo_mode_auto")),
                ("detour_shortcut", t("wb_hypo_mode_shortcut")),
                ("cluster_bridge", t("wb_hypo_mode_bridge")),
                ("feedback_homeostasis", t("wb_hypo_mode_homeostasis")),
                ("peripheral_growth", t("wb_hypo_mode_growth")),
                ("intent_exploration", t("wb_hypo_mode_intent"))
            ]
            mode_keys = [m[0] for m in mode_options]
            mode_labels = {m[0]: m[1] for m in mode_options}

            selected_hypo_mode = st.selectbox(
                t("wb_hypo_mode_select"),
                options=mode_keys,
                format_func=lambda k: mode_labels.get(k, k),
                key="wb_hypo_mode_select"
            )

        temp_engine = HypothesisEngine(
            entities=master_ents,
            relations=master_rels,
            workbench_state=st.session_state.get("workbench_state", {}),
            lang=st.session_state.get("ui_language", "zh")
        )

        with col_m2:
            if selected_hypo_mode == "auto":
                auto_info = temp_engine.detect_auto_mode()
                st.info(f"🤖 **{auto_info['label']}** — {auto_info['reason']}")
            elif selected_hypo_mode == "intent_exploration":
                hypo_intent_query = st.text_input(
                    t("wb_hypo_intent_label"),
                    value=st.session_state.get("wb_hypo_intent_input", ""),
                    placeholder=t("wb_hypo_intent_placeholder"),
                    key="wb_hypo_intent_input"
                )
            elif selected_hypo_mode == "detour_shortcut":
                st.caption("ℹ️ " + ("基于消息传递向量流场，识别高维语义极近但拓扑必须绕远路（3~4步）的节点对，推演直接跨越的隐性机制捷径" if not is_en else "Identifies semantically proximal nodes separated by multi-step topological detours, hypothesizing direct regulatory shortcuts"))
            elif selected_hypo_mode == "cluster_bridge":
                st.caption("ℹ️ " + ("识别模块化社区或独立子簇间的脆弱瓶颈与结构洞，推演建立跨簇核心冗余调控桥梁" if not is_en else "Identifies fragile bottlenecks and structural holes between communities, hypothesizing reinforced cross-cluster bridges"))
            elif selected_hypo_mode == "feedback_homeostasis":
                st.caption("ℹ️ " + ("针对多步正向级联单向驱动链，依据对称平衡与对偶补全原理，推演缺失的负反馈自抑回路以维持动态稳态" if not is_en else "Identifies unidirectional activation cascades and hypothesizes missing negative feedback loops for homeostatic balance"))
            else:
                st.caption("ℹ️ " + ("识别周界外围低度数实体，推演通过已知中介节点与核心中枢深入级联的薄层生长" if not is_en else "Identifies peripheral frontier nodes and hypothesizes cascade growth towards core network hubs"))

        btn_mine = st.button(t("wb_hypo_btn_mine"), type="primary", key="wb_hypo_btn_mine", width="stretch")

    # 2. 触发探勘推演
    if btn_mine:
        current_key = api_key.strip() if api_key else ""
        is_local = "ollama" in str(st.session_state.get("api_provider", "")).lower()
        if not current_key and not is_local:
            st.error(t("err_missing_api_key"))
            st.stop()

        from LLM_SYS import BioBrainAgent
        llm_agent = BioBrainAgent(api_key=current_key, model=selected_model_id, base_url=base_url)

        intent_text = st.session_state.get("wb_hypo_intent_input", "") if selected_hypo_mode == "intent_exploration" else ""
        active_engine = HypothesisEngine(
            entities=master_ents,
            relations=master_rels,
            workbench_state=st.session_state.get("workbench_state", {}),
            agent=llm_agent,
            searcher=searcher,
            lang=st.session_state.get("ui_language", "zh")
        )

        with st.spinner(t("msg_generating_pathways") if is_en else "🔬 正在进行拓扑寻孔与假说机制推演..."):
            cards = active_engine.mine_and_formulate(
                mode=selected_hypo_mode,
                intent=intent_text,
                top_k=5
            )
            st.session_state.hypothesis_cards = cards
            st.session_state.active_hypo_mode = selected_hypo_mode
            st.session_state.active_hypo_intent = intent_text
            st.rerun()

    # 3. 假说卡片列表渲染与交互区 (两阶段解耦，按需并发核验)
    hypo_cards = st.session_state.get("hypothesis_cards", [])
    if hypo_cards:
        st.markdown(t("wb_hypo_list_title").format(count=len(hypo_cards)))

        # 批量核验操作条
        unverified_cards = [c for c in hypo_cards if c.get("verification_status") != "verified"]
        if unverified_cards:
            col_b1, col_b2 = st.columns([3, 1])
            with col_b1:
                if st.button(t("wb_hypo_btn_verify_batch"), key="wb_hypo_btn_verify_batch", width="stretch"):
                    current_key = api_key.strip() if api_key else ""
                    is_local = "ollama" in str(st.session_state.get("api_provider", "")).lower()
                    if not current_key and not is_local:
                        st.error(t("err_missing_api_key"))
                        st.stop()

                    from LLM_SYS import BioBrainAgent
                    llm_agent = BioBrainAgent(api_key=current_key, model=selected_model_id, base_url=base_url)
                    engine = HypothesisEngine(
                        entities=master_ents,
                        relations=master_rels,
                        workbench_state=st.session_state.get("workbench_state", {}),
                        agent=llm_agent,
                        searcher=searcher,
                        lang=st.session_state.get("ui_language", "zh")
                    )
                    with st.spinner("⚡ 正在多线程并发检索 PubMed 文献并由裁判大模型终审..."):
                        updated_list = engine.verify_hypotheses_concurrent(
                            hypotheses=hypo_cards,
                            intent=st.session_state.get("active_hypo_intent", ""),
                            max_workers=4
                        )
                        st.session_state.hypothesis_cards = updated_list
                        st.toast("✅ 全部假说文献裁判终审已并发完成！", icon="🎉")
                        st.rerun()

        # 逐卡片呈现
        for idx, card in enumerate(hypo_cards):
            cid = card.get("id", f"hypo_{idx}")
            status = card.get("verification_status", "unverified")
            node_a = card.get("node_a", "")
            node_b = card.get("node_b", "")
            cat_a = card.get("category_a", "Gene_Protein")
            cat_b = card.get("category_b", "Gene_Protein")

            with st.container(border=True):
                h_col1, h_col2 = st.columns([4, 1])
                with h_col1:
                    h_title_prefix = "Hypothesis" if is_en else "假说"
                    st.markdown(f"#### {h_title_prefix} #{idx+1}：`{node_a}` $\\longleftrightarrow$ `{node_b}`")
                    st.caption(f"📌 **{card.get('hypothesis_statement', '')}**")
                with h_col2:
                    motif_lbl = card.get("motif_label", "")
                    if not motif_lbl:
                        mode_tag = card.get("mode", "")
                        if "shortcut" in mode_tag:
                            motif_lbl = t("wb_hypo_badge_shortcut")
                        elif "bridge" in mode_tag or "bottleneck" in mode_tag:
                            motif_lbl = t("wb_hypo_badge_bottleneck")
                        elif "homeostasis" in mode_tag or "feedback" in mode_tag:
                            motif_lbl = t("wb_hypo_badge_homeostasis")
                        elif "growth" in mode_tag or "frontier" in mode_tag:
                            motif_lbl = t("wb_hypo_badge_frontier")
                        else:
                            motif_lbl = t("wb_hypo_badge_intent")
                    st.markdown(f"`{motif_lbl}`")

                # 微型机制模体路径展示 (3~4节点：已知中介路径背景 + 待证实假说链路)
                circuit_edges = card.get("circuit_edges", [])
                if circuit_edges:
                    edge_pills = []
                    for ce in circuit_edges:
                        s_name = ce.get("source", "")
                        t_name = ce.get("target", "")
                        rel_name = ce.get("relation", "")
                        is_hypo = (ce.get("status") == "hypothesized")
                        if is_hypo:
                            hypo_prefix = "To Verify" if is_en else "待证实"
                            edge_pills.append(f"**`{s_name}`** ──┄*({hypo_prefix}: {rel_name})*┄──▶ **`{t_name}`**")
                        else:
                            known_prefix = "Known" if is_en else "已知"
                            edge_pills.append(f"`{s_name}` ──*({known_prefix}: {rel_name})*──▶ `{t_name}`")
                    st.info(" ｜ ".join(edge_pills))

                st.markdown(f"**{t('wb_hypo_card_rationale')}** {card.get('biological_rationale', '')}")
                st.markdown(f"**{t('wb_hypo_card_direction')}** {card.get('research_direction', '')}")

                # 阶段 B：裁判核验与结果展示
                if status != "verified":
                    if st.button(t("wb_hypo_btn_verify_single"), key=f"btn_verify_single_{cid}"):
                        current_key = api_key.strip() if api_key else ""
                        is_local = "ollama" in str(st.session_state.get("api_provider", "")).lower()
                        if not current_key and not is_local:
                            st.error(t("err_missing_api_key"))
                            st.stop()

                        from LLM_SYS import BioBrainAgent
                        llm_agent = BioBrainAgent(api_key=current_key, model=selected_model_id, base_url=base_url)
                        engine = HypothesisEngine(
                            entities=master_ents,
                            relations=master_rels,
                            workbench_state=st.session_state.get("workbench_state", {}),
                            agent=llm_agent,
                            searcher=searcher,
                            lang=st.session_state.get("ui_language", "zh")
                        )
                        with st.spinner(f"🔬 正在检索 PubMed 并审读 `{node_a}` 与 `{node_b}` 的机制文献..."):
                            updated_card = engine.verify_single_hypothesis(
                                card,
                                intent=st.session_state.get("active_hypo_intent", "")
                            )
                            st.session_state.hypothesis_cards[idx] = updated_card
                            st.rerun()
                else:
                    adj = card.get("adjudication", {}) or {}
                    verdict = adj.get("verdict", "novel_gap")
                    articles = card.get("retrieved_articles", [])

                    st.markdown("---")
                    if verdict == "known":
                        st.success(f"**{t('wb_hypo_verdict_known')}** (置信度: {adj.get('confidence', 0.9):.2f})")
                        pmid = adj.get("direct_evidence_pmid", "")
                        if pmid:
                            st.markdown(f"📄 **关键确证文献**: [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        quote = adj.get("evidence_quote", "")
                        if quote:
                            st.caption(f"📝 *原文依据*: “{quote}”")
                        st.markdown(f"⚖️ **裁判理由**: {adj.get('adjudication_rationale', '')}")

                        # 采纳确证文献证据并触发智能拓展抽取流水线入库 (占满卡片全宽)
                        if st.button(t("wb_hypo_btn_merge_return"), type="primary", key=f"btn_merge_{cid}", width="stretch"):
                            current_key = api_key.strip() if api_key else ""
                            is_local = "ollama" in str(st.session_state.get("api_provider", "")).lower()
                            if not current_key and not is_local:
                                st.error(t("err_missing_api_key"))
                                st.stop()

                            with st.spinner(f"🔬 正在基于确证文献触发实体反思抽取、关系推导与知识图谱融合流水线..."):
                                ingest_verified_hypothesis_via_pipeline(
                                    card=card,
                                    api_key=current_key,
                                    selected_model_id=selected_model_id,
                                    base_url=base_url,
                                    searcher=searcher
                                )
                            st.toast(t("wb_hypo_toast_merged_return"), icon="🎉")
                            st.rerun()

                    elif verdict == "novel_gap":
                        st.info(f"**{t('wb_hypo_verdict_novel')}** (创新度高)")
                        st.markdown(f"⚖️ **裁判结论**: {adj.get('adjudication_rationale', '')}")
                        st.markdown(f"🔭 **科研前瞻建议**: {adj.get('research_direction', '')}")
                        st.caption(t("wb_hypo_novel_notice"))

                        # 导出为专题独立子工程 (.biokg) (占满卡片全宽)
                        sub_biokg = HypothesisEngine.export_hypothesis_as_biokg(
                            hypothesis=card,
                            master_entities=master_ents,
                            master_relations=master_rels,
                            project_id=proj_id,
                            project_name=proj_name,
                            workbench_state=st.session_state.get("workbench_state", {})
                        )
                        sub_bytes = json.dumps(sub_biokg, ensure_ascii=False, indent=2).encode('utf-8')
                        clean_fname = f"BioKG_Hypothesis_{node_a}_{node_b}.biokg"
                        st.download_button(
                            t("wb_hypo_btn_dl_biokg"),
                            data=sub_bytes,
                            file_name=clean_fname,
                            mime="application/json",
                            key=f"dl_hypo_sub_{cid}",
                            width="stretch"
                        )

                    elif verdict == "cascade":
                        st.warning(f"**{t('wb_hypo_verdict_cascade')}**")
                        pmid = adj.get("direct_evidence_pmid", "")
                        if pmid:
                            st.markdown(f"📄 **级联确证文献**: [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        quote = adj.get("evidence_quote", "")
                        if quote:
                            st.caption(f"📝 *原文依据*: “{quote}”")
                        st.markdown(f"⚖️ **裁判理由**: {adj.get('adjudication_rationale', '')}")
                        st.markdown(f"🧭 **级联推演方向**: {adj.get('research_direction', '')}")

                        # 采纳级联证据并触发智能拓展抽取流水线入库 (智能桥接中介并合并入主图谱，占满全宽)
                        if st.button(t("wb_hypo_btn_merge_cascade"), type="primary", key=f"btn_merge_cascade_{cid}", width="stretch"):
                            current_key = api_key.strip() if api_key else ""
                            is_local = "ollama" in str(st.session_state.get("api_provider", "")).lower()
                            if not current_key and not is_local:
                                st.error(t("err_missing_api_key"))
                                st.stop()

                            with st.spinner(f"🔬 正在基于级联文献抽取中介实体与调控关系，执行智能桥接合并流水线..."):
                                ingest_verified_hypothesis_via_pipeline(
                                    card=card,
                                    api_key=current_key,
                                    selected_model_id=selected_model_id,
                                    base_url=base_url,
                                    searcher=searcher
                                )
                            st.toast(t("wb_hypo_toast_merged_cascade"), icon="🎉")
                            st.rerun()

                        # 同时支持导出为级联机制独立子工程 (.biokg) (占满全宽)
                        sub_biokg = HypothesisEngine.export_hypothesis_as_biokg(
                            hypothesis=card,
                            master_entities=master_ents,
                            master_relations=master_rels,
                            project_id=proj_id,
                            project_name=proj_name,
                            workbench_state=st.session_state.get("workbench_state", {})
                        )
                        sub_bytes = json.dumps(sub_biokg, ensure_ascii=False, indent=2).encode('utf-8')
                        clean_fname = f"BioKG_Cascade_{node_a}_{node_b}.biokg"
                        st.download_button(
                            t("wb_hypo_btn_dl_biokg"),
                            data=sub_bytes,
                            file_name=clean_fname,
                            mime="application/json",
                            key=f"dl_hypo_sub_cascade_{cid}",
                            width="stretch"
                        )
                    else:
                        st.info(f"**{t('wb_hypo_verdict_inconclusive')}**")
                        st.markdown(f"⚖️ **裁判理由**: {adj.get('adjudication_rationale', '')}")

                    # 文献摘要展开面板
                    if articles:
                        with st.expander(t("wb_hypo_articles_title").format(count=len(articles)), expanded=False):
                            for a_idx, art in enumerate(articles):
                                a_pmid = art.get("pmid", "")
                                a_title = art.get("title", "")
                                a_year = art.get("year", "")
                                a_journal = art.get("journal", "")
                                a_abstract = art.get("abstract", "")
                                st.markdown(f"**[{a_idx+1}] {a_title}** ({a_year}, {a_journal}) [PMID: {a_pmid}](https://pubmed.ncbi.nlm.nih.gov/{a_pmid}/)")
                                st.caption(a_abstract[:300] + ("..." if len(a_abstract) > 300 else ""))

    # 9. 底部返回按钮
    st.markdown("---")
    cb_1, cb_2, cb_3 = st.columns([1, 2, 1])
    with cb_2:
        if st.button(t("wb_btn_bottom_return"), key="btn_wb_bottom_return", width="stretch"):
            st.session_state.current_view = "main"
            st.rerun()
# ==========================================
# 主界面：全新的左右双栏平衡布局
if st.session_state.get("current_view", "main") == "workbench":
    render_analysis_workbench(api_key, selected_model_id, base_url)
    st.stop()

# ==========================================
left_col, right_col = st.columns([1, 1], gap="large")

# 初始化全局触发变量
start_button = False
uploaded_file = None

# --- 左侧控制栏 ---
with left_col:
    st.subheader(t("col1_title"))
    uploaded_file = st.file_uploader(t("upload_pdf"), type=["pdf"])

    if uploaded_file:
        # 🛡️ 核心优化 1：文件与页数双重缓存锁
        if "tmp_path" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state["tmp_path"] = tmp.name
                st.session_state["file_name"] = uploaded_file.name

            # 💡 性能修复：PdfReader 必须放在 if 内部！
            reader = PdfReader(st.session_state["tmp_path"])
            st.session_state["total_pages"] = len(reader.pages)

        total_pages = st.session_state["total_pages"]
        tmp_path = st.session_state["tmp_path"]
        st.markdown("---")
        st.info(t("pdf_read_success").format(pages=total_pages))
        st.divider()

    # ==========================================
    # 🌟 零基础冷启动 (Cold Start) 探索模块
    # ==========================================
    if not st.session_state.master_entities and not st.session_state.master_relations and not uploaded_file:
        st.markdown(t("cold_start_title"))
        st.caption(t("cold_start_caption"))

        user_topic = st.text_input(t("cold_start_placeholder"), key="cold_start_topic")

        if st.button(t("btn_ai_search"), type="primary", use_container_width=True):
            current_api_key = api_key.strip()
            if not current_api_key:
                st.error(t("err_no_api_key"))
            elif not user_topic.strip():
                st.warning(t("warn_no_topic"))
            else:
                from LLM_SYS import BioBrainAgent
                from WebSearcher import get_searcher

                current_db = st.session_state.get("search_database", "PubMed (生物医学权威)")
                searcher = get_searcher(current_db)
                agent = BioBrainAgent(api_key=current_api_key, model=selected_model_id, base_url=base_url)

                with st.spinner(t("msg_generating_query")):
                    query = agent.generate_topic_query(user_topic)

                if "[INVALID_TOPIC]" in query:
                    st.error(t("err_invalid_topic"))
                else:
                    st.info(t("msg_generated_query").format(query=query))

                    db_name = "PubMed" if "PubMed" in current_db else "Semantic Scholar"
                    with st.spinner(t("msg_fetching_papers").format(db=db_name)):
                        results = searcher.search_articles(query, max_results=10)

                    if not results:
                        st.warning(t("warn_no_papers"))
                    else:
                        st.success(t("msg_fetch_success").format(count=len(results)))

                        batch_size = 3
                        success = False
                        valid_pmids = []

                        # 🔄 开启滑动窗口排查
                        for i in range(0, len(results), batch_size):
                            chunk = results[i: i + batch_size]
                            curr_start = i + 1
                            curr_end = min(i + batch_size, len(results))

                            with st.spinner(t("msg_validating_batch").format(start=curr_start, end=curr_end)):
                                combined_abstracts = ""
                                current_pmids = []

                                for res in chunk:
                                    pmid = res.get('pmid')
                                    abs_text = searcher.fetch_abstract(pmid)
                                    if len(abs_text) > 50:
                                        combined_abstracts += f"\n\n--- PMID: {pmid} ---\n{abs_text}"
                                        current_pmids.append(pmid)

                                if not combined_abstracts:
                                    continue
                                current_ui_lang = st.session_state.get("language", "zh")
                                eval_result = agent.evaluate_abstracts_relevance(user_topic, combined_abstracts,output_lang=current_ui_lang)

                            if eval_result.get("is_relevant"):
                                st.success(t("msg_batch_pass").format(start=curr_start, end=curr_end))
                                st.info(t("msg_ai_review").format(reason=eval_result.get('reason')))
                                valid_pmids = current_pmids
                                success = True
                                break
                            else:
                                st.warning(t("msg_batch_fail").format(start=curr_start, end=curr_end,
                                                                      reason=eval_result.get('reason')))

                        # ==========================================
                        # 循环结束后的最终处理
                        # ==========================================
                        if success and valid_pmids:
                            progress_text = t("msg_extracting_graph")
                            my_bar = st.progress(0, text=progress_text)

                            total_docs = len(valid_pmids)
                            for idx, pmid in enumerate(valid_pmids):
                                my_bar.progress((idx) / total_docs,
                                                text=t("msg_parsing_paper_progress").format(idx=idx + 1,
                                                                                            total=total_docs,
                                                                                            pmid=pmid))
                                abs_text = searcher.fetch_abstract(pmid)

                                current_use_ref = use_reflection if 'use_reflection' in locals() or 'use_reflection' in globals() else True
                                current_lang = entity_language if 'entity_language' in locals() or 'entity_language' in globals() else "English"

                                new_entities = agent.extract_entities_with_reflection(abs_text,
                                                                                      use_reflection=current_use_ref,
                                                                                      entity_lang=current_lang)
                                pmid_tag = f"PubMed:{pmid}"
                                v_hash = f"pmid:{pmid}"
                                register_or_update_paper(
                                    get_current_project_id(),
                                    doc_hash=v_hash,
                                    display_name=pmid_tag,
                                    aliases=[pmid_tag, str(pmid)],
                                    pmid=str(pmid),
                                    source_type="online_pmid"
                                )
                                for ent in new_entities:
                                    ent["doc_source"] = pmid_tag
                                    ent["doc_hash"] = v_hash
                                    st.session_state.master_entities.append(ent)

                                if pmid_tag not in st.session_state.analyzed_files:
                                    st.session_state.analyzed_files.append(pmid_tag)
                                st.session_state.analyzed_files = canonicalize_project_papers(get_current_project_id(), st.session_state.analyzed_files)

                                new_relations = agent.extract_relations(abs_text, st.session_state.master_entities)
                                for rel in new_relations:
                                    rel["doc_source"] = pmid_tag
                                    rel["doc_hash"] = v_hash
                                    rel["weight"] = 1
                                    st.session_state.master_relations.append(rel)

                            st.session_state.master_relations = consolidate_homogeneous_relations(
                                st.session_state.master_relations
                            )
                            my_bar.progress(1.0, text=t("msg_graph_done"))
                            st.toast(t("toast_start_success"), icon="🎉")
                            st.session_state.show_results = True
                            redraw_and_update()
                            save_local_vault()
                            st.rerun()
                        elif not success:
                            st.error(t("err_all_validation_fail").format(count=len(results)))

    st.subheader(t("extract_settings_title"))
    is_summary_only = st.toggle(t("toggle_summary_only"), value=st.session_state.get("is_summary_only", False), key="is_summary_only", persist_state="session", on_change=save_config)
    use_reflection = st.toggle(t("toggle_reflection"), value=st.session_state.get("use_reflection", True), key="use_reflection", persist_state="session", on_change=save_config)
    append_mode = st.toggle(t("toggle_append_mode"), value=st.session_state.get("append_mode", True), key="append_mode", persist_state="session", on_change=save_config)
    enable_relation_grounding = st.toggle(
        t("toggle_relation_grounding"),
        value=st.session_state.get("enable_relation_grounding", True),
        key="enable_relation_grounding",
        help=t("toggle_relation_grounding_help"),
        persist_state="session",
        on_change=save_config
    )
    if "concurrency_workers" not in st.session_state:
        st.session_state["concurrency_workers"] = 4
    concurrency_workers = st.slider(
        t("slider_concurrency"),
        min_value=1,
        max_value=8,
        value=st.session_state.get("concurrency_workers", 4),
        key="concurrency_workers",
        help=t("slider_concurrency_help"),
        persist_state="session",
        on_change=save_config
    )

    st.markdown(t("entity_lang_title"))

    # 💡 内部固定的只读 Key（永不随语言切换而改变，彻底杜绝 KeyError 和缓存越界）
    lang_internal_keys = ["opt_original", "opt_zh", "opt_en"]

    # 💡 显示映射函数：只有在渲染到屏幕的那一瞬间，才把 Key 变成中/英文
    def get_lang_display(key):
        if key == "opt_original": return t("entity_lang_opt_original")
        if key == "opt_zh": return t("entity_lang_opt_zh")
        if key == "opt_en": return t("entity_lang_opt_en")
        return key

    if st.session_state.get("entity_language_ui") not in lang_internal_keys:
        st.session_state["entity_language_ui"] = "opt_original"

    selected_lang_key = st.radio(
        t("entity_lang_label"),
        options=lang_internal_keys,  # 缓存里存的永远是 "opt_xxx"
        format_func=get_lang_display,  # 界面上显示的是翻译后的文字
        key="entity_language_ui",  # 彻底安全了！
        horizontal=True
    )

    # 将内部 Key 映射回系统最初需要的硬编码指令（确保大模型行为不突变）
    backend_lang_map = {
        "opt_original": "关闭 (保持原文语言)",
        "opt_zh": "中文 (强制翻译为中文)",
        "opt_en": "English (强制翻译为英文)"
    }
    entity_language = backend_lang_map.get(selected_lang_key, "关闭 (保持原文语言)")
    st.session_state["entity_language"] = entity_language

    # 📷 机制图/通路图解析策略 (深度融合，置于提取模式设置下，保证未上传文件时也能渲染和配置)
    st.markdown(f"**{t('vision_strategy_title')}**")
    v_strategy_keys = ["opt_off", "opt_match", "opt_all"]
    v_strategy_labels = {
        "opt_off": t("vision_strategy_opt_off"),
        "opt_match": t("vision_strategy_opt_match"),
        "opt_all": t("vision_strategy_opt_all")
    }
    cur_v_strat = st.session_state.get("vision_strategy", "opt_off")
    if cur_v_strat not in v_strategy_keys:
        if "关" in str(cur_v_strat) or "Off" in str(cur_v_strat):
            cur_v_strat = "opt_off"
        elif "匹配" in str(cur_v_strat) or "Match" in str(cur_v_strat):
            cur_v_strat = "opt_match"
        elif "全" in str(cur_v_strat) or "All" in str(cur_v_strat):
            cur_v_strat = "opt_all"
        else:
            cur_v_strat = "opt_off"
        st.session_state["vision_strategy"] = cur_v_strat
    cur_v_idx = v_strategy_keys.index(cur_v_strat) if cur_v_strat in v_strategy_keys else 0

    col_v1, col_v2 = st.columns([3, 2])
    with col_v1:
        selected_v_strat = st.radio(
            t("vision_strategy_title"),
            options=v_strategy_keys,
            format_func=lambda k, m=v_strategy_labels: m.get(k, k),
            key="vision_strategy",
            horizontal=True,
            label_visibility="collapsed",
            on_change=save_config
        )

    vision_keyword_val = st.session_state.get("vision_match_keyword", "")
    if selected_v_strat == "opt_match":
        with col_v2:
            vision_keyword_val = st.text_input(
                t("vision_keyword_label"),
                key="vision_match_keyword",
                placeholder=t("vision_keyword_placeholder"),
                label_visibility="collapsed",
                on_change=save_config
            )

    final_vision_mode = "off" if selected_v_strat == "opt_off" else ("match" if selected_v_strat == "opt_match" else "all")

    if append_mode and len(st.session_state.master_entities) > 0:
        st.info(t("append_info_ready").format(count=len(st.session_state.master_entities)))
    elif not append_mode and len(st.session_state.master_entities) > 0:
        st.warning(t("append_warn_off"))

    if uploaded_file:
        default_end = min(2, total_pages) if is_summary_only else total_pages

        st.markdown(t("parse_range_title"))
        col_inputs1, col_inputs2 = st.columns(2)
        with col_inputs1:
            start_page = st.number_input(t("start_page"), min_value=1, max_value=total_pages, value=1)
        with col_inputs2:
            safe_default_end = max(start_page, default_end)
            end_page = st.number_input(t("end_page"), min_value=start_page, max_value=total_pages,
                                       value=safe_default_end)

        if is_summary_only and (end_page - start_page > 2):
            st.warning(t("warn_summary_range"))

        st.divider()
        start_button = st.button(t("btn_start_parsing"), use_container_width=True, type="primary")

# ==========================================
# 📁 后置侧边栏扩展区
# ==========================================
if uploaded_file and start_button:
    file_bytes = uploaded_file.getvalue()
    effective_disp = save_analyzed_paper(uploaded_file.name, file_bytes)
    cur_pid = get_current_project_id()
    if append_mode:
        cur_files = st.session_state.get("analyzed_files", [])
        st.session_state.analyzed_files = canonicalize_project_papers(cur_pid, cur_files + [effective_disp])
    else:
        # 非追加模式：仅将当前会话分析列表重置为该文件，不删除磁盘上的历史物理文件
        st.session_state.analyzed_files = [effective_disp]

# (已由 render_sidebar_history_docs 统一前置持久渲染)

# --- 右侧高清预览栏 ---
with right_col:
    st.subheader(t("preview_col_title"))
    if uploaded_file:
        def get_page_image_bytes(page_idx):
            doc = fitz.open(tmp_path)
            page = doc.load_page(page_idx)
            zoom_matrix = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=zoom_matrix)
            img_bytes = pix.tobytes("png")
            doc.close()
            return img_bytes


        with st.expander(t("preview_expander"), expanded=True):
            col_preview1, col_preview2 = st.columns(2)

            with col_preview1:
                st.markdown(t("preview_start_page").format(page=start_page))
                try:
                    img_start = get_page_image_bytes(start_page - 1)
                    st.image(img_start, use_container_width=True)
                except Exception as e:
                    st.error(t("preview_render_fail").format(e=e))

            with col_preview2:
                if start_page != end_page:
                    st.markdown(t("preview_end_page").format(page=end_page))
                    try:
                        img_end = get_page_image_bytes(end_page - 1)
                        st.image(img_end, use_container_width=True)
                    except Exception as e:
                        st.error(t("preview_render_fail").format(e=e))
                else:
                    st.markdown("**🔴 " + t("end_page") + "**")
                    st.info(t("preview_same_page"))
    else:
        st.info(t("preview_no_pdf"))

    # ==========================================
    # 💾 记忆库与工程管理区 (右栏底部)
    # ==========================================
    st.markdown("---")
    st.subheader(t("memory_mgr_title"))

    # 🏷️ 项目课题自由命名输入框（作为存储属性值，不直接用作文件名或物理目录）
    curr_proj_name = get_current_project_name()
    # 保证在 widget 实例化前与当前工程名称安全同步，避免在实例化后修改抛出 StreamlitAPIException
    st.session_state["project_name_input_widget"] = curr_proj_name

    def on_project_name_change():
        val = str(st.session_state.get("project_name_input_widget", "")).strip()
        if val and val != get_current_project_name():
            st.session_state.current_project_name = val
            save_local_vault()

    st.text_input(
        t("project_name_label"),
        key="project_name_input_widget",
        on_change=on_project_name_change,
        help=t("project_name_help")
    )

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button(t("btn_new_project"), use_container_width=True):
            st.session_state.show_new_confirm = not st.session_state.get("show_new_confirm", False)

    with col_btn2:
        export_placeholder = st.empty()
        with export_placeholder:
            if st.button(t("btn_export_project"), use_container_width=True, key="fake_export_btn"):
                st.toast(t("toast_export_empty"), icon="🈳")

    if st.session_state.get("show_new_confirm", False):
        st.info(t("info_new_project_confirm"))
        col_y, col_n = st.columns(2)
        with col_y:
            if st.button(t("btn_confirm_new_project"), use_container_width=True):
                # 🛡️ 步骤 1：新建空白工程前，先自动固化当前工程，避免未保存修改丢失
                save_local_vault()

                st.session_state.master_entities = []
                st.session_state.master_relations = []
                st.session_state.analyzed_files = []
                st.session_state.html_data = ""
                st.session_state.show_results = False
                html_file = ".bio_knowledge_graph.html"
                if os.path.exists(html_file):
                    os.remove(html_file)

                # 生成全新项目 ID 与项目默认显示名，历史工程在本地目录完整安全隔离保留
                new_pid = generate_project_id()
                new_pname = datetime.now().strftime("课题_%Y%m%d_%H%M%S")
                st.session_state.current_project_id = new_pid
                st.session_state.current_project_name = new_pname

                save_local_vault()
                st.session_state.project_loaded_success = False
                st.session_state.show_new_confirm = False
                st.rerun()

        with col_n:
            if st.button(t("btn_cancel"), use_container_width=True):
                st.session_state.show_new_confirm = False
                st.rerun()

    if "project_uploader_key" not in st.session_state:
        st.session_state.project_uploader_key = "project_uploader_init"

    uploaded_project = st.file_uploader(
        t("upload_project"),
        type=["biokg", "json"],
        key=st.session_state.project_uploader_key
    )

    if uploaded_project is not None:
        try:
            loaded_data = json.load(uploaded_project)

            # 兼容老版记忆库 (Auto-Migration)：若缺失 version 或 project_id，自动升格补齐
            file_stem = os.path.splitext(uploaded_project.name)[0]
            if "version" not in loaded_data:
                loaded_data["version"] = "1.0"
            if not loaded_data.get("project_id"):
                loaded_data["project_id"] = generate_project_id()
            if not loaded_data.get("project_name"):
                loaded_data["project_name"] = file_stem

            stored_pid = loaded_data.get("project_id", "")
            stored_pname = loaded_data.get("project_name", file_stem)

            if st.button(t("btn_confirm_load"), type="primary", use_container_width=True, help=t("help_confirm_load")):
                cur_pid = get_current_project_id()
                # 🛡️ 步骤 1：若载入的是不同工程（如导出的独立通路子工程），先自动保存当前活动工程，确保父工程绝对安全
                if stored_pid and stored_pid != cur_pid:
                    save_local_vault()
                    matched_pid = stored_pid
                    st.session_state.current_project_id = matched_pid
                else:
                    matched_pid = cur_pid

                # 🔄 智能文献映射与哈希自愈
                loaded_data = reconcile_loaded_vault(matched_pid, loaded_data)

                st.session_state.current_project_name = stored_pname

                st.session_state.master_entities = loaded_data.get("entities", [])
                st.session_state.master_relations = loaded_data.get("relations", [])
                st.session_state.figure_transcriptions = loaded_data.get("figure_transcriptions", {})
                st.session_state.workbench_state = loaded_data.get("workbench_state", {})
                loaded_files = loaded_data.get("analyzed_files", [])

                # 文件夹对齐与老文献挂载/合并 (严禁同目录自拷贝克隆旧文件)
                target_papers_dir = get_project_papers_dir_by_id(matched_pid)
                os.makedirs(target_papers_dir, exist_ok=True)
                for fname in loaded_files:
                    ident = resolve_paper_identity(matched_pid, fname)
                    if ident and ident.get("physical_file"):
                        local_phys = os.path.join(target_papers_dir, ident["physical_file"])
                        if os.path.exists(local_phys):
                            # 本地已存在该文献物理文件，绝不自拷贝克隆出旧名字副本
                            continue

                    target_file_path = os.path.join(target_papers_dir, get_physical_pdf_filename(fname))
                    if not os.path.exists(target_file_path):
                        found_path = find_project_paper(fname, proj_id=matched_pid)
                        if found_path and os.path.exists(found_path):
                            # 严格限制：仅当源文件位于外部其他工程目录时才拷贝，杜绝同目录内自我克隆
                            if os.path.dirname(os.path.abspath(found_path)) != os.path.abspath(target_papers_dir):
                                try:
                                    shutil.copy2(found_path, target_file_path)
                                except Exception:
                                    pass

                existing_in_target = [get_clean_display_title(f) for f in os.listdir(target_papers_dir) if os.path.isfile(os.path.join(target_papers_dir, f))] if os.path.exists(target_papers_dir) else []
                merged_files = canonicalize_project_papers(matched_pid, loaded_files + existing_in_target)
                st.session_state.analyzed_files = merged_files

                html_file = ".bio_knowledge_graph.html"
                if st.session_state.master_entities:
                    from back_logic import GraphVisualizer
                    visualizer = GraphVisualizer()
                    if os.path.exists(html_file):
                        os.remove(html_file)
                    visualizer.generate_html(
                        st.session_state.master_entities,
                        st.session_state.master_relations,
                        output_file=html_file,
                        output_lang=st.session_state.get("ui_language", "zh")
                    )
                    if os.path.exists(html_file):
                        with open(html_file, "r", encoding="utf-8") as f:
                            st.session_state.html_data = f.read()
                    st.session_state.show_results = True
                else:
                    st.session_state.show_results = False
                    st.session_state.html_data = ""
                    if os.path.exists(html_file):
                        os.remove(html_file)

                # 原地以新版本标准格式持久化
                save_local_vault()
                st.session_state.project_uploader_key = f"project_uploader_{uuid.uuid4().hex}"
                st.toast(t("toast_load_success").format(name=stored_pname), icon="🔄")
                st.rerun()
        except Exception as e:
            st.error(t("err_load_project").format(e=e))

    # ==========================================
    # 🗄️ 磁盘治理与工程归档管理区 (位于记忆库管理工具正下方)
    # ==========================================
    st.markdown("---")
    st.markdown(f"##### {t('archive_mgr_title')}")

    # 1. 清理当前工程孤儿文献 (Orphaned Files Cleanup)
    cur_papers_dir = get_current_papers_dir()
    cur_pid = get_current_project_id()
    all_cur_files = [f for f in os.listdir(cur_papers_dir) if os.path.isfile(os.path.join(cur_papers_dir, f))] if os.path.exists(cur_papers_dir) else []

    referenced_sources = set()
    for ent in st.session_state.get("master_entities", []):
        ds = ent.get("doc_source", "")
        if ds:
            referenced_sources.add(str(ds).strip())
    for rel in st.session_state.get("master_relations", []):
        ds = rel.get("doc_source", "")
        if ds:
            referenced_sources.add(str(ds).strip())

    def is_file_referenced(fname):
        ident = resolve_paper_identity(cur_pid, fname)
        disp = ident.get("display_name") if ident else get_clean_display_title(fname)
        for ref in referenced_sources:
            if fname in ref or ref in fname or (disp and (disp in ref or ref in disp)):
                return True
        return False

    orphan_files = [f for f in all_cur_files if not is_file_referenced(f)]

    col_clean_btn, col_clean_info = st.columns([3, 2])
    with col_clean_btn:
        if st.button(t("btn_clean_orphans"), use_container_width=True, help=t("help_clean_orphans")):
            if orphan_files:
                deleted_cnt = 0
                for of in orphan_files:
                    try:
                        f_del_path = os.path.join(cur_papers_dir, of)
                        if os.path.exists(f_del_path):
                            os.remove(f_del_path)
                        deleted_cnt += 1
                        if of in st.session_state.analyzed_files:
                            st.session_state.analyzed_files.remove(of)
                    except Exception:
                        pass
                save_local_vault()
                st.toast(t("toast_orphans_cleaned").format(count=deleted_cnt), icon="🧹")
                st.rerun()
            else:
                st.toast(t("toast_no_orphans"), icon="✨")
    with col_clean_info:
        if orphan_files:
            st.caption(t("orphan_found_count").format(count=len(orphan_files)))
        else:
            st.caption(t("orphan_none"))

    # 2. 本地历史工程库管理 (Manage & Delete Inactive Projects)
    with st.expander(t("manage_projects_expander"), expanded=False):
        if os.path.exists(PROJECTS_ROOT_DIR):
            all_projects = [d for d in os.listdir(PROJECTS_ROOT_DIR) if os.path.isdir(os.path.join(PROJECTS_ROOT_DIR, d))]
        else:
            all_projects = []

        if not all_projects:
            st.info(t("no_saved_projects"))
        else:
            cur_pid = get_current_project_id()
            for proj_id in sorted(all_projects):
                p_path = os.path.join(PROJECTS_ROOT_DIR, proj_id)
                p_papers_dir = os.path.join(p_path, "papers")
                p_papers = [f for f in os.listdir(p_papers_dir) if os.path.isfile(os.path.join(p_papers_dir, f))] if os.path.exists(p_papers_dir) else []

                # 读取存储的课题显示名称（存储值）
                p_display_name = proj_id
                pkg_file = os.path.join(p_path, "project.biokg")
                if os.path.exists(pkg_file):
                    try:
                        with open(pkg_file, "r", encoding="utf-8") as f:
                            p_meta = json.load(f)
                        p_display_name = p_meta.get("project_name", proj_id)
                    except Exception:
                        pass

                total_size = 0
                for dirpath, _, filenames in os.walk(p_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            try:
                                total_size += os.path.getsize(fp)
                            except Exception:
                                pass
                size_mb = total_size / (1024 * 1024)

                is_active = (proj_id == cur_pid)
                p_missing = get_project_missing_papers(proj_id)
                col_pname, col_pact = st.columns([7, 3])
                with col_pname:
                    active_badge = f" `[{t('badge_current')}]`" if is_active else ""
                    missing_badge = f" <span style='color:#e67e22;font-weight:bold;'>⚠️ [{t('badge_missing_docs').format(count=len(p_missing))}]</span>" if p_missing else ""
                    st.markdown(f"**📁 {p_display_name}**{active_badge}{missing_badge}<br><small style='color:gray;'>ID: <code>{proj_id}</code> | {len(p_papers)} {t('unit_papers')} | {size_mb:.2f} MB</small>", unsafe_allow_html=True)
                with col_pact:
                    if not is_active:
                        c_sw, c_del = st.columns(2)
                        with c_sw:
                            if st.button("🚀", key=f"switch_proj_{proj_id}", help=t("btn_switch_proj_help").format(name=p_display_name)):
                                if switch_to_project(proj_id):
                                    if p_missing:
                                        st.toast(f"{t('toast_proj_switched').format(name=p_display_name)} (⚠️ {t('badge_missing_docs').format(count=len(p_missing))})", icon="🚀")
                                    else:
                                        st.toast(t("toast_proj_switched").format(name=p_display_name), icon="🚀")
                                    st.rerun()
                        with c_del:
                            if st.button("🗑️", key=f"del_proj_{proj_id}", help=t("btn_del_proj_help").format(name=p_display_name)):
                                st.session_state[f"confirm_del_proj_{proj_id}"] = True
                    else:
                        st.write("")

                if st.session_state.get(f"confirm_del_proj_{proj_id}", False):
                    st.warning(t("warn_del_proj_confirm").format(name=p_display_name))
                    col_del_yes, col_del_no = st.columns(2)
                    with col_del_yes:
                        if st.button(t("btn_confirm_del"), key=f"btn_del_yes_{proj_id}", use_container_width=True):
                            try:
                                shutil.rmtree(p_path)
                                st.toast(t("toast_proj_deleted").format(name=p_display_name), icon="🗑️")
                                st.session_state[f"confirm_del_proj_{proj_id}"] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {e}")
                    with col_del_no:
                        if st.button(t("btn_cancel"), key=f"btn_del_no_{proj_id}", use_container_width=True):
                            st.session_state[f"confirm_del_proj_{proj_id}"] = False
                            st.rerun()

    # 3. 当前工程缺失文献快捷补全 (紧随工程库管理下方)
    cur_missing = get_project_missing_papers(get_current_project_id())
    if cur_missing:
        st.markdown("---")
        st.markdown(f"##### {t('missing_docs_section_title')}")
        st.warning(t("missing_docs_alert").format(name=get_current_project_name(), count=len(cur_missing)))
        with st.container():
            col_sel, col_up = st.columns([1, 1])
            with col_sel:
                target_missing_file = st.selectbox(
                    t("lbl_select_missing_doc"),
                    cur_missing,
                    key="select_missing_doc_replenish"
                )
            with col_up:
                replenish_nonce = st.session_state.get("replenish_nonce", 0)
                replenish_uploaded = st.file_uploader(
                    t("lbl_upload_replenish_doc"),
                    type=["pdf"],
                    key=f"replenish_uploader_{target_missing_file}_{replenish_nonce}"
                )
            if replenish_uploaded is not None:
                save_replenished_paper(target_missing_file, replenish_uploaded.getvalue())
                st.session_state["replenish_nonce"] = replenish_nonce + 1
                st.toast(t("toast_replenish_success").format(fname=target_missing_file), icon="🎉")
                st.rerun()

# ==========================================
# 核心引擎触发与结果渲染区（全览画布排版）
# ==========================================
if uploaded_file and start_button:
    if not api_key:
        st.error(t("err_missing_api_key"))
    else:
        # 渲染在双栏下方，让最终的巨幅知识图谱能占满整张屏幕，视觉极其震撼
        st.markdown("---")

        # 🛠️ 核心 UI 修复：将 try 提到最外层，内部所有的 spinner 扁平化独立，避免重叠！
        try:
            # 实例化 Pipeline 并传入动态模型
            pipeline = BioGraphPipeline(api_key=api_key, model=selected_model_id,base_url=base_url)

            # 🟢 阶段一：独立的解析转圈
            with st.spinner(t("msg_parsing_pages").format(start=start_page, end=end_page)):

                # ✨ 新增 1：在前端画布准备好一个进度条组件
                ui_progress_bar = st.progress(0.0, text=t("msg_init_engine"))

                # ✨ 新增 2：定义一个“监听器”函数 (结合了进度条和气泡)
                def update_ui_progress(current, total, message):
                    # 计算安全百分比 (0.0 到 1.0 之间)
                    percent = min(max(current / total, 0.0), 1.0)
                    # 更新进度条
                    ui_progress_bar.progress(percent, text=message)
                    # 弹出右上角气泡 (模拟后端日志流)
                    st.toast(message, icon="⏳")

                # 1. 跑当前这篇文献，拿到新数据
                new_entities, new_relations = pipeline.run(
                    tmp_path,
                    start_page=start_page - 1,
                    end_page=end_page,
                    is_summary_only=is_summary_only,
                    use_reflection=use_reflection,
                    source_name=uploaded_file.name,
                    entity_lang=entity_language,
                    progress_callback=update_ui_progress,
                    output_lang=st.session_state.get("ui_language", "zh"),
                    concurrency=concurrency_workers,
                    vision_strategy=final_vision_mode if 'final_vision_mode' in locals() else "off",
                    vision_keyword=vision_keyword_val if 'vision_keyword_val' in locals() else "",
                    vision_model=resolved_vision_model_id if 'resolved_vision_model_id' in locals() else None,
                    enable_relation_grounding=enable_relation_grounding
                )
                if hasattr(pipeline, "last_figure_transcriptions") and pipeline.last_figure_transcriptions:
                    cur_trans = st.session_state.setdefault("figure_transcriptions", {})
                    cur_trans.update(pipeline.last_figure_transcriptions)

            # 🟢 阶段二：独立的融合转圈 (上一个转圈已经销毁)
            if append_mode and len(st.session_state.master_entities) > 0:
                with st.spinner(t("msg_aligning_entities")):
                    # 1. 呼叫大模型裁判，获取同义词映射表 (如: {"TP53": "p53"})
                    alignment_map = pipeline.agent.align_global_entities(
                        st.session_state.master_entities,
                        new_entities
                    )

                    if alignment_map:
                        st.success(t("msg_found_synonyms").format(count=len(alignment_map)))

                    # ====================================================
                    # 2. 遍历新实体，执行融合手术 (双重拦截：精确匹配 + 语义对齐)
                    # ====================================================
                    aligned_new_entities = []
                    master_names = {ent["standard_name"]: ent for ent in st.session_state.master_entities}

                    for new_ent in new_entities:
                        new_name = new_ent.get("standard_name")
                        new_source = new_ent.get("doc_source", "")

                        # 🛡️ 拦截网 1：如果名字一模一样（代码级精准匹配）
                        if new_name in master_names:
                            master_ent = master_names[new_name]
                            aliases = master_ent.setdefault("aliases", [])
                            for new_alias in new_ent.get("aliases", []):
                                if new_alias not in aliases and new_alias != new_name:
                                    aliases.append(new_alias)
                            old_source = master_ent.get("doc_source", "")
                            if new_source and new_source not in old_source:
                                master_ent["doc_source"] = f"{old_source} | {new_source}"

                        # 🛡️ 拦截网 2：名字不一样，但大模型裁判认为它们是同一个东西
                        elif alignment_map and new_name in alignment_map:
                            master_name = alignment_map[new_name]
                            if master_name in master_names:
                                master_ent = master_names[master_name]
                                aliases = master_ent.setdefault("aliases", [])
                                if new_name not in aliases:
                                    aliases.append(new_name)
                                for new_alias in new_ent.get("aliases", []):
                                    if new_alias not in aliases and new_alias != master_name:
                                        aliases.append(new_alias)
                                old_source = master_ent.get("doc_source", "")
                                if new_source and new_source not in old_source:
                                    master_ent["doc_source"] = f"{old_source} | {new_source}"
                            else:
                                aligned_new_entities.append(new_ent)
                        else:
                            aligned_new_entities.append(new_ent)

                    # 3. 移花接木：将新文献抽出的【关系网】强行接入全局标准节点
                    if alignment_map:
                        for rel in new_relations:
                            if rel.get("source") in alignment_map:
                                rel["source"] = alignment_map[rel["source"]]
                            if rel.get("target") in alignment_map:
                                rel["target"] = alignment_map[rel["target"]]

                    # ====================================================
                    # 4. 🔗 核心大招：同种关系折叠合并 (去重与对称折叠，不预先自动提纯)
                    # ====================================================
                    all_rels = st.session_state.master_relations + new_relations
                    st.session_state.master_relations = consolidate_homogeneous_relations(all_rels)

                    # 5. 将处理后的“干净”实体数据并入全局中枢
                    st.session_state.master_entities.extend(aligned_new_entities)

            elif append_mode:
                st.session_state.master_entities.extend(new_entities)
                st.session_state.master_relations = consolidate_homogeneous_relations(
                    st.session_state.master_relations + new_relations
                )
            else:
                st.session_state.master_entities = new_entities
                st.session_state.master_relations = consolidate_homogeneous_relations(new_relations)

            # 🟢 阶段三：独立的渲染转圈
            with st.spinner(t("msg_rendering_graph")):
                visualizer = GraphVisualizer()
                html_file = ".bio_knowledge_graph.html"

                # 🛡️ 破除幻觉机制：生成前强制删掉旧文件！
                if os.path.exists(html_file):
                    os.remove(html_file)

                # ✨ 漏洞修复：抓取当前控制台的所有开关和滑块状态（带默认值防呆）
                current_show_shortcuts = st.session_state.get("show_shortcuts_toggle", False)

                empower_ontology = st.session_state.get("empower_ontology", False)
                alpha_ontology = st.session_state.get("alpha_ontology", 0.5)

                empower_node = st.session_state.get("empower_node", False)
                beta_node = st.session_state.get("beta_node", 0.2)

                empower_edge = st.session_state.get("empower_edge", False)
                gamma_edge = st.session_state.get("gamma_edge", 0.1)

                # ✨ 将所有状态参数全部传给渲染器，保证新生成的图谱继承当前的视觉设置！
                visualizer.generate_html(
                    st.session_state.master_entities,
                    st.session_state.master_relations,
                    output_file=html_file,
                    show_shortcuts=current_show_shortcuts,
                    empower_ontology=empower_ontology,
                    alpha_ontology=alpha_ontology,
                    empower_node=empower_node,
                    beta_node=beta_node,
                    empower_edge=empower_edge,
                    gamma_edge=gamma_edge,
                    output_lang=st.session_state.get("ui_language", "zh")
                )

            # 🟢 阶段四：所有处理完成，将结果存入全局保险箱
            if os.path.exists(html_file):
                st.success(t("msg_gen_success"))
                with open(html_file, "r", encoding="utf-8") as f:
                    st.session_state.html_data = f.read()

                # 开启展示开关
                st.session_state.show_results = True
                save_local_vault()
            else:
                st.error(t("err_gen_fail"))
                st.session_state.show_results = False

        except Exception as e:
            # 所有阶段任何一步报错，都会统一跳到这里处理，代码极其优雅
            st.error(t("err_process_fail").format(e=e))

# ==========================================
# 📥 独立渲染区：结果多元化导出与图谱展示（防白屏刷新机制）
# ==========================================
# 🌟 注意：下面这段代码必须完全没有缩进，贴紧最左侧！
if st.session_state.show_results and st.session_state.html_data:

    st.markdown(t("export_section_title"))

    entities_json = json.dumps(st.session_state.master_entities, ensure_ascii=False, indent=2)
    relations_json = json.dumps(st.session_state.master_relations, ensure_ascii=False, indent=2)

    df_entities = pd.DataFrame(st.session_state.master_entities)
    if 'aliases' in df_entities.columns:
        df_entities['aliases'] = df_entities['aliases'].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x)
    csv_entities = df_entities.to_csv(index=False).encode('utf-8')

    df_relations = pd.DataFrame(st.session_state.master_relations)
    csv_relations = df_relations.to_csv(index=False).encode('utf-8') if not df_relations.empty else b""

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        st.download_button(t("btn_export_ent_json"), entities_json, "bio_entities.json", "application/json",
                           use_container_width=True)
    with btn_col2:
        st.download_button(t("btn_export_ent_csv"), csv_entities, "bio_entities.csv", "text/csv", use_container_width=True)
    with btn_col3:
        st.download_button(t("btn_export_rel_csv"), csv_relations, "bio_relations.csv", "text/csv", use_container_width=True)
    with btn_col4:
        st.download_button(t("btn_export_graph_html"), st.session_state.html_data, "bio_graph.html", "text/html",
                           use_container_width=True)

    st.markdown("---")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        st.toggle(t("toggle_show_shortcuts"), key="show_shortcuts_toggle", on_change=redraw_and_update)

    with ctrl_col2:
        st.toggle(t("toggle_empower_ontology"), key="empower_ontology", on_change=redraw_and_update)
        if st.session_state.get("empower_ontology", False):
            st.slider(t("slider_alpha_ontology"), 0.1, 1.0, 0.5, 0.1, key="alpha_ontology", on_change=redraw_and_update)

    with ctrl_col3:
        # ✨ 节点互相辐射
        st.toggle(t("toggle_empower_node"), key="empower_node", on_change=redraw_and_update)
        if st.session_state.get("empower_node", False):
            st.slider(t("slider_beta_node"), 0.1, 1.0, 0.2, 0.1, key="beta_node", on_change=redraw_and_update)

    with ctrl_col4:
        # ✨ 纯粹的连线加粗
        st.toggle(t("toggle_empower_edge"), key="empower_edge", on_change=redraw_and_update)
        if st.session_state.get("empower_edge", False):
            st.slider(t("slider_gamma_edge"), 0.05, 0.5, 0.1, 0.05, key="gamma_edge", on_change=redraw_and_update)


    # 🛡️ 渲染图谱 (原有代码，作为定位点)
    components.html(st.session_state.html_data, height=800, scrolling=True)

    st.markdown("---")

    # ====================================================================
    # 💡 核心位置：回填真实导出按钮 (放在 Tabs 前面防止 DOM 崩溃)
    # ====================================================================
    if len(st.session_state.master_entities) > 0:
        with export_placeholder:
            current_pid = get_current_project_id()
            current_pname = get_current_project_name()
            clean_export_name = re.sub(r'[\\/*?:"<>|]', "_", str(current_pname).strip()).strip(". ") or "project"
            export_filename = f"{clean_export_name}.biokg"
            project_data = {
                "version": "1.0",
                "project_id": current_pid,
                "project_name": current_pname,
                "entities": st.session_state.master_entities,
                "relations": st.session_state.master_relations,
                "analyzed_files": st.session_state.analyzed_files
            }
            json_bytes = json.dumps(project_data, ensure_ascii=False, indent=2).encode('utf-8')

            st.download_button(
                label=t("btn_export_project"),
                data=json_bytes,
                file_name=export_filename,
                mime="application/json",
                use_container_width=True,
                key="real_export_btn"
            )

    # ==========================================
    # 🎛️ 终极图谱数据管理中心
    # ==========================================

    if ENABLE_EDITOR and len(st.session_state.master_entities) > 0:
        st.subheader(t("editor_title"))

        # 🚀 终极架构修复：解耦 Navigation 渲染
        cur_tab_lang = st.session_state.get("ui_language", "zh")
        nav_keys = ["nav_quick", "nav_node", "nav_rel"]
        main_tab_key = st.radio(
            t("nav_select_module"),
            options=nav_keys,
            format_func=lambda k, l=cur_tab_lang: t(k, lang=l),
            horizontal=True,
            label_visibility="collapsed",
            key="main_nav_radio"
        )
        st.markdown("---")

        # -----------------------------------------
        # 方向一：快捷功能
        # -----------------------------------------
        if main_tab_key == "nav_quick":
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(t("quick_clear_title"))
                st.info(t("quick_clear_desc"))
                if st.button(t("btn_clear_isolated"), use_container_width=True):
                    connected_nodes = set()
                    for rel in st.session_state.master_relations:
                        connected_nodes.add(rel.get("source"))
                        connected_nodes.add(rel.get("target"))

                    old_count = len(st.session_state.master_entities)
                    st.session_state.master_entities = [
                        ent for ent in st.session_state.master_entities
                        if ent.get("standard_name") in connected_nodes
                    ]
                    if old_count - len(st.session_state.master_entities) > 0:
                        st.toast(
                            t("toast_clear_success").format(count=old_count - len(st.session_state.master_entities)),
                            icon="🧹")
                        redraw_and_update()
                        st.rerun()
                    else:
                        st.toast(t("toast_no_isolated"), icon="✨")

                st.markdown(t("quick_rename_title"))
                raw_sources = []
                for ent in st.session_state.master_entities:
                    if "doc_source" in ent and ent["doc_source"]:
                        raw_sources.extend(str(ent["doc_source"]).split("|"))
                for rel in st.session_state.master_relations:
                    if "doc_source" in rel and rel["doc_source"]:
                        raw_sources.extend(str(rel["doc_source"]).split("|"))
                if "analyzed_files" in st.session_state:
                    raw_sources.extend(st.session_state.analyzed_files)

                cur_pid = get_current_project_id()
                all_sources = sorted(canonicalize_project_papers(cur_pid, raw_sources))

                if not all_sources:
                    st.info(t("lbl_no_source"))
                else:
                    with st.form("rename_source_form"):
                        old_source = st.selectbox(t("lbl_select_source"), all_sources)
                        new_source = st.text_input(t("lbl_new_source"))

                        if st.form_submit_button(t("btn_replace_source"), type="primary", use_container_width=True):
                            if not new_source.strip():
                                st.error(t("err_empty_name"))
                            elif new_source.strip() != old_source:
                                new_src_clean = get_clean_display_title(new_source.strip())
                                ok = rename_project_paper(old_source, new_src_clean)
                                if ok:
                                    st.toast(t("toast_rename_success").format(old=old_source, new=new_src_clean), icon="🎉")
                                    redraw_and_update()
                                    st.rerun()
                                else:
                                    st.error("更名失败，请检查文件系统或重试。")

            with col2:
                st.markdown(t("quick_add_title"))
                with st.form("manual_add_node_form"):
                    new_node_name = st.text_input(t("lbl_new_node_name"))
                    new_node_aliases = st.text_input(t("lbl_new_node_aliases"))
                    new_node_source = st.text_input(t("lbl_new_node_source"))

                    if st.form_submit_button(t("btn_add_node"), type="primary", use_container_width=True):
                        if not new_node_name.strip():
                            st.error(t("err_empty_name"))
                        else:
                            existing_names = [e.get("standard_name") for e in st.session_state.master_entities]
                            if new_node_name.strip() in existing_names:
                                st.warning(t("err_node_exists"))
                            else:
                                aliases_list = [a.strip() for a in
                                                new_node_aliases.split(",")] if new_node_aliases.strip() else []
                                final_source = new_node_source.strip() if new_node_source.strip() else t(
                                    "default_manual_add")
                                st.session_state.master_entities.append({
                                    "standard_name": new_node_name.strip(),
                                    "aliases": aliases_list,
                                    "doc_source": final_source
                                })
                                st.toast(t("toast_add_node_success"), icon="🎉")
                                redraw_and_update()
                                st.rerun()

        # -----------------------------------------
        # 🎯 方向二：节点操作
        # -----------------------------------------
        elif main_tab_key == "nav_node":
            all_node_names = sorted([e.get("standard_name") for e in st.session_state.master_entities])

            if not all_node_names:
                st.info(t("err_empty_graph"))
            else:
                # 💡 防崩溃下拉框设计：内部存 "__SELECT__"，外部显示翻译
                node_opts = ["__SELECT__"] + all_node_names
                target_node_name = st.selectbox(
                    t("lbl_search_node"),
                    node_opts,
                    format_func=lambda x: t("opt_select") if x == "__SELECT__" else x
                )

                if target_node_name != "__SELECT__":
                    target_ent = next(
                        (e for e in st.session_state.master_entities if e.get("standard_name") == target_node_name),
                        None)

                    if target_ent:
                        st.markdown(t("lbl_current_selected").format(node=target_node_name))

                        action_keys = ["act_edit", "act_merge", "act_split", "act_delete"]
                        action = st.radio(
                            t("lbl_select_action"),
                            options=action_keys,
                            format_func=lambda k: t(k),
                            horizontal=True,
                            label_visibility="collapsed",
                            key="node_action_radio"
                        )

                        # -- 修改信息 --
                        if action == "act_edit":
                            with st.form("edit_node_form"):
                                new_name = st.text_input(t("lbl_new_std_name"), value=target_ent.get("standard_name"))
                                current_aliases = ", ".join(target_ent.get("aliases", []))
                                new_aliases = st.text_input(t("lbl_new_aliases"), value=current_aliases)

                                if st.form_submit_button(t("btn_save_edit"), type="primary"):
                                    clean_new_name = new_name.strip()
                                    if clean_new_name:
                                        if clean_new_name != target_node_name:
                                            for rel in st.session_state.master_relations:
                                                if rel.get("source") == target_node_name: rel["source"] = clean_new_name
                                                if rel.get("target") == target_node_name: rel["target"] = clean_new_name

                                        target_ent["standard_name"] = clean_new_name
                                        target_ent["aliases"] = [a.strip() for a in new_aliases.split(",") if a.strip()]
                                        st.toast(t("toast_edit_success"), icon="🎉")
                                        redraw_and_update()
                                        st.rerun()

                        # -- 合并节点 --
                        elif action == "act_merge":
                            st.info(t("desc_merge").format(node=target_node_name))
                            merge_target = st.selectbox(t("lbl_select_merge_target"),
                                                        [n for n in all_node_names if n != target_node_name])

                            if st.button(t("btn_confirm_merge"), type="primary", use_container_width=True):
                                for rel in st.session_state.master_relations:
                                    if rel.get("source") == target_node_name: rel["source"] = merge_target
                                    if rel.get("target") == target_node_name: rel["target"] = merge_target

                                dest_ent = next(e for e in st.session_state.master_entities if
                                                e.get("standard_name") == merge_target)
                                combined_aliases = set(
                                    dest_ent.get("aliases", []) + target_ent.get("aliases", []) + [target_node_name])
                                dest_ent["aliases"] = list(combined_aliases)

                                st.session_state.master_entities = [e for e in st.session_state.master_entities if
                                                                    e.get("standard_name") != target_node_name]
                                # 🧹 实体合并后，执行同种关系折叠合并
                                st.session_state.master_relations = consolidate_homogeneous_relations(
                                    st.session_state.master_relations
                                )
                                st.toast(t("toast_merge_success").format(target=merge_target), icon="🎉")
                                redraw_and_update()
                                st.rerun()

                        # -- 节点拆分 --
                        elif action == "act_split":
                            st.info(t("desc_split").format(node=target_node_name))

                            related_rels = []
                            for i, rel in enumerate(st.session_state.master_relations):
                                if rel.get("source") == target_node_name or rel.get("target") == target_node_name:
                                    related_rels.append((i, rel))

                            with st.form("split_node_form"):
                                st.markdown(t("split_step1"))
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    name_a = st.text_input(t("lbl_new_node_a"), value=f"{target_node_name}_A")
                                    aliases_a = st.text_input(t("lbl_new_aliases_a"),
                                                              value=", ".join(target_ent.get("aliases", [])))
                                with col_b:
                                    name_b = st.text_input(t("lbl_new_node_b"), value=f"{target_node_name}_B")
                                    aliases_b = st.text_input(t("lbl_new_aliases_b"), value="")

                                st.markdown(t("split_step2"))
                                rel_choices = {}
                                if not related_rels:
                                    st.caption(t("desc_no_rel_to_split"))
                                else:
                                    split_opt_keys = ["split_a", "split_b", "split_both", "split_discard"]


                                    # 💡 这里必须用闭包获取内部函数正确的上下文字典映射
                                    def get_safe_rel_display(val):
                                        if val == "正作用": return t("rel_positive")
                                        if val == "负作用": return t("rel_negative")
                                        if val == "相关": return t("rel_correlation")
                                        return val


                                    for idx, rel in related_rels:
                                        src = rel.get("source")
                                        tgt = rel.get("target")
                                        rel_type_str = rel.get("relation", "相关")

                                        # 界面显示的漂亮文字，如: p53 -[Positive]-> MDM2
                                        display_text = f"**{src}** ─[{get_safe_rel_display(rel_type_str)}]▶ **{tgt}**"

                                        rel_choices[idx] = st.radio(
                                            display_text,
                                            options=split_opt_keys,
                                            format_func=lambda k: t(k),
                                            horizontal=True,
                                            key=f"split_rel_{idx}"
                                        )

                                if st.form_submit_button(t("btn_confirm_split"), type="primary",
                                                         use_container_width=True):
                                    c_name_a = name_a.strip()
                                    c_name_b = name_b.strip()

                                    if not c_name_a or not c_name_b:
                                        st.error(t("err_split_names_empty"))
                                    elif c_name_a == c_name_b:
                                        st.error(t("err_split_names_same"))
                                    else:
                                        import copy

                                        st.session_state.master_entities = [e for e in st.session_state.master_entities
                                                                            if
                                                                            e.get("standard_name") != target_node_name]

                                        st.session_state.master_entities.append({
                                            "standard_name": c_name_a,
                                            "aliases": [x.strip() for x in aliases_a.split(",") if x.strip()],
                                            "doc_source": target_ent.get("doc_source", t("default_manual_split"))
                                        })
                                        st.session_state.master_entities.append({
                                            "standard_name": c_name_b,
                                            "aliases": [x.strip() for x in aliases_b.split(",") if x.strip()],
                                            "doc_source": target_ent.get("doc_source", t("default_manual_split"))
                                        })

                                        new_relations = []
                                        for i, rel in enumerate(st.session_state.master_relations):
                                            if i in rel_choices:
                                                choice_key = rel_choices[i]
                                                if choice_key == "split_discard":
                                                    continue

                                                is_source = (rel.get("source") == target_node_name)
                                                is_target = (rel.get("target") == target_node_name)


                                                def make_rel(node_name):
                                                    new_r = copy.deepcopy(rel)
                                                    if is_source: new_r["source"] = node_name
                                                    if is_target: new_r["target"] = node_name
                                                    return new_r


                                                if choice_key == "split_a":
                                                    new_relations.append(make_rel(c_name_a))
                                                elif choice_key == "split_b":
                                                    new_relations.append(make_rel(c_name_b))
                                                elif choice_key == "split_both":
                                                    new_relations.append(make_rel(c_name_a))
                                                    new_relations.append(make_rel(c_name_b))
                                            else:
                                                new_relations.append(rel)

                                        st.session_state.master_relations = new_relations
                                        st.toast(t("toast_split_success").format(a=c_name_a, b=c_name_b), icon="🎉")
                                        redraw_and_update()
                                        st.rerun()

                        # -- 彻底删除 --
                        elif action == "act_delete":
                            st.error(t("warn_delete_node").format(node=target_node_name))
                            if st.button(t("btn_confirm_delete"), type="primary", use_container_width=True):
                                st.session_state.master_entities = [e for e in st.session_state.master_entities if
                                                                    e.get("standard_name") != target_node_name]
                                st.session_state.master_relations = [r for r in st.session_state.master_relations if
                                                                     r.get("source") != target_node_name and r.get(
                                                                         "target") != target_node_name]
                                st.toast(t("toast_delete_success"), icon="🗑️")
                                redraw_and_update()
                                st.rerun()

        # -----------------------------------------
        # 🎯 方向三：关系操作
        # -----------------------------------------
        elif main_tab_key == "nav_rel":
            all_node_names = sorted([e.get("standard_name") for e in st.session_state.master_entities])

            # 💡 数据底层必须是中文以兼容模型设定，UI显示靠 format_func
            SAFE_RELATIONS_INTERNAL = ["正作用", "负作用", "相关"]


            def get_rel_display(internal_val):
                if internal_val == "正作用": return t("rel_positive")
                if internal_val == "负作用": return t("rel_negative")
                if internal_val == "相关": return t("rel_correlation")
                return internal_val


            if not all_node_names:
                st.info(t("err_empty_graph"))
            else:
                st.markdown(t("lbl_search_rel_title"))
                col_a, col_b = st.columns(2)

                node_opts = ["__SELECT__"] + all_node_names
                with col_a:
                    node_a = st.selectbox(t("lbl_select_node_a"), node_opts,
                                          format_func=lambda x: t("opt_select") if x == "__SELECT__" else x)
                with col_b:
                    node_b = st.selectbox(t("lbl_select_node_b"), node_opts,
                                          format_func=lambda x: t("opt_select") if x == "__SELECT__" else x)

                if node_a != "__SELECT__" and node_b != "__SELECT__":
                    if node_a == node_b:
                        st.warning(t("warn_same_node"))
                    else:
                        st.markdown("---")

                        matching_rels = []
                        for i, r in enumerate(st.session_state.master_relations):
                            src = r.get("source")
                            tgt = r.get("target")
                            if (src == node_a and tgt == node_b) or (src == node_b and tgt == node_a):
                                matching_rels.append((i, r))

                        rel_mode_keys = ["rel_mode_edit", "rel_mode_add"]
                        rel_op_mode = st.radio(
                            t("lbl_rel_op_mode").format(a=node_a, b=node_b),
                            options=rel_mode_keys,
                            format_func=lambda k: t(k),
                            horizontal=True
                        )

                        # ==========================================
                        # 子功能 A：管理 / 修改 / 删除 现有关系
                        # ==========================================
                        if rel_op_mode == "rel_mode_edit":
                            if not matching_rels:
                                st.info(t("info_no_direct_rel").format(a=node_a, b=node_b))
                            else:
                                rel_options = {}
                                for idx, r in matching_rels:
                                    src = r.get("source")
                                    tgt = r.get("target")
                                    rel_type = r.get("relation", "相关")
                                    display_text = f"[{idx}] {src} ─[{get_rel_display(rel_type)}]▶ {tgt}"
                                    rel_options[display_text] = idx

                                selected_rel_text = st.selectbox(t("lbl_select_specific_rel"), list(rel_options.keys()))

                                rel_idx = rel_options[selected_rel_text]
                                target_rel = st.session_state.master_relations[rel_idx]

                                with st.form("edit_rel_form"):
                                    current_src = target_rel.get('source')

                                    dir_keys = ["dir_a_to_b", "dir_b_to_a"]


                                    def dir_display(k):
                                        if k == "dir_a_to_b": return t("dir_x_to_y").format(x=node_a, y=node_b)
                                        return t("dir_x_to_y").format(x=node_b, y=node_a)


                                    new_direction = st.selectbox(
                                        t("lbl_change_dir"),
                                        options=dir_keys,
                                        format_func=dir_display,
                                        index=0 if current_src == node_a else 1
                                    )

                                    current_rel = target_rel.get("relation", "")
                                    default_idx = SAFE_RELATIONS_INTERNAL.index(
                                        current_rel) if current_rel in SAFE_RELATIONS_INTERNAL else 0

                                    selected_rel_type = st.selectbox(
                                        t("lbl_change_rel_type"),
                                        options=SAFE_RELATIONS_INTERNAL,
                                        format_func=get_rel_display,
                                        index=default_idx
                                    )

                                    new_evidence = st.text_area(t("lbl_evidence"), value=target_rel.get("evidence", ""))
                                    new_reason = st.text_area(t("lbl_reason"), value=target_rel.get("reason", ""))
                                    new_doc_source = st.text_input(t("lbl_doc_source"),
                                                                   value=target_rel.get("doc_source",
                                                                                        t("default_manual_add")))

                                    if st.form_submit_button(t("btn_save_rel"), type="primary",
                                                             use_container_width=True):
                                        final_src, final_tgt = (node_a, node_b) if new_direction == "dir_a_to_b" else (
                                        node_b, node_a)

                                        st.session_state.master_relations[rel_idx]["source"] = final_src
                                        st.session_state.master_relations[rel_idx]["target"] = final_tgt
                                        st.session_state.master_relations[rel_idx]["relation"] = selected_rel_type
                                        st.session_state.master_relations[rel_idx]["evidence"] = new_evidence.strip()
                                        st.session_state.master_relations[rel_idx]["reason"] = new_reason.strip()
                                        st.session_state.master_relations[rel_idx][
                                            "doc_source"] = new_doc_source.strip() if new_doc_source.strip() else t(
                                            "default_manual_edit")

                                        st.toast(t("toast_rel_mod_success"), icon="🎉")
                                        redraw_and_update()
                                        st.rerun()

                                st.markdown("---")
                                if st.button(t("btn_delete_rel"), type="primary", use_container_width=True):
                                    st.session_state.master_relations.pop(rel_idx)
                                    st.toast(t("toast_rel_del_success"), icon="✂️")
                                    redraw_and_update()
                                    st.rerun()

                        # ==========================================
                        # 子功能 B：手动搭桥，安全新增
                        # ==========================================
                        elif rel_op_mode == "rel_mode_add":
                            with st.form("add_rel_form"):

                                dir_keys = ["dir_a_to_b", "dir_b_to_a"]


                                def dir_display_add(k):
                                    if k == "dir_a_to_b": return t("dir_x_to_y").format(x=node_a, y=node_b)
                                    return t("dir_x_to_y").format(x=node_b, y=node_a)


                                direction = st.radio(
                                    t("lbl_confirm_dir"),
                                    options=dir_keys,
                                    format_func=dir_display_add,
                                    horizontal=True
                                )

                                selected_new_rel = st.selectbox(
                                    t("lbl_select_rel_type"),
                                    options=SAFE_RELATIONS_INTERNAL,
                                    format_func=get_rel_display
                                )

                                new_evidence = st.text_area(t("lbl_rel_evidence_opt"))
                                new_doc_source = st.text_input(t("lbl_rel_source_opt"))

                                if st.form_submit_button(t("btn_add_rel"), type="primary", use_container_width=True):
                                    final_src, final_tgt = (node_a, node_b) if direction == "dir_a_to_b" else (
                                    node_b, node_a)

                                    exist = any(r.get("source") == final_src and
                                                r.get("target") == final_tgt and
                                                r.get("relation") == selected_new_rel
                                                for r in st.session_state.master_relations)

                                    if exist:
                                        st.warning(t("warn_rel_exists").format(src=final_src, tgt=final_tgt,
                                                                               type=get_rel_display(selected_new_rel)))
                                    else:
                                        final_source_str = new_doc_source.strip() if new_doc_source.strip() else t(
                                            "default_manual_add")

                                        st.session_state.master_relations.append({
                                            "source": final_src,
                                            "target": final_tgt,
                                            "relation": selected_new_rel,
                                            "evidence": new_evidence.strip(),
                                            "reason": "用户手动添加",
                                            "doc_source": final_source_str
                                        })
                                        st.toast(t("toast_rel_add_success"), icon="🎉")
                                        redraw_and_update()
                                        st.rerun()

# ==========================================
# 🤖 AI 智能图谱清洗中心 (真机驱动版)
# ==========================================

if ENABLE_AI_CLEANER and len(st.session_state.master_entities) > 0:
    st.markdown("---")
    st.subheader(t("ai_tools_title"))

    # 🚀 优化 1：使用带记忆锁的 Radio 导航替代失忆的 st.tabs
    cur_ai_tab_lang = st.session_state.get("ui_language", "zh")
    ai_nav_keys = ["ai_nav_pruning", "ai_nav_expansion", "ai_nav_bridging", "ai_nav_ask"]
    ai_nav = st.radio(
        t("ai_nav_select"),
        options=ai_nav_keys,
        format_func=lambda k, l=cur_ai_tab_lang: t(k, lang=l),
        horizontal=True,
        label_visibility="collapsed",
        key="ai_nav_radio"
    )
    st.markdown("---")

    # -----------------------------------------
    # 模块一：智能洗树 (Pruning)
    # -----------------------------------------
    if ai_nav == "ai_nav_pruning":
        st.info(t("pruning_info"))

        col_btn, col_toggle = st.columns([1, 1])
        with col_btn:
            st.write(t("pruning_desc"))
            if st.button(t("btn_start_pruning"), type="primary", use_container_width=True):
                current_api_key = api_key.strip()

                if not current_api_key:
                    st.error(t("err_pruning_no_key"))
                    st.stop()

                with st.spinner(t("msg_pruning_diagnosing")):
                    from LLM_SYS import BioBrainAgent

                    agent = BioBrainAgent(api_key=current_api_key,model=selected_model_id,base_url=base_url)
                    suggestions = agent.diagnose_graph(st.session_state.master_entities,
                                                       st.session_state.master_relations,
                                                       output_lang=st.session_state.get("ui_language", "zh"))

                    st.session_state.ai_suggestions = suggestions
                    st.toast(t("toast_pruning_done"), icon="🤖")
                    st.rerun()

        with col_toggle:
            st.info(t("pruning_tip_shortcut"))

        # -----------------------------------------
        # 📋 渲染真实的审查清单并执行
        # -----------------------------------------
        if hasattr(st.session_state, "ai_suggestions") and st.session_state.ai_suggestions:
            st.markdown(t("review_list_title"))

            with st.form("ai_prune_review_form"):
                selected_actions = []

                # 按操作类型优先级排序：高确信度结构/环路修复置顶，粗糙关系机制提纯(PURIFY)默认置于最下方
                ACTION_PRIORITY = {
                    "MERGE": 10,
                    "RESOLVE_CYCLE": 12,
                    "HIERARCHY": 15,
                    "FOLD_SYMMETRIC": 20,
                    "DOWNGRADE": 40,
                    "REMOVE": 50,
                    "PURIFY": 100,
                }
                sorted_suggestions = sorted(
                    st.session_state.ai_suggestions,
                    key=lambda x: ACTION_PRIORITY.get(x.get("action", ""), 60)
                )

                for i, sug in enumerate(sorted_suggestions):
                    action = sug.get("action")
                    reason = sug.get("reason", t("no_reason"))

                    default_checked = True

                    if action == "MERGE":
                        target = sug.get("target_node")
                        removes = sug.get("nodes_to_remove", [])
                        label = t("rev_merge").format(removes=removes, target=target, reason=reason)
                    elif action == "HIERARCHY":
                        parent = sug.get("parent")
                        child = sug.get("child")
                        label = t("rev_hierarchy").format(parent=parent, child=child, reason=reason)
                    elif action == "RESOLVE_CYCLE":
                        parent = sug.get("parent")
                        child = sug.get("child")
                        label = t("rev_resolve_cycle").format(parent=parent, child=child, reason=reason)
                        default_checked = True
                    elif action == "FOLD_SYMMETRIC":
                        # 🔀 双向相关连线折叠：结构无损去重，默认勾选！
                        node_a = sug.get("source") or sug.get("node_a")
                        node_b = sug.get("target") or sug.get("node_b")
                        label = t("rev_fold_symmetric").format(node_a=node_a, node_b=node_b, reason=reason)
                        default_checked = True
                    elif action == "DOWNGRADE":
                        src = sug.get("source")
                        tgt = sug.get("target")
                        rel = sug.get("relation")
                        label = t("rev_downgrade").format(src=src, rel=rel, tgt=tgt, reason=reason)
                    elif action == "PURIFY":
                        src = sug.get("source")
                        tgt = sug.get("target")
                        rel = sug.get("relation", "相关")
                        target_rel = sug.get("target_relation")
                        if target_rel:
                            label = t("rev_purify_with_target").format(src=src, rel=rel, tgt=tgt, target_rel=target_rel, reason=reason)
                        else:
                            label = t("rev_purify").format(src=src, rel=rel, tgt=tgt, reason=reason)
                        # ⚠️ 机制提纯合并粗糙线交由学者自行决断，默认不勾选！
                        default_checked = False
                    elif action == "REMOVE":
                        src = sug.get("source")
                        tgt = sug.get("target")
                        rel = sug.get("relation")
                        label = t("rev_remove").format(src=src, rel=rel, tgt=tgt, reason=reason)
                        # 若针对的是“相关”连线或原因中提及机制覆盖/粗糙冗余，同样默认不勾选！
                        if rel in ["相关", "关联"] or "机制覆盖" in str(reason) or "粗糙" in str(reason) or "低信息量" in str(reason):
                            default_checked = False
                    else:
                        continue

                    if st.checkbox(label, value=default_checked, key=f"sug_{i}"):
                        selected_actions.append(sug)

                st.markdown("---")
                if st.form_submit_button(t("btn_confirm_pruning"), type="primary", use_container_width=True):

                    # 核心执行引擎
                    for act in selected_actions:
                        if act["action"] == "MERGE":
                            target = act.get("target_node")
                            removes = act.get("nodes_to_remove", [])
                            for r in st.session_state.master_relations:
                                if r.get("source") in removes: r["source"] = target
                                if r.get("target") in removes: r["target"] = target
                            st.session_state.master_entities = [e for e in st.session_state.master_entities if
                                                                e.get("standard_name") not in removes]

                        elif act["action"] == "HIERARCHY":
                            parent = act.get("parent")
                            child = act.get("child")
                            reason = act.get("reason", "")
                            st.session_state.master_relations = merge_hierarchy_relation(
                                st.session_state.master_relations, parent=parent, child=child, reason=reason
                            )

                        elif act["action"] == "RESOLVE_CYCLE":
                            # 🔄 专职破除循环包含：确立规范父子单向层级，无损合并逆向包含证据与权重
                            parent = act.get("parent")
                            child = act.get("child")
                            reason = act.get("reason", "")
                            st.session_state.master_relations = merge_hierarchy_relation(
                                st.session_state.master_relations, parent=parent, child=child, reason=reason
                            )

                        elif act["action"] == "FOLD_SYMMETRIC":
                            # 🔀 双向相关连线折叠：单向规范化，无损合并双方证据、出处并累加热度
                            node_a = act.get("source") or act.get("node_a")
                            node_b = act.get("target") or act.get("node_b")
                            st.session_state.master_relations = fold_symmetric_relations(
                                st.session_state.master_relations, node_a=node_a, node_b=node_b, relation="相关"
                            )

                        elif act["action"] == "PURIFY":
                            # 🧪 机制提纯合并：将粗糙关系的证据、来源、热度无损合并入目标机制关系中！
                            src = act.get("source")
                            tgt = act.get("target")
                            rel = act.get("relation", "相关")
                            target_rel = act.get("target_relation")
                            st.session_state.master_relations = merge_coarse_relation_into_target(
                                st.session_state.master_relations, src, tgt, coarse_rel=rel, target_rel=target_rel
                            )

                        elif act["action"] == "REMOVE":
                            src = act.get("source")
                            tgt = act.get("target")
                            rel = act.get("relation")
                            # 🛡️ 信息防漏保护 1：若删除的是粗糙“相关”连线且同对节点间已存在明确机制，绝对不简单丢弃，执行提纯合并！
                            has_mechanism = any(
                                ((r.get("source") == src and r.get("target") == tgt) or (r.get("source") == tgt and r.get("target") == src))
                                and is_mechanism_relation(r.get("relation"))
                                for r in st.session_state.master_relations
                            )
                            # 🛡️ 信息防漏保护 2：若删除的是反向“包含”连线且对端已存在正向“包含”层级，绝对不简单丢弃，无损合并入正向层级！
                            has_opposite_containment = (
                                rel == "包含" and any(
                                    r.get("source") == tgt and r.get("target") == src and r.get("relation") == "包含"
                                    for r in st.session_state.master_relations
                                )
                            )
                            # 🛡️ 信息防漏保护 3：若删除的是“相关”连线且存在反向“相关”连线，绝对不简单丢弃，执行对称折叠合并！
                            has_opposite_related = (
                                rel in ["相关", "关联"] and any(
                                    r.get("source") == tgt and r.get("target") == src and r.get("relation") == rel
                                    for r in st.session_state.master_relations
                                )
                            )

                            if rel in ["相关", "关联"] and has_mechanism:
                                st.session_state.master_relations = merge_coarse_relation_into_target(
                                    st.session_state.master_relations, src, tgt, coarse_rel=rel
                                )
                            elif has_opposite_containment:
                                st.session_state.master_relations = merge_hierarchy_relation(
                                    st.session_state.master_relations, parent=tgt, child=src, reason=act.get("reason", "")
                                )
                            elif has_opposite_related:
                                st.session_state.master_relations = fold_symmetric_relations(
                                    st.session_state.master_relations, node_a=src, node_b=tgt, relation=rel
                                )
                            else:
                                for i in range(len(st.session_state.master_relations) - 1, -1, -1):
                                    r = st.session_state.master_relations[i]
                                    if r.get("source") == src and r.get("target") == tgt and r.get(
                                            "relation") == rel:
                                        st.session_state.master_relations.pop(i)

                        elif act["action"] == "DOWNGRADE":
                            for r in st.session_state.master_relations:
                                if r.get("source") == act.get("source") and r.get("target") == act.get(
                                        "target") and r.get("relation") == act.get("relation"):
                                    r["is_shortcut"] = True

                    # ====================================================
                    # 🧹 洗树后的终极清理（仅执行同种关系去重与对称折叠，保留学者未勾选的连线）
                    # ====================================================
                    st.session_state.master_relations = consolidate_homogeneous_relations(
                        st.session_state.master_relations
                    )
                    st.session_state.ai_suggestions = []
                    st.toast(t("toast_pruning_exec_done"), icon="🎉")
                    redraw_and_update()
                    st.rerun()

        # ==========================================
        # ✂️ 新增模块：基于意图的智能节点剪枝
        # ==========================================
        st.markdown("---")
        st.subheader(t("intent_pruning_title"))
        st.info(t("intent_pruning_info"))

        user_interest = st.text_area(t("lbl_user_interest"), key="user_interest_prune")

        if st.button(t("btn_find_irrelevant"), type="primary", use_container_width=True):
            current_api_key = api_key.strip()
            if not current_api_key:
                st.error(t("err_missing_api_key"))
            elif not user_interest.strip():
                st.warning(t("warn_empty_interest"))
            else:
                with st.spinner(t("msg_finding_irrelevant")):
                    from LLM_SYS import BioBrainAgent

                    agent = BioBrainAgent(api_key=current_api_key,model=selected_model_id,base_url=base_url)
                    prune_suggestions = agent.prune_nodes_by_intent(
                        user_interest,
                        st.session_state.master_entities,
                        st.session_state.master_relations,
                        output_lang=st.session_state.get("ui_language", "zh")
                    )
                    st.session_state.node_prune_suggestions = prune_suggestions
                    st.toast(t("toast_find_irrelevant_done"), icon="🤖")
                    st.rerun()

        if "node_prune_suggestions" in st.session_state and st.session_state.node_prune_suggestions:
            st.markdown(t("purge_list_title"))
            st.caption(t("purge_list_caption"))

            # 💡 动态生成带翻译的列名，以确保 DataFrame 绑定的 key 正确
            col_del_txt = t("col_delete")
            col_name_txt = t("col_node_name")
            col_neigh_txt = t("col_neighbors")
            col_risk_txt = t("col_risk_level")
            col_reason_txt = t("col_ai_reason")

            df_prune_data = []
            for item in st.session_state.node_prune_suggestions:
                risk = item.get("risk_level", "")
                node_name = item.get("node_name", "")
                to_delete = True if risk in ["完全无关", "可能无关", "Completely_Irrelevant",
                                             "Potentially_Irrelevant"] else False

                neighbors = set()
                for r in st.session_state.master_relations:
                    if r.get("source") == node_name:
                        neighbors.add(r.get("target"))
                    elif r.get("target") == node_name:
                        neighbors.add(r.get("source"))

                df_prune_data.append({
                    col_del_txt: to_delete,
                    col_name_txt: node_name,
                    col_neigh_txt: ", ".join(list(neighbors))[:60] + (
                        "..." if len(", ".join(list(neighbors))) > 60 else ""),
                    col_risk_txt: risk,
                    col_reason_txt: item.get("reason", "")
                })

            df_prune = pd.DataFrame(df_prune_data)
            edited_prune_df = st.data_editor(
                df_prune,
                use_container_width=True,
                hide_index=True,
                disabled=[col_name_txt, col_neigh_txt, col_risk_txt, col_reason_txt],
                key="prune_data_editor"
            )

            col_prune_y, col_prune_n = st.columns(2)
            with col_prune_y:
                if st.button(t("btn_execute_purge"), type="primary", use_container_width=True):
                    nodes_to_delete = edited_prune_df[edited_prune_df[col_del_txt] == True][
                        col_name_txt].tolist()
                    if nodes_to_delete:
                        st.session_state.master_entities = [
                            e for e in st.session_state.master_entities if
                            e.get("standard_name") not in nodes_to_delete
                        ]
                        st.session_state.master_relations = [
                            r for r in st.session_state.master_relations
                            if r.get("source") not in nodes_to_delete and r.get("target") not in nodes_to_delete
                        ]
                        st.toast(t("toast_purge_success").format(count=len(nodes_to_delete)), icon="🎉")

                    st.session_state.node_prune_suggestions = None
                    redraw_and_update()
                    st.rerun()
            with col_prune_n:
                if st.button(t("btn_cancel_purge"), use_container_width=True):
                    st.session_state.node_prune_suggestions = None
                    st.rerun()

        # ==========================================
        # 🏷️ 新增模块：文献名称智能识别与规范化重命名
        # ==========================================
        st.markdown("---")
        st.subheader(t("paper_rename_title"))
        st.info(t("paper_rename_info"))

        # 全盘聚合所有可规范化的文献与出处来源（通过注册表与哈希身份规范化，杜绝曾用名残余）
        raw_candidates = []
        for f in st.session_state.get("analyzed_files", []):
            if f and str(f).strip():
                raw_candidates.append(str(f).strip())
        for ent in st.session_state.get("master_entities", []):
            ds = ent.get("doc_source", "")
            if ds:
                raw_candidates.extend(str(ds).split("|"))
        for rel in st.session_state.get("master_relations", []):
            ds = rel.get("doc_source", "")
            if ds:
                raw_candidates.extend(str(ds).split("|"))

        cur_pid = get_current_project_id()
        all_target_sources = sorted(canonicalize_project_papers(cur_pid, raw_candidates))

        if not all_target_sources:
            st.caption(t("no_papers_to_rename"))
        else:
            # 持久展示重命名变更详情看板（明确展示“什么换为了什么”）
            if st.session_state.get("rename_history_report"):
                rep = st.session_state["rename_history_report"]
                succ = rep.get("success", [])
                skip = rep.get("skipped", [])
                fail = rep.get("failed", [])

                if succ:
                    report_lines = []
                    for old_f, new_f, pmid_tag in succ:
                        icon = "📄" if (new_f.lower().endswith(".pdf") or old_f.lower().endswith(".pdf")) else "🌐"
                        pmid_str = f" *(PMID: {pmid_tag})*" if (pmid_tag and f"PMID: {pmid_tag}" not in new_f) else ""
                        report_lines.append(f"- {icon} `{old_f}` ➔ 🏷️ **`{new_f}`**{pmid_str}")
                    st.success(f"🎉 **文献与出处规范化完成（共更新 {len(succ)} 项）：**\n\n" + "\n".join(report_lines))
                if skip:
                    st.info(f"ℹ️ {len(skip)} 项文献/出处名称已是规范学术标题，无需更名。")
                if fail:
                    for old_f, reason in fail:
                        st.warning(f"⚠️ `{old_f}` 未能成功识别: {reason}")

                if st.button("✖️ 清除变更提示", key="btn_clear_rename_report"):
                    del st.session_state["rename_history_report"]
                    st.rerun()

            cur_rn_lang = st.session_state.get("ui_language", "zh")
            def format_source_label(item_str: str, l=cur_rn_lang) -> str:
                if item_str == "__ALL_SOURCES__":
                    return f"🌟 {t('rename_opt_all', lang=l)} ({len(all_target_sources)}项)"
                fpath = find_project_paper(item_str)
                if (fpath and os.path.exists(fpath)) or item_str.lower().endswith(".pdf"):
                    return f"📄 [本地文件] {item_str}" if l == "zh" else f"📄 [Local] {item_str}"
                elif is_pmid_or_virtual_source(item_str):
                    return f"🌐 [线上PMID] {item_str}" if l == "zh" else f"🌐 [Online PMID] {item_str}"
                return f"🏷️ {item_str}"

            col_rn_opt, col_rn_btn = st.columns([3, 2])
            with col_rn_opt:
                rn_choices = ["__ALL_SOURCES__"] + all_target_sources
                cur_rn = st.session_state.get("sel_rename_target")
                if cur_rn not in rn_choices:
                    st.session_state["sel_rename_target"] = "__ALL_SOURCES__"
                selected_rn_target = st.selectbox(
                    t("lbl_select_rename_target"),
                    rn_choices,
                    format_func=format_source_label,
                    key="sel_rename_target"
                )
            with col_rn_btn:
                st.write("")
                st.write("")
                btn_start_rename = st.button(t("btn_start_rename"), type="primary", use_container_width=True)

            if btn_start_rename:
                current_api_key = api_key.strip()
                if not is_local and not current_api_key:
                    st.error(t("err_missing_api_key"))
                else:
                    targets = all_target_sources if selected_rn_target == "__ALL_SOURCES__" else [selected_rn_target]
                    success_list = []
                    skipped_list = []
                    failed_list = []

                    from LLM_SYS import BioBrainAgent
                    agent_key = current_api_key if current_api_key else "ollama"
                    agent = BioBrainAgent(api_key=agent_key, model=selected_model_id, base_url=base_url)

                    cur_pid = get_current_project_id()
                    papers_dir = get_current_papers_dir()

                    with st.spinner(t("msg_renaming_papers")):
                        for target_name in targets:
                            found_path = find_project_paper(target_name)
                            is_physical = bool((found_path and os.path.exists(found_path)) or target_name.lower().endswith(".pdf"))

                            if is_physical:
                                fpath = os.path.join(papers_dir, target_name)
                                if not os.path.exists(fpath):
                                    if found_path and os.path.exists(found_path):
                                        fpath = found_path
                                    else:
                                        failed_list.append((target_name, t("err_file_not_found")))
                                        continue

                                title, pmid_tag, method = extract_paper_title_and_pmid(fpath, max_pages=1, agent=agent)
                                if not title:
                                    failed_list.append((target_name, t("err_title_unrecognized")))
                                    continue
                                new_name = get_clean_display_title(title)
                            else:
                                # 线上 PMID 或其他出处来源
                                title, pmid_tag, method = extract_paper_title_and_pmid(target_name, agent=agent)
                                if not title:
                                    failed_list.append((target_name, t("err_title_unrecognized")))
                                    continue
                                new_name = get_clean_display_title(title)

                            if new_name == get_clean_display_title(target_name):
                                skipped_list.append(target_name)
                            else:
                                ok = rename_project_paper(target_name, new_name, cur_pid)
                                if ok:
                                    success_list.append((target_name, new_name, pmid_tag))
                                else:
                                    failed_list.append((target_name, "写入更新失败"))

                    # 持久化记录重命名看板报告
                    st.session_state.rename_history_report = {
                        "success": success_list,
                        "skipped": skipped_list,
                        "failed": failed_list
                    }

                    if success_list:
                        for old_f, new_f, pmid_tag in success_list:
                            pmid_str = f" (PMID: {pmid_tag})" if (pmid_tag and f"PMID: {pmid_tag}" not in new_f) else ""
                            st.toast(t("toast_rename_success").format(old=old_f, new=new_f) + pmid_str, icon="🏷️")
                        redraw_and_update()
                        save_local_vault()
                        st.rerun()
                    elif skipped_list or failed_list:
                        st.rerun()

    # -----------------------------------------
    # 模块二：🔍 智能拓展 (Smart Expansion)
    # -----------------------------------------
    elif ai_nav == "ai_nav_expansion":
        current_db_name = "PubMed" if "PubMed" in st.session_state.get("search_database",
                                                                       "PubMed") else "Semantic Scholar"
        st.info(t("expansion_info").format(db=current_db_name))

        all_node_names = [e.get("standard_name") for e in st.session_state.master_entities]
        selected_nodes = st.multiselect(t("lbl_select_nodes_exp"), all_node_names)

        strategy_keys = ["strategy_direct", "strategy_smart", "strategy_bg"]
        search_mode_key = st.radio(
            t("lbl_search_strategy"),
            options=strategy_keys,
            format_func=lambda k: t(k),
            horizontal=False
        )
        max_results = st.slider(t("lbl_max_results"), min_value=5, max_value=20, value=10)

        if st.button(t("btn_web_search"), type="primary"):
            if not selected_nodes:
                st.warning(t("warn_select_nodes_first"))
            else:
                with st.spinner(t("msg_fetching_api").format(db=current_db_name)):
                    from WebSearcher import get_searcher

                    current_db = st.session_state.get("search_database", "PubMed (生物医学权威)")
                    searcher = get_searcher(current_db)
                    query = ""

                    if search_mode_key == "strategy_direct":
                        query_parts = [f'("{node}"[Title/Abstract])' for node in selected_nodes]
                        query = " AND ".join(query_parts)
                        st.session_state.last_generated_query = query
                    else:
                        current_api_key = api_key.strip()
                        if not current_api_key:
                            st.error(t("err_need_key_for_ai_search"))
                            st.stop()

                        from LLM_SYS import BioBrainAgent

                        agent = BioBrainAgent(api_key=current_api_key,model=selected_model_id,base_url=base_url)

                        selected_nodes_info = []
                        for e in st.session_state.master_entities:
                            if e.get("standard_name") in selected_nodes:
                                selected_nodes_info.append({
                                    "name": e.get("standard_name"),
                                    "aliases": e.get("aliases", [])
                                })

                        if search_mode_key == "strategy_smart":
                            query = agent.generate_pubmed_query(selected_nodes_info)
                        elif search_mode_key == "strategy_bg":
                            relevant_relations = [r for r in st.session_state.master_relations
                                                  if r.get("source") in selected_nodes or r.get(
                                    "target") in selected_nodes]
                            query = agent.generate_pubmed_query(selected_nodes_info,
                                                                context_relations=relevant_relations)

                        st.session_state.last_generated_query = query

                    results = searcher.search_articles(query, max_results=max_results)
                    st.session_state.pubmed_search_results = results
                    st.toast(t("toast_search_success").format(count=len(results)), icon="🎉")
                    st.rerun()

        if st.session_state.pubmed_search_results:
            st.markdown(t("search_results_title"))

            df = pd.DataFrame(st.session_state.pubmed_search_results)

            # 💡 动态表头绑定
            col_include_txt = t("col_include_graph")
            col_title_txt = t("col_paper_title")
            col_year_txt = t("col_year")

            if col_include_txt not in df.columns:
                # ⚠️ 注意这里底层用英文键兼容原来的逻辑，但是在显示的时候渲染成当地语言
                df.insert(0, col_include_txt, False)

            edited_df = st.data_editor(
                df,
                column_config={
                    col_include_txt: st.column_config.CheckboxColumn(col_include_txt, default=False),
                    "pmid": st.column_config.TextColumn("PMID", disabled=True),
                    "title": st.column_config.TextColumn(col_title_txt, disabled=True),
                    "year": st.column_config.TextColumn(col_year_txt, disabled=True),
                },
                disabled=["pmid", "title", "year", "authors", "doi"],
                hide_index=True,
                use_container_width=True,
                key="pubmed_data_editor"
            )

            st.markdown("---")
            if st.button(t("btn_add_to_queue"), type="primary", use_container_width=True):
                selected_docs = edited_df[edited_df[col_include_txt] == True]

                if selected_docs.empty:
                    st.warning(t("warn_check_one_paper"))
                else:
                    st.session_state.expansion_queue = selected_docs.to_dict('records')
                    st.session_state.expansion_idx = 0
                    st.toast(t("toast_added_to_queue"), icon="📥")
                    st.rerun()


            def bg_reset_expansion_task():
                st.session_state.expansion_queue = []
                st.session_state.expansion_idx = 0
                if "auto_run_expansion" in st.session_state:
                    st.session_state.auto_run_expansion = False


            if "expansion_queue" in st.session_state and st.session_state.expansion_queue:
                queue = st.session_state.expansion_queue
                idx = st.session_state.expansion_idx
                total = len(queue)

                st.markdown("---")
                st.markdown(t("queue_title").format(idx=idx, total=total))

                if idx < total:
                    current_doc = queue[idx]
                    pmid = current_doc['pmid']
                    title = current_doc['title']

                    st.info(t("queue_current_doc").format(title=title, pmid=pmid))

                    auto_run = st.toggle(t("toggle_auto_run"), value=False, key="auto_run_expansion")
                    if auto_run:
                        st.caption(t("warn_auto_running"))

                    col_run, col_skip, col_stop = st.columns(3)

                    btn_run = col_run.button(t("btn_run_this"), use_container_width=True, type="primary",
                                             disabled=auto_run)
                    btn_skip = col_skip.button(t("btn_skip_this"), use_container_width=True, disabled=auto_run)
                    btn_stop = col_stop.button(t("btn_stop_queue"), use_container_width=True, disabled=auto_run,
                                               on_click=bg_reset_expansion_task)

                    if btn_run or auto_run:
                        current_api_key = api_key.strip()
                        if not current_api_key:
                            st.error(t("err_missing_api_key"))
                            st.stop()

                        from WebSearcher import get_searcher

                        current_db = st.session_state.get("search_database", "PubMed (生物医学权威)")
                        searcher = get_searcher(current_db)
                        from LLM_SYS import BioBrainAgent

                        agent = BioBrainAgent(api_key=current_api_key,model=selected_model_id,base_url=base_url)

                        with st.spinner(t("msg_deep_parsing").format(pmid=pmid)):
                            abstract = searcher.fetch_abstract(pmid)
                            if len(abstract) < 50:
                                st.toast(t("toast_skip_short_abstract").format(pmid=pmid), icon="⏭️")
                                st.session_state.expansion_idx += 1
                                st.rerun()
                            else:
                                new_entities = agent.extract_entities_with_reflection(abstract,
                                                                                      use_reflection=use_reflection,
                                                                                      entity_lang=entity_language)
                                pmid_tag = f"PubMed:{pmid}"
                                v_hash = f"pmid:{pmid}"
                                register_or_update_paper(
                                    get_current_project_id(),
                                    doc_hash=v_hash,
                                    display_name=pmid_tag,
                                    aliases=[pmid_tag, str(pmid)],
                                    pmid=str(pmid),
                                    source_type="online_pmid"
                                )
                                for ent in new_entities:
                                    ent["doc_source"] = pmid_tag
                                    ent["doc_hash"] = v_hash

                                if pmid_tag not in st.session_state.analyzed_files:
                                    st.session_state.analyzed_files.append(pmid_tag)
                                st.session_state.analyzed_files = canonicalize_project_papers(get_current_project_id(), st.session_state.analyzed_files)

                                combined_entities_dict = st.session_state.master_entities + new_entities
                                new_relations = agent.extract_relations(abstract, combined_entities_dict)
                                for rel in new_relations:
                                    rel["doc_source"] = pmid_tag
                                    rel["doc_hash"] = v_hash
                                    rel["weight"] = 1

                                alignment_map = agent.align_global_entities(
                                    st.session_state.master_entities, new_entities)

                                aligned_new_entities = []
                                master_names = {ent["standard_name"]: ent for ent in
                                                st.session_state.master_entities}

                                for new_ent in new_entities:
                                    new_name = new_ent.get("standard_name")
                                    new_source = new_ent.get("doc_source", "")

                                    if new_name in master_names:
                                        master_ent = master_names[new_name]
                                        aliases = master_ent.setdefault("aliases", [])
                                        for new_alias in new_ent.get("aliases", []):
                                            if new_alias not in aliases and new_alias != new_name:
                                                aliases.append(new_alias)
                                        old_source = master_ent.get("doc_source", "")
                                        if new_source and new_source not in old_source:
                                            master_ent["doc_source"] = f"{old_source} | {new_source}"
                                    elif alignment_map and new_name in alignment_map:
                                        master_name = alignment_map[new_name]
                                        if master_name in master_names:
                                            master_ent = master_names[master_name]
                                            aliases = master_ent.setdefault("aliases", [])
                                            if new_name not in aliases:
                                                aliases.append(new_name)
                                            for new_alias in new_ent.get("aliases", []):
                                                if new_alias not in aliases and new_alias != master_name:
                                                    aliases.append(new_alias)
                                            old_source = master_ent.get("doc_source", "")
                                            if new_source and new_source not in old_source:
                                                master_ent["doc_source"] = f"{old_source} | {new_source}"
                                        else:
                                            aligned_new_entities.append(new_ent)
                                    else:
                                        aligned_new_entities.append(new_ent)

                                if alignment_map:
                                    for rel in new_relations:
                                        if rel.get("source") in alignment_map:
                                            rel["source"] = alignment_map[rel["source"]]
                                        if rel.get("target") in alignment_map:
                                            rel["target"] = alignment_map[rel["target"]]

                                all_rels = st.session_state.master_relations + new_relations
                                st.session_state.master_relations = consolidate_homogeneous_relations(all_rels)
                                st.session_state.master_entities.extend(aligned_new_entities)

                                # 📝 记录拓展历史日志（全局与工程专属）
                                log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] PMID: {pmid} | Title: {title}\n"
                                try:
                                    log_dir = ".expanded_logs"
                                    os.makedirs(log_dir, exist_ok=True)
                                    with open(os.path.join(log_dir, "expansion_history.txt"), "a", encoding="utf-8") as f:
                                        f.write(log_entry)
                                except Exception:
                                    pass

                                try:
                                    proj_dir = get_project_dir_by_id(get_current_project_id())
                                    with open(os.path.join(proj_dir, "expansion_history.txt"), "a", encoding="utf-8") as f:
                                        f.write(log_entry)
                                except Exception:
                                    pass

                        # 🚀 推进队列、弹出提示、自动重绘落盘并刷新
                        st.session_state.expansion_idx += 1
                        st.toast(t("toast_fused_success").format(pmid=pmid), icon="🎉")
                        redraw_and_update()
                        st.rerun()

                    elif btn_skip:
                        # ⏭️ 处理“跳过这篇”按钮逻辑
                        st.session_state.expansion_idx += 1
                        st.rerun()

                else:
                    # 🏁 队列全部处理完毕分支
                    st.success(t("msg_queue_all_done").format(total=total))
                    if "auto_run_expansion" in st.session_state and st.session_state.auto_run_expansion:
                        st.session_state.auto_run_expansion = False

                    if st.button(t("btn_clear_queue_and_results"), type="primary", use_container_width=True):
                        st.session_state.expansion_queue = []
                        st.session_state.pubmed_search_results = []
                        st.session_state.expansion_idx = 0
                        st.rerun()
    # -----------------------------------------
    # 模块三：🔗 智能桥接 (Intelligent Bridging)
    # -----------------------------------------
    elif ai_nav == "ai_nav_bridging":
        st.info(t("bridge_info"))
        all_node_names = [e.get("standard_name") for e in st.session_state.master_entities]

        col_a, col_b = st.columns(2)
        node_opts = ["__SELECT__"] + all_node_names
        with col_a:
            node_a = st.selectbox(t("lbl_bridge_node_a"), options=node_opts,
                                  format_func=lambda x: t("opt_select") if x == "__SELECT__" else x,
                                  key="bridge_node_a")
        with col_b:
            node_b = st.selectbox(t("lbl_bridge_node_b"), options=node_opts,
                                  format_func=lambda x: t("opt_select") if x == "__SELECT__" else x,
                                  key="bridge_node_b")

        max_depth = st.slider(t("lbl_max_depth"), min_value=2, max_value=5, value=4)

        if st.button(t("btn_start_bridge"), type="primary", use_container_width=True):
            if node_a == node_b and node_a != "__SELECT__":
                st.warning(t("warn_same_nodes_bridge"))
            elif node_a == "__SELECT__" or node_b == "__SELECT__":
                st.warning(t("warn_select_nodes_bridge"))
            else:
                from back_logic import GraphMiner

                with st.spinner(t("msg_scanning_paths")):
                    paths = GraphMiner.find_paths(st.session_state.master_relations, node_a, node_b,
                                                  max_depth)
                    st.session_state.bridge_paths = paths
                    st.session_state.bridge_nodes = (node_a, node_b)
                    st.session_state.bridge_report = None
                    st.rerun()

        # --- 状态机：展示桥接结果与三挡分流 ---
        if "bridge_paths" in st.session_state and st.session_state.get("bridge_nodes") == (node_a, node_b):
            paths = st.session_state.bridge_paths

            # 🌿 分流三：毫无联系（孤岛）
            if len(paths) == 0:
                st.warning(t("msg_island").format(depth=max_depth, a=node_a, b=node_b))
                needs_network_search = True
                search_prompt_msg = t("prompt_island_search").format(a=node_a, b=node_b)

            else:
                max_path_len = max(len(p) for p in paths)

                # 🌿 分流二：只有直接联系
                if max_path_len == 1:
                    st.info(t("msg_direct_only").format(a=node_a, b=node_b))
                    needs_network_search = True
                    search_prompt_msg = t("prompt_direct_search").format(a=node_a, b=node_b)

                # 🌿 分流一：存在多步复杂机制
                else:
                    st.success(t("msg_complex_paths").format(count=len(paths)))
                    needs_network_search = False

                    if st.session_state.bridge_report is None:
                        current_api_key = api_key.strip()
                        if not current_api_key:
                            st.error(t("err_need_key_for_summary"))
                        else:
                            with st.spinner(t("msg_writing_report")):
                                from LLM_SYS import BioBrainAgent

                                agent = BioBrainAgent(api_key=current_api_key,model=selected_model_id,base_url=base_url)
                                report = agent.explain_mechanism(node_a, node_b, paths,output_lang=st.session_state.get("ui_language", "zh"))
                                st.session_state.bridge_report = report
                                st.rerun()

                if st.session_state.bridge_report:
                    st.markdown(t("report_title"))
                    st.markdown(t("report_subtitle").format(a=node_a, b=node_b, count=len(paths)))
                    st.markdown(st.session_state.bridge_report)

            # --- 动态联网检索模块 (针对分流二和三) ---
            if needs_network_search:
                if st.button(f"🌐 {search_prompt_msg}", type="primary", key="btn_bridge_search"):
                    current_api_key = api_key.strip()
                    if not current_api_key:
                        st.error(t("err_missing_api_key"))
                    else:
                        with st.spinner(t("msg_evaluating_strategy")):
                            from LLM_SYS import BioBrainAgent
                            from WebSearcher import get_searcher

                            agent = BioBrainAgent(api_key=current_api_key,model=selected_model_id,base_url=base_url)
                            query = agent.generate_bridge_query(node_a, node_b)

                            if "[NO_RELATION_ERROR]" in query:
                                st.error(t("err_ai_rejected").format(a=node_a, b=node_b))
                            else:
                                st.info(t("msg_search_strategy").format(query=query))
                                current_db = st.session_state.get("search_database", "PubMed (生物医学权威)")
                                searcher = get_searcher(current_db)
                                results = searcher.search_articles(query, max_results=6)

                                if not results:
                                    st.warning(t("warn_no_mechanism_papers"))
                                else:
                                    st.session_state.bridge_found_docs = results
                                    st.session_state.pending_bridge_data = None
                                    st.rerun()

        # --- 核心改进：专属文献筛选表格 (让用户过滤噪音) ---
        if "bridge_found_docs" in st.session_state and st.session_state.bridge_found_docs:
            st.markdown("---")
            st.markdown(t("search_res_title_bridge").format(a=node_a, b=node_b))
            st.caption(t("search_res_caption_bridge"))

            b_df = pd.DataFrame(st.session_state.bridge_found_docs)
            col_sel_txt = t("col_deep_read")
            col_title_txt = t("col_paper_title")
            col_date_txt = t("col_pub_date")

            if col_sel_txt not in b_df.columns:
                b_df.insert(0, col_sel_txt, True)

            edited_b_df = st.data_editor(
                b_df,
                column_config={
                    col_sel_txt: st.column_config.CheckboxColumn(col_sel_txt, default=True),
                    "pmid": "PMID",
                    "title": col_title_txt,
                    "pub_date": col_date_txt
                },
                disabled=["pmid", "title", "pub_date"],
                hide_index=True,
                key="bridge_docs_editor",
                use_container_width=True
            )

            if st.button(t("btn_start_sniper_extract"), type="primary", use_container_width=True):
                selected_b_docs = edited_b_df[edited_b_df[col_sel_txt] == True]
                if selected_b_docs.empty:
                    st.warning(t("warn_check_one_paper_bridge"))
                else:
                    with st.spinner(t("msg_sniper_extracting")):
                        from WebSearcher import get_searcher
                        from LLM_SYS import BioBrainAgent

                        current_db = st.session_state.get("search_database", "PubMed (生物医学权威)")
                        searcher = get_searcher(current_db)
                        agent = BioBrainAgent(api_key=api_key.strip(),model=selected_model_id,base_url=base_url)

                        combined_abstracts = ""
                        for _, row in selected_b_docs.iterrows():
                            pmid = row['pmid']
                            abs_text = searcher.fetch_abstract(pmid)
                            if len(abs_text) > 50:
                                combined_abstracts += "\n\n" + t("fmt_paper_title").format(pmid=pmid,
                                                                                           title=row[
                                                                                               'title']) + "\n" + abs_text

                        bridge_result = agent.extract_bridge_mechanism(
                            node_a, node_b, combined_abstracts, entity_lang=entity_language
                        )
                        st.session_state.pending_bridge_data = bridge_result
                        st.session_state.bridge_found_docs = None
                        st.rerun()

        # --- 最终阶段：桥接结果展示与图谱融合审批区 ---
        if "pending_bridge_data" in st.session_state and st.session_state.pending_bridge_data:
            bridge_data = st.session_state.pending_bridge_data
            st.markdown("---")
            st.markdown(t("ai_bridge_report_title"))
            st.info(bridge_data.get("explanation", t("no_text_explanation")))

            new_rels = bridge_data.get("relations", [])
            new_ents = bridge_data.get("entities", [])

            if not new_rels:
                st.warning(t("warn_no_clear_chain"))
                if st.button(t("btn_clear_report")):
                    st.session_state.pending_bridge_data = None
                    st.rerun()
            else:
                st.success(t("msg_extracted_links").format(count=len(new_rels)))
                st.json(new_rels)

                col_acc, col_rej = st.columns(2)
                if col_acc.button(t("btn_accept_merge"), type="primary", use_container_width=True):
                    with st.spinner(t("msg_bulletproof_merging")):
                        from LLM_SYS import BioBrainAgent

                        agent = BioBrainAgent(api_key=api_key.strip(),model=selected_model_id,base_url=base_url)

                        alignment_map = agent.align_global_entities(st.session_state.master_entities,
                                                                    new_ents)

                        for new_ent in new_ents:
                            if new_ent["standard_name"] not in [e["standard_name"] for e in
                                                                st.session_state.master_entities]:
                                st.session_state.master_entities.append(new_ent)

                        if alignment_map:
                            for rel in new_rels:
                                if rel.get("source") in alignment_map: rel["source"] = alignment_map[
                                    rel["source"]]
                                if rel.get("target") in alignment_map: rel["target"] = alignment_map[
                                    rel["target"]]

                        all_rels = st.session_state.master_relations + new_rels
                        st.session_state.master_relations = consolidate_homogeneous_relations(all_rels)

                        for existing_rel in st.session_state.master_relations:
                            src = existing_rel.get("source", "").strip()
                            tgt = existing_rel.get("target", "").strip()
                            if (src == node_a and tgt == node_b) or (src == node_b and tgt == node_a):
                                if not existing_rel.get("is_shortcut", False):
                                    existing_rel["is_shortcut"] = True
                                    old_reason = existing_rel.get("reason", "")
                                    existing_rel["reason"] = t("fmt_downgrade_reason").format(
                                        old=old_reason)

                    st.session_state.pending_bridge_data = None
                    st.toast(t("toast_bridge_merged"), icon="🎉")
                    redraw_and_update()
                    st.rerun()

                if col_rej.button(t("btn_reject_clear"), use_container_width=True):
                    st.session_state.pending_bridge_data = None
                    st.rerun()

    # -----------------------------------------
    # 模块四：🤖 问 AI (Ask AI)
    # -----------------------------------------
    elif ai_nav == "ai_nav_ask":
        st.info(t("ask_ai_info"))

        ask_mode_keys = ["explain_node", "explain_edge"]
        explain_mode = st.radio(
            t("lbl_what_to_explain"),
            options=ask_mode_keys,
            format_func=lambda k: t(k),
            horizontal=True
        )

        if explain_mode == "explain_node":
            if not st.session_state.master_entities:
                st.warning(t("warn_no_nodes_in_graph"))
            else:
                node_list = sorted([e.get("standard_name") for e in st.session_state.master_entities])
                node_opts = ["__SELECT__"] + node_list
                target_node = st.selectbox(t("lbl_select_node_to_explain"), node_opts,
                                           format_func=lambda x: t(
                                               "opt_select") if x == "__SELECT__" else x,
                                           key="ask_ai_target_node")

                if target_node != "__SELECT__":
                    if st.button(t("btn_explain_node"), type="primary"):
                        current_api_key = api_key.strip()
                        if not current_api_key:
                            st.error(t("err_missing_api_key"))
                        else:
                            with st.spinner(t("msg_extracting_context").format(node=target_node)):
                                context_lines = []


                                # 为了兼容内部保存逻辑，获取关系的中文并翻译
                                def get_safe_rel_display(val):
                                    if val == "正作用": return t("rel_positive")
                                    if val == "负作用": return t("rel_negative")
                                    if val == "相关": return t("rel_correlation")
                                    return val


                                for r in st.session_state.master_relations:
                                    rel_type = r.get('relation_type', r.get('relation', '关联'))
                                    disp_rel = get_safe_rel_display(rel_type)
                                    safe_reason = str(r.get('reason', '无')).replace('{', '(').replace('}',
                                                                                                      ')')
                                    raw_doc = r.get('doc_source', '未知')
                                    safe_doc = re.sub(r'[^\w\u4e00-\u9fa5.-]', '_', str(raw_doc))[:60]

                                    if r.get("source") == target_node:
                                        context_lines.append(
                                            f"<edge>起点:[{target_node}] -> 终点:[{r.get('target')}] | 关系:[{disp_rel}] | 提取依据:[{safe_reason}] | 来源:[{safe_doc}]</edge>")
                                    elif r.get("target") == target_node:
                                        context_lines.append(
                                            f"<edge>起点:[{r.get('source')}] -> 终点:[{target_node}] | 关系:[{disp_rel}] | 提取依据:[{safe_reason}] | 来源:[{safe_doc}]</edge>")

                                local_ctx = "\n".join(context_lines) if context_lines else t(
                                    "msg_no_connections")

                                from LLM_SYS import BioBrainAgent

                                agent = BioBrainAgent(api_key=current_api_key,model=selected_model_id,base_url=base_url)
                                explanation = agent.explain_graph_element("节点", target_node, local_ctx,output_lang=st.session_state.get("ui_language", "zh"))

                                st.markdown(t("report_node_title"))
                                st.markdown(explanation)

        else:
            if not st.session_state.master_entities:
                st.info(t("err_empty_graph"))
            else:
                all_node_names = sorted([e.get("standard_name") for e in st.session_state.master_entities])
                st.markdown(t("lbl_select_two_nodes"))

                col_a, col_b = st.columns(2)
                node_opts = ["__SELECT__"] + all_node_names
                with col_a:
                    node_a = st.selectbox(t("lbl_select_node_a"), node_opts,
                                          format_func=lambda x: t("opt_select") if x == "__SELECT__" else x,
                                          key="ask_ai_edge_node_a")
                with col_b:
                    node_b = st.selectbox(t("lbl_select_node_b"), node_opts,
                                          format_func=lambda x: t("opt_select") if x == "__SELECT__" else x,
                                          key="ask_ai_edge_node_b")

                if node_a != "__SELECT__" and node_b != "__SELECT__":
                    if node_a == node_b:
                        st.warning(t("warn_same_node_ask"))
                    else:
                        matching_rels = []
                        for i, r in enumerate(st.session_state.master_relations):
                            src = r.get("source")
                            tgt = r.get("target")
                            if (src == node_a and tgt == node_b) or (src == node_b and tgt == node_a):
                                matching_rels.append((i, r))

                        if not matching_rels:
                            st.info(t("info_no_rel_between").format(a=node_a, b=node_b))
                        else:
                            def get_safe_rel_display(val):
                                if val == "正作用": return t("rel_positive")
                                if val == "负作用": return t("rel_negative")
                                if val == "相关": return t("rel_correlation")
                                return val


                            rel_options = {}
                            for idx, r in matching_rels:
                                rel_type = r.get('relation_type', r.get('relation', '关联'))
                                disp_rel = get_safe_rel_display(rel_type)
                                label = f"[{idx}] {r.get('source')} --[{disp_rel}]--> {r.get('target')}"
                                rel_options[label] = r

                            target_rel_label = st.selectbox(t("lbl_select_rel_to_explain"),
                                                            list(rel_options.keys()),
                                                            key="ask_ai_target_rel_line")

                            if st.button(t("btn_explain_edge"), type="primary"):
                                current_api_key = api_key.strip()
                                if not current_api_key:
                                    st.error(t("err_missing_api_key"))
                                else:
                                    target_rel = rel_options[target_rel_label]
                                    with st.spinner(t("msg_analyzing_edge")):
                                        rel_type = target_rel.get('relation_type',
                                                                  target_rel.get('relation', '关联'))
                                        disp_rel = get_safe_rel_display(rel_type)
                                        safe_reason = str(target_rel.get('reason', '无')).replace('{',
                                                                                                 '(').replace(
                                            '}', ')')
                                        raw_doc = target_rel.get('doc_source', '未知')
                                        safe_doc = re.sub(r'[^\w\u4e00-\u9fa5.-]', '_', str(raw_doc))[:60]

                                        local_ctx = f"<edge_data>\n起点: {target_rel.get('source')}\n终点: {target_rel.get('target')}\n关系类型: {disp_rel}\n文献提取依据: {safe_reason}\n文献来源: {safe_doc}\n</edge_data>"

                                        from LLM_SYS import BioBrainAgent

                                        agent = BioBrainAgent(api_key=current_api_key,model=selected_model_id,base_url=base_url)
                                        explanation = agent.explain_graph_element("关系连线", target_rel_label,
                                                                                  local_ctx,output_lang=st.session_state.get("ui_language", "zh"))

                                        st.markdown(t("report_edge_title"))
                                        st.markdown(explanation)

# ==========================================
# 📊 页面最底部常驻：前往图谱分析工作台
# ==========================================
if len(st.session_state.get("master_entities", [])) > 0:
    st.markdown("---")
    cb_nav1, cb_nav2, cb_nav3 = st.columns([1, 3, 1])
    with cb_nav2:
        if st.button(t("wb_nav_bottom_btn"), type="primary", key="btn_bottom_to_wb", width="stretch"):
            st.session_state.current_view = "workbench"
            st.rerun()

