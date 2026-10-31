<p align="right">
   <strong>English</strong> &nbsp;｜&nbsp; <a href="README.md">中文</a>
</p>

<div align="center">

# 🧬 Bio-Graph Agent

**A Cross-Literature Incremental Biological Knowledge Graph & Multimodal Mechanism Discovery Engine (v2.0)**

[![Multi-LLM Powered](https://img.shields.io/badge/LLM_Engine-Multi--Model_Ecosystem-blue.svg)](#)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)

</div>

This project focuses on addressing long-text structural extraction, top-tier multimodal mechanism pathway transcription, and scientific knowledge grounding in the biomedical domain. Leveraging the long-text context understanding, multimodal vision transcription, and rigorous reasoning capabilities of modern Large Language Models (LLMs), we built a **Bio-Knowledge Graph LLM Agent** that automatically extracts information from literature (PDFs), performs dual-layer reflective quality audits, transcribes upfront pathway diagrams multimodally, aligns cross-literature semantics, and generates highly interactive dynamic network topology graphs.

---

## 🎯 Project Motivation & Scientific Value (Why this project?)

A classic pain point in biomedical research is: **How do we find highly valuable novel research topics in a vast sea of literature?**

In biology, it is common that "Molecule A regulates Protein B" and "Protein B regulates Biological Phenotype/Physiological System C" are isolated findings scattered across papers from different years and journals. However, **proving the regulatory cascade pathway "A mediates and influences C via B" is often a research topic of immense scientific value—one that can directly support a major grant proposal or breakthrough study.**

Relying on manual reading to piece together such cross-literature cascade relationships is exceedingly slow and prone to oversight.

**Bio-Graph Agent** was created precisely to conquer this challenge as an "intelligent scientific research counselor":
1. **Multi-Paper Ingestion**: Rapidly and accurately parses complex, long academic papers, sniffing and multimodally transcribing Graphical Abstracts and core pathway figures upfront.
2. **Bridging Knowledge Silos**: Through AI semantic alignment and online PubMed/PMID validation, entities (molecules, proteins, physiological systems) and their hidden relations across disparate papers are cross-linked into a unified network.
3. **Intuitive Topological Rendering**: Automatically generates scientifically standardized dynamic regulatory topologies, enabling researchers to discover latent `A ➔ B ➔ C` cascade pathways at a single glance.

---

## 🎨 Topological Network Visual Specification

This system strictly follows biomedical regulatory network standards for visual rendering:
* **🟢 Green Solid Line + Arrow**: Positive regulation / Activation / Promotion ($A \to B$).
* **🔴 Red Solid Line + Arrow**: Negative regulation / Inhibition / Downregulation ($A \dashv B$).
* **⚪ Gray Solid Line**: General correlation / Undirected association ($A \sim B$).
* **⚠️ Gray Dashed Line (---)**: Mechanism shortcut edges (condensed macro pathways).
* **🟣 Purple Solid Line**: Hierarchical containment / Subunit / Family membership ($A \subset B$).
* **🔮 Dynamic Node Popularity**: Node radius scales dynamically with its degree centrality across papers. Hovering over any node reveals cross-literature aliases, supporting textual evidence, and exact literature provenance.

---

## 🌟 Core Features & Operational Guide

### ⚙️ I. Initial Environment & Retrieval Setup
* 🌐 **Flexible LLM Engines**: Supports cloud APIs (Tencent Cloud, OpenRouter, SiliconFlow, etc.) and native **Ollama** offline execution for handling confidential biomedical data without internet access.
* 📚 **Intelligent Retrieval Sources**: Built-in dual search engines for PubMed (authoritative biomedical literature) and Semantic Scholar (all-discipline AI-driven high-speed retrieval with automatic rate-limit fallback).
* ⚡ **Multi-Threaded Concurrency (Concurrency Workers)**: Configurable thread pool (1-8 workers) driving parallel chunk extraction and concurrent pathway figure transcriptions, slashing parsing time by 5-10x.
* 🛡️ **Relation Node Literature Grounding**: Automatically regex-verifies unregistered relation endpoints against the original paper text after chunk analysis. Real entities are recovered; hallucinated edges are strictly dropped.
* 👁️ **Dedicated Multimodal Vision Models**: Mount dedicated vision models (such as Gemini 2.5 Flash-Lite, Gemini 3.8 Flash, or Qwen 2.5 VL 72B) to transcribe pathway figures with precision.

### 📂 II. Knowledge Graph Construction (Dual Startup)
* 💡 **Zero-Based Cold Start**: Have no papers on hand? Simply input your research direction in natural language (e.g., *"Role of macrophage polarization in tumor microenvironment"*). AI formulates professional queries, retrieves cutting-edge papers, and constructs your initial graph.
* 📄 **Precise Local Literature Analysis**: Drag in a PDF, specify page ranges, or enable abstract-only mode. With "🔗 Append Mode" enabled, newly parsed literature integrates and grows continuously into the existing graph.
* 📷 **Upfront Multimodal Vision Strategy**: Choose between `Off (Pure Text)`, `🎯 Keyword Match` (e.g., *Fig 4*, *pathway*), and `⚡ Extract All in Pages` to capture graphical abstracts and mechanism figures.

### 🎨 III. Visual Empowerment Engine & Topological Standards
* 👁️ **Mechanism Shortcuts**: Reveals express pathways bypassing intermediate nodes to view macro upstream-to-downstream regulatory outcomes.
* 🧬 **Ontology Empowerment (Containment Feedback)**: Feed back weights of micro-entities to macro parent categories (e.g., Kinase Families), amplifying key signaling families.
* 🔥 **Node Empowerment (Degree Centrality)**: Hub genes with dense molecular connections dynamically scale up in node size.
* 🔗 **Edge Empowerment (Consensus Boldening)**: Regulatory edges validated by multiple papers and rich textual evidence chains are automatically bolded.

### 🤖 IV. AI Intelligent Research Assistant
In the right-hand 🤖 **Ask AI** panel, four automated tools are integrated:
* 🧹 **Smart Pruning (Tree Cleaning)**:
  - **Topological Diagnosis**: Scans whole-graph logic to recommend synonymous node merges, hierarchical containment, and shortcut downgrading.
  - **🧲 Symmetric Relation Folding**: Automatically merges bidirectional $A \leftrightarrow B$ correlation lines into a single undirected edge, summing popularity and preserving evidence chains without loss.
  - **🔄 Mutual Containment Deadlock Resolution**: Detects and resolves reciprocal containment loops ($A \subset B$ and $B \subset A$).
  - **🧪 Coarse Relation Purification**: Selectively consolidates or strips vague correlation lines without early information loss.
  - **✂️ Intent-Based Node Cleaning (Intent Pruning)**: Enter your research focus to eliminate off-topic organs, drugs, or bystander molecules.
* 🔍 **Smart Expansion**: Select core nodes; AI formulates advanced background queries to pull the latest PubMed papers and enqueues them for **Autopilot Continuous Assimilation**.
* 🔗 **Smart Bridging**: Select any two seemingly unconnected molecules A and B; AI navigates the network to find hidden multi-step cascades or searches external literature to uncover unverified regulatory hypotheses.
* 📖 **Ask AI (Micro-Mechanism Explanations)**: Click any node or edge on the canvas to pass its micro-context (raw extraction evidence and literature origin) to the LLM, producing an in-depth biological mechanism essay directly suitable for grant proposals.

### ✂️ V. Expert Manual Intervention (Data Management Center)
Provides surgical precision over the graph:
* **Global Cleanup**: Eliminate isolated orphan nodes or batch-rename messy literature source tags across all connected nodes and edges.
* **Node Surgery**: Rename nodes, **Force Merge** (transfer all edges of Node A to Node B and remove A), or **Split Node** (reallocate edges to two newly split sub-nodes).
* **Edge Operations**: Reverse upstream/downstream directions, switch relation categories, edit textual evidence, sever false-positive edges, or manually add newly confirmed experimental links.

### 💾 VI. Project Workspaces & Central Memory Vault (Self-Healing Architecture)
* 🗂️ **Project Isolation (`projects/<project_id>/`)**: Every research topic has an independent workspace directory storing its private `graph_state.json` and attached PDFs. Freely create blank projects, switch between historical topics, or permanently delete old topics to reclaim disk space.
* 🏛️ **Central Memory Vault (`vault/`)**: All imported literature is registered in `literature_registry.json` via SHA-256 physical cryptographic hashes, preventing duplicate file storage across different projects.
* 🌐 **Automatic Standard Literature Renaming & Online PMID Alignment**: One-click online alignment with PubMed/DOI. Cryptic or temporary PDF filenames are automatically reconstructed into academic standards (e.g., `PMID: 35123456 | Article_Title.pdf`), and all node/edge provenance tags are updated seamlessly.
* 🩹 **Missing Literature Detection & Self-Healing Replenishment**: When moving projects across machines or relocating PDFs, the system automatically detects missing physical files, searches the Central Vault by file hash, and restores physical links with a single click.
* 📷 **Multimodal Figure Source Exemption**: Nodes and edges originating from mechanism figures carry tags such as `[Figure: Page1_Fig1 (P.1)]` and are permanently tied to their parent paper, enjoying auto-healing exemption.
* 💾 **Legacy Project Compatibility**: Seamless export and import of `.biokg` archive files.

### 📷 VII. Upfront Multimodal Vision Pathway Transcription & Spatial Clustering
In high-impact journals (Nature, Science, Cell), the densest regulatory conclusions are concentrated in Graphical Abstracts and molecular pathway diagrams:
* **Spatial Clustering Composite Algorithm (`_cluster_page_figures`)**: Automatically detects multi-layered and sliced raster layers in PDFs, merging overlapping fragments via disjoint-set clustering and rendering crisp 150-DPI composite images.
* **Balanced Vector Diagram Detection**: Accommodates single-page pathway posters (`drawings >= 10`) and multi-page papers (requiring figure keywords and `drawings >= 25`), avoiding false positives on plain text pages while reliably preserving real vector pathways.
* **Multi-Threaded Parallel Vision Transcription**: Concurrently transcribes candidate figures using dedicated multimodal models (Gemini 2.5/3.8 Flash, Qwen 2.5 VL 72B), injecting transcribed mechanism texts into the downstream extraction stream.

### ⚡ VIII. Chunk-Centric High-Purity Closed-Loop Pipeline & Multi-Threaded Concurrency
* **Chunk-Centric Localized Pipeline**: Solves the severe hallucination problem caused by global dictionary contamination. Each chunk extracts local entities and derives relations strictly grounded in local context.
* **High-Throughput Concurrency**: Built upon `concurrent.futures.ThreadPoolExecutor`, processing chunks and figures in parallel to achieve 5-10x parsing acceleration.

### 📊 IX. Graph Analysis Workbench & 4-Tier Escalation State Machine
* **Dedicated Second-Stage Scientific Workbench**: Decoupled from the primary builder canvas with smooth bidirectional navigation, dedicated to granular entity taxonomy classification and deep mechanism verb distillation (e.g. catalytic phosphorylation, competitive binding, transcriptional activation).
* **4-Tier Escalation State Machine**:
  - **Round 0 (0-Token Rule Pre-Filter)**: Leverages INN pharmaceutical suffixes, kinase/enzyme patterns, non-coding RNA naming conventions, and canonical gene databases for sub-millisecond classification without spending any LLM tokens.
  - **Round 1 (Topological Micro-Context Reasoning)**: Gathers node aliases, connected triples, and local edge evidence for compact, high-precision prompting. Ambiguous cases output `Uncertain`.
  - **Round 2 (Targeted In-Paper Regex Re-fetch)**: Only unresolved candidates trigger fast regex scanning over full PDF texts to retrieve focused sentences, bypassing expensive whole-paper reads.
  - **Round 3 (Biomedical World Knowledge Fallback)**: Engages global domain knowledge for final adjudication of stubborn concepts.
* **Authoritative Ontology Presets & Custom Schemas**: Pre-packaged with NIH BioRED universal standard (6+1 entity categories, 6 relation mechanisms), molecular signaling cascades, pharmacology & drug targets, and tumor microenvironment (TME) immunology, plus one-click custom JSON schema imports.
* **Sparse Dynamic Metrics Dashboard**: Exclusively renders categories with `count > 0` and provides expandable accordion drill-downs listing all member nodes, sub-types, round tags, and reasoning traces.
* **Scholar Manual Override Console (Human-in-the-Loop)**: Allows researchers to interactively reclassify entities or modify mechanism verbs with instant cache update and visual re-rendering.
* **Scientific Multi-Shape Categorical Topology & Collapsible Legend**: Maps entity types to distinctive geometric shapes (dot, diamond, triangle, square, hexagon, star, ellipse) and colors, featuring an embedded, semi-transparent, collapsible legend overlay.

---

## 🤖 The Core Role of the Multi-Agent System Across the Lifecycle

Rather than using LLMs as mere "text summarizers," this system builds an autonomous multi-agent framework tailored for rigorous biomedical research across model ecosystems (supporting Gemini, DeepSeek, GLM, Qwen, Tencent Hunyuan, and local private Ollama). Eight specialized agents coordinate across the graph lifecycle:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              🧠 LLM Agent Architecture Core                            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
       ┌────────────────────────────┬───────┴────────────────────┬───────────────────────┐
       ▼                            ▼                            ▼                       ▼
【Phase 1: Ingestion & Vision】 【Phase 2: Audit & Alignment】 【Phase 3: Topology Pruning】 【Phase 4: Expansion & Vault】
 1. Long-Context Ingestion Hub   3. Dual-Layer Reflection Insp. 5. Topology Pruning Expert    7. Web Exploration & Bridging
 2. Multimodal Figure Transcrib. 4. Cross-Literature Align Judge 6. Symmetric & Loop Resolver  8. Vault Self-Healing Auditor
```

### 1. Long-Context Ingestion Hub
Leverages modern long-context windows to act as a "scientific speed-reader," extracting standardized biological entities and regulatory mechanisms from multi-page PDFs without losing inter-paragraph context.

### 2. Multimodal Figure Transcriber
Collaborates with multimodal vision models to transcribe Graphical Abstracts and signaling pathway diagrams into structured biochemical cascade texts.

### 3. Dual-Layer Reflection Inspector
Employs a two-stage "Initial Extraction - Reflection Audit" pipeline. The reflection auditor cross-examines the draft against original sentences, retaining only edges with incontrovertible evidence.

### 4. Cross-Literature Semantic Alignment Judge
Overcomes surface lexical barriers. By comparing entity names and alias lists across papers, it executes semantic-level reconciliation (e.g., mapping `p53` and `TP53`), weaving isolated findings into a cohesive network.

### 5. Topology Pruning Expert
Reads full-graph topological logic to recommend shortcut downgrades, hierarchical memberships, and intent-based pruning of off-topic branches.

### 6. Symmetric & Hierarchy Loop Resolver
Automatically merges bidirectional correlation redundancies ($A \leftrightarrow B$) into undirected bonds and resolves reciprocal containment deadlocks ($A \subset B$ and $B \subset A$).

### 7. Web-Browsing Explorer (Expansion & Bridging Guide)
Generates targeted PubMed queries to autonomously retrieve literature for isolated nodes, and charts multi-step bridging cascades between distant molecules.

### 8. Literature Registry & Auto-Healing Auditor
Tracks literature via SHA-256 physical cryptographic hashes, aligns papers with official PubMed PMIDs, and executes automated self-healing across devices.

### 9. Ontology & Mechanism Escalation Architect
Orchestrates the 4-tier escalation state machine across zero-token rule filters, micro-context reasoning, in-paper sentence re-fetching, and domain knowledge fallbacks to elevate coarse graphs into fine-grained scientific mechanism networks.

---

## 🌟 Core Technical Highlights

1. **Long-Document Structured Extraction**: Ingests multi-page biomedical PDFs directly to produce structured JSON containing standard names, aliases, interaction types, and verbatim evidence.
2. **First-of-its-Kind Agent Dual-Layer Reflection**: An independent quality auditor checks every extraction against the source text, discarding unsupported claims.
3. **Cross-Literature Alignment & Automatic Online PMID Renaming**: Dynamically meshes multi-paper graphs and automatically renames literature files into standard `PMID: XXXXXXXX | Title.pdf` formats via PubMed/DOI.
4. **Dynamic Topological Visualization**: Powered by Pyvis, featuring degree-centrality node scaling and consensus-weighted edge boldening without external network dependencies.
5. **Intelligent Pruning, Symmetric Folding & Intent Cleaning**: Diagnoses graph topology, folds bidirectional correlation lines, resolves containment deadlocks, and strips off-topic nodes via natural language prompts.
6. **Autonomous Expansion & Hidden Path Bridging**: Expands peripheral nodes in autopilot mode and mines multi-step cascade pathways between unlinked targets.
7. **Micro-Context Biological Interpretation**: Select any node or edge to feed its evidence and citation provenance to the LLM, producing grant-ready mechanism reports.
8. **Project Isolation & Central Memory Vault**: Workspace-level physical isolation with cross-project SHA-256 deduplication and one-click self-healing for missing files.
9. **Chunk-Centric Localized Pipeline & Multi-Threading**: Independent chunk extraction eliminates global hallucination contamination while concurrency boosts throughput by 5-10x.
10. **Spatial Clustering & Balanced Vector Pathway Extraction**: Disjoint-set clustering merges fragmented image slices into pristine mechanism figures, while balanced vector thresholds accommodate both posters and papers.
11. **Graph Analysis Workbench & 4-Tier Escalation State Machine**: Features zero-token pre-filtering, topological micro-context reasoning, targeted in-paper regex re-fetch, sparse dynamic dashboards, and expert manual override consoles.

---

## 📂 Core Code Architecture

```text
├── app.py                      # Streamlit frontend UI (workspaces, project switching, analysis workbench)
├── back_logic.py               # Central execution pipeline (chunk-centric concurrency & vision workflow)
├── workbench_engine.py         # Graph Analysis Workbench engine (rule pre-filters, ontology management, 4-tier escalation)
├── ontology_presets/           # Authoritative biomedical ontology presets (NIH BioRED, signaling, pharmacology, TME)
├── LLM_SYS.py                  # LLM Agent core (reflection, symmetric pruning, alignment, transcription)
├── IO_SYS.py                   # Infrastructure layer (PyMuPDF spatial clustering, vector rendering, workbench Pyvis visualizer)
├── WebSearcher.py              # Web retrieval module (PubMed / Semantic Scholar search & PMID lookup)
├── projects/                   # Dedicated project workspaces
│   └── <project_id>/
│       ├── graph_state.json    # Project graph persistence state
│       └── papers/             # Bound physical PDF files
├── vault/                      # Central Memory Vault
│   └── literature_registry.json # Global SHA-256 fingerprint & metadata registry
├── requirements.txt            # Project dependencies
├── run.sh                      # Linux / WSL startup script
├── start.bat                   # Windows startup script
└── README.md                   # Project documentation (Chinese)
```

---

## 🚀 Quick Start

### 1. Clone Repository & Install Dependencies

```bash
git clone https://github.com/24238-T-34/Bio-Graph-Agent
cd Bio-Graph-Agent
pip install -r requirements.txt
```

### 2. Configure and Run Project

* **Method 1 (Linux / WSL)**:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```
* **Method 2 (Windows)**:
  Double-click `start.bat` (or `start_wsl.bat`).
* **Method 3 (Standard Streamlit Command)**:
  ```bash
  streamlit run app.py
  ```

Open the Streamlit interface in your browser, enter your API Key in the sidebar (or run key-free using local Ollama), upload PDFs, and watch your dynamic biological knowledge graph emerge!

---

## 📅 Future Work

* [x] **Secondary Correlation Filter & Pruning**: LLM-driven topology diagnostics, intent-based pruning, and orphan elimination. (✅ **v1.0 Completed & Merged**)
* [x] **Multi-Threaded Concurrency & Parallel Pipeline**: Chunk-centric parallel extraction and concurrent pathway figure transcriptions, boosting throughput by 5-10x. (✅ **v2.0 Completed & Merged**)
* [x] **Multimodal Pathway Figure Extraction**: Spatial clustering for composite figures, balanced vector diagram detection, and parallel vision transcription with Gemini/Qwen-VL. (✅ **v2.0 Completed & Merged**)
* [x] **Project Isolation & Central Memory Vault**: Physical project isolation, global SHA-256 deduplication, automatic online PMID renaming, and physical self-healing. (✅ **v2.0 Completed & Merged**)
* [x] **Symmetric Relation Folding & Containment Deadlock Resolution**: Bidirectional correlation consolidation and hierarchical loop resolution. (✅ **v2.0 Completed & Merged**)
* [ ] **Multi-Attribute Topological Coloring**: Categorical coloring based on entity types (genes, diseases, compounds, phenotypes) for heightened biological intuition.
* [ ] **LLM Dynamic Routing Engine / MoE**: Two-tier model routing directing routine text cleaning to high-speed models, while semantic alignment, reasoning, and vision tasks run on premier LLMs.
* [ ] **Hypothesis & Writing Copilot**: Multi-page AIGC writing bay capable of autonomous structural hole exploration and one-click grant background draft generation.
* [ ] **Edge Metadata & Biological Conditions**: Conditional constraints (cell lines, mutations, experimental parameters) attached to edges as hover tooltips, separating empirical facts from topological hypotheses.
