<p align="right">
    <strong>English</strong> &nbsp;｜&nbsp; <a href="README.md">中文</a>
</p>

<div align="center">

# 🧬 Bio-Graph Agent

**A Cross-Literature Incremental Biological Knowledge Graph & Multimodal Mechanism Discovery Engine (v2.0)**

[![Multi-LLM Powered](https://img.shields.io/badge/LLM_Engine-Multi--Model_Ecosystem-blue.svg)](#)
[![Graph Theory Engine](https://img.shields.io/badge/Graph_Theory-Pathway_Decomposition-orange.svg)](#)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
* **🔴 Red Solid Line + Arrow**: Negative regulation / Inhibition / Downregulation ($A \dashv B$, bolded to highlight blocking effects).
* **⚪ Gray Line (No Arrow)**: General correlation / Undirected association ($A \sim B$) (rendered as solid gray `#aaaaaa` on the main canvas; rendered as muted dashed gray `#9ca3af` in the Stage 1 analysis topology).
* **⚠️ Gray Dashed Line (---)**: Mechanism shortcut edges (condensed macro pathways). Displayed as dark gray dashed lines (`#555555`) with a warning tooltip on the main graph, toggleable on/off.
* **🟣 Purple Solid Line**: Hierarchical containment / Subunit / Family membership ($A \subset B$).
* **🔮 Dynamic Node Heat**: Node size scales dynamically with degree centrality in the multi-paper network. Hovering over any node reveals cross-literature aliases, supporting textual evidence, and literature provenance.
* **🔀 Topological Dynamic Layout Standards**:
  - **Main Knowledge Graph**: Driven by the **ForceAtlas2 physics engine** for organic community clustering;
  - **Local Signaling Pathways (Stage 2)**: Exclusively features one-click switching between **Hierarchical Left-to-Right DAG Layout** (aligning with Nature/Cell standards where cascades flow from upstream to downstream) and ForceAtlas2 physics layout.

---

## 🌟 Core Features & Operational Guide

### ⚙️ I. Sidebar Configuration & Retrieval Environment (Sidebar Settings)
* 🌐 **Multi-Model Ecosystem**: Supports cloud APIs (Tencent Cloud, OpenRouter, SiliconFlow, custom Base URLs) and native **Ollama** offline execution for handling confidential biomedical data key-free without internet access.
* 🌍 **Full-Stack Native Bilingual Support (i18n)**: Seamless one-click English/Chinese switching in the sidebar configuration popover, covering UI labels, node tooltips, Stage 1/2/3 state machines, edge polarities, and literature referee verdicts with robust reverse-lookup guards against KeyError.
* 📚 **Authoritative Retrieval Sources**: Built-in dual search engines for PubMed (authoritative biomedical literature) and Semantic Scholar (all-discipline AI-driven high-speed retrieval with automatic rate-limit fallback).
* 👁️ **Dedicated Multimodal Vision Models**: Mount dedicated vision models (Gemini 2.5 Flash-Lite, Gemini 3.8 Flash, Qwen 2.5 VL 72B, or custom model IDs) with custom model suffix support (e.g. `:free`) or auto-follow the main model (`auto`).
* 🛠️ **Module Safeguards**: Toggle the "Expert Data Management Center" and "AI Research Copilot" modules on or off.
* 📚 **Literature History**: Top-level sidebar lists all PDFs currently bound to the active project.

### 📂 II. Knowledge Graph Construction & Extraction Settings (Extraction Pipeline)
* 💡 **Zero-Based Cold Start**: Displayed when the graph is empty and no local PDF is uploaded. Simply input your research direction in natural language (e.g., *"Role of macrophage polarization in tumor microenvironment"*). AI formulates professional queries, retrieves cutting-edge papers, performs sliding-window relevance auditing, and constructs your initial graph.
* 📄 **Precise Local Literature Analysis**: Drag in an academic PDF, specify page ranges, or enable abstract-only mode. With "🔗 Append Mode" enabled, newly parsed literature integrates and grows continuously into the existing graph.
* 🧬 **Abstract Phenotype & Behavioral Endpoints (`enable_abstract_entities`)**: Transcends molecular-only extraction by identifying cellular fates (apoptosis, autophagy, pyroptosis), tissue microenvironment pathologies, and systemic/behavioral endpoints (cognitive decline, locomotor deficits, cytokine storms).
* 🏷️ **Concept Synthesis & Alias Backfill (`enable_concept_synthesis`)**: A secondary normalization toggle paired with abstract entities. Summarizes scattered, verbose descriptive outcomes into canonical standardized entities, backfills original sentences into the entity's `aliases` list, and attaches inferred markers `(总结)` / `(inferred)`.
* 🛡️ **Relation Node Literature Grounding**: Automatically regex-verifies unregistered relation endpoints against the original paper text after chunk analysis. Real entities are recovered; hallucinated edges are strictly dropped.
* ⚡ **Multi-Threaded Concurrency (Concurrency Workers)**: Configurable thread pool (1-8 workers) driving parallel chunk extraction and concurrent pathway figure transcriptions, slashing parsing time by 5-10x.
* 🌐 **Entity & Relation Language Control**: Select `Keep Original Language`, `Force Translation to Chinese`, or `Force Translation to English`.
* 🔄 **Dual-Layer Reflection Audit Toggle**: Turn rigorous verbatim cross-examination on or off to balance precision with speed.
* 📷 **Upfront Multimodal Vision Strategy**: Choose between `🚫 Off (Text Only)`, `🎯 Keyword Match` (e.g., *Fig 4*, *pathway*), and `⚡ Extract All in Pages` to capture graphical abstracts and mechanism figures.

### 🎨 III. Visual Empowerment Engine & Topological Standards
* 👁️ **Show Mechanism Shortcuts**: Reveals express pathways bypassing intermediate nodes to view macro upstream-to-downstream regulatory outcomes.
* 🧬 **Ontology Empower (Parent Node Enlarge)**: Feed back weights of micro-entities to macro parent categories (e.g., Kinase Families), amplifying key signaling families.
* 🌟 **Node Empower (Hub Nodes Enlarge)**: Hub genes with dense molecular connections dynamically scale up in node size.
* 🔗 **Edge Empower (High-Freq Edges Thicken)**: Regulatory edges validated by multiple papers and rich textual evidence chains are automatically bolded.

### 🤖 IV. AI Intelligent Research Assistant (AI Research Copilot)
Located directly beneath the main canvas, featuring four primary analytical modules:
* 🧹 **Smart Pruning (Tree Cleaning)**:
  Click `🚀 Start AI Smart Graph Diagnosis` to send full-graph entities, relations, and textual evidence to the LLM for structured diagnosis and review:
  - **7 Structured Graph Cleaning Action Directives**:
    1. 🧲 **Merge Synonyms (`MERGE`)**: Identifies synonymous entities referring to the exact same concept (e.g., "Urinary Tract Infection" and "UTI"), merging them and redirecting all incident edges;
    2. 🌳 **Build Hierarchy (`HIERARCHY`)**: Detects category/containment relationships (e.g., "E. coli" contains "UTI89"), adding a `[Contains]` edge; preserves functional mechanisms via the **Mechanism Coexistence Rule** (ensuring pharmacological or signaling links connected to both parent and child are strictly retained);
    3. 🔄 **Resolve Containment Cycles (`RESOLVE_CYCLE`)**: Resolves reciprocal containment deadlocks ($A \subset B$ and $B \subset A$), identifying canonical parent vs. child tiers and merging reverse evidence, provenance, and heat without loss;
    4. 🔀 **Fold Symmetric Relations (`FOLD_SYMMETRIC`)**: Collapses bidirectional $A \leftrightarrow B$ correlation lines into a single canonical edge, summing popularity and merging evidence chains without loss;
    5. 📉 **Downgrade Shortcuts (`DOWNGRADE`)**: Flags direct bypass edges induced by multi-step cascades ($A \to B \to C$ causing $A \to C$) as mechanism shortcuts, applying dashed visual downgrading (`is_shortcut = True`);
    6. ✂️ **Remove Transitive Edges (`REMOVE`)**: Permanently strips redundant shortcut containment edges across 3+ nested hierarchical tiers (e.g., removing transitive edges between grandparent and grandchild); backed by **3-tier data loss prevention safeguards** (automatically converting to purification if mechanisms overlap, to hierarchy merging if opposite containment exists, or to symmetric folding if reverse correlation is present);
    7. 🧪 **Purify Coarse Relations (`PURIFY`)**: Where an explicit biological mechanism (positive, negative, or containment) already connects two nodes, merges accompanying vague "related" edges into the explicit mechanism, transferring evidence and heat without loss.
  - **Action Priority & Scholar Safe Decision Strategy**:
    Suggestions are intelligently ranked by structural certainty: structural repairs (MERGE, RESOLVE_CYCLE, HIERARCHY, FOLD_SYMMETRIC, DOWNGRADE) are placed at the top and **checked by default**; invasive removals (REMOVE) and coarse relation purification (PURIFY, placed at the very bottom) are **unchecked by default**, leaving the ultimate decision on edge pruning entirely to the researcher;
  - **Post-Pruning Automatic Deduplication**:
    After executing optimizations, the engine automatically runs homogeneous relation consolidation (`consolidate_homogeneous_relations`), merging duplicate edges along identical directions with full evidence aggregation.
  - ✂️ **Intent-Based Node Cleaning (Intent Pruning)**:
    Input your core research focus prompt; the LLM examines each node alongside its **neighbor topological context**, categorizing nodes across 3 risk tiers:
    - `Completely_Irrelevant`: Zero relevance to research focus or cross-domain homonyms (pre-checked by default);
    - `Potentially_Irrelevant`: Divergent from core focus or weakly connected bystanders (pre-checked by default);
    - `Low_Relevance`: Relevant to general discipline but peripheral to specific questions (unchecked by default, leaving decisions to the user);
    - `Core Relevant`: Automatically protected and omitted from the purge list.
    Results are rendered in an interactive **`st.data_editor` table** ("Off-topic Nodes Purge List" with delete toggles, node names, connected neighbors, risk tiers, and AI rationale). Clicking `🚨 Execute Purge on Checked Nodes` erases selected entities and **cascades to strip all connected edges**.
  - 🏷️ **Smart Paper Title Identification & Renaming**:
    Inside the pruning accordion, click `🏷️ Start Identification & Renaming` to query PubMed/DOI online, sanitize copyright noise with a lightweight LLM, reconstruct filenames into academic standards (e.g., `PMID: 35123456 | Article_Title.pdf`), and seamlessly synchronize all provenance tags.
* 🔍 **Smart Expansion**: Select core nodes; AI formulates advanced background queries to pull the latest PubMed papers and enqueues them for **Autopilot Continuous Assimilation**.
* 🔗 **Smart Bridging**: Select any two seemingly unconnected molecules A and B; AI navigates the network to find hidden multi-step cascades or searches external literature to uncover unverified regulatory hypotheses.
* 🤖 **Ask AI (Micro-Mechanism Explanations)**: Click any node or edge on the canvas to pass its micro-context (raw extraction evidence and literature origin) to the LLM, producing an in-depth biological mechanism essay directly suitable for grant proposals.

### ✂️ V. Expert Manual Intervention (Data Management Center)
Provides surgical precision over the graph:
* **Quick Tools**: Eliminate isolated orphan nodes or batch-rename literature source tags across all connected nodes and edges.
* **Node Surgery**: Rename nodes, **Force Merge** (transfer all edges of Node A to Node B and remove A), or **Split Node** (reallocate edges to two newly split sub-nodes).
* **Edge Operations**: Reverse upstream/downstream directions, switch relation categories, sever false-positive edges, or manually add newly confirmed experimental links.

### 💾 VI. Project Workspaces & Local Registry (Project Vault)
* 🗂️ **Project Isolation (`projects/<project_id>/`)**: Every research topic has an independent workspace directory storing its private `project.biokg` (the single source of truth for graphs, analysis states, and workbench caches), `paper_registry.json` (local literature hash registry), and attached PDFs. Freely create blank projects, switch between historical topics, or permanently delete old topics.
* 🏛️ **Project-Level Literature Hash Registry (`paper_registry.json`)**: All imported literature is indexed via SHA-256 physical cryptographic hashes, preventing duplicates and tracking aliases within the project.
* 🩹 **Missing Literature Detection & Self-Healing Replenishment**: When moving projects across machines or relocating PDFs, the system automatically detects missing physical files, searches by file hash, and restores physical links.
* 📷 **Multimodal Figure Source Exemption**: Nodes and edges originating from mechanism figures carry tags such as `[Figure: Page1_Fig1 (P.1)]` and are permanently tied to their parent paper, enjoying auto-healing exemption.
* 💾 **Legacy Project Compatibility**: Seamless export and import of `.biokg` and `json` archive files.

### 📷 VII. Upfront Multimodal Vision Pathway Transcription & Spatial Clustering
In high-impact journals (Nature, Science, Cell), the densest regulatory conclusions are concentrated in Graphical Abstracts and molecular pathway diagrams:
* **Spatial Clustering Composite Algorithm (`_cluster_page_figures`)**: Automatically detects multi-layered and sliced raster layers in PDFs, merging overlapping fragments via disjoint-set clustering and rendering crisp 150-DPI composite images.
* **Balanced Vector Diagram Detection**: Accommodates single-page pathway posters (`drawings >= 10`) and multi-page papers (requiring figure keywords and `drawings >= 25`), avoiding false positives on plain text pages while reliably preserving real vector pathways.
* **Multi-Threaded Parallel Vision Transcription**: Concurrently transcribes candidate figures using dedicated multimodal models (Gemini 2.5/3.8 Flash, Qwen 2.5 VL 72B), injecting transcribed mechanism texts into the downstream extraction stream.

### ⚡ VIII. Chunk-Centric High-Purity Closed-Loop Pipeline & Multi-Threaded Concurrency
* **Chunk-Centric Localized Pipeline**: Solves the severe hallucination problem caused by global dictionary contamination. Each chunk extracts local entities and derives relations strictly grounded in local context.
* **High-Throughput Concurrency**: Built upon `concurrent.futures.ThreadPoolExecutor`, processing chunks and figures in parallel to achieve 5-10x parsing acceleration.

### 📊 IX. Graph Analysis Workbench & Three-Tier Classification (Stage 1)
As an advanced auxiliary analytical platform, it decouples from the primary builder canvas with smooth bidirectional navigation, dedicated to granular entity taxonomy classification and deep mechanism verb distillation (e.g. catalytic phosphorylation, competitive binding, transcriptional activation):
* **Three-Tier Escalation Architecture**:
  - **Tier 1 (0-Token Rule Pre-Filter, Round 0)**: Leverages INN pharmaceutical suffixes (`-mab`, `-nib`, `-stat`), kinase/enzyme patterns (`-ase`, `-kinase`), ncRNA naming conventions, and canonical gene databases for sub-millisecond classification without spending any LLM tokens (40%+ resolution rate on standard graphs);
  - **Tier 2 (Micro-Context Reasoning & Targeted In-Paper Grounding, Round 1 & 2)**: Gathers node aliases, connected triples, and local edge evidence for compact, high-precision prompting. If ambiguous (`Uncertain`), scans the raw PDF using targeted regex to retrieve only sentences containing the entity and its neighbors;
  - **Tier 3 (Biomedical World Knowledge Fallback & Human-in-the-Loop Override, Round 3 & Human Override)**: Resolves edge cases via global domain knowledge, paired with a **Scholar Manual Override Console** allowing researchers to interactively reclassify entities or modify mechanism verbs with instant cache update and visual re-rendering.
* **6 Built-in Authoritative Ontology Presets & Cross-Disciplinary Schemas**: Built-in NIH BioRED universal benchmark (6+1 entity types, 6 mechanism types), alongside molecular signaling, pharmacology targets, tumor microenvironment, ecosystem science, and ethology schemas; click `📁 Open Rules Folder` to inspect bilingual specification guides or hot-load custom `.json` rules.
* **Sparse Dynamic Dashboard & Conditional Rendering**: Renders metrics, canvas, and override consoles only when graph analysis results exist, showing exclusively categories with `count > 0` and interactive accordions.
* **Scientific Multi-Shape Categorical Topology & Bilingual Legend**: Maps entity types to distinct geometric shapes and colors, featuring an embedded, semi-transparent, collapsible bilingual legend overlay.

---

### 🔬 X. Signaling Pathway & Autoregulatory Circuit Decomposition (Stage 2)

As an auxiliary module for downstream causal extraction and secondary analysis, it leverages graph theory algorithms and LLM research focus prompts to distill cohesive local signaling pathways from complex macro networks:

#### 1. In-Depth Graph-Theoretic Algorithms & Theoretical Foundations
To overcome the common pitfalls of naive path searches—namely, "whole-graph collapse into a single high-frequency hub molecule (e.g., TP53, AKT)" and "combinatorial explosion of long paths"—Stage 2 implements a hybrid four-stage graph-theoretic architecture:
* **Hybrid Seed Scoring Model**:
  Computes a composite driver score for every node in the graph (out of 10.0 points):

$$
\text{TotalScore}(v) = \text{GraphScore}(v) + \text{SemanticScore}(v)
$$

  - **Graph-Theoretic Topological Base (5.0 Points Max)**:

$$
\text{GraphScore}(v) = 2.5 \times \frac{\text{Deg}(v)}{\max_{u} \text{Deg}(u)} + 2.5 \times \text{Betweenness}(v)
$$

    Balances local degree connectivity with global information flow bottleneck control;
  - **Scientific Intent & Biological Category Weighting (5.0 Points Max)**: If researchers enter a prompt in "Academic Focus Guidance", LLM batch scores candidate seeds (0.0 to 5.0 points); if left blank, physical biological driver weights are assigned based on Stage 1 ontology categories (receptors and transcription factors $+2.0$, kinases/enzymes $+1.8$, adaptor/messenger proteins $+1.6$, chemical drugs $+1.4$, biological processes $+1.2$).
* **MMR Kinetic Impedance & Exploration Diversity Penalty**:
  During multi-pathway generation, introduces a dynamic frequency-based damping penalty $\text{Score}\sb{\text{decay}}(v) = \text{BaseScore}(v) \times (1 - \lambda)^{\text{Count}\sb{\text{visited}}(v)} \quad (\lambda = 0.5)$, paired with a biochemical kinetic impedance model (activation/promotion assigned low impedance, inhibition/blocking assigned high impedance, vague correlations penalized), heuristic causal flow traversal forces the algorithm to explore differentiated branch cascades rather than repeating the same dominant hubs.
* **Strict Induced Subgraph Invariance**:
  Each decomposed pathway satisfies the mathematical definition of an induced subgraph $E\sb{\text{sub}} = \{(u, v) \in E \mid u, v \in V\sb{\text{sub}}\}$, where all connections strictly inherit empirical evidence and mechanism attributes from the master graph, **guaranteeing zero hallucinated edges**.
* **Tarjan / Johnson Autoregulatory Feedback Loop Mining**:
  Executes simple directed cycle search within the induced subgraph, filtering for compact circuits of length $2 \le L \le 5$, and computing the product of edge regulatory polarities $\text{Polarity}\sb{\text{loop}} = \prod\sb{e \in \text{Loop}} \text{Sign}(e)$, which automatically distinguishes **Negative Feedback Loops (Homeostasis)** (maintaining cellular balance) from **Positive Feedback Loops (Cascading Amplification)** (driving irreversible cell fate commitments).

#### 2. Workflow & Export
1. Click `📊 Go to Graph Analysis Workbench ➔` from the main canvas and scroll to `🔬 Signaling Pathway & Autoregulation Circuit Decomposition`;
2. (Optional) Enter your research focus prompt (e.g., *"Focus on mitochondrial dysfunction and Tau aggregation"*), or leave blank for autonomous graph-theoretic exploration;
3. Specify the target pathway limit (range 1-5, default 3), and click `🚀 Run Pathway Decomposition & Modeling`;
4. Review each decomposed pathway card displaying AI-generated biochemical titles, mechanistic summaries, autoregulatory feedback loop diagnostics, and interactive canvases (**supporting one-click switching between Hierarchical LR DAG and ForceAtlas2 layouts**);
5. **Export & Secondary Downstream Analysis**: Export standalone interactive webpages (`HTML`), structured graph data (`JSON`), or **`🧬 Export as Project (.biokg)`**—which packages the circuit's nodes, edges, cited literature, and inherited ontology annotations, ready to be re-loaded into the main workspace as an independent sub-project for targeted autopilot expansion and bridging!

---

### 💡 XI. Hypothesis Deduction & Scientific Discovery Copilot (Stage 3)

As the core engine elevating the platform from "static knowledge extraction" to "autonomous scientific discovery," the system integrates a **Graph Topology Message Passing and Micro-Circuit Discovery Framework**, directly assisting researchers in uncovering high-value latent mechanisms and grant-ready proposals:

#### 1. Vectorized Graph Message Passing & Micro-Circuit Topologies
To transcend simplistic two-node direct link heuristics, the engine executes high-dimensional vectorized message passing flows across the entire network, capturing 3-to-4 node high-order causal fields:
* **0-Token Dense Semantic Embedding Initialization (24-dim)**:
  Zero API token cost. Stage 1 ontology categories (Gene_Protein, Chemical_Drug, Disease, Phenotype, Biological_Process, Organism_Taxon, etc.) are projected into orthogonal base vectors and combined with token hash perturbations (ensuring homologous terms share local affinity), normalized via L2 into a compact 24-dimensional semantic base $h\sb{v}^{(0)}$.
* **Polarity-Aware Graph Topology Message Passing Flow**:
  Semantic vectors propagate directionally along regulatory edges according to biological polarity weights:

$$
\text{Pol}(e) = \begin{cases}
+1.0 & (\text{Positive / Activation / Promotion}) \\
-1.0 & (\text{Negative / Inhibition / Suppression}) \\
+0.5 & (\text{Correlation / Association}) \\
+0.3 & (\text{Hierarchy / Part\_Of})
\end{cases}
$$

  A 2-hop neighborhood aggregation and residual update is performed globally:

$$
h_v^{(t+1)} = \text{Normalize}\left(0.5 \cdot h_v^{(t)} + 0.5 \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \text{Pol}(u, v) \cdot \text{Dir}(u, v) \cdot h_u^{(t)}\right)
$$

  After propagation, each node vector absorbs its local 2-hop regulatory micro-environment field. Computing vector cosine similarity $\cos(h\sb{u}, h\sb{v}) = \frac{h\sb{u} \cdot h\sb{v}}{\|h\sb{u}\| \|h\sb{v}\|}$ allows the engine to accurately detect latent functional affinity between unconnected entities.
* **6 Scientific Exploration Micro-Circuit Topologies**:
  1. 🤖 **Auto Mode (Auto-Perception Smart Recommendation)**: Adaptively generates an ensemble portfolio (2 shortcuts + 1 cluster bridge + 1 homeostasis loop + 1 peripheral growth), establishing an instant panoramic hypothesis matrix;
  2. ⚡ **Detour Shortcuts (High-Dimensional Detour Shortcuts)**: Mines pairs that currently require intermediate mediators ($A \to M \to B$) but exhibit high semantic similarity $\cos(h\sb{A}, h\sb{B})$ and biological compatibility, hypothesizing direct bypass links ($A \dashrightarrow B$);
  3. 🔗 **Cluster Bridging (Autonomous Cluster Bridging)**: Identifies connected components and graph communities, scoring cross-cluster hub compatibility and topological bridging potential $\text{Score}\sb{\text{bridge}} = 2.2 \times \text{Compat}(C\sb{i}, C\sb{j}) + 0.3 \times (\text{Deg}\sb{i} + \text{Deg}\sb{j}) + 1.5 \times \cos(h\sb{i}, h\sb{j})$, bridging disparate pathway islands;
  4. ⚖️ **Homeostatic Feedback (Reverse Homeostatic Loops)**: Investigates multi-step feedforward cascades ($A \to B \to C$) to deduce reverse compensatory inhibition from terminal downstream to initial upstream ($C \dashrightarrow A$), uncovering self-balancing loops;
  5. 🌱 **Peripheral Growth (Peripheral Thin-Layer Growth)**: Extrapolates outward from peripheral low-degree nodes ($\text{Deg} \le 2$) toward core hub networks, inspiring high-novelty, pioneering grant proposals;
  6. 🎯 **Intent Exploration (Intent-Driven Directional Exploration)**: Uses natural language research queries as directional potential fields to prioritize and rank matching micro-circuits.

#### 2. Autonomous PubMed Literature Referee & 4-Class Adjudication
Directly links with online PubMed literature search pools. For candidate micro-circuits, it generates multi-angle academic search queries, retrieving abstracts for rigorous LLM-based adjudication:
* 🟢 **Known Science Fact**: Empirically documented in literature, backed by confidence scoring and verbatim citations;
* 🚀 **Novel Scientific Gap**: Rigorously validated as an unaddressed, high-potential research hypothesis with zero direct literature precedent;
* 🔗 **Cascade Pathway**: Indirect multi-step cascade substantiated across disparate papers, clarifying mediator transmission;
* ❓ **Inconclusive / Disputed**: Lacking sufficient published evidence or subject to conflicting findings.

#### 3. Deep Literature Expansion, Pipeline Ingestion & State Safeguards
* **Deep Literature Expansion (~8 Papers Re-Adjudication)**: For Inconclusive or Novel Gap cards, click `📖 Deep Search & Read More Literature (~8 papers re-adjudication)` to concurrently retrieve 8-10 PubMed abstracts for secondary authoritative re-adjudication;
* **Ingest Verified Hypothesis via Pipeline**: Click `📥 Ingest via Pipeline & Merge to Master Graph` to automatically trigger the entity and relation extraction pipeline, merging newly deduced intermediary nodes, regulatory edges, and biological attributes into the master graph;
* **Independent Sub-Project Export (`.biokg`)**: Export any hypothesis micro-circuit as a standalone `.biokg` project archive;
* **Workbench Safety Fallbacks & Bilingual Robustness**:
  - **Pruning Guidance**: Persistent entrance banners advise researchers to prune noise nodes first to maximize graph purity;
  - **Fallback Confirmation Dialogs**: Explicit warnings and confirmation popovers guard Stages 2 and 3 if invoked before Stage 1 (Ontology Categorization);
  - **Full-Width UI & Zero KeyError State Guards**: Full-width button styling (`width="stretch"`) and comprehensive reverse-lookup dictionary guards ensure seamless, 100% crash-free English/Chinese language toggling.

---

## 🤖 The Core Role of the Multi-Agent System Across the Lifecycle

This system coordinates specialized agents across the graph lifecycle:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              🧠 LLM Agent Architecture Core                            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
       ┌────────────────────────────┬───────┴────────────────────┬───────────────────────┐
       ▼                            ▼                            ▼                       ▼
【Phase 1: Ingestion & Vision】 【Phase 2: Audit & Alignment】 【Phase 3: Topology Pruning】 【Phase 4: Auxiliary Analysis】
 1. Long-Context Ingestion Hub   3. Dual-Layer Reflection Insp. 5. Topology Pruning Expert    7. Web Exploration & Bridging
 2. Multimodal Figure Transcrib. 4. Cross-Literature Align Judge 6. Symmetric & Loop Resolver  8. Literature Canonical Auditor
                                                                                               9. Ontology & Mechanism Architect
                                                                                               10. Pathway & Circuit Mining Expert
                                                                                               11. Hypothesis Deduction Copilot
```

### 1. Long-Context Ingestion Hub
Leverages modern long-context windows to extract standardized biological entities, abstract phenotype endpoints, and regulatory mechanisms from multi-page PDFs without losing inter-paragraph context.

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

### 8. Literature Canonical & Provenance Auditor
Tracks literature via SHA-256 physical cryptographic hashes, aligns papers with official PubMed PMIDs, updates canonical filenames, and synchronizes node/edge provenance tags.

### 9. Ontology & Mechanism Escalation Architect
Orchestrates the three-tier classification architecture across zero-token rule filters, micro-context reasoning, in-paper sentence re-fetching, and domain knowledge fallbacks to elevate coarse graphs into fine-grained scientific mechanism networks.

### 10. Pathway & Circuit Mining Expert
Combines graph-theoretic algorithms with LLM scientific reasoning to induce compact subgraphs, isolate shortcut edges, and produce peer-review-grade causal pathway reports and feedback diagnostics.

### 11. Hypothesis Deduction & Scientific Discovery Copilot
Transcends simplistic two-node bounds by executing multi-hop message passing flows across the global network to identify 3-to-4 node micro-circuits, coordinating 6 exploration modes, orchestrating online PubMed literature referees for 4-class adjudication, and driving deep literature expansion and smart bridging integration.

---

## 🌟 Core Technical Highlights

1. **Long-Document Structured Extraction**: Ingests multi-page biomedical PDFs directly to produce structured JSON containing standard names, aliases, interaction types, verbatim evidence, abstract phenotype endpoints, and synthesized canonical concepts.
2. **First-of-its-Kind Agent Dual-Layer Reflection**: An independent quality auditor checks every extraction against the source text, discarding unsupported claims.
3. **Cross-Literature Alignment & Automatic Online PMID Renaming**: Dynamically meshes multi-paper graphs and automatically renames literature files into standard `PMID: XXXXXXXX | Title.pdf` formats via PubMed/DOI.
4. **Dynamic Topological Visualization**: Powered by Pyvis, featuring degree-centrality node scaling and consensus-weighted edge boldening without external network dependencies, supporting ForceAtlas2 physics and Stage 2 Hierarchical LR DAG layouts.
5. **Intelligent Pruning, Symmetric Folding & Intent Cleaning**: Supports a 7-action structured cleaning system (MERGE, HIERARCHY, FOLD_SYMMETRIC, RESOLVE_CYCLE, DOWNGRADE, REMOVE, PURIFY) with 3-tier data loss safeguards, mechanism coexistence protection, priority-ranked review, and an interactive 3-tier risk intent purge table.
6. **Autonomous Expansion & Hidden Path Bridging**: Expands peripheral nodes in autopilot mode and mines multi-step cascade pathways between unlinked targets.
7. **Micro-Context Biological Interpretation**: Select any node or edge to feed its evidence and citation provenance to the LLM, producing grant-ready mechanism reports.
8. **Project Isolation & Local Literature Self-Healing**: Workspace-level physical isolation with `project.biokg` as single source of truth, SHA-256 physical hash deduplication, and one-click self-healing for missing files.
9. **Chunk-Centric Localized Pipeline & Multi-Threading**: Independent chunk extraction eliminates global hallucination contamination while configurable concurrency (1-8 workers) boosts throughput by 5-10x.
10. **Spatial Clustering & Balanced Vector Pathway Extraction**: Disjoint-set clustering merges fragmented image slices into pristine mechanism figures, while balanced vector thresholds accommodate both posters and papers.
11. **Three-Tier Rule-LLM Hybrid Classification (Workbench)**: Fuses a 0-token rule pre-filter (sub-millisecond regex matching of INN stems, enzyme suffixes, and ncRNAs) with micro-context reasoning, targeted in-paper grounding, and scholar manual overrides, cutting token consumption and latency by 40%+.
12. **Graph-Theoretic Pathway Decomposition & Circuit Mining**: Combines topological centrality metrics with LLM research-guided scoring, strictly guarantees subgraph invariance (zero hallucinated connections), applies MMR kinetic impedance to ensure exploration breadth, identifies closed autoregulatory feedback loops using Tarjan/Johnson algorithms, and supports exporting `.biokg` sub-projects for secondary deep dives.
13. **Graph Message Passing Hypothesis Engine & Autonomous Literature Referee**: Transcends two-node link heuristics by running 24-dimensional polarity-aware graph message passing flows to detect 3-4 node micro-circuits, supporting 6 discovery modes (Auto Mode, Detour Shortcut, Cluster Bridge, Homeostatic Feedback, Peripheral Growth, Intent-Driven); leverages online PubMed literature refereeing for 4-class adjudication (Known Fact / Novel Gap / Cascade / Inconclusive), backed by 8-10 paper deep literature expansion and one-click pipeline master graph ingestion.

---

## 📂 Core Code Architecture

```text
├── app.py                      # Streamlit frontend UI (workspaces, project switching, extraction settings, workbench views)
├── back_logic.py               # Central execution pipeline (chunk-centric concurrency & vision workflow)
├── workbench_engine.py         # Graph Analysis Workbench engine (rule pre-filters, ontology management, pathway mining)
├── hypothesis_engine.py        # Hypothesis deduction & discovery engine (message passing flow, micro-circuits, referee & expansion)
├── ontology_presets/           # Authoritative biomedical ontology presets (NIH BioRED, signaling, ecology, ethology)
├── LLM_SYS.py                  # LLM Agent core (reflection, symmetric pruning, alignment, transcription)
├── IO_SYS.py                   # Infrastructure layer (PyMuPDF spatial clustering, vector rendering, workbench Pyvis visualizer)
├── WebSearcher.py              # Web retrieval module (PubMed / Semantic Scholar search & PMID lookup)
├── projects/                   # Dedicated project workspaces:
│   └── <project_id>/
│       ├── project.biokg       # Single source of truth (BioKG 2.0 standard archive storing graph & workbench state)
│       ├── paper_registry.json # Project literature SHA-256 fingerprint & metadata registry
│       ├── graph.html          # Main graph Pyvis visualization HTML cache
│       ├── workbench_graph.html# Workbench analysis topology HTML cache
│       ├── papers/             # Bound physical PDF files
│       └── pathways/           # Pathway decomposition persistence snapshot directory
├── .bio_graph_config.json      # Local environment configuration (models, API providers, language & toggles cache)
├── requirements.txt            # Project dependencies
├── run.sh                      # Linux / WSL startup script
├── start.bat                   # Windows startup script
├── start_wsl.bat               # Windows WSL startup script
├── README.md                   # Project documentation (Chinese)
└── README_EN.md                # Project documentation (English)
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
* [x] **Project Isolation & Local Literature Self-Healing**: Physical project isolation, project-level SHA-256 registry, automatic online PMID renaming, and physical self-healing. (✅ **v2.0 Completed & Merged**)
* [x] **Symmetric Relation Folding & Containment Deadlock Resolution**: Bidirectional correlation consolidation and hierarchical loop resolution. (✅ **v2.0 Completed & Merged**)
* [x] **Graph Analysis Workbench & Multi-Attribute Topological Coloring**: Second-stage scientific workbench with 0-token pre-filtering, micro-context reasoning, targeted in-paper grounding, manual override consoles, and categorical visual coloring based on biological attributes. (✅ **v2.0 Completed & Merged**)
* [x] **Graph-Theoretic Pathway Decomposition & Autoregulation Circuit Mining**: Blends hybrid seed scoring, master subgraph induction, kinetic impedance MMR traversal, cycle mining, shortcut filtering, and `.biokg` secondary project export. (✅ **v2.0 Completed & Merged**)
* [x] **Cross-Disciplinary Multi-Scale Ontology Expansion**: Built-in rules and regex patterns for ecosystem science and ethology. (✅ **v2.0 Completed & Merged**)
* [x] **Hypothesis Deduction & Scientific Discovery Copilot (Stage 3)**: Mines 3-4 node micro-circuits via 24-dimensional graph message passing flow across 6 exploration modes (Auto Mode, Detour Shortcut, Cluster Bridge, Homeostatic Feedback, Peripheral Growth, Intent-Driven), integrated with online PubMed literature refereeing (Known Fact / Novel Gap / Cascade / Inconclusive), 8-10 paper deep literature expansion, and one-click smart bridging master graph ingestion. (✅ **v2.0 Completed & Merged**)
* [x] **Full-Stack Native Bilingual Support (i18n)**: One-click English/Chinese switching across UI, node tooltips, Stage 1/2/3 state machines, edge polarities, and literature referee verdicts with robust reverse-lookup guards against KeyError. (✅ **v2.0 Completed & Merged**)
* [x] **Abstract Phenotype Extraction & Concept Synthesis**: Identifies cellular/behavioral phenotypes, normalizes verbose outcomes into canonical concepts, and backfills original text into aliases with inference tags. (✅ **v2.0 Completed & Merged**)
* [ ] **Edge Metadata & Biological Conditions**: Conditional constraints (cell lines, mutations, experimental parameters) attached to edges as hover tooltips, separating empirical facts from topological hypotheses.
