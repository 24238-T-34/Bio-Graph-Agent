================================================================================
          Biomedical Knowledge Graph Workbench · Ontology Framework Guide
================================================================================

[README FIRST / QUICK OVERVIEW]
Welcome to the extensible ontology schema system of the Bio-RAG Analysis Workbench.
This system adopts a streamlined, rigorous "Minimalist 3-Element" data-driven architecture:
- Schema definitions and the analysis execution engine are completely decoupled.
  No classification rules or domain dictionaries are hardcoded in the codebase.
- Simply create or edit any UTF-8 encoded .json file in this folder (ontology_presets/),
  and the system will immediately scan, discover, and expose the standard in the
  "Select Ontology Preset" dropdown on the web interface.
- Text documents (*.txt) in this directory serve as reference manuals and are
  automatically ignored by the JSON scanner without interfering with presets.

--------------------------------------------------------------------------------
1. Core Design Philosophy: The Minimalist 3-Element Schema
--------------------------------------------------------------------------------
In conventional graph platforms, users are burdened with manually configuring
complex node colors, icons, geometric shapes, and hundreds of lines of flattened rules.
In this workbench, we strictly enforce the "Minimalist 3-Element" scientific standard:

1. Each entity category (entity_categories) contains ONLY three essential elements:
   (1) Name: name (Chinese display name) and en_name (English display name)
   (2) Description: description (Chinese definition) and en_description (English definition)
   (3) Regex Patterns: patterns (a list of precise regular expression strings)

2. Elimination of Emoji / Icon Clutter:
   To align with international biomedical research platforms (Cytoscape, Gephi, NCBI),
   no "icon" fields are used or declared in the JSON schemas.

3. Visualization Color & Geometry Dynamically Allocated by Engine:
   Do NOT declare "color" or "shape" attributes in the JSON files!
   The visualization engine (IO_SYS.py) embeds high-clarity Nature/Science-inspired palettes
   and Vis.js geometric shape pools:
   - Entity Node Color Palette (expanded to 12 neutral academic colors, strictly excluding red and green
     to avoid semantic polarity conflict with biological inhibition/activation lines):
     Royal Blue (#2563eb), Amber Gold (#d97706), Royal Violet (#7c3aed), Cyan Teal (#0891b2),
     Coral Orange (#ea580c), Magenta Pink (#db2777), Deep Indigo (#4f46e5), Sky Blue (#0284c7),
     Orchid Purple (#9333ea), Ochre Brown (#b45309), Deep Teal (#0e7490), Fuchsia (#c026d3).
   - Entity Geometry Shape Pool: dot, diamond, square, triangle, hexagon, star, box, triangleDown.
   - The engine automatically cycles through the palettes based on the sequential reading order
     of categories defined in your JSON schema.
   - The default fallback category "Uncertain" is consistently assigned neutral gray (#9ca3af)
     and smooth ellipse geometry.

4. Relation Edge Semantics & Line Style Specification (Biological Paper Standards):
   Edge line colors and shapes adhere strictly to international biological pathway conventions:
   - Positive Regulation / Activation / Stimulation: Scientific Emerald Green (#10b981) + Solid line + Arrow
   - Negative Regulation / Inhibition / Blockade: Scientific Ruby Red (#ef4444) + Bold solid line + Arrow
   - Hierarchy / Component / Complex Subunit: Tech Violet (#8b5cf6) + Solid line + Arrow
   - Association / Binding / Correlation / Pending: Neutral Gray (#9ca3af) + Dashed line + No arrow
   When deep mechanism predicates are distilled by LLM (e.g., "Catalytic phosphorylation"),
   they are rendered directly as readable labels on the curve.

--------------------------------------------------------------------------------
2. Regular Expression (patterns) Engineering & Best Practices
--------------------------------------------------------------------------------
The Stage 0 pre-filter engine (RulePreFilter) operates deterministically with zero LLM
token cost, identifying biomedical entities conforming to nomenclature within milliseconds.

By default, the engine compiles patterns with case-insensitivity enabled (re.IGNORECASE).

[Typical Regex Pattern Examples]:
1. Stem & Affix Suffix Matching (using .* with end-of-string $):
   - Kinases / Receptors: ".*(?:kinase|receptor|synthase|polymerase)$"
   - Targeted Small Molecule Drug Suffixes (INN international nonproprietary names):
     ".*(?:tinib|rafenib|lisib|ciclib|parib|zomib|statin|prazole|sartan|dipine)$"
   - Monoclonal Antibodies & ADCs: ".*(?:mab|cept|vedotin|deruxtecan)$"
   - Pathological Phenotypes & Diseases: ".*(?:itis|oma|carcinoma|sarcoma|syndrome|disease)$"

2. Exact Full-Name or Abbreviation Sets (using ^ and $ boundary anchors):
   - Landmark Cancer Genes: "^(?:TP53|EGFR|KRAS|MYC|STAT3|AKT1|MTOR|BRAF|PIK3CA|PTEN)$"
   - Key Immune Checkpoints: "^(?:PD-1|PD-L1|CTLA-4|LAG-3|TIM-3|TIGIT|4-1BB|OX40)$"
   - Endogenous Metabolites & Cofactors: "^(?:ATP|ADP|AMP|GTP|CAMP|NADH|NADPH|ROS|NO|GLUCOSE|LACTATE)$"

3. Standardized Prefixes:
   - Non-coding RNAs: "^(?:mir|mirna|microrna|lncrna|circrna|sirna)[-_].*"

4. Sequence Variants & Point Mutations:
   - Protein Single Amino Acid Substitutions (e.g., p.T790M, L858R): "^p\\.[a-z]\\d+[a-z]$" or "^[a-z]\\d+[a-z]$"
   - Genomic SNP Identifiers (e.g., rs123456): "^rs\\d+$"

[JSON Escaping Notice]:
- In standard JSON strings, the backslash "\" is an escape control character. Therefore,
  native regex backslashes (such as \\d, \\., \\+) MUST be written as double backslashes "\\\\".
  Example: To match "p.T790M", write "^p\\.[a-z]\\d+[a-z]$".
           To match positive ion "Ca2+", write ".*ca2\\+.*".

--------------------------------------------------------------------------------
3. Ready-to-Use Schema Template (Copy & Customize)
--------------------------------------------------------------------------------
You can copy the template below, save it as "custom_oncology.json" in this directory:

{
  "id": "custom_oncology",
  "name": "自定义肿瘤精准医学分类标准",
  "en_name": "Custom Precision Oncology Taxonomy",
  "description": "面向肿瘤转化医学与靶向治疗的定制实体分类与机制层级体系。",
  "en_description": "Custom ontology tailored for translational oncology and targeted therapeutics.",
  "entity_categories": {
    "Target_Protein": {
      "name": "靶点蛋白 / 驱动基因",
      "en_name": "Target Protein / Driver Gene",
      "description": "肿瘤发生发展的驱动基因产物、激酶受体及功能性蛋白复合体",
      "en_description": "Driver oncogene products, kinase receptors, and functional protein complexes.",
      "patterns": [
        ".*(?:kinase|receptor|phosphatase)$",
        "^(?:EGFR|KRAS|BRAF|PIK3CA|TP53|MYC|MET|ALK|RET|HER2)$"
      ]
    },
    "Targeted_Therapeutic": {
      "name": "小分子靶向药 / ADC药物",
      "en_name": "Targeted Therapeutic / ADC",
      "description": "合成小分子抑制剂、激酶抑制剂、单克隆抗体及抗体偶联药物(ADC)",
      "en_description": "Small molecule inhibitors, kinase inhibitors, monoclonal antibodies, and ADCs.",
      "patterns": [
        ".*(?:tinib|rafenib|lisib|ciclib|parib|nib|mab|cept|vedotin|deruxtecan)$"
      ]
    },
    "Neoplasm_Condition": {
      "name": "肿瘤分型 / 临床适应症",
      "en_name": "Neoplasm Subtype / Indication",
      "description": "实体瘤、恶性血液病、转移灶及临床病理诊断分型",
      "en_description": "Solid tumors, hematologic malignancies, metastatic lesions, and clinical diagnoses.",
      "patterns": [
        ".*(?:carcinoma|sarcoma|melanoma|leukemia|lymphoma|cancer|tumor)$"
      ]
    },
    "Uncertain": {
      "name": "待定 / 无法判断",
      "en_name": "Pending / Uncertain",
      "description": "上下文证据不充分或未命中特定类别的实体，由系统统一分流管理",
      "en_description": "Entities with insufficient context or pending escalation analysis."
    }
  },
  "relation_categories": {
    "Direct_Inhibition": {
      "name": "直接抑制 / 靶向阻断",
      "en_name": "Direct Inhibition / Blockade",
      "description": "小分子结合变构阻断、抗体中和或降解诱导",
      "en_description": "Allosteric inhibition, antibody neutralization, or degradation induction."
    },
    "Downstream_Activation": {
      "name": "下游激活 / 信号级联",
      "en_name": "Downstream Activation / Cascade",
      "description": "信号传导增强、催化激活、转录激活或通路驱动",
      "en_description": "Signal transduction enhancement, catalytic or transcriptional activation."
    },
    "Resistance_Mechanism": {
      "name": "耐药突变 / 旁路代偿",
      "en_name": "Resistance / Bypass Activation",
      "description": "获得性耐药突变、旁路信号转导代偿激活",
      "en_description": "Acquired resistance mutation or bypass signaling compensation."
    }
  }
}

--------------------------------------------------------------------------------
4. Quick Start & Verification
--------------------------------------------------------------------------------
1. Create and save your custom `.json` file inside this directory.
2. Go back to the web application and click [🔄 Refresh List] in the configuration area
   (or refresh your browser tab).
3. Select your custom taxonomy in the dropdown menu.
4. Click [🚀 Run Analysis Pipeline] to perform automated knowledge graph ontology analysis!
