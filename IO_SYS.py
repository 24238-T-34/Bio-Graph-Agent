import os
import webbrowser
import pypdf
from pyvis.network import Network
import re
from pypdf import PdfReader
import hashlib
import json
import datetime
from typing import Optional, List, Dict, Union, Tuple


# =====================================================================
# 1. PDF 处理器类 (双引擎增强版)
# =====================================================================
class PDFProcessor:
    def __init__(self, chunk_size=1200, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_chunks(self, pdf_path, start_page=0, end_page=None):
        """读取PDF指定页码范围并切块"""
        print(f"📄 [PDFProcessor] 正在读取文献: {pdf_path}")
        text = ""

        # 🌟 首选引擎：尝试使用容错率极高的 PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            if end_page is None or end_page > total_pages:
                end_page = total_pages

            print(f"📌 [PDFProcessor] (PyMuPDF引擎) 计划解析第 {start_page + 1} 页 到 第 {end_page} 页...")
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                extracted = page.get_text()
                if extracted:
                    text += extracted + "\n"
            doc.close()

        except Exception as fitz_err:
            print(f"⚠️ [PDFProcessor] PyMuPDF 解析受阻，自动降级至 pypdf 引擎... ({fitz_err})")

            # 🌟 备用引擎：Fallback 回 pypdf
            try:
                with open(pdf_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    total_pages = len(reader.pages)

                    if end_page is None or end_page > total_pages:
                        end_page = total_pages

                    print(f"📌 [PDFProcessor] (pypdf引擎) 计划解析第 {start_page + 1} 页 到 第 {end_page} 页...")
                    for page_num in range(start_page, end_page):
                        page = reader.pages[page_num]
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            except Exception as e:
                print(f"❌ [PDFProcessor] 双引擎读取均告失败: {e}")
                return []

        # 清理多余空格并切块
        text = " ".join(text.split())
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.overlap)

        print(f"✅ [PDFProcessor] 文献切分完成，共 {len(chunks)} 个片段。")
        return chunks

    def get_raw_text(self, pdf_path, start_page, end_page):
        """直接获取指定页码的纯文本，不进行分块，用于寻找摘要"""
        text = ""
        # 摘要提取区同样配置双引擎
        try:
            import fitz
            doc = fitz.open(pdf_path)
            for i in range(start_page, min(end_page, len(doc))):
                text += doc[i].get_text() + "\n"
            doc.close()
        except Exception:
            try:
                reader = PdfReader(pdf_path)
                for i in range(start_page, min(end_page, len(reader.pages))):
                    text += reader.pages[i].extract_text() + "\n"
            except Exception as e:
                print(f"❌ [PDFProcessor] 获取原文本失败: {e}")
        return text

    def extract_abstract_regex(self, text):
        """第一道防线：正则捕获摘要 (中英双语增强版)"""
        pattern = re.compile(
            r'(?i)(?:abstract|summary|摘要)\s*[:：\.\n]*\s*(.*?)(?=(?:\n\s*keywords|\bintroduction\b|1\.\s*introduction|\bbackground\b|关键词|引言|背景|1\.\s*引言))',
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(text)
        if match:
            extracted = match.group(1).strip()
            if len(extracted) > 50:
                return extracted
        return None

    def _cluster_page_figures(self, doc, page, min_dim=35, min_area=2500, padding=12):
        """
        对单页内的切片/分层光栅位图进行空间聚类，将重叠或邻近的图片切片合并为完整的机制图复合区域
        :param min_dim: 单个切片最小宽/高（低于此值视为微型图标或分割线）
        :param min_area: 单个切片最小面积（平方点）
        :param padding: 邻近切片合并的膨胀外边距（点）
        :return: List[dict] [{'rect': fitz.Rect, 'expanded_rect': fitz.Rect, 'items': [{'xref': int, 'rect': fitz.Rect}]}]
        """
        img_list = page.get_images()
        if not img_list:
            return []

        import fitz
        rect_entries = []
        for img_info in img_list:
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            for r in rects:
                if r.width < min_dim or r.height < min_dim or (r.width * r.height < min_area):
                    continue
                rect_entries.append({'xref': xref, 'rect': r})

        if not rect_entries:
            return []

        clusters = []
        for item in rect_entries:
            r = item['rect']
            expanded_r = fitz.Rect(r.x0 - padding, r.y0 - padding, r.x1 + padding, r.y1 + padding)
            matched_indices = []
            for c_idx, c in enumerate(clusters):
                if c['expanded_rect'].intersects(expanded_r):
                    matched_indices.append(c_idx)

            if not matched_indices:
                clusters.append({
                    'rect': fitz.Rect(r),
                    'expanded_rect': expanded_r,
                    'items': [item]
                })
            else:
                first_idx = matched_indices[0]
                clusters[first_idx]['rect'] |= r
                clusters[first_idx]['items'].append(item)
                for m_idx in reversed(matched_indices[1:]):
                    merged = clusters.pop(m_idx)
                    clusters[first_idx]['rect'] |= merged['rect']
                    clusters[first_idx]['items'].extend(merged['items'])
                cr = clusters[first_idx]['rect']
                clusters[first_idx]['expanded_rect'] = fitz.Rect(cr.x0 - padding, cr.y0 - padding, cr.x1 + padding, cr.y1 + padding)

        return clusters

    def _find_nearest_caption(self, page, cluster_rect):
        """
        根据聚类边界框，查找页面上物理距离最近的 Figure / Scheme / Graphical abstract 标题与说明
        """
        import fitz
        blocks = page.get_text("blocks")
        caption_candidates = []
        fig_pattern = re.compile(r'(?:^|\n)\s*(?:fig(?:ure)?\.?\s*\d+|graphical\s+abstract|scheme\s*\d*)', re.IGNORECASE)

        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            b_rect = fitz.Rect(b[:4])
            if fig_pattern.search(text) or ("Figure " in text or "Fig." in text or "Graphical abstract" in text):
                if b_rect.y0 >= cluster_rect.y1:
                    dist = b_rect.y0 - cluster_rect.y1
                elif b_rect.y1 <= cluster_rect.y0:
                    dist = cluster_rect.y0 - b_rect.y1
                else:
                    dist = 0
                caption_candidates.append((dist, text))

        if caption_candidates:
            caption_candidates.sort(key=lambda x: x[0])
            best_text = caption_candidates[0][1]
            return " ".join(best_text.split()[:60])
        return ""

    def extract_figures(self, pdf_path, start_page=0, end_page=None, mode="off", keyword="", min_size=200):
        """
        从选定页码中提取目标插图及关联图注（支持空间聚类复合图提取与矢量通路图高清机制渲染）
        :param mode: 'off' | 'match' | 'all'
        :param keyword: 精准匹配词（如 'Fig 4', 'Figure 2', 'pathway'）
        :param min_size: 最小像素宽高过滤阀值
        :return: List[dict] [{'image_bytes': b'...', 'ext': 'png', 'page': 5, 'caption': '...', 'label': 'Fig 4'}]
        """
        if mode == "off" or not os.path.exists(pdf_path):
            return []

        matched_figures = []
        try:
            import fitz
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            if end_page is None or end_page > total_pages:
                end_page = total_pages

            kw_clean = keyword.strip() if keyword else ""

            # 模式 2：精准匹配模式
            if mode == "match" and kw_clean:
                kw_num_match = re.search(r"(?:fig(?:ure)?\.?\s*)?(\d+[a-zA-Z]?)", kw_clean, re.IGNORECASE)
                if kw_num_match and kw_num_match.group(1):
                    fig_num = kw_num_match.group(1)
                    pattern_str = rf"(?:fig(?:ure)?\.?\s*{re.escape(fig_num)}\b|{re.escape(kw_clean)})"
                else:
                    pattern_str = re.escape(kw_clean)

                kw_pattern = re.compile(pattern_str, re.IGNORECASE)

                for page_idx in range(start_page, end_page):
                    page = doc[page_idx]
                    page_num = page_idx + 1
                    page_text = page.get_text()

                    captions = list(kw_pattern.finditer(page_text))
                    is_single_page = (total_pages == 1)

                    if captions or is_single_page:
                        matched_caption = captions[0].group(0).strip() if captions else (kw_clean or f"Page {page_num} Diagram")
                        clusters = self._cluster_page_figures(doc, page)

                        # 如果聚类成功捕获了位图区域
                        for c_idx, cl in enumerate(clusters):
                            clip = cl['rect'] & page.rect
                            if clip.width < 50 or clip.height < 50 or (clip.width * clip.height < 5000):
                                continue
                            pix = page.get_pixmap(clip=clip, dpi=150)
                            cap_detail = self._find_nearest_caption(page, cl['rect'])
                            full_caption = f"{matched_caption}: {cap_detail}" if cap_detail and matched_caption not in cap_detail else (cap_detail or matched_caption)
                            matched_figures.append({
                                "image_bytes": pix.tobytes("png"),
                                "ext": "png",
                                "page": page_num,
                                "caption": full_caption,
                                "label": kw_clean or f"Page{page_num}_Fig{c_idx+1}"
                            })
                            break

                        # 💡 矢量通路图/单页海报兜底：若命中了关键词或为单页海报，但未找到光栅位图，且页面包含矢量绘图（门槛适中，防止文本页规则线条误判）
                        dw_count = len(page.get_drawings())
                        is_vector = (is_single_page and dw_count >= 10) or (not is_single_page and dw_count >= 25)
                        if not matched_figures and is_vector:
                            pix = page.get_pixmap(dpi=150)
                            matched_figures.append({
                                "image_bytes": pix.tobytes("png"),
                                "ext": "png",
                                "page": page_num,
                                "caption": matched_caption,
                                "label": kw_clean or f"Page{page_num}_VectorFig"
                            })

                        if matched_figures:
                            break

            # 模式 3：全开模式 (空间聚类复合图 + 均衡矢量检测)
            elif mode == "all":
                for page_idx in range(start_page, end_page):
                    page = doc[page_idx]
                    page_num = page_idx + 1
                    clusters = self._cluster_page_figures(doc, page)
                    found_raster = False

                    for c_idx, cl in enumerate(clusters):
                        clip = cl['rect'] & page.rect
                        if clip.width < 60 or clip.height < 60 or (clip.width * clip.height < 6000):
                            continue
                        pix = page.get_pixmap(clip=clip, dpi=150)
                        cap = self._find_nearest_caption(page, cl['rect']) or f"Page {page_num} Figure {c_idx + 1}"
                        matched_figures.append({
                            "image_bytes": pix.tobytes("png"),
                            "ext": "png",
                            "page": page_num,
                            "caption": cap,
                            "label": f"Page{page_num}_Fig{c_idx + 1}"
                        })
                        found_raster = True

                    # 💡 矢量机制图/单页海报兜底：兼顾单页图与多页文献，防止普通文本页误判，同时不设过高门槛
                    if not found_raster:
                        drawings = page.get_drawings()
                        dw_count = len(drawings)
                        text = page.get_text()
                        has_fig_kw = bool(re.search(r'(?:fig(?:ure)?\.?\s*\d+|graphical\s+abstract|scheme\s*\d*|pathway|model)', text, re.IGNORECASE))
                        is_single = (total_pages == 1)

                        is_vector = False
                        if is_single and dw_count >= 10:
                            is_vector = True
                        elif not is_single:
                            if has_fig_kw and dw_count >= 25:
                                is_vector = True
                            elif dw_count >= 75:
                                is_vector = True

                        if is_vector:
                            print(f"📐 [PDFProcessor] 页面 {page_num} 检测到矢量绘图 ({dw_count} 个路径)，自动执行 150-DPI 高清机制图渲染...")
                            pix = page.get_pixmap(dpi=150)
                            cap = self._find_nearest_caption(page, page.rect) or f"Page {page_num} 机制通路图 (矢量渲染)"
                            matched_figures.append({
                                "image_bytes": pix.tobytes("png"),
                                "ext": "png",
                                "page": page_num,
                                "caption": cap,
                                "label": f"Page{page_num}_VectorFig"
                            })

            doc.close()
        except Exception as e:
            print(f"⚠️ [PDFProcessor] 机制图提取遇到异常: {e}")

        return matched_figures


# =====================================================================
# 3. 图谱可视化类
# =====================================================================
class GraphVisualizer:
    def __init__(self, bgcolor="#1a1a1a", font_color="white"):
        self.bgcolor = bgcolor
        self.font_color = font_color

    def generate_html(self, global_entities, global_relations, output_file="bio_knowledge_graph.html",
                      show_shortcuts=False,
                      empower_ontology=False, alpha_ontology=0.5,
                      empower_node=False, beta_node=0.2,
                      empower_edge=False, gamma_edge=0.1,
                      output_lang="zh"):
        print(f"🎨 [GraphVisualizer] 正在绘制动态网络拓扑图...")
        from pyvis.network import Network
        from collections import Counter
        import os

        # 🌐 悬停面板双语词典
        is_en = "en" in output_lang.lower()
        lbl_std_name = "🏷️ Standard Name" if is_en else "🏷️ 标准名称"
        lbl_aliases = "📚 Aliases" if is_en else "📚 别名"
        lbl_source = "📄 Source" if is_en else "📄 来源文献"
        lbl_heat = "🔥 Overall Heat" if is_en else "🔥 综合热度"
        lbl_weight = "🔥 Merge Count" if is_en else "🔥 证据合并次数"
        lbl_times = "times" if is_en else "次"
        lbl_rel = "🔍 Relation Type" if is_en else "🔍 关系类型"
        lbl_evi = "📝 Evidence" if is_en else "📝 证据"
        lbl_doc_src = "📄 Source" if is_en else "📄 源自"
        lbl_shortcut = "⚠️ [AI judged as mechanism shortcut]" if is_en else "⚠️ [AI 判定为机制捷径]"

        # 🔥 绝杀修复 1 & 2：保持 CDN 和网络配置不变
        net = Network(
            height="750px",
            width="100%",
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=True,
            cdn_resources="remote"
        )
        net.force_atlas_2based()

        # ==========================================
        # 1. 构建超强的信息映射字典 (包含别名和来源)
        # ==========================================
        entity_info_map = {}
        for ent in global_entities:
            std_name = ent.get("standard_name", "未知实体")
            entity_info_map[std_name] = {
                "aliases": ent.get("aliases", []),
                "doc_source": ent.get("doc_source", "未知文献")
            }

        # ==========================================
        # 2. 🧮 赋权引擎 Phase 0：统计基础节点热度 (保留原有的 weight 机制)
        # ==========================================
        base_heat = Counter()
        for item in global_relations:
            weight = item.get("weight", 1)
            base_heat[item.get("source")] += weight
            base_heat[item.get("target")] += weight

        final_heat = Counter(base_heat)  # 创建快照字典，后续操作在此累加

        # ==========================================
        # 🌟 赋权引擎 Phase 1 & 2：本体聚合与互相辐射
        # ==========================================
        for item in global_relations:
            source = item.get("source")
            target = item.get("target")
            rel_type = item.get("relation", "相关")  # 注意这里使用原来的 key

            # 🛡️ 致命 Bug 修复：拦截自环线！
            if source == target:
                continue

            # 🛡️ 强转洗净捷径参数，判断该线是否被隐藏
            raw_shortcut = item.get("is_shortcut", False)
            is_shortcut = (str(raw_shortcut).lower() == "true") or (raw_shortcut is True)
            safe_show = (str(show_shortcuts).lower() == "true") or (show_shortcuts is True)
            is_hidden = is_shortcut and not safe_show

            # 【开关 A】：包含关系反哺 (父节点变大)
            if empower_ontology and rel_type == "包含":
                final_heat[target] += base_heat[source] * alpha_ontology

            # 【开关 B】：节点互相辐射 (普通连线两端互相做大，排除隐藏线和包含关系)
            if empower_node and rel_type != "包含" and not is_hidden:
                final_heat[source] += base_heat[target] * beta_node
                final_heat[target] += base_heat[source] * beta_node

        # ==========================================
        # 3. 🎨 统一绘制节点 (应用计算好的 final_heat)
        # ==========================================
        for std_name, info in entity_info_map.items():
            heat_val = final_heat[std_name]
            node_title = f"{lbl_std_name} {std_name}\n{lbl_aliases} {', '.join(info['aliases'])}\n{lbl_source} {info['doc_source']}\n{lbl_heat} {heat_val:.1f}"

            # 动态调整大小 (保留原有的缩放比例逻辑)
            node_size = 20 + heat_val * 3
            net.add_node(std_name, label=std_name, title=node_title, color="#4da6ff", size=node_size)

        # ==========================================
        # 4. 🎨 绘制关系连线 (保留所有防御与颜色逻辑，加入加粗引擎)
        # ==========================================
        for item in global_relations:
            source = item.get("source")
            target = item.get("target")

            # 🛡️ 再次拦截自环线
            if source == target:
                continue

            rel_type = item.get("relation", "相关")
            evidence = item.get("evidence", "无")
            doc_source = item.get("doc_source", "未知文献")


            # 🛡️ 终极安全强转 (保持原样)
            raw_shortcut = item.get("is_shortcut", False)
            is_shortcut = (str(raw_shortcut).lower() == "true") or (raw_shortcut is True)
            safe_show = (str(show_shortcuts).lower() == "true") or (show_shortcuts is True)

            # 拦截被隐藏的捷径
            if is_shortcut and not safe_show:
                continue

            # 🌟 基础连线宽度 (保留你原有的缩放系数)
            rel_weight = item.get("weight", 1)
            scale_factor = 0.5
            display_width = 1 + (rel_weight * scale_factor)

            # 🔗 【开关 C】：核心主干加粗 (高热度节点间的非包含线变粗)
            if empower_edge and rel_type != "包含":
                display_width += gamma_edge * (final_heat[source] + final_heat[target])

            # 🛡️ 防御性编程：幻觉节点探测 (保持原样)
            for node in [source, target]:
                if node not in entity_info_map:
                    net.add_node(node, label=node, color="#ff9999", size=20,
                                 title=f"⚠️ 模型幻觉节点\n📄 来源: {doc_source}")
                    entity_info_map[node] = {"aliases": [], "doc_source": doc_source}


            # 强化悬停提示框
            edge_title = f"{lbl_weight}: {rel_weight} {lbl_times}\n{lbl_rel}: {rel_type}\n{lbl_evi}: {evidence}\n{lbl_doc_src}: {doc_source}"

            if is_shortcut:
                edge_title = f"{lbl_shortcut}\n" + edge_title

            # ==========================================
            # 🎨 视觉分支判断：保持你原有的颜色、虚线与箭头处理
            # ==========================================
            shortcut_color = "#555555"
            force_dashed = True if is_shortcut else False

            if rel_type == "正作用":
                final_color = shortcut_color if is_shortcut else "#00ff00"
                net.add_edge(source, target, title=edge_title, color=final_color, arrows="to",
                             width=display_width, dashes=force_dashed)

            elif rel_type == "负作用":
                final_color = shortcut_color if is_shortcut else "#ff3333"
                net.add_edge(source, target, title=edge_title, color=final_color, arrows="to",
                             width=display_width + 1, dashes=force_dashed)

            elif rel_type == "包含":
                final_color = shortcut_color if is_shortcut else "#9b59b6"
                net.add_edge(source, target, title=edge_title, color=final_color, arrows="to",
                             width=display_width, dashes=force_dashed)

            else:
                final_color = shortcut_color if is_shortcut else "#aaaaaa"
                net.add_edge(source, target, title=edge_title, color=final_color, arrows="",
                             dashes=True, width=display_width)

        net.save_graph(output_file)
        print(f"✨ [GraphVisualizer] 拓扑图已成功保存至: {output_file}")

    def generate_workbench_html(self, global_entities, global_relations,
                                entity_annotations=None, relation_annotations=None,
                                ontology=None, output_file="bio_knowledge_workbench.html",
                                output_lang="zh"):
        """
        绘制图谱分析工作台的科学层级拓扑图：
        - 节点按实体分类呈现不同几何形态 (dot, diamond, triangle, square, hexagon, star, ellipse) 与主题色彩
        - 关系连线显示深层机制谓词动词标签 (如 催化磷酸化, 竞争性结合, 转录激活) 与机制色彩
        - 包含丰富科学元数据的悬停卡片 (分类、细分子类、功能标签、推导轮次、置信度、AI推断依据)
        - 自动在页面右上角嵌入半透明、可折叠的本体图例控制板
        """
        print(f"🎨 [GraphVisualizer] 正在绘制分析工作台科学网络拓扑图...")
        from pyvis.network import Network
        from collections import Counter
        import os
        import json

        is_en = "en" in str(output_lang).lower()
        entity_annotations = entity_annotations or {}
        relation_annotations = relation_annotations or {}

        # 默认本体兜底
        if not ontology:
            try:
                from workbench_engine import OntologyManager
                ontology = OntologyManager().get_preset("universal_biored") or {}
            except Exception:
                ontology = {}

        ent_cat_defs = ontology.get("entity_categories", {})
        rel_cat_defs = ontology.get("relation_categories", {})

        # 学术科研实体调色板 (扩充至 12 种高对比中立学术色彩，严格剔除容易产生生物极性混淆的红与绿)
        SCIENTIFIC_PALETTE = [
            "#2563eb",  # 皇家深蓝 (Royal Blue)
            "#d97706",  # 暖金琥珀 (Amber Gold)
            "#7c3aed",  # 稳重紫罗兰 (Royal Violet)
            "#0891b2",  # 清爽青蓝 (Cyan Teal)
            "#ea580c",  # 活力珊瑚橙 (Coral Orange)
            "#db2777",  # 优雅洋红粉 (Magenta Pink)
            "#4f46e5",  # 经典深靛蓝 (Deep Indigo)
            "#0284c7",  # 晴空蔚蓝 (Sky Blue)
            "#9333ea",  # 兰花深紫 (Orchid Purple)
            "#b45309",  # 赭石棕黄 (Ochre Brown)
            "#0e7490",  # 深松青 (Deep Teal)
            "#c026d3",  # 亮紫红 (Fuchsia)
        ]
        SCIENTIFIC_SHAPES = [
            "dot",
            "diamond",
            "square",
            "triangle",
            "hexagon",
            "star",
            "box",
            "triangleDown"
        ]

        # 动态映射实体分类色彩与形态（按 JSON 定义顺序自动分配）
        entity_style_map = {}
        ent_idx = 0
        for cat_id, cat_def in ent_cat_defs.items():
            if cat_id == "Uncertain":
                entity_style_map[cat_id] = {
                    "color": cat_def.get("color", "#9ca3af"),
                    "shape": cat_def.get("shape", "ellipse")
                }
            else:
                c = cat_def.get("color") or SCIENTIFIC_PALETTE[ent_idx % len(SCIENTIFIC_PALETTE)]
                s = cat_def.get("shape") or SCIENTIFIC_SHAPES[ent_idx % len(SCIENTIFIC_SHAPES)]
                entity_style_map[cat_id] = {"color": c, "shape": s}
                ent_idx += 1

        net = Network(
            height="750px",
            width="100%",
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=True,
            cdn_resources="remote"
        )
        net.force_atlas_2based()

        # 1. 实体基本信息映射
        entity_info_map = {}
        for ent in global_entities:
            std_name = ent.get("standard_name", "未知实体")
            entity_info_map[std_name] = {
                "aliases": ent.get("aliases", []),
                "doc_source": ent.get("doc_source", "未知文献")
            }

        # 2. 统计节点热度
        base_heat = Counter()
        for item in global_relations:
            weight = item.get("weight", 1)
            src = item.get("source")
            tgt = item.get("target")
            if src:
                base_heat[src] += weight
            if tgt:
                base_heat[tgt] += weight

        # 记录出现过的类别以动态构建图例
        present_entity_cats = set()
        present_relation_cats = set()
        present_relation_polarities = set()

        # 3. 绘制带有几何形态与颜色的实体节点
        for std_name, info in entity_info_map.items():
            heat_val = base_heat[std_name]
            annot = entity_annotations.get(std_name, {})
            primary_type = annot.get("primary_type", "Uncertain")
            sub_type = annot.get("sub_type", "")
            tags = annot.get("tags", [])
            confidence = annot.get("confidence", 0.0)
            reasoning = annot.get("reasoning", "")
            round_info = annot.get("round", "")
            status = annot.get("status", "uncertain")

            present_entity_cats.add(primary_type)

            cat_def = ent_cat_defs.get(primary_type, {})
            cat_style = entity_style_map.get(primary_type, {"color": "#9ca3af", "shape": "dot"})
            node_shape = cat_style["shape"]
            node_color = cat_style["color"]
            cat_label = ((cat_def.get("en_name") or cat_def.get("en_label") or primary_type) if is_en else (cat_def.get("name") or cat_def.get("label") or primary_type))

            # 构建科学元数据悬停卡片 (学术文字排版，无 Emoji)
            lbl_type = "Category" if is_en else "实体分类"
            lbl_subtype = "Sub-type / Function" if is_en else "细分子类/功能"
            lbl_status = "Analysis Status" if is_en else "分析状态"
            lbl_conf = "Confidence" if is_en else "置信度"
            lbl_reason = "Reasoning" if is_en else "推断依据"
            lbl_aliases = "Aliases" if is_en else "别名"
            lbl_source = "Source" if is_en else "文献来源"
            lbl_heat = "Centrality Heat" if is_en else "拓扑热度"

            status_str = f"Round {round_info}" if round_info != "manual" else ("Manual Override" if is_en else "专家人工调换")
            if status == "uncertain" and primary_type == "Uncertain":
                status_str += " (" + ("Pending / Uncertain" if is_en else "待定 / 无法判断") + ")"

            title_lines = [
                f"{std_name} ({cat_label})",
                f"{lbl_type}: {cat_label}",
            ]
            if sub_type:
                title_lines.append(f"{lbl_subtype}: {sub_type}")
            if tags:
                title_lines.append(f"Tags: {', '.join(tags)}")
            title_lines.append(f"{lbl_status}: {status_str} ({lbl_conf}: {int(confidence * 100)}%)")
            if reasoning:
                title_lines.append(f"{lbl_reason}: {reasoning}")
            if info["aliases"]:
                title_lines.append(f"{lbl_aliases}: {', '.join(info['aliases'])}")
            title_lines.append(f"{lbl_source}: {info['doc_source']}")
            title_lines.append(f"{lbl_heat}: {heat_val:.1f}")

            node_title = "\n".join(title_lines)
            node_size = 20 + heat_val * 2.5

            net.add_node(
                std_name,
                label=std_name,
                title=node_title,
                color=node_color,
                shape=node_shape,
                size=node_size
            )

        # 4. 绘制连线并标注深层机制谓词动词
        for item in global_relations:
            source = item.get("source")
            target = item.get("target")
            if not source or not target or source == target:
                continue

            rel_type = item.get("relation", "相关")
            evidence = item.get("evidence", "无")
            doc_source = item.get("doc_source", "未知文献")
            rel_weight = item.get("weight", 1)

            # 探测幻觉节点
            for node in [source, target]:
                if node not in entity_info_map:
                    net.add_node(node, label=node, color="#ff9999", shape="ellipse", size=20,
                                 title=f"模型幻觉节点\n来源: {doc_source}" if not is_en else f"Hallucinated Node\nSource: {doc_source}")
                    entity_info_map[node] = {"aliases": [], "doc_source": doc_source}

            edge_key = f"{source}|{target}|{rel_type}"
            annot = relation_annotations.get(edge_key, {})

            mech_cat = annot.get("mechanism_category")
            mech_verb = annot.get("mechanism_verb")
            mech_conf = annot.get("confidence", 0.0)
            mech_reason = annot.get("reasoning", "")
            mech_round = annot.get("round", "")

            if mech_cat:
                present_relation_cats.add(mech_cat)

            rel_def = rel_cat_defs.get(mech_cat, {}) if mech_cat else {}
            rel_cat_lbl = ((rel_def.get("en_name") or rel_def.get("en_label") or mech_cat) if is_en else (rel_def.get("name") or rel_def.get("label") or mech_cat)) or rel_type

            display_width = 1.5 + (rel_weight * 0.6)
            edge_label = mech_verb if mech_verb else rel_type

            # 确定生物学连线极性、颜色与线型规范 (严格对齐经典生物论文与原图谱表达)
            is_positive = (rel_type == "正作用") or (mech_cat and any(k in str(mech_cat).lower() for k in ["positive", "activation", "induction", "agonism", "stimulat"]))
            is_negative = (rel_type == "负作用") or (mech_cat and any(k in str(mech_cat).lower() for k in ["negative", "inhibition", "degradation", "lysis", "exhaustion", "inactivation", "block", "suppress"]))
            is_component = (rel_type == "包含") or (mech_cat and any(k in str(mech_cat).lower() for k in ["hierarchy", "component", "scaffold", "subunit", "complex"]))

            if is_positive and not is_negative:
                edge_color = "#10b981"  # 经典科研绿 (正向调控 / 激活 / 促进)
                is_dashed = False
                arrows_val = "to"
                edge_width = display_width
                polarity_type = "positive"
            elif is_negative:
                edge_color = "#ef4444"  # 经典科研红 (负向调控 / 抑制 / 阻断)
                is_dashed = False
                arrows_val = "to"
                edge_width = display_width + 1.0  # 抑制实线加粗以突出阻断特征
                polarity_type = "negative"
            elif is_component:
                edge_color = "#8b5cf6"  # 经典科技紫 (包含 / 复合体组成)
                is_dashed = False
                arrows_val = "to"
                edge_width = display_width
                polarity_type = "component"
            else:
                edge_color = "#9ca3af"  # 中性灰色 (相关 / 结合伴随 / 未决)
                is_dashed = True
                arrows_val = ""
                edge_width = display_width
                polarity_type = "neutral"

            present_relation_polarities.add(polarity_type)

            # 构建连线科学悬停说明 (学术文字排版，无 Emoji)
            lbl_edge_mech = "Mechanism" if is_en else "深层机制"
            lbl_orig_rel = "Original Relation" if is_en else "原抽取关系"
            lbl_edge_evi = "Evidence" if is_en else "文献原文证据"
            lbl_conf_txt = "Confidence" if is_en else "置信度"
            lbl_reason_txt = "Reasoning" if is_en else "依据"

            edge_title_lines = [
                f"{lbl_edge_mech}: {rel_cat_lbl} ({edge_label})",
                f"{lbl_orig_rel}: {rel_type}"
            ]
            if mech_reason:
                edge_title_lines.append(f"{lbl_reason_txt}: {mech_reason}")
            if mech_conf > 0:
                edge_title_lines.append(f"{lbl_conf_txt}: {int(mech_conf * 100)}% (Round {mech_round})")
            edge_title_lines.append(f"{lbl_edge_evi}: {evidence}")
            edge_title_lines.append(f"{lbl_source}: {doc_source}")

            edge_title = "\n".join(edge_title_lines)

            net.add_edge(
                source,
                target,
                label=edge_label,
                title=edge_title,
                color=edge_color,
                arrows=arrows_val,
                width=edge_width,
                dashes=is_dashed,
                font={"color": "#d1d5db", "size": 11, "align": "middle", "strokeWidth": 2, "strokeColor": "#111827"}
            )

        net.save_graph(output_file)

        # 5. 注入半透明、响应式、可折叠的本体图例控制面板 (纯学术色块与符号，无 Emoji)
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            legend_title = "Ontology Legend" if is_en else "本体分类图例"
            entity_sec_title = "Entity Categories" if is_en else "实体分类 (形态 & 色彩)"
            rel_sec_title = "Relation Semantics" if is_en else "关系作用极性 (连线语义)"
            toggle_tip = "Click to collapse / expand" if is_en else "点击折叠/展开"
            btn_collapse = "Collapse" if is_en else "折叠"
            btn_expand = "Expand" if is_en else "展开"

            # 动态生成实体图例项目 (无 Emoji，纯学术几何形态与色标)
            shape_map = {
                "dot": "●", "diamond": "◆", "triangle": "▲", "triangleDown": "▼",
                "square": "■", "box": "▣", "hexagon": "⬡", "star": "★",
                "circle": "○", "ellipse": "⬭"
            }
            ent_legend_items = []
            for cat_id, cat_info in ent_cat_defs.items():
                if cat_id in present_entity_cats or not present_entity_cats:
                    lbl = (cat_info.get("en_name") or cat_info.get("en_label", cat_id)) if is_en else (cat_info.get("name") or cat_info.get("label", cat_id))
                    style = entity_style_map.get(cat_id, {"color": "#9ca3af", "shape": "dot"})
                    color = style["color"]
                    shape = style["shape"]
                    shape_symbol = shape_map.get(shape, "●")
                    ent_legend_items.append(f"""
                    <div style="display:flex;align-items:center;gap:8px;margin:3px 0;">
                        <span style="color:{color};font-size:14px;width:16px;text-align:center;">{shape_symbol}</span>
                        <span style="font-size:12px;color:#e5e7eb;font-weight:500;">{lbl}</span>
                    </div>
                    """)

            # 动态生成关系图例项目 (经典生物学四类极性规范，无 Emoji)
            rel_legend_items = []
            rel_polarity_specs = [
                ("positive", "Positive / Activation" if is_en else "正向调控 / 激活 / 促进", "#10b981", False),
                ("negative", "Negative / Inhibition" if is_en else "负向调控 / 抑制 / 阻断", "#ef4444", False),
                ("component", "Component / Complex" if is_en else "包含 / 复合体组成", "#8b5cf6", False),
                ("neutral", "Correlation / Association" if is_en else "相关 / 伴随 / 未决", "#9ca3af", True),
            ]
            for pol_key, pol_lbl, pol_color, pol_dash in rel_polarity_specs:
                if pol_key in present_relation_polarities or not present_relation_polarities:
                    dash_css = "border-top:2px dashed " + pol_color + ";height:0px;" if pol_dash else f"background:{pol_color};height:3px;"
                    rel_legend_items.append(f"""
                    <div style="display:flex;align-items:center;gap:8px;margin:3px 0;">
                        <span style="{dash_css}width:16px;display:inline-block;border-radius:2px;"></span>
                        <span style="font-size:12px;color:#e5e7eb;font-weight:500;">{pol_lbl}</span>
                    </div>
                    """)

            legend_html = f"""
            <!-- 图谱分析工作台悬浮动态图例 (无 Emoji 极简学术规范) -->
            <div id="workbench-legend-panel" style="position:fixed;top:16px;right:16px;background:rgba(17,24,39,0.88);backdrop-filter:blur(8px);border:1px solid rgba(75,85,99,0.6);border-radius:10px;padding:12px 16px;color:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:12px;box-shadow:0 10px 25px rgba(0,0,0,0.5);z-index:9999;max-width:280px;user-select:none;">
                <div id="legend-header" style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;font-weight:600;font-size:13px;border-bottom:1px solid rgba(75,85,99,0.5);padding-bottom:6px;margin-bottom:8px;" title="{toggle_tip}">
                    <span>{legend_title}</span>
                    <span id="legend-toggle-btn" style="font-size:11px;color:#9ca3af;background:rgba(55,65,81,0.6);padding:2px 6px;border-radius:4px;">{btn_collapse}</span>
                </div>
                <div id="legend-body" style="transition:all 0.2s ease;">
                    <div style="font-weight:600;color:#9ca3af;font-size:11px;text-transform:uppercase;margin:4px 0;">{entity_sec_title}</div>
                    {''.join(ent_legend_items)}
                    <div style="font-weight:600;color:#9ca3af;font-size:11px;text-transform:uppercase;margin:8px 0 4px 0;">{rel_sec_title}</div>
                    {''.join(rel_legend_items)}
                </div>
            </div>
            <script>
                (function() {{
                    var panel = document.getElementById('workbench-legend-panel');
                    var header = document.getElementById('legend-header');
                    var body = document.getElementById('legend-body');
                    var btn = document.getElementById('legend-toggle-btn');
                    var collapsed = false;
                    header.onclick = function() {{
                        collapsed = !collapsed;
                        if (collapsed) {{
                            body.style.display = 'none';
                            btn.innerText = '{btn_expand}';
                        }} else {{
                            body.style.display = 'block';
                            btn.innerText = '{btn_collapse}';
                        }}
                    }};
                }})();
            </script>
            """

            if "</body>" in html_content:
                html_content = html_content.replace("</body>", f"{legend_html}\n</body>")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(html_content)
        except Exception as e:
            print(f"⚠️ [GraphVisualizer] 注入图例组件遇到小异常: {e}")

        print(f"✨ [GraphVisualizer] 工作台科学拓扑图已成功保存至: {output_file}")

    def generate_pathway_html(self, pathway_entities, pathway_relations,
                              feedback_loops=None, cross_paper_bridges=None,
                              entity_annotations=None, relation_annotations=None,
                              ontology=None, output_file="pathway_subgraph.html",
                              output_lang="zh", height="480px",
                              layout_mode="hierarchical"):
        """
        绘制信号/因果通路拆解与建模的精简独立交互子图 (Mini-Subgraph)：
        - 聚焦呈现该通路 5~10 个关键种子节点与直接中继边
        - 突出标注自调节闭环连线 (Feedback Loop) 与跨文献中枢节点 (Cross-Paper Pivot)
        - 保持与主工作台一致的高对比学术中立色彩与生物极性箭头规范
        - 生成响应式、独立可脱离系统全屏打开的 PyVis HTML
        """
        from pyvis.network import Network
        from collections import Counter
        import os

        is_en = "en" in str(output_lang).lower()
        entity_annotations = entity_annotations or {}
        relation_annotations = relation_annotations or {}
        feedback_loops = feedback_loops or []
        cross_paper_bridges = cross_paper_bridges or {}

        # 默认本体兜底
        if not ontology:
            try:
                from workbench_engine import OntologyManager
                ontology = OntologyManager().get_preset("universal_biored") or {}
            except Exception:
                ontology = {}

        ent_cat_defs = ontology.get("entity_categories", {})
        rel_cat_defs = ontology.get("relation_categories", {})

        # 学术科研实体调色板 (高对比中立色彩)
        SCIENTIFIC_PALETTE = [
            "#2563eb",  # 皇家深蓝
            "#d97706",  # 暖金琥珀
            "#7c3aed",  # 稳重紫罗兰
            "#0891b2",  # 清爽青蓝
            "#ea580c",  # 活力珊瑚橙
            "#db2777",  # 优雅洋红粉
            "#4f46e5",  # 经典深靛蓝
            "#0284c7",  # 晴空蔚蓝
            "#9333ea",  # 兰花深紫
            "#b45309",  # 赭石棕黄
            "#0e7490",  # 深松青
            "#c026d3",  # 亮紫红
        ]
        SCIENTIFIC_SHAPES = ["dot", "diamond", "square", "triangle", "hexagon", "star", "box", "triangleDown"]

        entity_style_map = {}
        ent_idx = 0
        for cat_id, cat_def in ent_cat_defs.items():
            if cat_id == "Uncertain":
                entity_style_map[cat_id] = {
                    "color": cat_def.get("color", "#9ca3af"),
                    "shape": cat_def.get("shape", "ellipse")
                }
            else:
                c = cat_def.get("color") or SCIENTIFIC_PALETTE[ent_idx % len(SCIENTIFIC_PALETTE)]
                s = cat_def.get("shape") or SCIENTIFIC_SHAPES[ent_idx % len(SCIENTIFIC_SHAPES)]
                entity_style_map[cat_id] = {"color": c, "shape": s}
                ent_idx += 1

        # 提取跨文献桥接中枢节点集合
        pivot_nodes_raw = cross_paper_bridges.get("pivot_nodes", [])
        pivot_node_names = set()
        for p in pivot_nodes_raw:
            if isinstance(p, dict):
                pivot_node_names.add(p.get("node", ""))
            elif isinstance(p, str):
                pivot_node_names.add(p)

        # 提取闭环连线集合 (u, v)
        cycle_edge_set = set()
        for loop in feedback_loops:
            c_nodes = loop.get("nodes", [])
            for i in range(len(c_nodes)):
                u = c_nodes[i]
                v = c_nodes[(i + 1) % len(c_nodes)]
                cycle_edge_set.add((u, v))

        # 预先扫描识别空间/复合物场域容器节点
        container_nodes = set()
        for r in pathway_relations:
            r_rel = str(r.get("relation", r.get("relation_type", ""))).strip()
            r_src = str(r.get("source", "")).strip()
            edge_k = f"{r_src}|{r.get('target', '')}|{r_rel}"
            ann = relation_annotations.get(edge_k, {})
            m_cat = str(ann.get("mechanism_category", "")).lower()
            m_verb = str(ann.get("mechanism_verb", "")).lower()
            if (r_rel in ["包含", "属于", "part_of", "hierarchy", "component"]) or any(k in (m_cat + " " + m_verb) for k in ["hierarchy", "component", "complex", "包含", "属于", "复合体", "亚型", "分类"]):
                container_nodes.add(r_src)

        net = Network(
            height=height,
            width="100%",
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=True,
            cdn_resources="remote"
        )
        if str(layout_mode).lower() == "hierarchical":
            hierarchical_options = {
                "layout": {
                    "hierarchical": {
                        "enabled": True,
                        "direction": "LR",
                        "sortMethod": "directed",
                        "levelSeparation": 190,
                        "nodeSpacing": 110,
                        "treeSpacing": 160,
                        "blockShifting": True,
                        "edgeMinimization": True,
                        "parentCentralization": True
                    }
                },
                "physics": {
                    "hierarchicalRepulsion": {
                        "centralGravity": 0.0,
                        "springLength": 100,
                        "springConstant": 0.01,
                        "nodeDistance": 120,
                        "damping": 0.09
                    },
                    "solver": "hierarchicalRepulsion"
                }
            }
            import json
            net.set_options(json.dumps(hierarchical_options))
        else:
            net.force_atlas_2based(gravity=-60, central_gravity=0.015, spring_length=120, spring_strength=0.08, damping=0.9)

        # 1. 实体节点添加
        entity_info_map = {}
        for ent in pathway_entities:
            std_name = ent.get("standard_name", "")
            if not std_name and isinstance(ent, str):
                std_name = ent
            if std_name:
                entity_info_map[std_name] = {
                    "aliases": ent.get("aliases", []) if isinstance(ent, dict) else [],
                    "doc_source": ent.get("doc_source", "未知文献") if isinstance(ent, dict) else "未知文献"
                }

        # 统计节点在当前子图内的连接度
        sub_heat = Counter()
        for r in pathway_relations:
            sub_heat[r.get("source")] += 1
            sub_heat[r.get("target")] += 1

        # 渲染节点
        for std_name in list(entity_info_map.keys()):
            annot = entity_annotations.get(std_name, {})
            primary_type = annot.get("primary_type", "Uncertain")
            sub_type = annot.get("sub_type", "")
            aliases = entity_info_map[std_name]["aliases"]
            doc_src = entity_info_map[std_name]["doc_source"]

            style = entity_style_map.get(primary_type, {"color": "#0284c7", "shape": "dot"})
            node_color = style["color"]
            node_shape = style["shape"]
            node_size = 22 + min(12, sub_heat[std_name] * 3)

            is_pivot = std_name in pivot_node_names
            is_container = std_name in container_nodes
            border_width = 3 if is_pivot else (2.5 if is_container else 1.5)
            border_color = "#f59e0b" if is_pivot else ("#8b5cf6" if is_container else "#ffffff")

            # 悬停说明卡片
            lbl_type = "Category" if is_en else "实体分类"
            lbl_pivot = "🌟 [Cross-Literature Pivot]" if is_en else "🌟 [跨文献桥接中枢]"
            lbl_container = "🏛️ [Spatial/Complex Container]" if is_en else "🏛️ [空间/复合物场域容器]"
            lbl_src = "Source" if is_en else "来源文献"

            title_lines = [f"{std_name}"]
            if is_pivot:
                title_lines.append(lbl_pivot)
            if is_container:
                title_lines.append(lbl_container)
            title_lines.append(f"{lbl_type}: {primary_type}" + (f" ({sub_type})" if sub_type else ""))
            if aliases:
                title_lines.append(f"Aliases: {', '.join(aliases[:4])}")
            title_lines.append(f"{lbl_src}: {doc_src}")
            node_title = "\n".join(title_lines)

            net.add_node(
                std_name,
                label=std_name,
                color=node_color,
                shape=node_shape,
                size=node_size,
                borderWidth=border_width,
                borderColor=border_color,
                title=node_title
            )

        # 2. 关系连线添加
        for r in pathway_relations:
            src = r.get("source", "").strip()
            tgt = r.get("target", "").strip()
            if not src or not tgt or src == tgt:
                continue

            # 保底补充缺失节点
            for nd in [src, tgt]:
                if nd not in entity_info_map:
                    net.add_node(nd, label=nd, color="#9ca3af", shape="ellipse", size=20, title=nd)
                    entity_info_map[nd] = {"aliases": [], "doc_source": "未知文献"}

            rel_type = r.get("relation", r.get("relation_type", "相关"))
            edge_key = f"{src}|{tgt}|{rel_type}"
            annot = relation_annotations.get(edge_key, {})
            mech_cat = annot.get("mechanism_category", "")
            mech_verb = annot.get("mechanism_verb", rel_type)
            evidence = r.get("evidence", r.get("reason", "无"))
            doc_source = r.get("doc_source", "未知文献")

            edge_label = mech_verb if mech_verb else rel_type
            is_in_loop = (src, tgt) in cycle_edge_set

            # 极性色彩判断与动力学层次表达
            is_positive = (rel_type == "正作用") or any(k in str(mech_cat).lower() or k in str(mech_verb).lower() for k in ["positive", "activation", "induction", "agonism", "stimulat", "促进", "激活", "上调", "催化"])
            is_negative = (rel_type == "负作用") or any(k in str(mech_cat).lower() or k in str(mech_verb).lower() for k in ["negative", "inhibition", "degradation", "lysis", "suppress", "block", "repress", "抑制", "降解", "阻断", "下调"])
            is_component = (rel_type in ["包含", "属于", "part_of", "hierarchy", "component"]) or any(k in str(mech_cat).lower() or k in str(mech_verb).lower() for k in ["hierarchy", "component", "complex", "包含", "属于", "从属", "复合体", "亚型", "分类"])

            if is_in_loop:
                # 闭环连线醒目高亮
                edge_color = "#ef4444" if is_negative else "#f59e0b"
                edge_width = 3.5
                is_dashed = False
            elif is_negative:
                edge_color = "#ef4444"  # 红色实线 (阻断/降解/抑制)
                edge_width = 2.5
                is_dashed = False
            elif is_positive:
                edge_color = "#10b981"  # 绿色实线 (激活/促进/催化)
                edge_width = 2.0
                is_dashed = False
            elif is_component:
                edge_color = "#a78bfa"  # 淡紫色虚线 (从属/结构组成/场域容器，不喧宾夺主)
                edge_width = 1.4
                is_dashed = True
                edge_label = f"[{edge_label}]"
            else:
                edge_color = "#9ca3af"  # 灰色虚线 (一般关联)
                edge_width = 1.4
                is_dashed = True

            edge_title_lines = [
                f"{src} ➔ {tgt} : {edge_label}"
            ]
            if is_in_loop:
                edge_title_lines.append("🔄 [Feedback Loop Edge / 自调节闭环连线]" if not is_en else "🔄 [Feedback Loop Edge]")
            if mech_cat:
                edge_title_lines.append(f"Category: {mech_cat}")
            if evidence and evidence != "无":
                edge_title_lines.append(f"Evidence: {evidence}")
            edge_title_lines.append(f"Source: {doc_source}")

            net.add_edge(
                src,
                tgt,
                label=edge_label,
                title="\n".join(edge_title_lines),
                color=edge_color,
                width=edge_width,
                dashes=is_dashed,
                arrows="to",
                font={"color": "#e5e7eb", "size": 11, "align": "middle", "strokeWidth": 2, "strokeColor": "#111827"}
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        net.save_graph(output_file)
        print(f"✨ [GraphVisualizer] 信号通路精简子图已成功生成至: {output_file}")
        return output_file



# =====================================================================
# 3. 文献学术标题提取与安全命名工具集
# =====================================================================
def sanitize_paper_filename(name: str, max_len: int = 160) -> str:
    """过滤操作系统非法字符并规范化学术文献文件名，清除版权与声明等杂质"""
    if not name:
        return ""
    clean = str(name).strip()
    # 剥离可能存在的 .pdf 扩展名，确保生成的学术标题纯净
    clean = re.sub(r'(?i)\.pdf$', '', clean).strip()

    # 清除不可见软连字符与零宽字符
    clean = clean.replace("\xad", "").replace("\u200b", "").replace("\ufeff", "")

    # 彻底清除版权与出版声明等杂质（如 some rights reserved, all rights reserved, copyright 等）
    clean = re.sub(r'(?i)\b(?:all|some)\s+rights?\s+reserved\.?', '', clean)
    clean = re.sub(r'(?i)(?:\bcopyright\b|©|\(c\)).*$', '', clean)
    clean = re.sub(r'(?i)\bopen\s+access\b\.?', '', clean)
    clean = re.sub(r'(?i)\bcreative\s+commons\b.*?$', '', clean)
    clean = re.sub(r'(?i)\[(?:article\s+in\s+[a-z]+|retracted|epub\s+ahead\s+of\s+print)\]', '', clean)

    # 精细标点规范化：
    # 1. 冒号统一规范为带空格的单破折号 ' - '（如 "Title: Subtitle" -> "Title - Subtitle"）
    clean = re.sub(r'\s*[:：]\s*', ' - ', clean)
    # 2. 斜杠/反斜杠/竖线统一替换为下划线 '_'（不注入空格，保留专有名词如 "TGF-β/Smad" -> "TGF-β_Smad"）
    clean = re.sub(r'[\\/|]', '_', clean)
    # 3. 彻底移除其他文件系统非法字符 * ? " < >
    clean = re.sub(r'[*?"<>]', '', clean)

    # 规范化多余空白与连字符（避免连续破折号或连续下划线）
    clean = re.sub(r'_+', '_', clean)
    clean = re.sub(r'\s*-\s*-\s*', ' - ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    clean = clean.strip(" ._-\n\r\t")

    # 长度截断控制（避免文件名超长无法保存，尽量在完整单词处截断）
    if len(clean) > max_len:
        truncated = clean[:max_len]
        if " " in truncated:
            clean = truncated.rsplit(" ", 1)[0]
        else:
            clean = truncated
    return clean.strip(" ._-")


def extract_paper_title_and_pmid(pdf_path: str, max_pages: int = 1, agent=None):
    """
    智能提取文献的真实学术标题与 PMID（双阶极简 Token 架构）
    流水线策略：
    1. 阶段一（免Token优先）：读取首页文本，容错正则扫描 DOI / PMID -> 优先从 PubMed 获取官方权威标题（0 Token）；
    2. 阶段一（降级兜底）：若未匹配到 DOI/PMID，截取第 1 页头部极短切块 (~300字符) 由模型极简提取候选；
    3. 阶段二（统一精洗）：无论何种方式获得的候选标题，均统一送交大模型做轻量级审校清洗（去版权、去期刊标签，~30-50 Token）；
    4. 阶段三（标点治理与安全重命名）：精细清洗非法标点与空格，防止期刊名混淆与格式破坏。
    :return: (sanitized_title, pmid_str, method_tag)
    """
    if not os.path.exists(pdf_path):
        # 兼容线上 PMID 来源（如 "PubMed:40014690", "PMID: 40014690", "40014690" 等无本地 PDF 的来源）
        src_str = str(pdf_path).strip()
        pmid_m = re.search(r"(?:^|[\W_])(?:pmid|pubmed)?[:\s#_-]*(\d{6,9})\b", src_str, re.I)
        if not pmid_m:
            pmid_m = re.search(r"^(\d{6,9})$", src_str)
        if pmid_m:
            online_pmid = pmid_m.group(1)
            try:
                from WebSearcher import PubMedSearcher
                ps = PubMedSearcher()
                res = ps.session.get(f"{ps.base_url}esummary.fcgi", params={
                    "db": "pubmed", "id": str(online_pmid), "retmode": "json", "email": ps.email
                }, timeout=6).json()
                t_off = res.get("result", {}).get(str(online_pmid), {}).get("title")
                if t_off and len(t_off.strip()) >= 6:
                    candidate = t_off.strip()
                    if agent and hasattr(agent, "clean_and_sanitize_title"):
                        try:
                            cleaned = agent.clean_and_sanitize_title(candidate)
                            if cleaned and len(cleaned) >= 6 and "unknown" not in cleaned.lower():
                                candidate = cleaned
                        except Exception as e:
                            print(f"⚠️ [extract_paper_title] PMID 线上标题大模型清洗受阻: {e}")
                    clean_final = sanitize_paper_filename(candidate)
                    if clean_final and len(clean_final) >= 6:
                        # 规范化学术来源格式：大标题 (PMID: xxxxxxxx)
                        return f"{clean_final} (PMID: {online_pmid})", online_pmid, "online_pmid+llm_clean"
            except Exception as e:
                print(f"⚠️ [extract_paper_title] 线上 PMID 检索失败: {e}")
            return None, online_pmid, "online_pmid_not_found"

        return None, None, "file_not_found"

    doc = None
    page_text = ""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if len(doc) > 0:
            page_text = doc[0].get_text()
            # 若第 1 页纯字符少于 80 (如空白封面)，容错读取第 2 页
            if len(page_text.strip()) < 80 and len(doc) > 1:
                page_text += "\n" + doc[1].get_text()
    except Exception:
        # Fallback to pypdf
        try:
            reader = PdfReader(pdf_path)
            if len(reader.pages) > 0:
                page_text = reader.pages[0].extract_text() or ""
                if len(page_text.strip()) < 80 and len(reader.pages) > 1:
                    page_text += "\n" + (reader.pages[1].extract_text() or "")
        except Exception:
            if doc:
                doc.close()
            return None, None, "pdf_open_error"

    if doc:
        doc.close()

    # ----------------------------------------------------
    # 预处理：融接跨行折叠的 URL/DOI，清理软连字符与零宽字符
    # ----------------------------------------------------
    norm_text = re.sub(r'/\s*\n\s*', '/', page_text)
    norm_text = norm_text.replace("\xad", "").replace("\u200b", "").replace("\ufeff", "")
    norm_text = re.sub(r'(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)', r'\1\2', norm_text)

    candidate_title = ""
    candidate_source = "none"
    pmid = None
    doi = None

    # ----------------------------------------------------
    # 阶段一（分支 A，0 Token 优先）：正则扫描 PMID 与 DOI 并请求 PubMed 官方数据
    # ----------------------------------------------------
    # 1. 扫描 PMID
    pmid_m = re.search(r"\b(?:PMID|PubMed\s*(?:ID)?)[:\s#]*(\d{6,9})\b", norm_text, re.I)
    if not pmid_m:
        # 文件名明确包含 pmid 前缀 (如 PMID_12345678.pdf)
        pmid_m = re.search(r"(?:^|[\W_])pmid[_\-\s:]*(\d{6,9})\b", os.path.basename(pdf_path), re.I)
    if not pmid_m:
        # 文件名纯数字 (如 12345678.pdf)
        pmid_m = re.search(r"^(\d{6,9})\.pdf$", os.path.basename(pdf_path), re.I)
    if pmid_m:
        pmid = pmid_m.group(1)

    # 2. 扫描 DOI
    doi_m = re.search(r"\b(?:doi\.org/|doi:\s*|https?://[^/]+/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", norm_text, re.I)
    if not doi_m:
        doi_m = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", norm_text, re.I)
    if doi_m:
        doi = doi_m.group(1).rstrip(" .;")

    # 3. 优先通过 PMID 检索 PubMed
    if pmid:
        try:
            from WebSearcher import PubMedSearcher
            ps = PubMedSearcher()
            res = ps.session.get(f"{ps.base_url}esummary.fcgi", params={
                "db": "pubmed", "id": str(pmid), "retmode": "json", "email": ps.email
            }, timeout=6).json()
            t_official = res.get("result", {}).get(str(pmid), {}).get("title")
            if t_official and len(t_official.strip()) >= 6:
                candidate_title = t_official.strip()
                candidate_source = "pubmed_pmid"
            else:
                # 若该 PMID 无法在 PubMed 查证，则重置为 None 避免误报
                pmid = None
        except Exception as err:
            print(f"⚠️ [extract_paper_title] PMID 在线检索受阻: {err}")

    # 4. 若未命中 PMID 则通过 DOI 检索 PubMed
    if not candidate_title and doi:
        try:
            from WebSearcher import PubMedSearcher
            ps = PubMedSearcher()
            res = ps.search_articles(doi, max_results=1)
            if res and res[0].get("title"):
                candidate_title = res[0]["title"].strip()
                if res[0].get("pmid"):
                    pmid = res[0]["pmid"]
                candidate_source = "pubmed_doi"
        except Exception as err:
            print(f"⚠️ [extract_paper_title] DOI 在线检索受阻: {err}")

    # ----------------------------------------------------
    # 阶段一（分支 B，极低 Token 降级）：无 DOI/PMID 时，从首页头部截取极短切块交由模型快提
    # ----------------------------------------------------
    if not candidate_title and agent:
        lines = [l.strip() for l in norm_text.split("\n") if l.strip()]
        short_chunk = "\n".join(lines[:10])[:400]
        if hasattr(agent, "fast_extract_title"):
            try:
                local_cand = agent.fast_extract_title(short_chunk)
                if local_cand and len(local_cand) >= 6:
                    candidate_title = local_cand
                    candidate_source = "local_fast_extract"
            except Exception as e:
                print(f"⚠️ [extract_paper_title] 本地模型快提受阻: {e}")

    # ----------------------------------------------------
    # 阶段二：统一大模型清洗（输入仅候选标题，耗费仅 ~30-50 Token，剥离版权与期刊杂质）
    # ----------------------------------------------------
    if candidate_title and agent:
        try:
            if hasattr(agent, "clean_and_sanitize_title"):
                cleaned = agent.clean_and_sanitize_title(candidate_title)
            elif hasattr(agent, "review_paper_title"):
                cleaned = agent.review_paper_title(candidate_title=candidate_title)
            else:
                cleaned = candidate_title

            if cleaned and len(cleaned) >= 6 and "unknown" not in cleaned.lower():
                candidate_title = cleaned
                candidate_source += "+llm_clean"
        except Exception as err:
            print(f"⚠️ [extract_paper_title] 大模型统一清洗受阻: {err}")

    # ----------------------------------------------------
    # 阶段三：安全护栏校验与文件名外科手术式规范化
    # ----------------------------------------------------
    if candidate_title:
        # 防期刊误判与纯栏目名安全护栏
        pure_check = candidate_title.strip(" ._-:").lower()
        junk_blacklist = {
            "science", "nature", "cell", "article", "research article", "review",
            "research article summary", "pathway diagram key", "molecular biology",
            "editorial", "brief report", "summary", "contents"
        }
        if pure_check in junk_blacklist:
            return None, pmid, "rejected_as_journal_or_junk"

        clean_final = sanitize_paper_filename(candidate_title)
        if clean_final and len(clean_final) >= 6:
            return clean_final, pmid, candidate_source

    return None, pmid, "unrecognized"


# =====================================================================
# 4. 文献不可变哈希身份与注册表系统 (Hash-Centric Paper Identity & Registry)
# =====================================================================
PROJECTS_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")


def resolve_project_dir(project_dir_or_id: str) -> str:
    """辅助解析工程目录绝对路径"""
    if not project_dir_or_id:
        return ""
    if os.path.isabs(project_dir_or_id) and os.path.exists(project_dir_or_id):
        return project_dir_or_id
    cand = os.path.join(PROJECTS_ROOT_DIR, project_dir_or_id)
    if os.path.exists(cand):
        return cand
    if os.path.exists(project_dir_or_id):
        return os.path.abspath(project_dir_or_id)
    return cand


def compute_file_content_hash(path_or_bytes: Union[str, bytes]) -> str:
    """计算物理文件的 SHA-256 内容哈希（作为跨更名操作的不可变唯一主键）"""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return f"sha256_{hashlib.sha256(path_or_bytes).hexdigest()}"
    if isinstance(path_or_bytes, str) and os.path.exists(path_or_bytes):
        hasher = hashlib.sha256()
        try:
            with open(path_or_bytes, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return f"sha256_{hasher.hexdigest()}"
        except Exception as e:
            print(f"⚠️ [compute_file_content_hash] 计算哈希受阻: {e}")
            return ""
    return ""


def compute_source_identity_hash(source_str: str) -> str:
    """针对线上 PMID 或系统虚拟来源生成规范不可变哈希"""
    if not source_str:
        return "manual:empty"
    src = str(source_str).strip()
    pmid_m = re.search(r"(?:^|[\W_])(?:pmid|pubmed)?[:\s#_-]*(\d{6,9})\b", src, re.I)
    if not pmid_m:
        pmid_m = re.search(r"^(\d{6,9})$", src)
    if pmid_m:
        return f"pmid:{pmid_m.group(1)}"

    clean_tag = src.lower().strip()
    return f"virtual:{hashlib.md5(clean_tag.encode('utf-8')).hexdigest()[:12]}"


FIGURE_TAG_PATTERN = re.compile(
    r'\s*\[(?:Figure|Fig|Visual Figure|图片|插图|机制图|VectorFig)[^\]]*\]',
    flags=re.IGNORECASE
)


def is_figure_doc_source(source_str: str) -> bool:
    """判定出处是否带有插图/机制图的子图来源标记"""
    if not source_str:
        return False
    return bool(FIGURE_TAG_PATTERN.search(str(source_str)))


def extract_parent_doc_source(source_str: str) -> str:
    """
    从带有插图、机制图或切块标注的出处字符串中，提取所属的母体文献标识。
    例如:
      'PIIS0092867425014874 [Figure: Page1_Fig1 (P.1)]' -> 'PIIS0092867425014874'
      'Cell_2025.pdf [Fig: 1]' -> 'Cell_2025.pdf'
      'PMID: 39218274 [Figure: VectorFig]' -> 'PMID: 39218274'
      '[Figure: Page1_Fig1]' -> '' (纯虚拟图片标记，无母体)
    """
    if not source_str:
        return ""
    s = str(source_str).strip()
    parent = FIGURE_TAG_PATTERN.sub('', s).strip()
    return parent


def get_clean_display_title(name: str) -> str:
    """获取纯净的前端展示标题（严格剥离末尾的 .pdf 后缀）"""
    if not name:
        return ""
    clean = str(name).strip()
    clean = re.sub(r'(?i)\.pdf$', '', clean).strip()
    return clean


def get_physical_pdf_filename(name: str) -> str:
    """获取合法的磁盘物理文件名（严格确保以 .pdf 结尾）"""
    if not name:
        return "unnamed_paper.pdf"
    clean = str(name).strip()
    if clean.lower().endswith(".pdf"):
        return clean
    return f"{clean}.pdf"


def load_project_paper_registry(project_dir_or_id: str) -> dict:
    """
    载入工程文献哈希注册表。若本地尚无注册表，则自动通过本地 papers/ 与 project.biokg 进行冷启动全盘建表
    """
    pdir = resolve_project_dir(project_dir_or_id)
    reg_file = os.path.join(pdir, "paper_registry.json")
    if os.path.exists(reg_file):
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "papers" in data:
                    return data
        except Exception as e:
            print(f"⚠️ [load_project_paper_registry] 读取注册表失败: {e}")

    # 冷启动全量扫描并构建
    registry = {"version": "2.0", "papers": {}}
    papers_dir = os.path.join(pdir, "papers")
    if os.path.exists(papers_dir):
        for fname in os.listdir(papers_dir):
            fpath = os.path.join(papers_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(".pdf"):
                fhash = compute_file_content_hash(fpath)
                if fhash:
                    disp_name = get_clean_display_title(fname)
                    registry["papers"][fhash] = {
                        "doc_hash": fhash,
                        "display_name": disp_name,
                        "physical_file": fname,
                        "aliases": list(dict.fromkeys([fname, disp_name])),
                        "pmid": None,
                        "doi": None,
                        "source_type": "pdf",
                        "file_size": os.path.getsize(fpath),
                        "updated_at": datetime.datetime.now().isoformat()
                    }

    # 检查 project.biokg 中可能存在的历史文献与 PMID 来源
    pkg_file = os.path.join(pdir, "project.biokg")
    if os.path.exists(pkg_file):
        try:
            with open(pkg_file, "r", encoding="utf-8") as f:
                pkg_data = json.load(f)
            # 若已有打包的 paper_registry，优先合并
            pkg_reg = pkg_data.get("paper_registry", {})
            if isinstance(pkg_reg, dict) and "papers" in pkg_reg:
                for h, item in pkg_reg.get("papers", {}).items():
                    if h not in registry["papers"]:
                        registry["papers"][h] = item
                    else:
                        merged_aliases = list(dict.fromkeys(registry["papers"][h].get("aliases", []) + item.get("aliases", [])))
                        registry["papers"][h]["aliases"] = merged_aliases

            # 兼容 analyzed_files
            for af in pkg_data.get("analyzed_files", []):
                af_str = str(af).strip()
                if not af_str:
                    continue
                pmid_m = re.search(r"(?:pmid|pubmed)?[:\s#_-]*(\d{6,9})\b", af_str, re.I)
                if pmid_m and ("pubmed" in af_str.lower() or "pmid" in af_str.lower()):
                    pmid_val = pmid_m.group(1)
                    v_hash = f"pmid:{pmid_val}"
                    if v_hash not in registry["papers"]:
                        registry["papers"][v_hash] = {
                            "doc_hash": v_hash,
                            "display_name": af_str,
                            "physical_file": None,
                            "aliases": [af_str, f"PubMed:{pmid_val}", pmid_val],
                            "pmid": pmid_val,
                            "doi": None,
                            "source_type": "online_pmid",
                            "file_size": 0,
                            "updated_at": datetime.datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"⚠️ [load_project_paper_registry] biokg 补充建表失败: {e}")

    save_project_paper_registry(pdir, registry)
    return registry


def save_project_paper_registry(project_dir_or_id: str, registry: dict):
    """原子保存文献哈希注册表到工程专属目录"""
    pdir = resolve_project_dir(project_dir_or_id)
    if not pdir:
        return
    os.makedirs(pdir, exist_ok=True)
    reg_file = os.path.join(pdir, "paper_registry.json")
    try:
        with open(reg_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ [save_project_paper_registry] 写入注册表失败: {e}")


def register_or_update_paper(project_dir_or_id: str, doc_hash: str, display_name: str,
                             physical_file: Optional[str] = None, aliases: Optional[List[str]] = None,
                             pmid: Optional[str] = None, doi: Optional[str] = None,
                             source_type: str = "pdf", file_size: int = 0) -> dict:
    """在注册表中登记或更新文献信息，自动追加并维护曾用名别名链"""
    registry = load_project_paper_registry(project_dir_or_id)
    papers = registry.setdefault("papers", {})

    clean_disp = get_clean_display_title(display_name)
    raw_aliases = aliases or []
    new_aliases = [clean_disp]
    if physical_file:
        new_aliases.append(physical_file)
        new_aliases.append(get_clean_display_title(physical_file))
    for a in raw_aliases:
        if a:
            new_aliases.append(str(a).strip())
            new_aliases.append(get_clean_display_title(str(a)))

    if doc_hash in papers:
        item = papers[doc_hash]
        old_disp = item.get("display_name", "")
        old_phys = item.get("physical_file", "")
        if old_disp:
            new_aliases.append(old_disp)
            new_aliases.append(get_clean_display_title(old_disp))
        if old_phys:
            new_aliases.append(old_phys)
            new_aliases.append(get_clean_display_title(old_phys))
        existing_aliases = item.get("aliases", [])
        combined = list(dict.fromkeys(existing_aliases + new_aliases))
        item["display_name"] = clean_disp
        if physical_file:
            item["physical_file"] = physical_file
        item["aliases"] = combined
        if pmid:
            item["pmid"] = pmid
        if doi:
            item["doi"] = doi
        if file_size > 0:
            item["file_size"] = file_size
        item["updated_at"] = datetime.datetime.now().isoformat()
    else:
        papers[doc_hash] = {
            "doc_hash": doc_hash,
            "display_name": clean_disp,
            "physical_file": physical_file,
            "aliases": list(dict.fromkeys(new_aliases)),
            "pmid": pmid,
            "doi": doi,
            "source_type": source_type,
            "file_size": file_size,
            "updated_at": datetime.datetime.now().isoformat()
        }

    save_project_paper_registry(project_dir_or_id, registry)
    return papers[doc_hash]


def resolve_paper_identity(project_dir_or_id: str, query_ref: str, registry: Optional[dict] = None) -> Optional[dict]:
    """
    多级文献身份智能解析器：
    输入 query_ref（哈希、当前展示名、带/不带 .pdf 的物理名、曾用名、PMID）
    精准解析并返回注册表项（含 doc_hash, display_name, physical_file, aliases 等）
    """
    if not query_ref:
        return None
    ref_clean = str(query_ref).strip()
    if not ref_clean:
        return None

    if registry is None:
        registry = load_project_paper_registry(project_dir_or_id)
    papers = registry.get("papers", {})

    # 1. 精确哈希匹配
    if ref_clean in papers:
        return papers[ref_clean]

    # 2. display_name 匹配 (支持忽略大小写与去首尾空格)
    ref_no_pdf = get_clean_display_title(ref_clean)
    ref_no_pdf_lower = ref_no_pdf.lower()
    for h, item in papers.items():
        disp = str(item.get("display_name", "")).strip()
        if disp == ref_no_pdf or disp == ref_clean or disp.lower() == ref_no_pdf_lower:
            return item

    # 3. physical_file 匹配
    ref_with_pdf = get_physical_pdf_filename(ref_clean)
    ref_with_pdf_lower = ref_with_pdf.lower()
    for h, item in papers.items():
        phys = str(item.get("physical_file", "")).strip()
        if phys == ref_clean or phys == ref_with_pdf or phys.lower() == ref_with_pdf_lower:
            return item

    # 4. aliases 曾用名/别名链命中 (支持不区分大小写匹配)
    for h, item in papers.items():
        aliases = item.get("aliases", [])
        for a in aliases:
            if not a:
                continue
            a_clean = str(a).strip()
            a_lower = a_clean.lower()
            if (ref_clean == a_clean or ref_no_pdf == a_clean or ref_with_pdf == a_clean or
                ref_clean.lower() == a_lower or ref_no_pdf_lower == a_lower or ref_with_pdf_lower == a_lower):
                return item

    # 5. PMID 匹配
    pmid_m = re.search(r"(?:^|[\W_])(?:pmid|pubmed)?[:\s#_-]*(\d{6,9})\b", ref_clean, re.I)
    if not pmid_m:
        pmid_m = re.search(r"^(\d{6,9})$", ref_clean)
    if pmid_m:
        pmid_digits = pmid_m.group(1)
        for h, item in papers.items():
            if str(item.get("pmid", "")).strip() == pmid_digits or h == f"pmid:{pmid_digits}":
                return item

    # 6. 物理磁盘哈希即时匹配（如用户在外部重命名了磁盘文件）
    pdir = resolve_project_dir(project_dir_or_id)
    papers_dir = os.path.join(pdir, "papers")
    if os.path.exists(papers_dir):
        cand_path = os.path.join(papers_dir, ref_clean)
        if not os.path.exists(cand_path) and os.path.exists(os.path.join(papers_dir, ref_with_pdf)):
            cand_path = os.path.join(papers_dir, ref_with_pdf)
        if os.path.exists(cand_path) and os.path.isfile(cand_path):
            disk_hash = compute_file_content_hash(cand_path)
            if disk_hash and disk_hash in papers:
                return papers[disk_hash]

    # 7. 若带有插图机制图子图标记（如 [Figure: ...]），自动提取母体文献并解析
    parent_ref = extract_parent_doc_source(ref_clean)
    if parent_ref and parent_ref != ref_clean:
        parent_ident = resolve_paper_identity(project_dir_or_id, parent_ref, registry=registry)
        if parent_ident:
            return parent_ident

    return None


def canonicalize_project_papers(project_dir_or_id: str, paper_list: List[str]) -> List[str]:
    """
    接收任意来源的文献名称列表（可能包含旧文件名、带.pdf名、曾用名、PubMed变体、复合管道字符串等），
    基于工程注册表哈希与别名链进行全局身份归一化与哈希折叠：
    1. 自动拆解管道符 '|' 并剔除系统虚拟标记（如'AI 分析'、'手动添加'等）；
    2. 对每个出处条目调用 resolve_paper_identity 进行身份解析；
    3. 命中相同 doc_hash 或文献实体的多个变体/曾用名自动折叠为当前唯一的规范 display_name；
    4. 未命中的条目自动清洗为去除 .pdf 的纯净展示名（若是在线 PMID 则保留规范格式）；
    5. 返回按原顺序首次出现去重后的规范纯净展示名列表。
    """
    if not paper_list:
        return []

    virtual_ignore_tags = {
        "AI 分析", "未知", "手动添加", "手动修改", "手动拆分",
        "AI Analysis", "Unknown", "Manual Add", "Manual Edit", "Manual Split",
        "无", "None", ""
    }

    registry = load_project_paper_registry(project_dir_or_id)
    seen_identities = set()
    result = []

    for item in paper_list:
        if not item:
            continue
        # 拆解管道符
        parts = [p.strip() for p in str(item).split("|") if p.strip()]
        for p in parts:
            if not p or p in virtual_ignore_tags:
                continue

            parent_p = extract_parent_doc_source(p)
            if not parent_p or parent_p in virtual_ignore_tags:
                continue

            ident = resolve_paper_identity(project_dir_or_id, p, registry=registry)
            if ident:
                ident_key = ident.get("doc_hash") or ident.get("display_name")
                canon_disp = ident.get("display_name") or get_clean_display_title(parent_p)
            else:
                canon_disp = get_clean_display_title(parent_p)
                ident_key = canon_disp

            if ident_key not in seen_identities and canon_disp not in seen_identities:
                seen_identities.add(ident_key)
                seen_identities.add(canon_disp)
                result.append(canon_disp)

    return result



