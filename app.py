"""
Persona-Driven Document Intelligence - Lightweight Web UI
Runs with native Python 3.10+ using Flask.
"""

import os
import sys
import json
import re
import math
import time
from pathlib import Path
from collections import Counter
from flask import Flask, request, jsonify, render_template_string, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max upload
UPLOAD_FOLDER = Path(__file__).resolve().parent / "input" / "web_uploads"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "output"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Detect if full ML pipeline can be imported
HAS_ML_PIPELINE = False
try:
    import fitz
    from sentence_transformers import SentenceTransformer
    HAS_ML_PIPELINE = True
except ImportError:
    HAS_ML_PIPELINE = False


# Lightweight Pure-Python Fallback Pipeline using pypdf & BM25/TF-IDF
def extract_text_pypdf(pdf_path):
    import pypdf
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text.strip()})
    return pages


def light_bm25_score(query_tokens, doc_tokens, avg_doc_len, doc_count, doc_freqs, k1=1.5, b=0.75):
    score = 0.0
    doc_len = len(doc_tokens)
    token_counts = Counter(doc_tokens)
    for q in query_tokens:
        if q not in token_counts:
            continue
        df = doc_freqs.get(q, 1)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
        tf = token_counts[q]
        denom = tf + k1 * (1.0 - b + b * (doc_len / (avg_doc_len or 1.0)))
        score += idf * (tf * (k1 + 1.0)) / denom
    return score


def process_with_fallback(pdf_paths, persona, job):
    all_chunks = []
    for pdf_path in pdf_paths:
        filename = Path(pdf_path).name
        pages = extract_text_pypdf(pdf_path)
        for p in pages:
            text = p["text"]
            if not text:
                continue
            # Split into heading candidates and paragraphs
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            first_line = lines[0] if lines else f"Page {p['page']} Overview"
            title = first_line[:80]
            all_chunks.append({
                "document": filename,
                "page": p["page"],
                "section_title": title,
                "text": text
            })

    if not all_chunks:
        return {"metadata": {"input_documents": [Path(p).name for p in pdf_paths], "persona": persona, "job_to_be_done": job}, "extracted_sections": [], "subsection_analysis": []}

    query_str = f"{persona} {job}".lower()
    query_tokens = [w for w in re.findall(r'\b\w{3,}\b', query_str)]

    tokenized_docs = [[w for w in re.findall(r'\b\w{3,}\b', c["text"].lower())] for c in all_chunks]
    doc_count = len(tokenized_docs)
    avg_len = sum(len(d) for d in tokenized_docs) / (doc_count or 1)

    doc_freqs = Counter()
    for d in tokenized_docs:
        doc_freqs.update(set(d))

    scored = []
    for i, chunk in enumerate(all_chunks):
        score = light_bm25_score(query_tokens, tokenized_docs[i], avg_len, doc_count, doc_freqs)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    extracted_sections = []
    subsection_analysis = []

    for rank, (score, chunk) in enumerate(scored[:10]):
        extracted_sections.append({
            "document": chunk["document"],
            "page_number": chunk["page"],
            "section_title": chunk["section_title"],
            "importance_rank": rank + 1
        })

        if rank < 5:
            paras = [p.strip() for p in chunk["text"].split("\n\n") if len(p.strip()) > 30]
            if not paras:
                paras = [chunk["text"][:300]]
            best_para = paras[0]
            best_para_score = -1
            for para in paras:
                p_tokens = [w for w in re.findall(r'\b\w{3,}\b', para.lower())]
                p_score = sum(1 for q in query_tokens if q in p_tokens)
                if p_score > best_para_score:
                    best_para_score = p_score
                    best_para = para

            subsection_analysis.append({
                "document": chunk["document"],
                "page_number": chunk["page"],
                "refined_text": best_para.strip()
            })

    return {
        "metadata": {
            "input_documents": [Path(p).name for p in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
            "engine": "Fast-Local-BM25 (Light Fallback)",
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }


# Create a sample PDF if none exists
def ensure_sample_pdf():
    sample_path = UPLOAD_FOLDER / "sample_quarterly_report.pdf"
    if not sample_path.exists() or sample_path.stat().st_size < 500:
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
            b"2 0 obj <</Type /Pages /Kids [3 0 R 4 0 R] /Count 2>> endobj\n"
            b"3 0 obj <</Type /Page /Parent 2 0 R /Resources <</Font <</F1 5 0 R>>>> /MediaBox [0 0 612 792] /Contents 6 0 R>> endobj\n"
            b"4 0 obj <</Type /Page /Parent 2 0 R /Resources <</Font <</F1 5 0 R>>>> /MediaBox [0 0 612 792] /Contents 7 0 R>> endobj\n"
            b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
            b"6 0 obj <</Length 350>> stream\n"
            b"BT\n/F1 16 Tf 50 720 Td (1. Executive Summary & Liquidity Position) Tj\n"
            b"/F1 11 Tf 0 -25 Td (The corporation maintained total available liquidity of $4.2 billion as of Q3.) Tj\n"
            b"0 -18 Td (Credit facilities remain undrawn with $1.5 billion in committed revolving credit lines.) Tj\n"
            b"0 -18 Td (Debt covenants require maintaining a minimum consolidated interest coverage ratio of 3.0x.) Tj\n"
            b"0 -18 Td (Working capital management and operational efficiency provided positive operating cash flows.) Tj\n"
            b"ET\nendstream\nendobj\n"
            b"7 0 obj <</Length 360>> stream\n"
            b"BT\n/F1 16 Tf 50 720 Td (2. Credit Facility and Debt Maturity Schedule) Tj\n"
            b"/F1 11 Tf 0 -25 Td (Senior unsecured notes of $800 million are due for final repayment in November 2027.) Tj\n"
            b"0 -18 Td (Management anticipates refinancing through long-term term loans subject to market rates.) Tj\n"
            b"0 -18 Td (Liquidity ratios reflect strong debt service capability and sustainable leverage ratios.) Tj\n"
            b"0 -18 Td (The company complies with all restrictive debt covenants and reporting milestones.) Tj\n"
            b"ET\nendstream\nendobj\n"
            b"xref\n0 8\n0000000000 65535 f \n"
            b"trailer <</Size 8 /Root 1 0 R>>\nstartxref\n800\n%%EOF\n"
        )
        sample_path.write_bytes(pdf_content)
    return sample_path


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocIntel - Persona-Driven Document Intelligence</title>
    <style>
        :root {
            --bg: #f8fafc;
            --surface: #ffffff;
            --border: #e2e8f0;
            --border-hover: #cbd5e1;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --primary-light: #eff6ff;
            --accent: #10b981;
            --accent-light: #ecfdf5;
            --badge-bg: #f1f5f9;
            --radius: 8px;
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: var(--font-sans); background: var(--bg); color: var(--text-main); line-height: 1.5; padding: 24px; }
        .container { max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }

        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
        .brand { display: flex; align-items: center; gap: 10px; }
        .logo { width: 36px; height: 36px; background: var(--primary); color: white; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 18px; }
        h1 { font-size: 20px; font-weight: 700; }
        .subtitle { font-size: 13px; color: var(--text-muted); }
        .engine-tag { font-size: 12px; padding: 4px 10px; border-radius: 12px; background: var(--primary-light); color: var(--primary); font-weight: 600; }

        .grid { display: grid; grid-template-columns: 380px 1fr; gap: 20px; }
        @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }

        .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; }
        .card-title { font-size: 15px; font-weight: 600; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }

        .form-group { margin-bottom: 14px; }
        label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-main); }
        input[type="text"], textarea { width: 100%; border: 1px solid var(--border); border-radius: 6px; padding: 9px 12px; font-size: 13px; font-family: inherit; transition: border-color 0.15s; }
        input[type="text"]:focus, textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px var(--primary-light); }
        textarea { resize: vertical; min-height: 72px; }

        .presets { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
        .preset-btn { font-size: 11px; padding: 3px 8px; background: var(--badge-bg); border: 1px solid var(--border); border-radius: 4px; cursor: pointer; color: var(--text-muted); transition: all 0.15s; }
        .preset-btn:hover { background: var(--primary-light); color: var(--primary); border-color: var(--primary); }

        .dropzone { border: 2px dashed var(--border); border-radius: 6px; padding: 20px; text-align: center; cursor: pointer; background: #fafafa; transition: border-color 0.2s; }
        .dropzone:hover { border-color: var(--primary); background: var(--primary-light); }
        .file-list { margin-top: 8px; font-size: 12px; color: var(--text-muted); max-height: 80px; overflow-y: auto; }
        .file-item { display: flex; justify-content: space-between; padding: 3px 6px; background: #f1f5f9; border-radius: 4px; margin-bottom: 4px; }

        .btn-submit { width: 100%; background: var(--primary); color: white; border: none; border-radius: 6px; padding: 11px; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; justify-content: center; align-items: center; gap: 8px; transition: background 0.15s; }
        .btn-submit:hover { background: var(--primary-hover); }
        .btn-submit:disabled { background: #94a3b8; cursor: not-allowed; }

        /* Results Pane */
        .results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .tabs { display: flex; gap: 8px; }
        .tab-btn { background: none; border: 1px solid transparent; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; cursor: pointer; color: var(--text-muted); }
        .tab-btn.active { background: var(--primary-light); color: var(--primary); border-color: var(--primary); font-weight: 600; }

        .rank-list { display: flex; flex-direction: column; gap: 10px; }
        .rank-item { border: 1px solid var(--border); border-radius: 6px; padding: 14px; display: flex; flex-direction: column; gap: 6px; background: #ffffff; transition: box-shadow 0.15s; }
        .rank-item:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.04); border-color: var(--border-hover); }
        .rank-top { display: flex; align-items: center; justify-content: space-between; }
        .rank-badge { font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; background: #e0e7ff; color: #4338ca; }
        .rank-badge.top-3 { background: #dcfce7; color: #15803d; }
        .rank-title { font-size: 14px; font-weight: 600; }
        .rank-meta { font-size: 12px; color: var(--text-muted); display: flex; gap: 12px; }
        .refined-box { background: #f8fafc; border-left: 3px solid var(--primary); padding: 8px 12px; font-size: 12.5px; line-height: 1.5; color: #334155; margin-top: 4px; border-radius: 0 4px 4px 0; }

        .json-box { background: #0f172a; color: #f8fafc; font-family: var(--font-mono); font-size: 12px; padding: 16px; border-radius: 6px; overflow-x: auto; max-height: 520px; white-space: pre-wrap; }
        .action-row { display: flex; gap: 8px; justify-content: flex-end; }
        .btn-secondary { background: white; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; }
        .btn-secondary:hover { background: #f1f5f9; }

        .empty-state { text-align: center; padding: 60px 20px; color: var(--text-muted); }
        .empty-icon { font-size: 36px; margin-bottom: 8px; opacity: 0.5; }
        .loading-spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #ffffff; border-radius: 50%; border-top-color: transparent; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <div class="logo">D</div>
                <div>
                    <h1>Persona-Driven Document Intelligence</h1>
                    <div class="subtitle">Extract and rank section intelligence based on user persona & job intent</div>
                </div>
            </div>
            <div class="engine-tag">
                {{ "YOLO + MRL Transformer" if has_ml else "Lightweight Pure-Python Engine" }}
            </div>
        </header>

        <div class="grid">
            <!-- Left Input Column -->
            <div class="card">
                <div class="card-title">Analysis Configuration</div>
                <form id="intelForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="persona">Persona</label>
                        <input type="text" id="persona" name="persona" placeholder="e.g. Chief Financial Officer" value="Financial Risk Analyst" required>
                        <div class="presets">
                            <span class="preset-btn" onclick="setPreset('Financial Risk Analyst', 'Identify liquidity risks, debt covenants, and cash burn')">Finance</span>
                            <span class="preset-btn" onclick="setPreset('Cybersecurity Auditor', 'Review data privacy, encryption standards, and breach protocols')">Security</span>
                            <span class="preset-btn" onclick="setPreset('ESG Compliance Director', 'Extract carbon emission disclosures, net-zero goals, and governance')">ESG</span>
                            <span class="preset-btn" onclick="setPreset('Product Strategy Lead', 'Analyze competitive landscape, market share, and product roadmap')">Strategy</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="job">Job-to-be-Done</label>
                        <textarea id="job" name="job" placeholder="What specific information do you need extracted?" required>Evaluate credit facility terms, debt maturities, and liquidity ratios across all documents.</textarea>
                    </div>

                    <div class="form-group">
                        <label>Input Document(s) (PDF)</label>
                        <div class="dropzone" id="dropzone" onclick="document.getElementById('pdfInput').click()">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin: 0 auto; color: var(--text-muted);">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="17 8 12 3 7 8"></polyline>
                                <line x1="12" y1="3" x2="12" y2="15"></line>
                            </svg>
                            <div style="font-size: 13px; font-weight: 600; margin-top: 6px;">Click or drag PDFs here</div>
                            <div style="font-size: 11px; color: var(--text-muted);">Supports single or multiple PDF files</div>
                            <input type="file" id="pdfInput" name="pdfs" accept=".pdf" multiple style="display: none;" onchange="updateFileList(this.files)">
                        </div>
                        <div class="file-list" id="fileList"></div>
                    </div>

                    <button type="submit" class="btn-submit" id="submitBtn">
                        <span id="btnText">Run Intelligence Pipeline</span>
                    </button>
                </form>
            </div>

            <!-- Right Results Column -->
            <div class="card">
                <div class="results-header">
                    <div class="card-title" style="margin: 0;">Extracted Intelligence</div>
                    <div class="tabs">
                        <button class="tab-btn active" id="tabSections" onclick="switchTab('sections')">Ranked Sections</button>
                        <button class="tab-btn" id="tabJson" onclick="switchTab('json')">JSON Payload</button>
                    </div>
                </div>

                <div id="loadingState" style="display: none; text-align: center; padding: 60px 20px;">
                    <div class="loading-spinner" style="border-color: var(--primary); border-top-color: transparent; width: 32px; height: 32px; margin-bottom: 12px;"></div>
                    <div style="font-size: 14px; font-weight: 600;">Analyzing documents against persona intent...</div>
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Segmenting sections, generating vector rankings, and refining subsections.</div>
                </div>

                <div id="emptyState" class="empty-state">
                    <div class="empty-icon">📄</div>
                    <div style="font-weight: 600; font-size: 15px;">No Analysis Run Yet</div>
                    <div style="font-size: 13px; margin-top: 4px;">Configure persona, query, and upload documents on the left, then click <strong>Run Intelligence Pipeline</strong>.</div>
                </div>

                <div id="resultsContent" style="display: none;">
                    <div class="action-row" style="margin-bottom: 12px;">
                        <button class="btn-secondary" onclick="copyJson()">📋 Copy JSON</button>
                        <button class="btn-secondary" onclick="downloadJson()">💾 Download JSON</button>
                    </div>

                    <div id="sectionsView" class="rank-list"></div>
                    <div id="jsonView" style="display: none;">
                        <pre class="json-box" id="jsonPre"></pre>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentResult = null;
        let selectedFiles = [];

        function setPreset(persona, job) {
            document.getElementById('persona').value = persona;
            document.getElementById('job').value = job;
        }

        function updateFileList(files) {
            selectedFiles = Array.from(files);
            const list = document.getElementById('fileList');
            list.innerHTML = '';
            if (selectedFiles.length === 0) return;
            selectedFiles.forEach((f, i) => {
                const item = document.createElement('div');
                item.className = 'file-item';
                item.innerHTML = `<span>📄 ${f.name}</span><span>${(f.size/1024).toFixed(0)} KB</span>`;
                list.appendChild(item);
            });
        }

        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--primary)'; });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--border)'; });
        });
        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length) {
                document.getElementById('pdfInput').files = e.dataTransfer.files;
                updateFileList(e.dataTransfer.files);
            }
        });

        function switchTab(tab) {
            document.getElementById('tabSections').classList.toggle('active', tab === 'sections');
            document.getElementById('tabJson').classList.toggle('active', tab === 'json');
            document.getElementById('sectionsView').style.display = tab === 'sections' ? 'flex' : 'none';
            document.getElementById('jsonView').style.display = tab === 'json' ? 'block' : 'none';
        }

        document.getElementById('intelForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            
            const formData = new FormData();
            formData.append('persona', document.getElementById('persona').value);
            formData.append('job', document.getElementById('job').value);

            const fileInput = document.getElementById('pdfInput');
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('pdfs', fileInput.files[i]);
            }

            submitBtn.disabled = true;
            btnText.innerHTML = '<span class="loading-spinner"></span> Processing...';
            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('resultsContent').style.display = 'none';
            document.getElementById('loadingState').style.display = 'block';

            try {
                const resp = await fetch('/api/analyze', { method: 'POST', body: formData });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Processing error');
                currentResult = data;
                renderResults(data);
            } catch (err) {
                alert('Analysis failed: ' + err.message);
                document.getElementById('emptyState').style.display = 'block';
            } finally {
                submitBtn.disabled = false;
                btnText.textContent = 'Run Intelligence Pipeline';
                document.getElementById('loadingState').style.display = 'none';
            }
        });

        function renderResults(data) {
            const container = document.getElementById('sectionsView');
            container.innerHTML = '';

            const subMap = {};
            (data.subsection_analysis || []).forEach(sub => {
                const key = `${sub.document}_p${sub.page_number}`;
                subMap[key] = sub.refined_text;
            });

            (data.extracted_sections || []).forEach(sec => {
                const item = document.createElement('div');
                item.className = 'rank-item';
                const isTop = sec.importance_rank <= 3;
                const subKey = `${sec.document}_p${sec.page_number}`;
                const refinedText = subMap[subKey];

                item.innerHTML = `
                    <div class="rank-top">
                        <div class="rank-title">${escapeHtml(sec.section_title || 'Untitled Section')}</div>
                        <span class="rank-badge ${isTop ? 'top-3' : ''}">Rank #${sec.importance_rank}</span>
                    </div>
                    <div class="rank-meta">
                        <span>📁 ${escapeHtml(sec.document)}</span>
                        <span>📄 Page ${sec.page_number}</span>
                    </div>
                    ${refinedText ? `<div class="refined-box"><strong>Refined Subsection:</strong> ${escapeHtml(refinedText)}</div>` : ''}
                `;
                container.appendChild(item);
            });

            document.getElementById('jsonPre').textContent = JSON.stringify(data, null, 2);
            document.getElementById('resultsContent').style.display = 'block';
            switchTab('sections');
        }

        function escapeHtml(str) {
            return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        function copyJson() {
            if (!currentResult) return;
            navigator.clipboard.writeText(JSON.stringify(currentResult, null, 2)).then(() => alert('JSON copied to clipboard!'));
        }

        function downloadJson() {
            if (!currentResult) return;
            const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'doc_intel_output.json';
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, has_ml=HAS_ML_PIPELINE)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    persona = request.form.get("persona", "").strip()
    job = request.form.get("job", "").strip()
    if not persona or not job:
        return jsonify({"error": "Persona and Job are required"}), 400

    uploaded_files = request.files.getlist("pdfs")
    saved_pdf_paths = []

    for f in uploaded_files:
        if f and f.filename and f.filename.lower().endswith(".pdf"):
            save_path = UPLOAD_FOLDER / f.filename
            f.save(str(save_path))
            saved_pdf_paths.append(save_path)

    # If no files uploaded, use demo sample PDF
    if not saved_pdf_paths:
        sample_path = ensure_sample_pdf()
        saved_pdf_paths.append(sample_path)

    # Process using full pipeline if available, otherwise fallback
    try:
        if HAS_ML_PIPELINE:
            from process_pdfs_1b import (
                load_yolo_model, process_pdf_1a, create_corpus_from_pdfs,
                encode_query_and_corpus, retrieve_and_rank_sections
            )
            BASE_DIR = Path(__file__).resolve().parent
            yolo_path = BASE_DIR / 'models' / 'yolov11s-doclaynet.pt'
            yolo_model = load_yolo_model(str(yolo_path))
            outline_data_map = {}
            for pdf_file in saved_pdf_paths:
                outline_data_map[pdf_file.name] = process_pdf_1a(pdf_file, yolo_model)
            corpus = create_corpus_from_pdfs(saved_pdf_paths, outline_data_map)
            model_dir = BASE_DIR / 'models' / 'static-similarity-mrl-multilingual-v1'
            model = SentenceTransformer(str(model_dir))
            q_emb, c_emb = encode_query_and_corpus(persona, job, corpus, model)
            sections, sub_analysis = retrieve_and_rank_sections(q_emb, c_emb, corpus, model)
            result = {
                "metadata": {
                    "input_documents": [p.name for p in saved_pdf_paths],
                    "persona": persona,
                    "job_to_be_done": job,
                    "engine": "Full YOLOv11 + MRL SentenceTransformer",
                    "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                },
                "extracted_sections": sections,
                "subsection_analysis": sub_analysis
            }
        else:
            result = process_with_fallback(saved_pdf_paths, persona, job)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting DocIntel Web UI on http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
