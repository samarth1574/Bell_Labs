import markdown2
import os
from pathlib import Path
import re

def generate_ieee_report():
    md_path = Path("reports/report_phase2.md")
    html_path = Path("reports/report_ieee.html")
    pdf_path = Path("report.pdf")
    
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        return
        
    with open(md_path, "r") as f:
        md_text = f.read()

    # --- IEEE CSS Template ---
    ieee_css = """
    <style>
        @page {
            size: letter;
            margin: 20mm 15mm;
        }
        body {
            font-family: "Times New Roman", Times, serif;
            font-size: 10pt;
            line-height: 1.15;
            color: #000;
            margin: 0;
            padding: 0;
            background: white;
        }
        .container {
            width: 100%;
        }
        .header-section {
            text-align: center;
            margin-bottom: 20px;
        }
        h1.main-title {
            font-size: 24pt;
            margin-bottom: 10px;
            font-weight: normal;
        }
        .authors {
            font-size: 11pt;
            margin-bottom: 20px;
        }
        .abstract-container {
            margin-bottom: 20px;
            font-weight: bold;
            font-style: italic;
        }
        .abstract-title {
            font-weight: bold;
            font-style: normal;
            display: inline;
            margin-right: 5px;
        }
        .content-columns {
            column-count: 2;
            column-gap: 5mm;
            column-fill: auto;
            text-align: justify;
        }
        h2 { 
            font-size: 12pt; 
            text-align: center; 
            text-transform: uppercase; 
            margin-top: 15px;
            margin-bottom: 10px;
            break-after: avoid;
        }
        h3 { 
            font-size: 11pt; 
            font-style: italic; 
            margin-top: 10px;
            margin-bottom: 5px;
            break-after: avoid;
        }
        p { margin-bottom: 8px; text-indent: 1em; }
        p:first-of-type { text-indent: 0; }
        
        table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 8pt; }
        table, th, td { border: 1px solid black; padding: 4px; text-align: center; }
        th { background-color: #f2f2f2; }
        
        img { width: 100%; height: auto; margin: 10px 0; border: 0.5px solid #ccc; }
        .full-width { column-span: all; text-align: center; }
        
        .references { font-size: 8pt; }
        .references h2 { text-align: left; text-transform: none; font-size: 10pt; }
        
        code { font-family: "Courier New", Courier, monospace; font-size: 9pt; background: #eee; }
        pre { font-family: "Courier New", Courier, monospace; font-size: 8pt; background: #f9f9f9; padding: 5px; border: 1px solid #ccc; white-space: pre-wrap; word-wrap: break-word; }
    </style>
    """

    # --- Pre-processing Markdown for IEEE ---
    # Separate title, authors, abstract from the rest
    lines = md_text.split("\n")
    title = ""
    authors = ""
    abstract = ""
    body_md = ""
    
    in_abstract = False
    for line in lines:
        if line.startswith("# ") and not title:
            title = line.replace("# ", "").strip()
        elif line.startswith("> **Authors:**"):
            authors = line.replace("> **Authors:**", "").strip()
        elif "Abstract" in line and len(line) < 20:
            in_abstract = True
        elif in_abstract and line.strip() and not line.startswith("---"):
            abstract += line.strip() + " "
        elif line.startswith("---") and in_abstract:
            in_abstract = False
        elif not in_abstract and title:
            if not authors in line:
                body_md += line + "\n"

    # Convert body to HTML
    body_html = markdown2.markdown(body_md, extras=["tables", "fenced-code-blocks"])
    
    # Clean up body HTML: remove the title if it leaked in
    body_html = re.sub(r'<h1>.*?</h1>', '', body_html, count=1, flags=re.DOTALL)

    # Wrap figures that should be full-width (Architecture diagrams)
    body_html = body_html.replace('<p><img src="file:///Users/samarthshekhar3541/Desktop/Bell_Labs/reports/figures/detector_output.png"', 
                                  '<div class="full-width"><img src="file:///Users/samarthshekhar3541/Desktop/Bell_Labs/reports/figures/detector_output.png"')
    body_html = body_html.replace('alt="Detector Output" /></p>', 'alt="Detector Output" /></div>')

    # Fix image paths
    project_root = Path.cwd().absolute()
    body_html = body_html.replace("file:///Users/samarthshekhar3541/Desktop/Bell_Labs/", str(project_root) + "/")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        {ieee_css}
    </head>
    <body>
        <div class="container">
            <div class="header-section">
                <h1 class="main-title">{title}</h1>
                <div class="authors">{authors}</div>
                <div class="abstract-container">
                    <span class="abstract-title">Abstract—</span>{abstract}
                </div>
            </div>
            
            <div class="content-columns">
                {body_html}
            </div>
        </div>
    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Generated IEEE HTML: {html_path}")

if __name__ == "__main__":
    generate_ieee_report()
