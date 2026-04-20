import markdown2
import os
from pathlib import Path

def generate_pdf():
    md_path = Path("reports/report_phase2.md")
    html_path = Path("reports/report_phase2.html")
    pdf_path = Path("report.pdf")
    
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        return
        
    with open(md_path, "r") as f:
        md_text = f.read()
        
    # Add Bell Labs Styling
    html_header = """
    <html>
    <head>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 40px auto; padding: 0 20px; }
        h1, h2, h3 { color: #000; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 24px; margin-bottom: 16px; }
        code { background-color: rgba(27,31,35,0.05); padding: 0.2em 0.4em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 85%; }
        pre { background-color: #f6f8fa; padding: 16px; border-radius: 3px; overflow: auto; line-height: 1.45; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        table, th, td { border: 1px solid #dfe2e5; padding: 8px 12px; }
        th { background-color: #f6f8fa; }
        img { max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .branding { text-align: center; color: #666; font-size: 0.9em; margin-bottom: 40px; }
    </style>
    </head>
    <body>
    <div class="branding">Bell Labs | Technical Validation Phase 2</div>
    """
    
    html_footer = "</body></html>"
    # Convert image links to local absolute paths for Safari to find them
    project_root = Path.cwd().absolute()
    md_text = md_text.replace("file:///Users/samarthshekhar3541/Desktop/Bell_Labs/", str(project_root) + "/")
    
    html_content = html_header + markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks"]) + html_footer
    
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"Generated {html_path}")
    
    # AppleScript to Print to PDF
    # Note: Safari needs the file to exist on disk.
    abs_html = html_path.absolute()
    abs_pdf = pdf_path.absolute()
    
    applescript = f'''
    tell application "Safari"
        activate
        open POSIX file "{abs_html}"
        delay 3 -- Wait for images to load
        print document 1 with properties {{target printer:"PDF", PDF file:POSIX file "{abs_pdf}"}}
        close document 1
    end tell
    '''
    
    # Actually, simpler: Use 'cupsfilter' on the HTML? No, it failed.
    # I'll use 'qlmanage' to generate a static image first? No.
    
    # Final Fallback for PDF: I'll inform the user I created the HTML and they can save it, 
    # but I'll try one more programmatic way if I can.
    
if __name__ == "__main__":
    generate_pdf()
