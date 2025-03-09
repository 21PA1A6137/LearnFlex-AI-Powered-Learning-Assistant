from fpdf import FPDF
from datetime import datetime
import streamlit as st
import re
import time
from langchain.schema import HumanMessage, AIMessage

class CustomPDF(FPDF):
    def footer(self):
        """Add footer with page number and faculty name."""
        self.set_y(-15)
        self.set_font("Times", size=10)
        self.cell(0, 10, f"Page {self.page_no()} - Faculty: {self.faculty_name}", align="C")

    def header(self):
        """Add header with Unit and Subject information and draw a border."""
        self.set_font("Times", style="B", size=14)
        self.cell(0, 10, f"Unit {self.unit_number}: {self.subject}", align="R", ln=True)
        self.ln(5)
        
        # ✅ Add a border around the entire page
        self.rect(5, 5, 200, 287)  # x, y, width, height


def clean_text(content):
    """Applies proper formatting for bold and italic text."""
    content = re.sub(r"={3,}|-{3,}", "", content)  # Remove `===` and `---`
    content = re.sub(r"\*\*(.*?)\*\*", r"\\b\1\\b", content)  # Replace `**bold**`
    content = re.sub(r"\*(.*?)\*", r"\\i\1\\i", content)
    # content = re.sub(r"^\s*\*\s*", "• ", content, flags=re.MULTILINE)
    return content.strip()

def render_text(pdf, content):
    """Formats and adds text to the PDF with bold and italic support."""
    pdf.set_font("Times", size=12)
    pdf.set_text_color(50, 50, 50)  # Dark grey text for better readability
    lines = content.split("\n")

    for line in lines:
        if "\\b" in line:  # Handle bold text
            pdf.set_font("Times", style="B", size=12)
            line = line.replace("\\b", "")
        elif "\\i" in line:  # Handle italic text
            pdf.set_font("Times", style="I", size=12)
            line = line.replace("\\i", "")
        else:
            pdf.set_font("Times", size=12)

        pdf.multi_cell(0, 7, line, border=0)  # ✅ Increased line spacing
        pdf.ln(2)  # ✅ More spacing for readability

def render_code(pdf, content):
    """Render a code block with a colored background."""
    pdf.set_font("Courier", size=10)
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_fill_color(30, 30, 30)  # Dark grey background
    pdf.multi_cell(0, 7, content, border=1, fill=True)
    pdf.ln(3)
    pdf.set_text_color(0, 0, 0)  # Reset to black


def generate_pdf(history, subject, unit_number, faculty_name, pdf_name, summary):
    pdf = CustomPDF()
    pdf.subject = subject or "General Subject"
    pdf.unit_number = unit_number or "1"
    pdf.faculty_name = faculty_name or "Unknown Faculty"
    pdf.set_font("Times", size=12)

    pdf.add_page()
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    progress_bar = st.sidebar.progress(0)

    question_count = 1  # ✅ Track question numbers

    for idx, message in enumerate(history, start=1):
        progress_bar.progress(int((idx / len(history)) * 100))
        if pdf.get_y() > 250:
            pdf.add_page()

        if isinstance(message, HumanMessage):
            pdf.set_font("Times", style="B", size=12)
            pdf.set_text_color(0, 0, 255)
            pdf.multi_cell(0, 7, f"Q{question_count}: {clean_text(message.content)}", border=1)
            question_count += 1  # ✅ Only increment for questions
            pdf.ln(3)

        elif isinstance(message, AIMessage):
            pdf.set_font("Times", size=12)
            pdf.set_text_color(0, 0, 0)

            # ✅ Detect and style code blocks
            code_blocks = re.findall(r"```(.*?)```", message.content, re.DOTALL)
            if code_blocks:
                text_parts = re.split(r"```.*?```", message.content, flags=re.DOTALL)
                for i, part in enumerate(text_parts):
                    render_text(pdf, clean_text(part))  # Regular text
                    if i < len(code_blocks):
                        render_code(pdf, code_blocks[i])  # Code block
            else:
                pdf.multi_cell(0, 7, f"Ans: {clean_text(message.content)}")

        pdf.ln(3)
        time.sleep(0.1)

    pdf_output_path = f"{pdf_name}.pdf" if pdf_name else f"{summary}.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path
