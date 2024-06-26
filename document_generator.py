import base64
from io import BytesIO
import pandas as pd
from docx import Document
from datetime import datetime
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH


def get_binary_file_downloader_html(bin_file, file_label='File', customer_name='Customer', file_extension='.docx'):
    file_name = f'{customer_name}_vm_results{file_extension}'
    bin_str = bin_file.read()
    href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(bin_str).decode()}" download="{file_name}">{file_label}</a>'
    return href


def generate_document_from_template(template_path, results, results_grade1, results_grade3, df_comparison,
                                    third_party_licenses, notes, input_table, customer_name, high_availability,
                                    server_specs, gpu_specs,
                                    first_year_storage_raid5, total_image_storage_raid5, num_studies):
    doc = Document(template_path)

    title = doc.add_heading(f'{customer_name} HW Recommendation', level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('Customer Information', level=2)
    doc.add_paragraph(f'Customer Name: {customer_name}')
    doc.add_paragraph(f'Date: {datetime.now().strftime("%Y-%m-%d")}')

    def add_table(df, heading, style_name):
        doc.add_heading(heading, level=2)
        table = doc.add_table(rows=1, cols=len(df.columns), style=style_name)
        hdr_cells = table.rows[0].cells
        for i, col_name in enumerate(df.columns):
            hdr_cells[i].text = col_name
        for index, row in df.iterrows():
            row_cells = table.add_row().cells
            for i, value in enumerate(row):
                row_cells[i].text = str(value)

    # Add the number of studies to the input_table first
    num_studies_df = pd.DataFrame({"Input": ["Number of Studies"], "Value": [num_studies]})
    input_table = pd.concat([num_studies_df, input_table], ignore_index=True)
    input_table['Value'] = input_table['Value'].replace({True: 'Required', False: 'Not Required'})

    add_table(input_table, 'Customer Load / Technical Inputs', 'CustomTableStyle')

    doc.add_heading('Storage Requirements', level=2)
    doc.add_paragraph(f'RAID 1 Storage (SSD): {first_year_storage_raid5 / 1024:.2f} TB')
    doc.add_paragraph(f'RAID 5 Storage (First Year): {first_year_storage_raid5 / 1024:.2f} TB')
    doc.add_paragraph(f'RAID 5 Storage (Full Contract Duration): {total_image_storage_raid5 / 1024:.2f} TB')

    # Remove the last two rows from results before adding to the document
    results = results.iloc[:-2]

    # Move "Windows Server 2019 or Higher" and add "Total"
    if not results.empty:
        results.at[results.index[-1], "Operating System"] = ""
        results.at[results.index[-1], results.columns[0]] = "Total"

    add_table(results, 'VM Recommendations', 'CustomTableStyle')
    add_table(df_comparison, 'Minimum vs. Recommended Resources', 'CustomTableStyle')
    add_table(third_party_licenses, 'Third Party Licenses', 'CustomTableStyle')

    def add_bullet_points(text, heading):
        doc.add_heading(heading, level=2)
        for line in text.strip().split('\n'):
            if line.strip():
                paragraph = doc.add_paragraph(line.strip())
                p = paragraph._element
                pPr = p.get_or_add_pPr()
                numPr = OxmlElement('w:numPr')
                numId = OxmlElement('w:numId')
                numId.set(qn('w:val'), '1')
                numPr.append(numId)
                pPr.append(numPr)

    add_bullet_points(notes['sizing_notes'].replace('-', ''), 'Sizing Notes')
    add_bullet_points(notes['technical_requirements'].replace('-', ''), 'Technical Requirements')
    add_bullet_points(notes['network_requirements'].replace('-', ''), 'Network Requirements (LAN)')

    design_heading = 'High Availability Design' if high_availability else 'Server Design'
    doc.add_heading(design_heading, level=2)
    doc.add_paragraph(server_specs)

    if gpu_specs:
        doc.add_heading('GPU Requirements', level=2)
        doc.add_paragraph(gpu_specs)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
