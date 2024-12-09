import base64
from io import BytesIO

import docx
import pandas as pd
from docx import Document
from datetime import datetime
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def add_paragraph_with_bold(doc, text):
    """
    Adds a paragraph to the document with text that can include **bold** segments.
    """
    paragraph = doc.add_paragraph()
    parts = text.split("**")
    for i, part in enumerate(parts):
        if i % 2 == 1:  # Odd indices are the bold segments
            run = paragraph.add_run(part)
            run.bold = True
        else:
            paragraph.add_run(part)
    return paragraph

def add_table_with_no_split(doc, df, title, description=None):
    """
    Adds a table to the document and ensures it does not split across pages.
    Allows an optional description to be included above the table with support for bold text formatting.

    Args:
        doc (Document): The Word document to add the table to.
        df (DataFrame): The data to populate the table.
        title (str): The title for the table.
        description (str, optional): An explanatory paragraph to add before the table. Default is None.
    """
    # Add heading
    doc.add_heading(title, level=3)

    # Add explanatory text if provided
    if description:
        add_paragraph_with_bold(doc, description)

    # Create the table
    table = doc.add_table(rows=1, cols=len(df.columns), style='CustomTableStyle')

    # Add table header
    header_cells = table.rows[0].cells
    for i, column_name in enumerate(df.columns):
        header_cells[i].text = column_name

    # Add table rows
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, value in enumerate(row):
            row_cells[i].text = str(value)

    # Prevent the table from splitting across pages
    for row in table.rows:
        tr = row._tr
        trPr = tr.get_or_add_trPr()
        cant_split = OxmlElement('w:cantSplit')
        trPr.append(cant_split)

    # Ensure the table as a whole does not split
    tbl = table._element
    tblPr = tbl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    tblPr.append(OxmlElement('w:tblInd'))  # Prevent splitting at the table level

    return table
def generate_introduction(customer_name, high_availability, nas_backup_details, tier_2_disks, gpu_specs):
    """
    Generate a narrative-style introduction with spaced paragraphs based on selected features.
    """
    # Base introduction paragraph
    intro_text = f"This document outlines the proposed IT infrastructure and hardware recommendations for **{customer_name}**. " \
                 "The design prioritizes reliability, scalability, and efficiency to meet the operational requirements " \
                 "of medical imaging systems in a modern healthcare environment."

    # Initialize a list to hold all sections
    sections = [intro_text]

    # Add sections dynamically based on selected features
    if high_availability:
        ha_text = "The proposed architecture includes a **High Availability (HA)** design to ensure continuous " \
                  "operation and fault tolerance, minimizing system downtime and maximizing reliability."
        sections.append(ha_text)

    if tier_2_disks is not None:
        tier_2_text = "To optimize performance, the design incorporates an intermediate tier of **high-speed storage**. " \
                      "This tier is configured with SSD RAID 5, ensuring fast access to frequently accessed images " \
                      "while maintaining data integrity and redundancy."
        sections.append(tier_2_text)

    if nas_backup_details:
        nas_text = "For long-term data protection, a **NAS backup solution** is included. This solution provides " \
                   "additional redundancy and facilitates data recovery in case of system failures or unexpected events."
        sections.append(nas_text)
    if gpu_specs:
        gpu_text = "The design incorporates **powerful GPU-enabled hardware** to support advanced AI functionalities, " \
                   "enhancing performance and enabling cutting-edge capabilities. These GPUs are optimized for resource-intensive " \
                   "tasks, ensuring seamless operation of AI modules and analytics tools."
        sections.append(gpu_text)

    # Conclude the introduction
    conclusion_text = "These recommendations are tailored to support the specific operational needs of the customer, " \
                      "ensuring an optimized and future-ready IT infrastructure."
    sections.append(conclusion_text)

    # Join all sections for the introduction
    return sections

# Add Introduction to Document
def add_introduction_to_document(doc, intro_sections):
    """
    Adds the introduction sections to the Word document with bold formatting
    and proper spacing between paragraphs.
    """
    doc.add_heading("Introduction", level=2)
    for section in intro_sections:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        paragraph_format = paragraph.paragraph_format
        paragraph_format.first_line_indent = Pt(0)
        paragraph_format.space_after = Pt(12)  # Add space after each paragraph

        # Split the section text by '**' to identify bold segments
        parts = section.split('**')
        for i, part in enumerate(parts):
            run = paragraph.add_run(part)
            if i % 2 == 1:  # Odd indices correspond to bold text
                run.bold = True
def get_binary_file_downloader_html(bin_file, file_label='File', customer_name='Customer', file_extension='.docx'):
    file_name = f'{customer_name}_IT Infrastructure, HW, and VM Design {file_extension}'
    bin_str = bin_file.read()
    href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(bin_str).decode()}" download="{file_name}">{file_label}</a>'
    return href


def generate_document_from_template(
    template_path,
    results,
    results_grade1,
    results_grade3,
    df_comparison,
    third_party_licenses,
    notes,
    input_table,
    customer_name,
    high_availability,
    server_specs,
    gpu_specs,
    first_year_storage_raid5,
    total_image_storage_raid5,
    num_studies,
    storage_title,
    shared_storage,
    raid_1_storage_tb,
    gateway_specs,
    diagnostic_specs=None,
    viewing_specs=None,
    ris_specs=None,
    project_grade=None,
    storage_table=None,
    physical_design=None,
    nas_backup_details=None,
    tier_2_disks=None,
    tier_2_disk_size=None,
    tier_3_disks=None,
    tier_3_disk_size=None
):
    """
    Generates a Word document based on the given inputs and template.
    """

    doc = Document(template_path)

    # Add Title

    title_text = f"{customer_name} IT Infrastructure, HW, and VM Design Recommendations for Medical Imaging Solutions"
    title = doc.add_heading(title_text, level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add introduction with bold formatting
    intro_text = generate_introduction(
        customer_name, high_availability, nas_backup_details, tier_2_disks, gpu_specs)

    # Add Introduction to Document
    add_introduction_to_document(doc, intro_text)

    description_text = (
        "The table below outlines the **key system inputs** and **design assumptions** that underpin "
        "the proposed IT infrastructure. These inputs ensure scalability and reliability for the project."
    )

    add_table_with_no_split(
        doc,
        input_table,
        title='System Inputs and Assumptions',
        description=description_text
    )

    # Utility to add tables

    if storage_table is not None:
        storage_description = (
            "The table below provides the **storage requirements** over the contract duration, "
            "based on the specified inputs such as the number of studies, growth rate, and duration of the contract."
        )
        add_table_with_no_split(
            doc,
            storage_table,
            title='Detailed Storage Allocation',
            description=storage_description
        )

    # Add VM Recommendations
    if not results.empty:
        # Modify the DataFrame to adjust total row content
        results = results.iloc[:-2]
        results.at[results.index[-1], "Operating System"] = ""
        results.at[results.index[-1], results.columns[0]] = "Total"

        # Define the description text for the VM table
        gpu_text = "including AI capabilities." if gpu_specs else ""
        vm_description_text = (
            f"The table below outlines the **Virtual Machine (VM) requirements** for the proposed solution. "
            f"These recommendations ensure optimal performance, scalability, and support for advanced functionalities {gpu_text}"
        )

        # Use the add_table_with_no_split function to add the table
        add_table_with_no_split(
            doc,
            results,
            title='VM Recommendations',
            description=vm_description_text
        )
    # Add Minimum vs Recommended Resources or Recommended Resources
    # Dynamically set the section title and description
    section_title = "Minimum vs. Recommended Resources" if project_grade == 1 else "Recommended Resources"

    # Description based on the section title
    if section_title == "Minimum vs. Recommended Resources":
        resource_description_text = (
            "The table below outlines the **minimum and recommended specifications** required to ensure "
            "optimal performance for the system under varying load conditions."
        )
    else:
        resource_description_text = (
            "The table below outlines the **recommended specifications** necessary for the system to "
            "operate efficiently and reliably."
        )

    # Add the table with the description
    add_table_with_no_split(
        doc,
        df_comparison,
        title=section_title,
        description=resource_description_text.strip()
    )

    def add_bullet_points_with_bold(doc, text, heading, bullet_symbol="•", force_bullet=False):
        """
        Adds a section with bullet points and handles bold text within the points.
        Ensures consistent bullet style using a specified bullet symbol.
        Removes leading symbols (like '-') from each line.
        Skips bullets for lines that are entirely bold or start with bold text unless forced.
        Skips adding the heading if it is empty or contains only whitespace.
        """
        # Add the section heading if it's not empty or whitespace
        if heading.strip():
            doc.add_heading(heading, level=3)

        # Process each line in the text
        for line in text.strip().split('\n'):
            if line.strip():  # Ensure the line isn't empty
                # Remove leading symbols like '-' or bullets
                cleaned_line = line.strip().lstrip('-').strip()

                # Check if the line starts with bold text
                if not force_bullet and cleaned_line.startswith("**") and "**" in cleaned_line[2:]:
                    # Add a paragraph with bold formatting but no bullet
                    parts = cleaned_line.split("**")
                    paragraph = doc.add_paragraph()
                    for i, part in enumerate(parts):
                        run = paragraph.add_run(part.strip())
                        if i % 2 == 1:  # Odd indices are bold text
                            run.bold = True
                else:
                    # Add a new paragraph for the bullet point
                    paragraph = doc.add_paragraph()
                    paragraph_format = paragraph.paragraph_format
                    paragraph_format.space_after = Pt(6)  # Add space after each bullet point

                    # Insert the bullet symbol
                    run = paragraph.add_run(f'{bullet_symbol} ')
                    run.bold = False  # Bullet itself is not bold

                    # Split the cleaned line to handle bold formatting
                    parts = cleaned_line.split("**")  # Handle bold segments
                    for i, part in enumerate(parts):
                        run = paragraph.add_run(part.strip())
                        if i % 2 == 1:  # Odd indices are bold text
                            run.bold = True

    # Add Physical System Design
    if physical_design:
        # Add main heading for Physical System Design
        doc.add_heading('Physical System Design', level=3)

        # Add server design description as a paragraph
        server_design_text = (
            "The system is designed with a **High Availability (HA)** configuration, ensuring fault tolerance and "
            "minimizing system downtime. This setup includes multiple servers working together to handle workloads "
            "seamlessly in the event of a hardware failure."
        ) if high_availability else (
            "The system is designed with a **single-server** configuration, optimized for cost-effectiveness while "
            "meeting the operational needs of the medical imaging infrastructure. "
            "This setup is ideal for environments with limited redundancy requirements."
        )
        add_paragraph_with_bold(doc, server_design_text)

        # Process the physical design details
        if physical_design.strip():
            design_lines = physical_design.strip().split("\n")
            if design_lines:
                # Use the first line as a subheading (e.g., "Standard Server Design")
                subheading = design_lines[0].strip().rstrip(":")  # Remove trailing colon for cleaner heading

                # Format the remaining lines using the `add_bullet_points_with_bold` function
                remaining_lines = "\n".join(design_lines[1:])
                add_bullet_points_with_bold(doc, remaining_lines, subheading)

    # Add Storage Design
    doc.add_heading('Storage Design', level=3)

    # Add descriptive text for storage design
    storage_design_text = (
        "The storage system is designed with **DAS/SAN (Direct-Attached Storage/Storage Area Network)** configurations "
        "to provide high availability and scalability for large-scale medical imaging data. This design supports "
        "redundant connections and fault tolerance to ensure continuous data availability."
    ) if high_availability else (
        "The storage system uses **built-in storage**, providing cost-effective and efficient data management for "
        "environments with single-server configurations. This design is tailored for smaller-scale operations."
    )
    add_paragraph_with_bold(doc, storage_design_text)

    # Add Storage Tiers
    tier_1_text = f"""
    **Tier 1**: OS & DB (SSD RAID 1)
    SSD Drives: 2x {raid_1_storage_tb:.2f} TB
    """

    if tier_2_disks is not None and tier_2_disk_size is not None:
        # Add Tier 2 and Tier 3 as separate sections
        tier_2_text = f"""
        **Tier 2**: Fast Image Storage (SSD RAID 5)
        SSD Drives: {tier_2_disks}x {tier_2_disk_size:.2f} TB
        """
        tier_3_text = f"""
        **Tier 3**: Long-Term Storage (HDD RAID 5)
        HDD Drives: {tier_3_disks}x {tier_3_disk_size:.2f} TB
        """
        add_bullet_points_with_bold(doc, tier_1_text + tier_2_text + tier_3_text, "")
    else:
        # Promote Tier 3 to Tier 2 if Tier 2 doesn't exist
        if tier_3_disks is not None and tier_3_disk_size is not None:
            tier_2_promoted_text = f"""
            **Tier 2**: Long-Term Storage (HDD RAID 5)
            HDD Drives: {tier_3_disks}x {tier_3_disk_size:.2f} TB
            """
            add_bullet_points_with_bold(doc, tier_1_text + tier_2_promoted_text, "")
        else:
            # Handle cases where no Tier 2 or Tier 3 details are provided
            add_bullet_points_with_bold(doc, tier_1_text, "")

    # Add NAS Backup Details
    if nas_backup_details:
        # Add main heading for NAS Backup Storage
        doc.add_heading('Backup Storage (NAS)', level=3)

        # Add descriptive text for NAS storage
        nas_backup_text = (
            "The NAS backup solution is designed to provide additional redundancy and facilitate seamless data recovery. "
            "This backup configuration ensures long-term protection for critical medical imaging data, safeguarding against "
            "unexpected system failures or data loss."
        )
        # Add descriptive text for NAS backup
        add_paragraph_with_bold(doc, nas_backup_text)

        # Add subheading for NAS technical specifications and process bullets in one go
        backup_lines = nas_backup_details.strip().split("\n")
        if len(backup_lines) > 1:
            # Extract the heading manually

            # Process the rest of the NAS details as bullet points
            remaining_lines = "\n".join(backup_lines[1:])
            add_bullet_points_with_bold(doc, remaining_lines,"",force_bullet=True)# No additional heading called here

    # Add Third Party Licenses with No Split
    description_text = (
        "The table below lists the third-party licenses required to support the proposed infrastructure, "
        "ensuring compatibility and reliability."
    )
    add_table_with_no_split(doc, third_party_licenses, title='Third Party Licenses', description=description_text)

    # Add Notes Section
    add_bullet_points_with_bold(doc, notes['licensing_notes'], 'Licensing Notes',force_bullet=True)
    add_bullet_points_with_bold(doc, notes['sizing_notes'], 'Sizing Notes')
    add_bullet_points_with_bold(doc, notes['technical_requirements'], 'Technical Requirements')
    add_bullet_points_with_bold(doc, notes['network_requirements'], 'Network Requirements (LAN)')
    # Add GPU Requirements (if provided)
    # GPU Requirements Section
    if gpu_specs:
        doc.add_heading('GPU Requirements', level=3)

        # Add descriptive text for GPU Requirements
        add_paragraph_with_bold(
            doc,
            "The system leverages powerful GPUs to support advanced AI functionalities, enabling seamless processing "
            "of resource-intensive tasks like segmentation and speech-to-text analysis. These GPUs are optimized for "
            "high-performance medical imaging workloads."
        )

        # Add GPU details as bullet points with bold formatting
        add_bullet_points_with_bold(doc, gpu_specs, "GPU Specifications", force_bullet=True)

    # Add Gateway Specifications (if provided)
    if gateway_specs:
        doc.add_heading('Gateway Specifications', level=3)

        # Add a descriptive paragraph for the Gateway section
        gateway_text = (
            "The gateway is responsible for acquiring and transmitting images and data from the modalities "
            "to the PACS system at the main site. This ensures secure, efficient data transfer and seamless system interoperability."
        )
        add_paragraph_with_bold(doc, gateway_text)

        # Add Gateway details as bullet points using the existing utility function
        add_bullet_points_with_bold(doc, gateway_specs, 'Recommended Specs ')

    # Workstation Specifications Section
    if diagnostic_specs is not None or viewing_specs is not None or ris_specs is not None:
        doc.add_page_break()
        doc.add_heading('Workstation Specifications', level=2)

        # Add descriptive text for workstation specifications
        add_paragraph_with_bold(
            doc,
            "The proposed workstations are tailored to meet the demands of high-resolution medical imaging and efficient "
            "clinical workflows. Each workstation is equipped with state-of-the-art hardware to ensure optimal performance "
            "and user satisfaction."
        )

        # Add Diagnostic Workstation Specifications
        if diagnostic_specs is not None:
            add_table_with_no_split(doc, diagnostic_specs, title='Diagnostic Workstation')

        # Add Viewing Workstation Specifications
        if viewing_specs is not None:
            add_paragraph_with_bold(doc, "**Viewing Workstation Specifications**:")
            add_table_with_no_split(doc, viewing_specs, title='Viewing Workstation')

        # Add RIS Workstation Specifications
        if ris_specs is not None:
            add_paragraph_with_bold(doc, "**RIS Workstation Specifications**:")
            add_table_with_no_split(doc, ris_specs, title='RIS Workstation')

    # Save to a buffer and return
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
