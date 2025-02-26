import os
import time
import logging
import math
from PIL import Image
import streamlit as st
import pandas as pd
from document_generator import generate_document_from_template, get_binary_file_downloader_html
from vm_calculations import calculate_vm_requirements, modality_sizes

logging.basicConfig(level=logging.INFO)


def add_server(servers, remaining_threads, remaining_ram, max_threads, max_ram):
    if remaining_threads == 0 or remaining_ram == 0:
        return remaining_threads, remaining_ram

    print(f"Entering add_server with remaining_threads={remaining_threads} and remaining_ram={remaining_ram}")

    # Calculate the number of servers needed
    num_servers_threads = math.ceil(remaining_threads / max_threads)
    num_servers_ram = math.ceil(remaining_ram / max_ram)
    num_servers = max(num_servers_threads, num_servers_ram)

    # Uniform allocation for threads and RAM
    threads_per_server = math.ceil(remaining_threads / num_servers)
    ram_per_server = math.ceil(remaining_ram / num_servers)

    # Ensure threads are divisible by 2
    threads_per_server -= threads_per_server % 2

    # Allocate resources for the current server
    if threads_per_server > 128:
        processors_to_allocate = 'Dual'
        cores_per_processor = min(math.ceil(threads_per_server / 4), 64)
        cores_per_processor = cores_per_processor if cores_per_processor % 2 == 0 else cores_per_processor + 1
        total_cores = cores_per_processor * 2
        total_threads = total_cores * 2
    else:
        if threads_per_server > 20:
            processors_to_allocate = 'Dual'
            cores_per_processor = min(math.ceil(threads_per_server / 4), 64)
            cores_per_processor = cores_per_processor if cores_per_processor % 2 == 0 else cores_per_processor + 1
            total_cores = cores_per_processor * 2
            total_threads = total_cores * 2
        else:
            processors_to_allocate = 'Single'
            cores_per_processor = min(math.ceil(threads_per_server / 2), 64)
            cores_per_processor = cores_per_processor if cores_per_processor % 2 == 0 else cores_per_processor + 1
            total_cores = cores_per_processor
            total_threads = total_cores * 2

    servers.append({
        "Processors": processors_to_allocate,
        "Total Cores": total_cores,
        "Total Threads": total_threads,
        "Cores per Processor": cores_per_processor,
        "Threads per Processor": total_threads // (2 if processors_to_allocate == 'Dual' else 1),
        "RAM": ram_per_server
    })

    logging.info(
        f"Allocated Server: Processors={processors_to_allocate}, Total Cores={total_cores}, Total Threads={total_threads}, RAM={ram_per_server} GB")
    print(
        f"Allocated Server: Processors={processors_to_allocate}, Total Cores={total_cores}, Total Threads={total_threads}, RAM={ram_per_server} GB")

    # Update remaining resources
    remaining_threads -= threads_per_server
    remaining_ram -= ram_per_server

    # Ensure no negative values
    remaining_threads = max(remaining_threads, 0)
    remaining_ram = max(remaining_ram, 0)

    return remaining_threads, remaining_ram

def format_server_specs(servers, base_name="Server"):
    """
    Format server specifications for display with dynamic naming.

    Args:
        servers (list): List of server dictionaries containing specs.
        base_name (str): Base name for the servers (e.g., "Test Server", "Management Server").

    Returns:
        str: Formatted server specifications as a string.
    """
    server_specs = ""
    for i, server in enumerate(servers):
        # Dynamically assign server name
        server_name = f"{base_name} {i + 1}"
        server_specs += f"""
        **{server_name}:**
          - Processors: {server['Processors']}
          - Total CPU: {server['Total Cores']} Cores / {server['Total Threads']} Threads
          - Per Processor: {server['Cores per Processor']} Cores / {server['Threads per Processor']} Threads
          - RAM: {server['RAM']} GB
        """
    return server_specs
def calculate_raid5_disks(usable_storage_tb, min_disks=3, max_disks=24):
    """
    Calculate the optimal RAID 5 disk configuration for the given usable storage requirements.

    Args:
        usable_storage_tb (float): The required usable storage in terabytes (TB).
        min_disks (int): The minimum number of disks in the RAID 5 array (default is 3).
        max_disks (int): The maximum number of disks in the RAID 5 array (default is 24).

    Returns:
        (int, float): A tuple containing the total number of disks and the size of each disk in TB.
                      Returns (None, None) if no valid configuration is found.
    """
    available_disk_sizes_tb = sorted([ 1.5,2.4,3,4, 8, 12, 16, 20, 22], reverse=True)

    best_total_disks = None
    best_disk_size = None
    smallest_excess = float('inf')

    for disk_size in available_disk_sizes_tb:
        for total_disks in range(min_disks, max_disks + 1):
            usable_storage_with_disks = (total_disks - 1) * disk_size  # RAID 5 formula

            # Check if the configuration meets the required storage
            if usable_storage_with_disks >= usable_storage_tb:
                excess_storage = usable_storage_with_disks - usable_storage_tb

                # Add a penalty for using more disks to prioritize fewer disks
                penalty = excess_storage + 0.01 * total_disks

                # Update the best configuration if this configuration is better
                if penalty < smallest_excess:
                    smallest_excess = penalty
                    best_total_disks = total_disks
                    best_disk_size = disk_size

    # Return the best configuration
    return best_total_disks, best_disk_size
# Test the optimized function with an input of 18 TB usable storage


def round_to_nearest_divisible_by_two(value):
    return round(value / 2) * 2


def format_number_with_commas(number):
    return f"{number:,}"


def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    logo_image = Image.open(os.path.join(app_dir, "assets", "logo.png"))

    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.image(logo_image)
    with col3:
        st.write("")

    st.title("PaxeraHealth Sizing Calculator")
    customer_name = st.text_input("Customer Name:")

    with st.expander("Project and Location Details"):
        num_machines = st.number_input("Number of Machines (Modalities):", min_value=1, value=1, format="%d")
        num_locations = st.number_input("Number of Locations:", min_value=1, value=1, format="%d")
        breakdown_per_modality = st.radio("Breakdown per Modality?", ["No", "Yes"])
        if breakdown_per_modality == "No":
            num_studies = st.number_input("Number of studies per year:", min_value=0, value=100000, format="%d")

            # Display the number of studies in green with commas
            st.metric(label="Number of Studies", value=f"{num_studies:,} studies", delta=None, delta_color="off")
            modality_cases = {}
        else:
            st.subheader("Modality Breakdown:")
            modality_cases = {
                "CT": st.number_input("CT Cases:", min_value=0, format="%d"),
                "MR": st.number_input("MR Cases:", min_value=0, format="%d"),
                "US": st.number_input("US Cases:", min_value=0, format="%d"),
                "NM": st.number_input("NM Cases:", min_value=0, format="%d"),
                "X-ray": st.number_input("X-ray Cases:", min_value=0, format="%d"),
                "MG": st.number_input("MG Cases:", min_value=0, format="%d"),
                "Cath": st.number_input("Cath Cases:", min_value=0, format="%d"),
            }
            num_studies = sum(modality_cases.values())

        contract_duration = st.number_input("Contract Duration (years):", min_value=1, value=3, format="%d")
        study_size_mb = st.number_input("Study Size (MB):", min_value=0, value=100, format="%d")
        annual_growth_rate = st.number_input("Annual Growth Rate (%):", min_value=0.0, value=10.0, format="%f")

    with st.expander("Project Grade"):
        st.markdown("""
          <style>
          .tooltip {
              position: relative;
              display: inline-block;
              cursor: pointer;
          }

          .tooltip .tooltiptext {
              visibility: hidden;
              width: 220px;
              background-color: #555;
              color: #fff;
              text-align: center;
              border-radius: 6px;
              padding: 5px 0;
              position: absolute;
              z-index: 1;
              bottom: 125%;
              left: 50%;
              margin-left: -110px;
              opacity: 0;
              transition: opacity 0.3s;
          }

          .tooltip:hover .tooltiptext {
              visibility: visible;
              opacity: 1;
          }
          </style>
          """, unsafe_allow_html=True)

        if num_studies <= 50000:
            st.markdown(
                "Project Grade: <span class='tooltip'>ℹ️<span class='tooltiptext'>Select the appropriate project grade based on the volume of studies. Grade 1 is for lower volume (up to 50,000), Grade 2 for medium volume (50,000 to 150,000), and Grade 3 for high volume (more than 150,000).</span></span>",
                unsafe_allow_html=True)
            project_grade = st.selectbox("", [1, 2, 3], index=0)
        elif 50000 < num_studies <= 150000:
            st.markdown(
                "Project Grade: <span class='tooltip'>ℹ️<span class='tooltiptext'>Select the appropriate project grade based on the volume of studies. Grade 1 is for lower volume (up to 50,000), Grade 2 for medium volume (50,000 to 150,000), and Grade 3 for high volume (more than 150,000).</span></span>",
                unsafe_allow_html=True)
            project_grade = st.selectbox("", [2, 3], index=0)
        else:
            st.markdown(
                "Project Grade: <span class='tooltip'>ℹ️<span class='tooltiptext'>Select the appropriate project grade based on the volume of studies. Grade 1 is for lower volume (up to 50,000), Grade 2 for medium volume (50,000 to 150,000), and Grade 3 for high volume (more than 150,000).</span></span>",
                unsafe_allow_html=True)
            project_grade = st.selectbox("", [3], index=0)

    with st.expander("CCU Details"):
        pacs_enabled = st.checkbox("Include PACS")
        if pacs_enabled:
            pacs_ccu = st.number_input("PACS CCU:", min_value=0, value=8, format="%d")
        else:
            pacs_ccu = 0

        ris_enabled = st.checkbox("Include RIS")
        if ris_enabled:
            ris_ccu = st.number_input("RIS CCU:", min_value=0, value=8, format="%d")
        else:
            ris_ccu = 0
        combine_pacs_ris = False
        if pacs_enabled and ris_enabled and pacs_ccu > 0 and ris_ccu > 0:
            combine_pacs_ris = st.checkbox("Combine PACS and RIS VMs")
            if combine_pacs_ris:
                st.warning("Combining PACS and RIS VMs will use shared resources for both systems.")
        else:
            combine_pacs_ris = False  # Default to False if conditions are not met

        ref_phys_enabled = st.checkbox("Include Referring Physician")
        if ref_phys_enabled:
            ref_phys_ccu = st.number_input("Referring Physician CCU:", min_value=0, value=8, format="%d")
            ref_phys_external_access = st.checkbox("External Access for Referring Physician Portal")
        else:
            ref_phys_ccu = 0
            ref_phys_external_access = False
        patient_portal_enabled = st.checkbox("Include Patient Portal")
        if patient_portal_enabled:
            patient_portal_ccu = st.number_input("Patient Portal CCU:", min_value=0, value=8, format="%d")
            patient_portal_external_access = True
        else:
            patient_portal_ccu= 0
            patient_portal_external_access = False

    with st.expander("Business Continuity & Gateway"):
        location_details = []
        mini_pacs_settings = []
        gateway_locations = []
        for i in range(2, num_locations + 1):
            location_type = st.selectbox(
                f"Select interconnection type for Location {i}",
                ["Gateway Uploader", "Business Continuity Mini PACS"]
            )

            if location_type == "Business Continuity Mini PACS":
                mini_pacs_num_studies = st.selectbox(f"Number of studies per year for Location {i}:", [5000, 10000, 15000], index=0, format_func=format_number_with_commas)
                mini_pacs_pacs_ccu = st.selectbox(f"PACS CCU for Location {i}:", [2, 4, 8], index=0)
                mini_pacs_ris_enabled = st.checkbox(f"Enable RIS for Location {i}", value=False)
                if mini_pacs_ris_enabled:
                    mini_pacs_ris_ccu = st.selectbox(f"RIS CCU for Location {i}:", [4, 8], index=0)
                else:
                    mini_pacs_ris_ccu = 0
                mini_pacs_broker_level = st.radio(f"Broker Level for Location {i}:", ["Not Required", "WL", "HL7 Unidirectional", "HL7 Bidirectional"], index=0)
                mini_pacs_high_availability = st.checkbox(f"High Availability HW Design Required for Location {i}", value=False)

                mini_pacs_settings.append({
                    "location": f"Location {i}",
                    "num_studies": mini_pacs_num_studies,
                    "pacs_ccu": mini_pacs_pacs_ccu,
                    "ris_enabled": mini_pacs_ris_enabled,
                    "ris_ccu": mini_pacs_ris_ccu,
                    "broker_level": mini_pacs_broker_level,
                    "high_availability": mini_pacs_high_availability
                })
            else:
                gateway_locations.append(i)

            location_details.append(location_type)

    with st.expander("Broker Details"):
        broker_required = st.checkbox("Broker VM Required (check if explicitly requested)", value=False)
        if broker_required:
            broker_level = st.radio("Broker Level:", ["WL", "HL7 Unidirectional", "HL7 Bidirectional"], index=0)
        else:
            broker_level = "Not Required"
    # Initialize AI module variables
    organ_segmentator = False
    lesion_segmentator_2d = False
    lesion_segmentator_3d = False
    speech_to_text_container = False

    # AI Features Section
    with st.expander("AI Features"):
        # U9.Ai Features
        aidocker_included = st.checkbox(
            "Include U9th Integrated AI modules",
            value=False,
            disabled=(num_studies == 0)
        )

        if aidocker_included:
            st.markdown("#### U9th AI Modules")
            organ_segmentator = st.checkbox("Organ Segmentator")
            lesion_segmentator_2d = st.checkbox("Lesion Segmentator 2D")
            lesion_segmentator_3d = st.checkbox("Lesion Segmentator 3D")
            speech_to_text_container = st.checkbox("Speech-to-text Container")

        # ARKAI
        ark_included = st.checkbox(
            "Include ARKAI",
            value=False,
            disabled=(num_studies == 0)
        )

    # High Availability Selection Section
    with st.expander("High Availability Design Options"):
        # General HA Design
        high_availability = st.checkbox("Enable High Availability on Hardware Level (Host, Server, Storage)",
                                        value=False)
        high_availability_vms = st.checkbox("Enable High Availability on VM Level", value=False)

    # Hardware and Design Features Section
    with st.expander("Hardware and Design Features"):
        training_vm_included = st.checkbox("Include Testing/Training VM", value=False)

        # NAS Backup Options
        use_nas_backup = st.checkbox("Include NAS Storage for Backup", value=False)
        if use_nas_backup:
            # Redundancy factor input
            nas_redundancy_factor = st.number_input(
                "NAS Redundancy Factor (e.g., 1.5 for 50% extra storage):",
                min_value=1.0, value=1.5, step=0.1
            )

            # Duration input for NAS storage backup
            nas_backup_years = st.number_input(
                "Number of Years for NAS Storage Backup:",
                min_value=contract_duration,  # Minimum value is the contract period
                value=contract_duration,  # Default to contract period
                step=1
            )

        # Intermediate/Short-Term Storage Options
        include_fast_tier_storage = st.checkbox(
            "Include High-Performance Tier for Short-Term Image Storage (e.g., SSD or High-Speed SATA)"
        )
        fast_tier_duration = None  # Initialize variable
        if include_fast_tier_storage:
            fast_tier_duration = st.radio(
                "Select Duration for High-Performance Tier Storage:",
                options=["6 Months", "1 Year"],
                index=0
            )
        else:
            fast_tier_duration = "Not Selected"  # Set default value if not selected

        # Workstation Specifications
        include_workstation_specs = st.checkbox("Include Workstation Specifications")


    calculate = st.button("Calculate")

    # Validation check
    if calculate:
        if (
                num_studies == 0 and pacs_ccu == 0 and ris_ccu == 0 and ref_phys_ccu == 0
                and not pacs_enabled and not ris_enabled and not ref_phys_enabled
                and not broker_required and not aidocker_included and not ark_included
                and not high_availability and not training_vm_included
        ):
            st.error("Please enter values for the number of studies, CCUs, or select additional features.")
        else:
            logging.info("Starting VM Requirements Calculation")
            start_calc_time = time.time()
            results, sql_license, first_year_storage_raid5, total_image_storage_raid5, total_vcpu, total_ram, total_storage = calculate_vm_requirements(
                num_studies=num_studies,
                pacs_ccu=pacs_ccu,
                ris_ccu=ris_ccu,
                ref_phys_ccu=ref_phys_ccu,
                project_grade=project_grade,
                broker_required=broker_required,
                broker_level=broker_level,
                num_machines=num_machines,
                contract_duration=contract_duration,
                study_size_mb=study_size_mb,
                annual_growth_rate=annual_growth_rate,
                breakdown_per_modality=breakdown_per_modality,
                aidocker_included=aidocker_included,
                ark_included=ark_included,
                u9_ai_features=[  # Updated: Dynamic U9.AI feature list
                    "Organ Segmentator" if organ_segmentator else None,
                    "Lesion Segmentator 2D" if lesion_segmentator_2d else None,
                    "Lesion Segmentator 3D" if lesion_segmentator_3d else None,
                    "Speech-to-text Container" if speech_to_text_container else None
                ],
                ref_phys_external_access=ref_phys_external_access,
                patient_portal_ccu=patient_portal_ccu,
                patient_portal_external_access=patient_portal_external_access,
                training_vm_included=training_vm_included,
                high_availability=high_availability,
                combine_pacs_ris=combine_pacs_ris,
                **modality_cases
            )

            logging.info(f"VM Requirements Calculation completed in {time.time() - start_calc_time:.2f} seconds")

            if not results.empty:
                # 🛡️ Set default Operating System and Other Software
                results["Operating System"] = "Windows Server 2019 or Higher"
                results["Other Software"] = ""

                for index in results.index:
                    if "TestVM" in index:
                        results.at[index, "Other Software"] = sql_license
                    if "DBServer" in index:
                        results.at[index, "Other Software"] = sql_license

                    # Update for specific AI Segmentation Dockers
                    if any(name in index for name in
                           ["OrganSegmentator", "LesionSegmentator2D", "LesionSegmentator3D", "SpeechToText"]):
                        results.at[index, "Operating System"] = "Ubuntu 20.4"
                        results.at[
                            index, "Other Software"] = "Nvidia Driver version 450.80.02 or higher\nNvidia driver to support CUDA version 11.4 or higher"

                    # Update for ARK Lab
                    if "AIARKLAB" in index:
                        results.at[index, "Operating System"] = "Ubuntu 20.4"
                        results.at[
                            index, "Other Software"] = "Nvidia Driver version 450.80.02 or higher\nNvidia driver to support CUDA version 11.4 or higher"

                    # Special case for ARK Manager
                    if "AIARKManager" in index:
                        results.at[index, "Operating System"] = "Windows Server 2019 or Higher"
                        results.at[index, "Other Software"] = ""

            display_results = results.drop(["RAID 1 (SSD)", "RAID 5 (HDD)"])

            last_index = display_results.tail(1).index
            display_results.loc[last_index, "Operating System"] = ""
            display_results.loc[last_index, "Other Software"] = ""

            st.subheader("VM Recommendations:")
            st.dataframe(display_results.style.apply(
                lambda x: ['background-color: yellow' if 'Test Environment VM' in x.name else '' for i in x],
                axis=1).format(precision=2))

            # 🛠️ Prepare Additional VMs
            additional_vms = []

            # Add Test Environment VM
            if training_vm_included:
                test_vm_specs = {
                    "VM Type": "Test Environment VM (Ultima, PACS, Broker)",
                    "vCores": 0,
                    "RAM (GB)": 0,
                    "Storage (GB)": 150,
                    "RAID 5 Storage (TB)": 0
                }
                if sql_license == "SQL Express":
                    test_vm_specs["vCores"] = 8
                    test_vm_specs["RAM (GB)"] = 16
                    test_vm_specs["RAID 5 Storage (TB)"] = 1.0
                elif sql_license == "SQL Standard":
                    test_vm_specs["vCores"] = 10
                    test_vm_specs["RAM (GB)"] = 32
                    test_vm_specs["RAID 5 Storage (TB)"] = 5.0
                elif sql_license == "SQL Enterprise":
                    test_vm_specs["vCores"] = 12
                    test_vm_specs["RAM (GB)"] = 64
                    test_vm_specs["RAID 5 Storage (TB)"] = 10.0

                additional_vms.append(test_vm_specs)

            # Add Management VM for High Availability
            if high_availability:
                management_vm_specs = {
                    "VM Type": "Management VM (Backup, Antivirus, vCenter)",
                    "vCores": 8,
                    "RAM (GB)": 32 if sql_license in ["SQL Express", "SQL Standard"] else 64,
                    "Storage (GB)": 150,
                    "RAID 5 Storage (TB)": 0
                }
                additional_vms.append(management_vm_specs)

            # Prepare Additional VMs Table
            if additional_vms:
                additional_vms_table = pd.DataFrame(additional_vms)

                if "VM Type" in additional_vms_table.columns:
                    if "Total" in additional_vms_table["VM Type"].values:
                        total_row_index = additional_vms_table[additional_vms_table["VM Type"] == "Total"].index[0]
                        additional_vms_table.at[total_row_index, "Operating System"] = ""
                        additional_vms_table.at[total_row_index, "Other Software"] = ""
            else:
                additional_vms_table = None


    
            additional_vm_notes = []
            # Display Additional VMs Table and Notes
            if additional_vms:
                st.subheader("Additional VMs:")
                additional_vms_table = pd.DataFrame(additional_vms)
                st.table(additional_vms_table.style.format(precision=2))

                # Display Notes for Additional VMs
                st.subheader("Additional VM Notes:")

                # Test Environment VM Note
                if any(vm["VM Type"] == "Test Environment VM (Ultima, PACS, Broker)" for vm in additional_vms):
                    additional_vm_notes.append("""
                - **Test Environment VM:**  
                  - It is recommended to host this VM on a **separate hardware pool** dedicated for testing and training purposes.  
                    This ensures that production workloads are not impacted by testing or training activities.
                    """)

                # Management VM Note
                if any(vm["VM Type"] == "Management VM (Backup, Antivirus, vCenter)" for vm in additional_vms):
                    additional_vm_notes.append("""
                - **Management VM:**  
                  - Hosted on a **dedicated hardware pool** to ensure exclusive resources are allocated for management tasks, minimizing any impact on production workloads.  
                  - Alternatively, deployed as a **Distributed Management Node Allocation**, where management tasks are shared across production servers with **High Availability (HA)** configured.  
                    This ensures seamless continuity in the event of a server failure, as the remaining server(s) automatically take over the management operations.
                    """)

                # Display notes only if they exist
                if additional_vm_notes:
                    st.markdown("\n".join(additional_vm_notes))

            # Display General Notes
            st.subheader("General Notes:")
            general_notes = """
            - **Virtual Core (vCore) Definition:**
                - Each **vCore** corresponds to a single thread within the physical processor.
                - Physical processors typically support **two threads per core**, meaning the number of vCores is twice the number of physical cores.
                - This 1:2 ratio between physical cores and vCores ensures optimal utilization of processor resources, enabling better performance and efficient workload distribution in virtualized environments.
            """
            st.markdown(general_notes)

            # Total Requirements for Additional VMs

            # Calculate VM Requirements (Grade 1 and Grade 3)
            results_grade1, _, first_year_storage_raid5_grade1, total_image_storage_raid5_grade1, total_vcpu_grade1, total_ram_grade1, total_storage_grade1 = calculate_vm_requirements(
                num_studies=num_studies,
                pacs_ccu=pacs_ccu,
                ris_ccu=ris_ccu,
                ref_phys_ccu=ref_phys_ccu,
                project_grade=1,
                broker_required=broker_required,
                broker_level=broker_level,
                num_machines=num_machines,
                contract_duration=contract_duration,
                study_size_mb=study_size_mb,
                annual_growth_rate=annual_growth_rate,
                breakdown_per_modality=breakdown_per_modality,
                aidocker_included=aidocker_included,
                ark_included=ark_included,
                u9_ai_features=[
                    "Organ Segmentator" if organ_segmentator else None,
                    "Lesion Segmentator 2D" if lesion_segmentator_2d else None,
                    "Lesion Segmentator 3D" if lesion_segmentator_3d else None,
                    "Speech-to-text Container" if speech_to_text_container else None
                ],
                ref_phys_external_access=ref_phys_external_access,
                patient_portal_ccu=patient_portal_ccu,
                patient_portal_external_access=patient_portal_external_access,
                training_vm_included=training_vm_included,
                high_availability=high_availability,
                combine_pacs_ris=combine_pacs_ris,
                **modality_cases
            )

            results_grade3, _, first_year_storage_raid5_grade3, total_image_storage_raid5_grade3, total_vcpu_grade3, total_ram_grade3, total_storage_grade3 = calculate_vm_requirements(
                num_studies=num_studies,
                pacs_ccu=pacs_ccu,
                ris_ccu=ris_ccu,
                ref_phys_ccu=ref_phys_ccu,
                project_grade=3,
                broker_required=broker_required,
                broker_level=broker_level,
                num_machines=num_machines,
                contract_duration=contract_duration,
                study_size_mb=study_size_mb,
                annual_growth_rate=annual_growth_rate,
                breakdown_per_modality=breakdown_per_modality,
                aidocker_included=aidocker_included,
                ark_included=ark_included,
                u9_ai_features=[
                    "Organ Segmentator" if organ_segmentator else None,
                    "Lesion Segmentator 2D" if lesion_segmentator_2d else None,
                    "Lesion Segmentator 3D" if lesion_segmentator_3d else None,
                    "Speech-to-text Container" if speech_to_text_container else None
                ],
                ref_phys_external_access=ref_phys_external_access,
                patient_portal_ccu=patient_portal_ccu,
                patient_portal_external_access=patient_portal_external_access,
                training_vm_included=training_vm_included,
                high_availability=high_availability,
                combine_pacs_ris=combine_pacs_ris,
                **modality_cases
            )

            # Define Storage Values for Display
            raid_1_storage_tb_g3 = results_grade3.loc["RAID 1 (SSD)", "Storage (GB)"] / 1024
            raid_5_storage_tb = total_image_storage_raid5_grade3 / 1024  # Total HDD storage for full duration
            raid_1_storage = results.loc["RAID 1 (SSD)", "Storage (GB)"]  # Tier 1 storage
            raid_1_storage_tb = raid_1_storage / 1024  # Convert to TB
            st.subheader("Storage Requirements:")

            # Initialize storage values
            years = range(1, contract_duration + 1)
            tier_1_storage = raid_1_storage_tb  # Constant for Tier 1
            tier_2_storage = None  # Initialize Tier 2 storage as None
            tier_3_storage = []  # Initialize Tier 3 storage list for accumulation

            # Calculate the storage for the first year (initial storage)
            initial_year_storage = (num_studies * study_size_mb) / 1024 / 1024  # Convert to TB

            if include_fast_tier_storage:
                # Determine storage multiplier based on selection (6 months = 0.5, 1 year = 1.0)
                fast_storage_multiplier = 0.5 if fast_tier_duration == "6 Months" else 1.0

                tier_2_storage = []  # Initialize Tier 2 storage as an empty list
                for year in years:
                    year_storage = initial_year_storage * (1 + annual_growth_rate / 100) ** (year - 1)
                    tier_2_storage.append(round(year_storage * fast_storage_multiplier, 2))
                tier_2_label = f"Tier 2: Fast Image Storage (SSD RAID 5) ({fast_tier_duration})"

                # Calculate cumulative Tier 3 storage for each year
                for year in years:
                    current_year_storage = initial_year_storage * (1 + annual_growth_rate / 100) ** (year - 1)
                    if year == 1:
                        tier_3_storage.append(round(current_year_storage, 2))  # First year storage
                    else:
                        cumulative_storage = tier_3_storage[-1] + current_year_storage
                        tier_3_storage.append(round(cumulative_storage, 2))
            else:
                # No intermediate Tier 2; promote Tier 3 to Tier 2
                tier_2_label = None  # No Tier 2 storage
                # Initialize and calculate Tier 3 storage
                for year in years:
                    current_year_storage = initial_year_storage * (1 + annual_growth_rate / 100) ** (year - 1)
                    if year == 1:
                        tier_3_storage.append(round(current_year_storage, 2))  # First year storage
                    else:
                        cumulative_storage = tier_3_storage[-1] + current_year_storage
                        tier_3_storage.append(round(cumulative_storage, 2))

            # Use the final Tier 3 storage value in the comparison tables
            final_tier_3_storage = tier_3_storage[-1]

            # Prepare storage data dictionary
            storage_data = {
                "Year": [f"Year {year}" for year in years],
                "Tier 1: OS & DB (SSD RAID 1)": [round(tier_1_storage, 2)] * len(years),
            }

            if tier_2_storage:  # Add Tier 2 only if it's present
                storage_data[tier_2_label] = tier_2_storage

            # Add Tier 3 to the storage table
            storage_data["Tier 3: Long-Term Storage (HDD RAID 5)"] = tier_3_storage

            # Create and display the Storage Table
            storage_table = pd.DataFrame(storage_data).reset_index(drop=True)
            st.table(storage_table.style.format(precision=2))

            # Resource Comparison Table
            # Minimum vs Recommended Resources for Grade 1
            if project_grade == 1:  # Minimum vs Recommended Resources for Grade 1
                comparison_data = {
                    "Specification": ["Total vCores", "Total RAM (GB)", "RAID 1 (SSD) (TB)",
                                      "RAID 5 (HDD) Full Duration (TB)"],
                    "Minimum Specs": [
                        round(total_vcpu_grade1, 1),
                        round(total_ram_grade1, 1),
                        round(raid_1_storage_tb, 1),
                        round(final_tier_3_storage, 1),  # Use final Tier 3 storage
                    ],
                    "Recommended Specs": [
                        round(total_vcpu_grade3, 1),
                        round(total_ram_grade3, 1),
                        round(raid_1_storage_tb_g3, 1),
                        round(final_tier_3_storage, 1),  # Use final Tier 3 storage
                    ],
                }
                df_comparison = pd.DataFrame(comparison_data)
                st.subheader("Minimum vs. Recommended Resources:")
                st.table(
                    df_comparison.style.set_table_styles(
                        [
                            {"selector": "thead th", "props": [("font-size", "16px"), ("font-weight", "bold")]},
                            {"selector": "tbody td", "props": [("font-size", "14px"), ("text-align", "center")]},
                            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f9f9f9")]},
                            {"selector": "tbody tr:hover", "props": [("background-color", "#f1f1f1")]},
                        ]
                    ).format(precision=1)
                )

            # Recommended Resources Table for Grade 2 or 3
            elif project_grade in [2, 3]:  # Only Recommended Resources for Grade 2 or 3
                recommended_data = {
                    "Specification": ["Total vCores", "Total RAM (GB)", "RAID 1 (SSD) (TB)",
                                      "RAID 5 (HDD) Full Duration (TB)"],
                    "Recommended Specs": [
                        round(total_vcpu_grade3, 1),
                        round(total_ram_grade3, 1),
                        round(raid_1_storage_tb_g3, 1),
                        round(final_tier_3_storage, 1),  # Use final Tier 3 storage
                    ],
                }
                df_comparison = pd.DataFrame(recommended_data)
                st.subheader("Recommended Resources:")
                st.table(
                    df_comparison.style.set_table_styles(
                        [
                            {"selector": "thead th", "props": [("font-size", "16px"), ("font-weight", "bold")]},
                            {"selector": "tbody td", "props": [("font-size", "14px"), ("text-align", "center")]},
                            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f9f9f9")]},
                            {"selector": "tbody tr:hover", "props": [("background-color", "#f1f1f1")]},
                        ]
                    ).format(precision=1)
                )

            # Calculate Additional VM Requirements
            if additional_vms:
                # Calculate totals for additional VMs
                total_additional_vcpu = sum(vm.get("vCores", 0) for vm in additional_vms)
                total_additional_ram = sum(vm.get("RAM (GB)", 0) for vm in additional_vms)
                total_additional_ssd = sum(vm.get("Storage (GB)", 0) for vm in additional_vms) / 1024  # Convert to TB
                total_additional_hdd = sum(vm.get("RAID 5 Storage (TB)", 0) for vm in additional_vms)  # HDD in TB

                # Define the additional requirements table
                additional_requirements_data = {
                    "Specification": ["Total vCores", "Total RAM (GB)", "Total SSD Storage (TB)",
                                      "Total HDD Storage (TB)"],
                    "Requirements": [
                        total_additional_vcpu,
                        total_additional_ram,
                        round(total_additional_ssd, 1),  # Round to 1 decimal point for consistency
                        round(total_additional_hdd, 1),
                    ],
                }
                additional_requirements_table = pd.DataFrame(additional_requirements_data)

                # Display in Streamlit (if applicable)
                st.subheader("Requirements for Additional VMs:")
                st.table(additional_requirements_table.style.format(precision=1))
            else:
                additional_requirements_table = None  # Ensure it is set to None if no additional VMs

            logging.info("Starting Physical System Allocation")
            total_vcpu = results.loc["Total", "vCores"]
            total_ram = results.loc["Total", "RAM (GB)"]

            # Include additional VMs in the physical hardware calculations
            total_vcpu += sum(vm.get("vCores", 0) for vm in additional_vms)
            total_ram += sum(vm.get("RAM (GB)", 0) for vm in additional_vms)

            max_threads_per_server = 128
            max_ram_per_server = 512

            # Section: Physical System Design
            st.subheader("Physical System Design")

            # Server Design for Production Servers
            servers = []
            if high_availability:
                logging.info("Calculating high availability resources")
                print("Length of servers:", len(servers))

                # Check the number of servers in normal design
                if total_vcpu > max_threads_per_server or total_ram > max_ram_per_server:
                    logging.info("Total workload exceeds max server specs. Using dynamic HA allocation logic.")

                    # Scale total resources for Active-Active HA
                    total_vcpu_ha = int(total_vcpu * 1.5)
                    total_ram_ha = int(total_ram * 1.5)

                    remaining_threads, remaining_ram = total_vcpu_ha, total_ram_ha
                    while remaining_threads > 0 or remaining_ram > 0:
                        if remaining_ram == 0:
                            logging.warning("No remaining RAM to allocate. Exiting loop.")
                            break
                        remaining_threads, remaining_ram = add_server(
                            servers,
                            remaining_threads,
                            remaining_ram,
                            max_threads_per_server,
                            max_ram_per_server
                        )
                else:
                    logging.info("Total workload is within max server specs. Enforcing two servers for HA redundancy.")

                    # Allocate two servers with 75% of total resources
                    threads_per_server = int(total_vcpu * 0.75)
                    ram_per_server = int(total_ram * 0.75)

                    # Ensure threads are divisible by 2
                    if threads_per_server % 2 != 0:
                        threads_per_server -= 1

                    # Clear any existing servers
                    servers.clear()

                    # Add two servers with adjusted resources
                    for _ in range(2):
                        remaining_threads, remaining_ram = add_server(
                            servers,
                            threads_per_server,
                            ram_per_server,
                            max_threads_per_server,
                            max_ram_per_server
                        )

                    # Fallback to ensure exactly two servers exist
                    while len(servers) < 2:
                        servers.append(
                            servers[0].copy())  # Duplicate the first server if fewer than 2 servers are added

                # Format and display servers
                server_specs = format_server_specs(servers, base_name="Production Server")
                st.markdown("### High Availability Server Design")
                st.markdown(server_specs)

                logging.info("High availability server allocation complete")
            else:
                logging.info("Calculating standard server resources")
                num_servers = (total_vcpu + max_threads_per_server - 1) // max_threads_per_server
                avg_ram_per_server = (total_ram + num_servers - 1) // num_servers

                remaining_threads, remaining_ram = total_vcpu, total_ram
                while remaining_threads > 0 or remaining_ram > 0:
                    if remaining_ram == 0:
                        logging.warning(f"No remaining RAM to allocate. Exiting loop.")
                        break
                    remaining_threads, remaining_ram = add_server(
                        servers, remaining_threads, remaining_ram, max_threads_per_server, avg_ram_per_server
                    )

                server_specs = format_server_specs(servers, base_name="Production Server")
                st.markdown("### Standard Server Design")
                st.markdown(server_specs)

                logging.info("Standard server allocation complete")
            additional_servers = []
            # Additional VMs on Separate Hardware Pool
            if additional_vms:
                st.markdown("### Additional VM Hardware Design")

                additional_total_vcpu = sum(vm.get("vCores", 0) for vm in additional_vms)
                additional_total_ram = sum(vm.get("RAM (GB)", 0) for vm in additional_vms)
                additional_total_ssd = sum(
                    vm.get("Storage (GB)", 0) for vm in additional_vms) / 1024  # Convert SSD to TB
                additional_total_hdd = sum(vm.get("RAID 5 Storage (TB)", 0) for vm in additional_vms)  # HDD in TB
                # Determine base name for additional servers
                if len(additional_vms) == 1:
                    additional_vm_type = additional_vms[0]["VM Type"]
                    if "Test" in additional_vm_type:
                        base_name = "Test Server"
                    elif "Management" in additional_vm_type:
                        base_name = "Management Server"
                elif any("Management" in vm["VM Type"] for vm in additional_vms):
                    base_name = "Test and Management Server"
                else:
                    base_name = "Additional Server"

                # Calculate physical servers for additional VMs
                remaining_threads, remaining_ram = additional_total_vcpu, additional_total_ram
                while remaining_threads > 0 or remaining_ram > 0:
                    if remaining_ram == 0:
                        logging.warning(f"No remaining RAM to allocate for additional VMs. Exiting loop.")
                        break
                    remaining_threads, remaining_ram = add_server(
                        additional_servers, remaining_threads, remaining_ram, max_threads_per_server, max_ram_per_server
                    )

                additional_server_specs = format_server_specs(additional_servers, base_name=base_name)

                # Display Additional VM Hardware Design
                st.markdown(additional_server_specs)

                # Display storage details under hardware design
                st.markdown(f"""
                **Storage Configuration for Additional Pool:**
                - **SSD Storage:** {round(additional_total_ssd * 1024, 2)} GB
                - **HDD Storage (RAID 5):** {round(additional_total_hdd, 2)} TB
                """)

            # Storage Details
            st.markdown("### Storage Design")

            # Tier 1: OS & DB (SSD RAID 1)
            raid_1_storage = results.loc["RAID 1 (SSD)", "Storage (GB)"]  # Tier 1 storage
            raid_1_storage_tb = raid_1_storage / 1024  # Convert to TB

            # Tier 2: Fast Image Storage (SSD RAID 5) or Long-Term Storage
            if include_fast_tier_storage:
                intermediate_tier_multiplier = 0.5 if fast_tier_duration == "6 Months" else 1.0
                intermediate_tier_storage_tb = math.ceil(
                    first_year_storage_raid5 * intermediate_tier_multiplier / 1024
                )  # Convert to TB
                tier_2_disks, tier_2_disk_size = calculate_raid5_disks(intermediate_tier_storage_tb)
                tier_2_label = "Tier 2: Fast Image Storage (SSD RAID 5)"
            else:
                tier_2_disks, tier_2_disk_size = calculate_raid5_disks(final_tier_3_storage)
                tier_2_label = "Tier 2: Long-Term Storage (HDD RAID 5)"
                tier_3_disks, tier_3_disk_size = 0, 0  # No separate Tier 3

            # Tier 3: Long-Term Storage (HDD RAID 5) if Tier 2 is SSD RAID 5
            if include_fast_tier_storage:
                tier_3_disks, tier_3_disk_size = calculate_raid5_disks(final_tier_3_storage)
            else:
                tier_3_disks, tier_3_disk_size = 0, 0

            # Determine storage type
            if len(servers) == 1 and not high_availability:
                storage_type = "Built-in Storage"
            else:
                storage_type = "Shared DAS/SAN Storage"

            # Generate Storage Details
            storage_details = f"""
            #### Storage Details ({storage_type})

            **Tier 1: OS & DB (SSD RAID 1):**
            - SSD Drives: 2x {raid_1_storage_tb:.2f} TB
            """

            # Add Tier 2 Storage
            if tier_2_disks and tier_2_disk_size:
                storage_details += f"""
            **{tier_2_label}:**
            - Drives: {tier_2_disks}x {tier_2_disk_size:.2f} TB
            """

            # Add Tier 3 Storage (if applicable)
            if tier_3_disks and tier_3_disk_size:
                storage_details += f"""
            **Tier 3: Long-Term Storage (HDD RAID 5):**
            - HDD Drives: {tier_3_disks}x {tier_3_disk_size:.2f} TB
            """

            # Handle cases where no details are available
            if not tier_2_disks and not tier_3_disks:
                storage_details += """
            **Details not available for some tiers.**
            """

            # Display Storage Details
            st.markdown(storage_details)

            # Backup Storage (NAS)
            if use_nas_backup:
                # Calculate NAS storage based on redundancy factor and backup years
                nas_storage_gb = round(
                    (total_image_storage_raid5 * nas_redundancy_factor * nas_backup_years) / contract_duration)
                nas_storage_tb = nas_storage_gb / 1024  # Convert to TB
                st.markdown("#### Backup Storage (NAS):")
                # Display the calculated NAS storage

                nas_backup_string = f"""
                - **NAS Storage:** {nas_storage_tb:.2f} TB
                - **Redundancy Factor Applied:** {nas_redundancy_factor:.1f}
                - **Backup Duration:** {nas_backup_years} years
                - **Network Ports:** Minimum 2 (1 Gigabit; 10 Gigabit recommended)
                - **Memory:** 8 GB (16 GB recommended)
                - **Power Supply:** Dual (for redundancy)
                - **RAID Configuration:** RAID 5 + hotspare for data protection
                """
                st.markdown(nas_backup_string)
            else:
                nas_backup_string = None

            non_total_display_results = display_results[display_results.index != "Total"]
            windows_count = len(non_total_display_results) - len(
                non_total_display_results[non_total_display_results["Operating System"] == "Ubuntu 20.4"])
            # Initialize GPU specs
            gpu_specs = ""

            if aidocker_included or ark_included:
                st.subheader("GPU Requirements:")
                total_gpu_memory = 0
                gpu_modules = []

                # Check and add U9.Ai modules
                if organ_segmentator:
                    gpu_modules.append("Organ Segmentator")
                    total_gpu_memory += 16
                if lesion_segmentator_2d:
                    gpu_modules.append("Lesion Segmentator 2D")
                    total_gpu_memory += 4
                if lesion_segmentator_3d:
                    gpu_modules.append("Lesion Segmentator 3D")
                    total_gpu_memory += 12
                if speech_to_text_container:
                    gpu_modules.append("Speech-to-Text Container")
                    total_gpu_memory += 4

                # Generate GPU specs for U9.Ai
                if aidocker_included:
                    gpu_specs = f"""
                    - **Total GPU Memory Required**: {total_gpu_memory} GB
                    - **NVIDIA GPUs**: Recommended
                    - **Modules**: {', '.join(gpu_modules)}
                    - **Driver Requirements**: NVIDIA Driver version 450.80.02 or higher, supporting CUDA version 11.4 or higher.
                    """

                # Handle ARKAI separately
                if ark_included:
                    if aidocker_included:
                        gpu_specs += "\n\n"
                    gpu_specs += """
                    - **ARKAI Requirements**:
                      - Dedicated GPU required for ARKAI workloads.
                      - Recommended GPU Memory for ARKAI: 16 GB.
                      - **Driver Requirements**: NVIDIA Driver version 450.80.02 or higher, supporting CUDA version 11.4 or higher.
                    """


                st.markdown(gpu_specs)
            # Filter Ubuntu Licenses
            ubuntu_vms_count = len(
                non_total_display_results[non_total_display_results["Operating System"] == "Ubuntu 20.4"])

            # Update the Third Party Licenses DataFrame
            third_party_licenses = pd.DataFrame({
                "Item Description": [
                    sql_license,
                    "MS Windows Server Standard 2019 or higher",
                    "Antivirus Server License",
                    "Virtualization Platform License(VMware or HyperV) (Supports High Availability)" if high_availability else "Virtualization Platform License",
                    "Backup Software (Compatible with VMs)",
                    "Ubuntu 20.4 Server License" if ubuntu_vms_count > 0 else None
                ],
                "Qty": [
                    int(2 if training_vm_included else 1),  # SQL licenses
                    int(len(non_total_display_results)-ubuntu_vms_count),  # One Windows license per VM
                    int(len(non_total_display_results)-ubuntu_vms_count),  # One Antivirus license per VM
                    1,  # Virtualization license
                    1,  # Backup software license
                    int(ubuntu_vms_count) if ubuntu_vms_count > 0 else None  # Ubuntu licenses
                ]
            })

            # Remove rows with None values
            third_party_licenses = third_party_licenses.dropna(subset=["Item Description", "Qty"])
            third_party_licenses["Qty"] = third_party_licenses["Qty"].astype(int)

            # Display the Third Party Licenses Table
            st.subheader("Third Party Licenses")
            st.dataframe(third_party_licenses.style.set_properties(subset=["Item Description"], **{'width': '300px'}))

            # Add Notes Section
            st.markdown("### Licensing Notes")
            notes = []

            # Windows Licensing Note
            notes.append(
                "- **Windows Server Licensing:** Each VM requires an active Windows Server license. For environments with a Data Center license, these licenses may be unlimited. Verify the licensing model in use.")

            # Virtualization Licensing Note (HA-Specific)
            if high_availability:
                notes.append(
                    "- **Virtualization Platform Licensing:** Ensure the selected platform (e.g., VMware or Hyper-V) supports High Availability features (e.g., vMotion, Live Migration).")

            # Ubuntu Licensing Note (Only if Ubuntu VMs Exist)
            if ubuntu_vms_count > 0:
                notes.append(
                    "- **Ubuntu Licensing:** Ensure sufficient Ubuntu Server licenses are available for VMs running on Linux.")

            # Antivirus Licensing Note
            notes.append("- **Antivirus Licensing:** One license per VM is required unless otherwise specified.")

            # Backup Software Note
            notes.append("- **Backup Software:** Ensure compatibility with selected VMs and storage solutions.")

            # Render Notes Dynamically
            for note in notes:
                st.markdown(note)

            st.subheader("Sizing Notes:")
            st.markdown("""
            - Provided VM sizing of the Virtual servers will be scaled up horizontally or vertically based on the expected volume of data and number of CCUs.
            - SSD Datastore recommended for all OS Virtual disks of all VMs.
            - SSD recommended for the data drive of the Database Server.
            - It's recommended to have SSD storage for the short Term Storage (STS) that keep last 6 months of data for higher performance in data access.
            """)

            st.subheader("Technical Requirements:")
            st.markdown("""
            - Provide remote access to the VMs (all VMs) for SW installation and configurations.
            - In case not using dongles, a connection from the VMs to (https://paxaeraultima.net:1435) for PaxeraHealth licensing except the database VM.
            - **Licensed Operating Systems & Third-Party Components:** Operating systems and all supporting software must be properly licensed and regularly updated to close security gaps and maintain system reliability.
            """)

            st.subheader("Network Requirements:")
            network_requirements = """
            - LAN bandwidth to be 1GB dedicated.
            - Paxera PACS VMs, Paxera Ultima Viewer VMs, Paxera Broker Integration VMs must be in the same vLAN.
            - 1 Gb/s dedicated bandwidth across the vLAN with the maximum number of two network hops.
            - Latency less than 1ms.
            - Site-to-Site VPN with any other connected branch.
            """

            # Add DMZ recommendation if Referring Physician External Access is enabled
            if ref_phys_external_access:
                network_requirements += """
            - Referring Physician Portal VMs must be placed in a **DMZ (Demilitarized Zone) network** to isolate external access from internal hospital systems and enhance security.
            """
            if patient_portal_external_access:
                network_requirements += """
            - Patient Portal VMs must be placed in a **DMZ (Demilitarized Zone) network** to isolate external access from internal hospital systems and enhance security.
            """

            st.markdown(network_requirements)

            st.subheader("Minimum Requirements & Recommendations:")
            st.markdown("""
            1. **Antivirus on All Backend Servers (Required):**
               Every server hosting our software components must have robust, up-to-date antivirus protection.
            2. **Firewall for External Publishing (Required):**
               Any system accessible from the internet—such as patient portals, referring physician portals, or tele-radiology services—must be protected by a properly configured firewall.
            3. **Backup Storage for Secondary Copies (Strongly Recommended):**
               Implementing a reliable backup strategy ensures a secondary copy of critical patient data and system files, enhancing business continuity and recovery capabilities.
            4. **Additional Security Measures (Recommended):**
               - Regular Security Updates & Patches: Keep all systems current with the latest patches.
               - Multi-Factor Authentication (MFA): Enforce MFA for all administrative and remote access accounts.
               - Network Segmentation & Isolation: Limit potential breaches by isolating critical systems.
            5. **Non-Compliance & Customer Responsibility:**
               Failure to adhere to these guidelines places customers at heightened risk for cyberattacks, including ransomware, data breaches, and service interruptions. If recommended security measures are not implemented, PaxeraHealth cannot be held responsible for any resulting damages or data loss. In such situations, any re-installation, re-implementation, or post-incident remediation will be billed as a separate professional service.
            6. **Transparency & Long-Term Value:**
               By openly communicating these requirements and their implications during the pre-sales process, we help customers make informed decisions that safeguard their systems and data. This level of transparency fosters trust, enhances the long-term value of our solutions, and ensures smoother implementations.
            """)

            mini_pacs_results = []
            mini_pacs_storage = []
            mini_pacs_servers = []
            for i, location in enumerate(mini_pacs_settings):
                if location["high_availability"]:
                    location["pacs_ccu"] *= 2
                    location["ris_ccu"] *= 2

                mini_pacs_result, _, mini_pacs_first_year_storage, mini_pacs_total_storage, mini_pacs_vcpu, mini_pacs_ram, _ = calculate_vm_requirements(
                    location["num_studies"], location["pacs_ccu"], location["ris_ccu"], 0, 1, broker_required, location["broker_level"], num_machines,
                    contract_duration, study_size_mb, annual_growth_rate, aidocker_included=False,
                    ark_included=False, ref_phys_external_access=False,
                    training_vm_included=False, high_availability=location["high_availability"], **modality_cases
                )
                mini_pacs_results.append(mini_pacs_result)
                mini_pacs_storage.append({
                    "first_year": mini_pacs_first_year_storage,
                    "total_storage": mini_pacs_total_storage
                })
                mini_pacs_servers.append((mini_pacs_vcpu, mini_pacs_ram))

                st.write(f"**Location {i + 2} (Mini PACS)**")

                mini_pacs_display_results = mini_pacs_result.drop(["RAID 1 (SSD)", "RAID 5 (HDD)"])
                last_index = mini_pacs_display_results.tail(1).index
                mini_pacs_display_results.loc[last_index, "Operating System"] = ""
                mini_pacs_display_results.loc[last_index, "Other Software"] = ""

                st.subheader(f"VM Recommendations for {location['location']}:")
                st.dataframe(mini_pacs_display_results.style.apply(
                    lambda x: ['background-color: yellow' if 'Test Environment VM' in x.name else '' for i in x],
                    axis=1).format(precision=2))

                mini_pacs_raid_1_storage_tb = mini_pacs_result.loc["RAID 1 (SSD)", "Storage (GB)"] / 1024
                mini_pacs_raid_5_storage_tb = mini_pacs_result.loc["RAID 5 (HDD)", "Storage (GB)"] / 1024

                st.subheader(f"Storage Requirements for {location['location']}:")
                st.markdown(f"**RAID 1 Storage:** {mini_pacs_raid_1_storage_tb:.2f} TB")
                st.markdown(f"**RAID 5 Storage (First Year):** {mini_pacs_storage[i]['first_year'] / 1024:.2f} TB")
                st.markdown(f"**RAID 5 Storage (Full Contract Duration):** {mini_pacs_storage[i]['total_storage'] / 1024:.2f} TB")

                mini_pacs_servers = []
                remaining_threads, remaining_ram = mini_pacs_vcpu, mini_pacs_ram
                while remaining_threads > 0 or remaining_ram > 0:
                    if remaining_ram == 0:
                        logging.warning("No remaining RAM to allocate. Exiting loop.")
                        break
                    remaining_threads, remaining_ram = add_server(
                        mini_pacs_servers, remaining_threads, remaining_ram, max_threads_per_server, max_ram_per_server
                    )

                mini_pacs_server_specs = format_server_specs(mini_pacs_servers)
                st.subheader(f"Server Design for {location['location']}:")
                st.markdown(mini_pacs_server_specs)

            if gateway_locations:
                gateway_specs = """
                **Recommended Hardware Specifications**  
                - Processor: Intel core i7, Xeon 2.5 GHz or higher processor type.  
                - RAM: 16GB  
                - Storage: 100GB SSD for operating system, 100 GB SSD for imaging data storage (This can be changed based on the retention policy)  
                - Network Interface: Gigabit Ethernet port  

                **Software Requirements**  
                - Operating System: Windows 10 or higher, Windows Server 2019 or higher  
                - DB Software: MSSQL 2019 or higher edition (Express edition can be used)  
                - Security Software: Firewall software, antivirus  

                **Security Considerations**  
                - Enable secure boot and ensure regular security updates.  
                - Implement role-based access controls for administrative tasks.  
                - Site to Site VPN with the main site. 

                **Internet Requirements**  
                - Minimum Required  bandwidth: 30 Mbps  
                - Recommended  bandwidth: 50 Mbps
                """
                st.subheader("Gateway Workstation Specs")
                st.markdown(f"<h4>Gateway Locations: {', '.join(map(str, gateway_locations))}</h4>",
                            unsafe_allow_html=True)
                st.markdown(gateway_specs)
            else:
                gateway_specs = None

                # Construct Licensing Notes Dynamically
            licensing_notes = []

            # Windows Licensing Note
            licensing_notes.append(
                "- **Windows Server Licensing:** Each VM requires an active Windows Server license. For environments with a Data Center license, these licenses may be unlimited. Verify the licensing model in use."
            )

            # Virtualization Licensing Note (HA-Specific)
            if high_availability:
                licensing_notes.append(
                    "- **Virtualization Platform Licensing:** Ensure the selected platform (e.g., VMware or Hyper-V) supports High Availability features (e.g., vMotion, Live Migration)."
                )

            # Ubuntu Licensing Note (Only if Ubuntu VMs Exist)
            if ubuntu_vms_count > 0:
                licensing_notes.append(
                    "- **Ubuntu Licensing:** Ensure sufficient Ubuntu Server licenses are available for VMs running on Linux."
                )

            # Antivirus Licensing Note
            licensing_notes.append(
                "- **Antivirus Licensing:** One license per VM is required unless otherwise specified."
            )

            # Backup Software Note
            licensing_notes.append(
                "- **Backup Software:** Ensure compatibility with selected VMs and storage solutions."
            )

            # Combine Licensing Notes into a Single String
            licensing_notes_text = "\n".join(licensing_notes)

            # Define the DMZ Recommendations
            dmz_recommendations = ""
            if ref_phys_external_access:
                dmz_recommendations += """
                                                       - Referring Physician Portal VMs must be placed in a **DMZ (Demilitarized Zone) network** to isolate external access from internal hospital systems and enhance security.
                                                   """
            if patient_portal_external_access:
                dmz_recommendations += """
                                                       - Patient Portal VMs must be placed in a **DMZ (Demilitarized Zone) network** to isolate external access from internal hospital systems and enhance security.
                                                   """

            # Define the Notes Dictionary
            notes = {
                "licensing_notes": licensing_notes_text.strip(),  # Add dynamically constructed Licensing Notes
                "sizing_notes": """
                                               - Provided VM sizing of the Virtual servers will be scaled up horizontally or vertically based on the expected volume of data and number of CCUs.
                                               - SSD Datastore recommended for all OS Virtual disks of all VMs.
                                               - SSD recommended for the data drive of the Database Server.
                                               - It's recommended to have SSD storage for the short Term Storage (STS) that keep last 6 month of data for higher performance in data access.
                                           """,
                "technical_requirements": """
                                               - Provide remote access to the VMs (all VMs) for SW installation and configurations.
                                               - In case not using dongles, a connection from the VMs to (https://paxaeraultima.net:1435) for PaxeraHealth licensing except the database VM.
                                           """,
                "network_requirements": f"""
                                               - LAN bandwidth to be 1GB dedicated.
                                               - Paxera PACS VMs, Paxera Ultima Viewer VMs, Paxera Broker Integration VMs must be in same vLAN.
                                               - 1 Gb/s dedicated bandwidth across the vLAN with the maximum number of two network hops.
                                               - Latency less than 1ms.
                                               - Site to Site VPN with any other connected branch.
                                               {dmz_recommendations.strip()}
                                           """,
                "minimum_requirements": """
                                               - **Antivirus on All Backend Servers (Required):**
                                                  Every server hosting our software components must have robust, up-to-date antivirus protection.
                                               - **Firewall for External Publishing (Required):**
                                                  Any system accessible from the internet—such as patient portals, referring physician portals, or tele-radiology services—must be protected by a properly configured firewall.
                                               - **Backup Storage for Secondary Copies (Strongly Recommended):**
                                                  Implementing a reliable backup strategy ensures a secondary copy of critical patient data and system files, enhancing business continuity and recovery capabilities.
                                               - **Additional Security Measures (Recommended):**
                                                  - Regular Security Updates & Patches: Keep all systems current with the latest patches.
                                                  - Multi-Factor Authentication (MFA): Enforce MFA for all administrative and remote access accounts.
                                                  - Network Segmentation & Isolation: Limit potential breaches by isolating critical systems.
                                               - **Non-Compliance & Customer Responsibility:**
                                                  Failure to adhere to these guidelines places customers at heightened risk for cyberattacks, including ransomware, data breaches, and service interruptions. If recommended security measures are not implemented, PaxeraHealth cannot be held responsible for any resulting damages or data loss. In such situations, any re-installation, re-implementation, or post-incident remediation will be billed as a separate professional service.
                                               - **Transparency & Long-Term Value:**
                                                  By openly communicating these requirements and their implications during the pre-sales process, we help customers make informed decisions that safeguard their systems and data. This level of transparency fosters trust, enhances the long-term value of our solutions, and ensures smoother implementations.
                                           """
            }

            ai_features = []
            if aidocker_included:
                if organ_segmentator:
                    ai_features.append("Organ Segmentator")
                if lesion_segmentator_2d:
                    ai_features.append("Lesion Segmentator 2D")
                if lesion_segmentator_3d:
                    ai_features.append("Lesion Segmentator 3D")
                if speech_to_text_container:
                    ai_features.append("Speech-to-Text Container")

            ark_feature = "Included" if ark_included else "Not Included"

            # Prepare AI-related values for input table
            ai_features = []
            if aidocker_included:
                if organ_segmentator:
                    ai_features.append("Organ Segmentator")
                if lesion_segmentator_2d:
                    ai_features.append("Lesion Segmentator 2D")
                if lesion_segmentator_3d:
                    ai_features.append("Lesion Segmentator 3D")
                if speech_to_text_container:
                    ai_features.append("Speech-to-Text Container")

            # Initialize the base Input and Value lists
            input_list = []
            value_list = []

            # Add items to input_list and value_list conditionally
            if num_studies > 0:
                input_list.append("Number of Studies")
                value_list.append(f"{num_studies:,}")  # Add comma as thousand separator

            if num_locations > 1:  # Only show if there is more than 1 location
                input_list.append("Number of Locations")
                value_list.append(num_locations)

            if num_machines > 1:  # Only show if there is more than 1 machine
                input_list.append("Number of Machines")
                value_list.append(num_machines)

            input_list.append("Contract Duration (years)")
            value_list.append(contract_duration)

            input_list.append("Study Size (MB)")
            value_list.append(study_size_mb)

            if annual_growth_rate > 0:
                input_list.append("Annual Growth Rate (%)")
                value_list.append(annual_growth_rate)

            # Add PACS CCU if > 0
            if pacs_ccu and pacs_ccu > 0:
                input_list.append("PACS CCU")
                value_list.append(pacs_ccu)

            # Add RIS CCU if > 0
            if ris_ccu and ris_ccu > 0:
                input_list.append("RIS CCU")
                value_list.append(ris_ccu)

            # Add Referring Physician CCU if > 0
            if ref_phys_ccu and ref_phys_ccu > 0:
                input_list.append("Referring Physician CCU")
                value_list.append(ref_phys_ccu)
            # Broker Logic
            if broker_level and broker_level != "Not Required":  # If broker_level is explicitly selected and not "Not Required"
                input_list.append("Broker VM")
                value_list.append(broker_level)
            elif ris_ccu and ris_ccu > 0:  # If RIS CCU > 0 and broker is not explicitly set
                broker_level = "WL"  # Default to "WL"
                input_list.append("Broker VM")
                value_list.append(broker_level)
            # If broker_level is "Not Required" and RIS CCU is 0, do not add Broker VM to the table

            # Conditionally add AI Features if any are selected
            if ai_features:
                input_list.append("AI Features")
                value_list.append(", ".join(ai_features))

            # Conditionally add ARKAI if it's included
            if ark_included:
                input_list.append("ARKAI")
                value_list.append("Included")

            # Construct the input values DataFrame
            input_values = pd.DataFrame({
                "Input": input_list,
                "Value": value_list
            })

            # Initialize Workstation Specifications
            diagnostic_specs = None
            viewing_specs = None
            ris_specs = None

            if include_workstation_specs:
                st.subheader("Workstation Specifications")

                diagnostic_specs = pd.DataFrame({
                    "Item": [
                        "Processor", "Memory Capacity", "Hard Drives",
                        "Internal Optical Drive", "Operating System",
                        "Medical Grade Monitor", "Monitor", "Graphics"
                    ],
                    "Description": [
                        "Intel Core i7 (6C, 12M Cache, base 3.1GHz, up to 4.5GHz)",
                        "16 GB, 2 x 8 GB, DDR5, 4400 MT/s, V2",
                        "512GB PCIe NVMe Class 40 M.2 SSD",
                        "8x DVD+/-RW 9.5mm Optical Disk Drive",
                        "Windows 10 Pro English",
                        "3MP (5MP for Mammo) color high brightness single head",
                        "24” LCD color monitor",
                        "Dedicated GPU with at least 4GB VRAM"
                    ]
                })
                st.markdown("### Diagnostic Workstation")
                st.dataframe(diagnostic_specs)

                # Viewing Workstation Table
                viewing_specs = pd.DataFrame({
                    "Item": [
                        "Processor", "Memory Capacity", "Hard Drives",
                        "Internal Optical Drive", "Operating System",
                        "Monitor"
                    ],
                    "Description": [
                        "Intel Core i5 (6C, 12M Cache, base 3.1GHz, up to 4.5GHz)",
                        "1 x 8 GB, DDR5, 4400 MT/s, V2",
                        "256GB PCIe NVMe Class 40 M.2 SSD",
                        "8x DVD+/-RW 9.5mm Optical Disk Drive",
                        "Windows 10 Pro English",
                        "24” LCD color monitor"
                    ]
                })
                st.markdown("### PACS Viewing Workstation")
                st.dataframe(viewing_specs)

                # RIS Workstation Table
                if ris_enabled:
                    ris_specs = pd.DataFrame({
                        "Item": [
                            "Processor", "Memory Capacity", "Hard Drives",
                            "Internal Optical Drive", "Operating System",
                            "Monitor"
                        ],
                        "Description": [
                            "10th Generation Intel® Core™ i5, 6 Cores, 12MB Cache, 3.2GHz",
                            "8GB 1 x 8GB DDR4-2666MHz",
                            "256GB PCIe NVMe Class 40 M.2 SSD (3.5 inch)",
                            "8x DVD+/-RW 9.5mm Optical Disk Drive",
                            "Windows 10 Pro English",
                            "24” LCD color monitor"
                        ]
                    })  # Set "Item" as the index
                    st.markdown("### RIS Workstation")
                    st.dataframe(ris_specs)

        if high_availability:
            physical_design_string = "High Availability Server Design:\n" + server_specs
        else:
            physical_design_string = "Standard Server Design:\n" + server_specs
        # Determine shared storage type
        if len(servers) == 1 and not high_availability:
            shared_storage = "Built-in Storage"
        else:
            shared_storage = "Shared DAS/SAN Storage"

        doc = generate_document_from_template(
            template_path=os.path.join(app_dir, "assets", "templates", "Temp.docx"),
            results=results,
            results_grade1=results_grade1,
            results_grade3=results_grade3,
            df_comparison=df_comparison,
            third_party_licenses=third_party_licenses,
            notes=notes,
            input_table=input_values,
            customer_name=customer_name,
            high_availability=high_availability,
            server_specs=server_specs,  # Updated production server specs
            gpu_specs=gpu_specs,
            first_year_storage_raid5=first_year_storage_raid5,
            total_image_storage_raid5=total_image_storage_raid5,
            num_studies=num_studies,
            storage_title=storage_type,
            shared_storage=shared_storage,
            raid_1_storage_tb=raid_1_storage_tb,
            gateway_specs=gateway_specs,
            diagnostic_specs=diagnostic_specs,
            viewing_specs=viewing_specs,
            ris_specs=ris_specs,
            project_grade=project_grade,
            storage_table=storage_table,
            physical_design=physical_design_string,
            nas_backup_details=nas_backup_string,
            tier_2_disks=tier_2_disks,
            tier_2_disk_size=tier_2_disk_size,
            tier_3_disks=tier_3_disks,
            tier_3_disk_size=tier_3_disk_size,
            additional_vm_table=additional_vms_table,  # Pass additional VMs as a DataFrame
            additional_vm_notes=additional_vm_notes,  # Notes for additional VMs
            general_notes=general_notes,  # General notes applicable to all VMs
            additional_servers=additional_servers,
            additional_vms=additional_vms,
            additional_requirements_table=additional_requirements_table  # Pass additional requirements table
        )

        download_link = get_binary_file_downloader_html(
            doc,
            file_label="Download Word Document",
            customer_name=customer_name,
            file_extension=".docx"
        )
        st.markdown(download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
