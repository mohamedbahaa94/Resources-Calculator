import os
import time
import logging
import math
from PIL import Image
import streamlit as st
import pandas as pd
from document_generator import generate_document_from_template, get_binary_file_downloader_html
from vm_calculations import calculate_vm_requirements, modality_sizes
from Bandwidth import calculate_layer4_throughput, calculate_layer7_throughput

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
    available_disk_sizes_tb = sorted([1.5, 2.4, 3, 4, 8, 12, 16, 20, 22])  # ascending order

    best_total_disks = None
    best_disk_size = None
    smallest_penalty = float('inf')

    # Loop to prioritize fewer disks and larger disk sizes
    for total_disks in range(min_disks, max_disks + 1):
        for disk_size in reversed(available_disk_sizes_tb):  # larger disks first
            usable_storage = (total_disks - 1) * disk_size  # RAID 5 usable capacity

            if usable_storage >= usable_storage_tb:
                # Penalty balances excess capacity and disk count
                excess = usable_storage - usable_storage_tb
                penalty = excess + 2 * total_disks  # higher weight on disk count

                if penalty < smallest_penalty:
                    smallest_penalty = penalty
                    best_total_disks = total_disks
                    best_disk_size = disk_size

    return best_total_disks, best_disk_size
# Test the optimized function with an input of 18 TB usable storage


def round_to_nearest_divisible_by_two(value):
    return round(value / 2) * 2


def format_number_with_commas(number):
    return f"{number:,}"


def main():
    import base64
    import os
    from PIL import Image

    app_dir = os.path.dirname(os.path.abspath(__file__))

    # Encode logo
    with open(os.path.join(app_dir, "assets", "logo.png"), "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
        <style>
        .black-ribbon {{
            width: 100vw;
            margin-left: -50vw;
            left: 50%;
            position: relative;
            background-color: black;
            padding: 30px 0;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .ribbon-inner {{
            display: flex;
            align-items: center;
            gap: 30px;
        }}
        .logo-img {{
            height: 100px;
        }}
        .divider {{
            width: 3px;
            height: 80px;
            background-color: #F37F21;
        }}
        .title-text {{
            color: #F37F21;
            font-size: 36px;
            font-weight: bold;
        }}
        </style>

        <div class="black-ribbon">
            <div class="ribbon-inner">
                <img src="data:image/png;base64,{logo_base64}" class="logo-img" />
                <div class="divider"></div>
                <div class="title-text">PaxeraHealth IT Infrastructure & VM Design Tool</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Customer name field — visually integrated
    with st.container():
     customer_name = st.text_input("Customer Name:", placeholder="Enter customer or site name")

    with st.expander("Project and Location Details"):
        num_machines = st.number_input("Number of Machines (Modalities):", min_value=1, value=1, format="%d")
        num_locations = st.number_input("Number of Locations:", min_value=1, value=1, format="%d")
        breakdown_per_modality = st.radio("Breakdown per Modality?", ["No", "Yes"])

        # ✅ Define Modality Average Study Sizes (in MB)
        modality_sizes_mb = {
            "CT": 1024,
            "MR": 100,
            "US": 10,
            "NM": 10,
            "X-ray": 30,
            "MG": 160,
            "Cath": 300
        }

        if breakdown_per_modality == "No":
            num_studies = st.number_input("Number of studies per year:", min_value=0, value=100000, format="%d")

            # ✅ Display the number of studies dynamically
            st.metric(label="Number of Studies", value=f"{num_studies:,} studies", delta=None, delta_color="off")
            modality_cases = {}
        else:
            st.subheader("Modality Breakdown:")
            modality_cases = {}

            for modality, size_mb in modality_sizes_mb.items():
                # ✅ Keep the old style but add the **average study size** next to each modality
                modality_cases[modality] = st.number_input(
                    f"{modality} Cases (📏 Avg. {size_mb:,} MB)", min_value=0, format="%d", key=modality
                )

            # ✅ Calculate the total number of studies dynamically
            num_studies = sum(modality_cases.values())

            # ✅ Display the updated number of studies dynamically
            st.metric(label="Total Number of Studies", value=f"{num_studies:,} studies", delta=None, delta_color="off")

        contract_duration = st.number_input("Contract Duration (years):", min_value=1, value=3, format="%d")
        study_size_mb = st.number_input("Study Size (MB):", min_value=0, value=100,
                                        format="%d")  # ✅ Leave this untouched
        annual_growth_rate = st.number_input("Annual Growth Rate (%):", min_value=0.0, value=10.0, format="%f")
        legacy_data_migration = st.checkbox("Migration of legacy studies required?", value=False)
        migration_data_tb = 0
        if legacy_data_migration:
            migration_data_tb = st.number_input("Estimated volume of legacy data to migrate (in TB):", min_value=0.0,
                                                value=1.0, step=0.5)
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
            broker_level = st.multiselect("Select Broker Level(s):", ["WL", "HL7"])
        else:
            broker_level = []  # No broker required
    with st.expander("Redundancy Options"):
        split_pacs = st.checkbox("Split PACS VM to Avoid Single Point of Failure", value=False)
        add_n_plus_one = st.checkbox("Add N+1 VM for High Availability (PACS/RIS/Ultima)", value=False)
        sql_always_on = st.checkbox("Enable SQL Always On (requires SQL Enterprise + 2 DB VMs)", value=False)
        load_balancers_enabled = st.checkbox("Deploy Layer 4 and Layer 7 Load Balancers", value=False)
        high_availability = st.checkbox("Enable High Availability on Hardware Level (Host, Server, Storage)",
                                        value=False)


    # AI Features Section
    with st.expander("AI Features"):
        aidocker_included = st.checkbox(
            "Include U9th Integrated AI Modules",
            value=False,
            disabled=(num_studies == 0)
        )

        organ_segmentator = False
        radiomics = False
        speech_to_text = False
        chatbot = False
        xray_nodule = False

        if aidocker_included:
            st.markdown("#### U9th AI Modules")
            organ_segmentator = st.checkbox("Organ Segmentation + Spine Labeling + Rib Counting")
            radiomics = st.checkbox("Radiomics")
            speech_to_text = st.checkbox("Speech to Text")
            chatbot = st.checkbox("Chatbot (All Services + Report Rephrasing)")
            xray_nodule = st.checkbox("X-ray Nodule & Consolidation")

        ark_included = st.checkbox(
            "Include ARKAI",
            value=False,
            disabled=(num_studies == 0)
        )

        # ➕ New ARK options (only shown if ARK is selected)
        share_segmentation = False
        share_chatbot = False
        ark_chatbot_enabled = False

        if ark_included:
            st.markdown("#### ARK AI Options")

            if organ_segmentator:
                share_segmentation = st.checkbox("Share Segmentation Module with ARK?")
            if chatbot:
                share_chatbot = st.checkbox("Share Chatbot Module with ARK?")

            ark_chatbot_enabled = st.checkbox("Enable Chatbot for ARK?")

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
        include_workstation_specs = 1


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
                share_segmentation=share_segmentation if ark_included and organ_segmentator else False,
                share_chatbot=share_chatbot if ark_included and chatbot else False,
                ark_chatbot_enabled=ark_chatbot_enabled if ark_included else False,
                u9_ai_features=[
                    feature for feature, selected in [
                        ("Organ Segmentation + Spine Labeling + Rib Counting", organ_segmentator),
                        ("Radiomics", radiomics),
                        ("Speech to Text", speech_to_text),
                        ("Chatbot (All Services + Report Rephrasing)", chatbot),
                        ("X-ray Nodule & Consolidation", xray_nodule),
                    ] if selected
                ] if aidocker_included else [],
                ref_phys_external_access=ref_phys_external_access,
                patient_portal_ccu=patient_portal_ccu,
                patient_portal_external_access=patient_portal_external_access,
                training_vm_included=training_vm_included,
                high_availability=high_availability,
                combine_pacs_ris=combine_pacs_ris,
                sql_always_on=sql_always_on,
                split_pacs=split_pacs,
                add_n_plus_one=add_n_plus_one,
                load_balancers_enabled=load_balancers_enabled,
                **modality_cases
            )

            # Save in session state
            st.session_state["results"] = results
            st.session_state["sql_license"] = sql_license
            st.session_state["calculation_done"] = True
            migration_vm = None

            if legacy_data_migration and not results.empty:
                if legacy_data_migration and not results.empty:
                    # Extract the PACS VM (first one if split)
                    pacs_vm_rows = results[results.index.str.contains("PACS")]
                    if not pacs_vm_rows.empty:
                        first_pacs_index = pacs_vm_rows.index[0]
                        pacs_vm_spec = results.loc[first_pacs_index]
                        migration_vm = {
                            "vCores": pacs_vm_spec["vCores"],
                            "RAM (GB)": pacs_vm_spec["RAM (GB)"],
                            "Storage (GB)": pacs_vm_spec["Storage (GB)"],
                            "Operating System": "Windows Server 2019 or Higher",
                            "Other Software": ""
                        }


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
                    # Special case for ARK DB
                    if "ARKDatabase" in index:
                        results.at[index, "Operating System"] = "Windows Server 2019 or Higher"
                        results.at[index, "Other Software"] = "SQL Server Express"
                    # Update for specific AI Segmentation Dockers
                    if any(name in index for name in
                           ["OrganSegmentationDocker",
                            "RadiomicsDocker",
                            "SpeechToTextDocker",
                            "ChatbotDocker",
                            "XrayNoduleDocker"]):
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
            # Build the migration VM table
            # 🛠️ Display the migration VM only if requested
            # 🧭 Migration Info (Keep this under the condition)
            if legacy_data_migration and not results.empty:
                st.markdown("### Temporary Migration Virtual Machine")
                st.markdown(
                    "To support the data migration process, a temporary virtual machine will be provisioned to facilitate smooth and efficient migration of studies, configurations, "
                    "and system data to the new environment."
                )

                # Build the migration VM table
                migration_vm_df = pd.DataFrame([{
                    "VM Type": "Migration VM",
                    "vCores": int(migration_vm["vCores"]),
                    "RAM (GB)": int(migration_vm["RAM (GB)"]),
                    "Storage (GB)": round(migration_vm["Storage (GB)"], 1),
                    "Operating System": migration_vm["Operating System"],

                    "Other Software": migration_vm["Other Software"]
                }])

                st.dataframe(migration_vm_df, use_container_width=True)

                st.markdown("**Migration VM Note:**")
                st.markdown(
                    "- The resources allocated to this temporary VM (vCores, RAM, and SSD storage) are **not included** in the overall system sizing,\n"
                    "  as the VM is only required during the migration phase and will be decommissioned afterward."
                )

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
                elif "SQL Standard" in sql_license:
                     test_vm_specs["vCores"] = 10
                     test_vm_specs["RAM (GB)"] = 32
                     test_vm_specs["RAID 5 Storage (TB)"] = 5.0
                elif "SQL Enterprise" in sql_license:
                    test_vm_specs["vCores"] = 12
                    test_vm_specs["RAM (GB)"] = 64
                    test_vm_specs["RAID 5 Storage (TB)"] = 10.0
                elif sql_license =="SQL Express (Recommended: SQL Standard)":
                    test_vm_specs["vCores"] = 10
                    test_vm_specs["RAM (GB)"] = 32
                    test_vm_specs["RAID 5 Storage (TB)"] = 5.0

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
                project_grade=project_grade,
                broker_required=1,
                broker_level=broker_level,
                num_machines=num_machines,
                contract_duration=contract_duration,
                study_size_mb=study_size_mb,
                annual_growth_rate=annual_growth_rate,
                breakdown_per_modality=breakdown_per_modality,
                aidocker_included=aidocker_included,
                ark_included=ark_included,
                share_segmentation=share_segmentation if ark_included and organ_segmentator else False,
                share_chatbot=share_chatbot if ark_included and chatbot else False,
                ark_chatbot_enabled=ark_chatbot_enabled if ark_included else False,
                u9_ai_features=[
                    feature for feature, selected in [
                        ("Organ Segmentation + Spine Labeling + Rib Counting", organ_segmentator),
                        ("Radiomics", radiomics),
                        ("Speech to Text", speech_to_text),
                        ("Chatbot (All Services + Report Rephrasing)", chatbot),
                        ("X-ray Nodule & Consolidation", xray_nodule),
                    ] if selected
                ] if aidocker_included else [],
                ref_phys_external_access=ref_phys_external_access,
                patient_portal_ccu=patient_portal_ccu,
                patient_portal_external_access=patient_portal_external_access,
                training_vm_included=training_vm_included,
                high_availability=high_availability,
                combine_pacs_ris=combine_pacs_ris,
                sql_always_on=sql_always_on,
                split_pacs=split_pacs,
                add_n_plus_one=add_n_plus_one,
                load_balancers_enabled=load_balancers_enabled,
                **modality_cases
            )

            results_grade2, _, first_year_storage_raid5_grade2, total_image_storage_raid5_grade2, total_vcpu_grade2, total_ram_grade2, total_storage_grade2 = calculate_vm_requirements(
                num_studies=num_studies,
                pacs_ccu=pacs_ccu,
                ris_ccu=ris_ccu,
                ref_phys_ccu=ref_phys_ccu,
                project_grade=2,
                broker_required=broker_required,
                broker_level=broker_level,
                num_machines=num_machines,
                contract_duration=contract_duration,
                study_size_mb=study_size_mb,
                annual_growth_rate=annual_growth_rate,
                breakdown_per_modality=breakdown_per_modality,
                aidocker_included=aidocker_included,
                ark_included=ark_included,
                share_segmentation=share_segmentation if ark_included and organ_segmentator else False,
                share_chatbot=share_chatbot if ark_included and chatbot else False,
                ark_chatbot_enabled=ark_chatbot_enabled if ark_included else False,
                u9_ai_features=[
                    feature for feature, selected in [
                        ("Organ Segmentation + Spine Labeling + Rib Counting", organ_segmentator),
                        ("Radiomics", radiomics),
                        ("Speech to Text", speech_to_text),
                        ("Chatbot (All Services + Report Rephrasing)", chatbot),
                        ("X-ray Nodule & Consolidation", xray_nodule),
                    ] if selected
                ] if aidocker_included else [],
                ref_phys_external_access=ref_phys_external_access,
                patient_portal_ccu=patient_portal_ccu,
                patient_portal_external_access=patient_portal_external_access,
                training_vm_included=training_vm_included,
                high_availability=high_availability,
                combine_pacs_ris=combine_pacs_ris,
                sql_always_on=sql_always_on,
                split_pacs=split_pacs,
                add_n_plus_one=add_n_plus_one,
                load_balancers_enabled=load_balancers_enabled,
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
                share_segmentation=share_segmentation if ark_included and organ_segmentator else False,
                share_chatbot=share_chatbot if ark_included and chatbot else False,
                ark_chatbot_enabled=ark_chatbot_enabled if ark_included else False,
                u9_ai_features=[
                    feature for feature, selected in [
                        ("Organ Segmentation + Spine Labeling + Rib Counting", organ_segmentator),
                        ("Radiomics", radiomics),
                        ("Speech to Text", speech_to_text),
                        ("Chatbot (All Services + Report Rephrasing)", chatbot),
                        ("X-ray Nodule & Consolidation", xray_nodule),
                    ] if selected
                ] if aidocker_included else [],
                ref_phys_external_access=ref_phys_external_access,
                patient_portal_ccu=patient_portal_ccu,
                patient_portal_external_access=patient_portal_external_access,
                training_vm_included=training_vm_included,
                high_availability=high_availability,
                combine_pacs_ris=combine_pacs_ris,
                sql_always_on=sql_always_on,
                split_pacs=split_pacs,
                add_n_plus_one=add_n_plus_one,
                load_balancers_enabled=load_balancers_enabled,
                **modality_cases
            )

            # Define Storage Values for Display
            raid_1_storage_tb_g3 = results_grade3.loc["RAID 1 (SSD)", "Storage (GB)"] / 1024
            raid_5_storage_tb = total_image_storage_raid5_grade3 / 1125  # Total HDD storage for full duration
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

                # ➕ Reflect migration data across all years in Tier 3
                if legacy_data_migration and migration_data_tb > 0:
                    tier_3_storage = [round(value + migration_data_tb, 2) for value in tier_3_storage]

                final_tier_3_storage = round(tier_3_storage[-1], 1)


            else:

                # No intermediate Tier 2; rename Tier 3 to Tier 2

                tier_2_label = "Tier 2: Long-Term Storage (HDD RAID 5)"

                # Initialize and calculate Tier 2 storage (same as Tier 3 would be)

                tier_2_storage = []

                for year in years:

                    current_year_storage = initial_year_storage * (1 + annual_growth_rate / 100) ** (year - 1)

                    if year == 1:

                        tier_2_storage.append(round(current_year_storage, 2))  # First year storage

                    else:

                        cumulative_storage = tier_2_storage[-1] + current_year_storage

                        tier_2_storage.append(round(cumulative_storage, 2))

                # ➕ Reflect migration data across all years in Tier 2

                if legacy_data_migration and migration_data_tb > 0:
                    tier_2_storage = [round(value + migration_data_tb, 2) for value in tier_2_storage]

                tier_3_storage = None  # Clear Tier 3 since it's now Tier 2

                final_tier_3_storage = round(tier_2_storage[-1], 1)  # ✅ FIX: Use actual storage value
            # Prepare storage data dictionary
            storage_data = {
                "Year": [f"Year {year}" for year in years],
                "Tier 1: OS & DB (SSD RAID 1)": [round(tier_1_storage, 2)] * len(years),
                tier_2_label: tier_2_storage
            }

            # Add Tier 3 only if `include_fast_tier_storage` is True
            if include_fast_tier_storage and tier_3_storage:
                storage_data["Tier 3: Long-Term Storage (HDD RAID 5)"] = tier_3_storage

            # Create and display the Storage Table
            storage_table = pd.DataFrame(storage_data).reset_index(drop=True)
            st.table(storage_table.style.format(precision=2))
            if legacy_data_migration and migration_data_tb > 0:
                st.markdown(
                    "**Note:** The long-term storage tiers above include provisioned capacity for migrated legacy studies."
                )

            # Resource Comparison Table
            # Minimum vs Recommended Resources for Grade 1
            # Resource Comparison Table

            # Resource Comparison Table

            if project_grade in [1, 2]:  # Grade 1 or 2: Show Minimum vs. Recommended Resources
                # Adjust RAID 5 with migration data if applicable
                adjusted_raid_5_storage_tb = raid_5_storage_tb + migration_data_tb if legacy_data_migration and migration_data_tb > 0 else raid_5_storage_tb

                comparison_data = {
                    "Specification": ["Total vCores", "Total RAM (GB)", "RAID 1 (SSD) (TB)",
                                      "RAID 5 (HDD) Full Duration (TB)"],
                    "Minimum Specs": [
                        round(total_vcpu_grade1 if project_grade == 1 else total_vcpu_grade2, 1),
                        round(total_ram_grade1 if project_grade == 1 else total_ram_grade2, 1),
                        round(raid_1_storage_tb if project_grade == 1 else raid_1_storage_tb_g3, 1),
                        round(adjusted_raid_5_storage_tb, 1),
                    ],
                    "Recommended Specs": [
                        round(total_vcpu_grade3, 1),
                        round(total_ram_grade3, 1),
                        round(raid_1_storage_tb_g3, 1),
                        round(adjusted_raid_5_storage_tb, 1),
                    ],
                }


                df_comparison = pd.DataFrame(comparison_data)
                st.subheader(f"Minimum vs. Recommended Resources :")

            else:  # Grade 3: Show Only Recommended Resources
                recommended_data = {
                    "Specification": ["Total vCores", "Total RAM (GB)", "RAID 1 (SSD) (TB)",
                                      "RAID 5 (HDD) Full Duration (TB)"],
                    "Recommended Specs": [
                        round(total_vcpu_grade3, 1),
                        round(total_ram_grade3, 1),
                        round(raid_1_storage_tb_g3, 1),
                        round(raid_5_storage_tb, 1),  # Handle None case
                    ],
                }
                df_comparison = pd.DataFrame(recommended_data)  # Keep format consistent
                st.subheader("Recommended Resources :")

            # Display table (common for all grades)
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
                    threads_per_server = int(total_vcpu_grade3 * 0.75)
                    ram_per_server = int(total_ram_grade3 * 0.75)

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
            print(f"[DEBUG] Tier 1 (RAID 1) Storage: {raid_1_storage_tb:.2f} TB")

            # Tier 2/3 Logic
            print(
                f"\n[DEBUG] Storage Scenario: {'3-tier (Fast+Long)' if include_fast_tier_storage else '2-tier (Long only)'}")
            print(f"[DEBUG] final_tier_3_storage (GB): {final_tier_3_storage}")

            if include_fast_tier_storage:
                # Intermediate (Fast Tier) Calculation
                intermediate_tier_multiplier = 0.5 if fast_tier_duration == "6 Months" else 1.0
                intermediate_tier_storage_tb = math.ceil(first_year_storage_raid5 * intermediate_tier_multiplier / 1024)
                print(f"[DEBUG] Intermediate Tier Multiplier: {intermediate_tier_multiplier}")
                print(f"[DEBUG] Intermediate Tier Storage (TB): {intermediate_tier_storage_tb}")

                tier_2_disks, tier_2_disk_size = calculate_raid5_disks(intermediate_tier_storage_tb)
                tier_2_label = "Tier 2: Fast Image Storage (SSD RAID 5)"
                print(f"[DEBUG] Tier 2 (Fast): {tier_2_disks}x {tier_2_disk_size:.2f} TB")

                # Long-Term (Tier 3) Calculation
                tier_3_disks, tier_3_disk_size = calculate_raid5_disks(final_tier_3_storage)
                print(f"[DEBUG] Tier 3 (Long): {tier_3_disks}x {tier_3_disk_size:.2f} TB")
            else:
                # Combined Long-Term Storage (as Tier 2)
                tier_3_disks, tier_3_disk_size = calculate_raid5_disks(final_tier_3_storage)
                print(f"[DEBUG] Raw Tier 3 Calculation: {tier_3_disks}x {tier_3_disk_size:.2f} TB")

                tier_2_disks, tier_2_disk_size = tier_3_disks, tier_3_disk_size
                tier_2_label = "Tier 2: Long-Term Storage (HDD RAID 5)"
                tier_3_disks, tier_3_disk_size = 0, 0
                print(f"[DEBUG] Tier 2 (Long): {tier_2_disks}x {tier_2_disk_size:.2f} TB")

            # Storage Type
            storage_type = "Built-in Storage" if len(
                servers) == 1 and not high_availability else "Shared DAS/SAN Storage"
            print(f"\n[DEBUG] Final Storage Configuration:")
            print(f"Tier 1: 2x {raid_1_storage_tb:.2f} TB (SSD RAID 1)")
            print(f"Tier 2: {tier_2_disks}x {tier_2_disk_size:.2f} TB ({tier_2_label})")
            if tier_3_disks:
                print(f"Tier 3: {tier_3_disks}x {tier_3_disk_size:.2f} TB (HDD RAID 5)")

            # Generate Storage Details
            storage_details = f"""
            #### Storage Details ({storage_type})

            **Tier 1: OS & DB (SSD RAID 1):**
            - SSD Drives: 2x {raid_1_storage_tb:.2f} TB
            """

            # Add Tier 2 Storage
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

            st.markdown(storage_details)
            # Backup Storage (NAS)
            if use_nas_backup:
                # Calculate NAS storage based on redundancy factor and backup years
                if use_nas_backup:
                    # Calculate NAS storage based on redundancy factor and backup years
                    if legacy_data_migration and migration_data_tb > 0:
                        nas_storage_gb = round(
                            ((
                                         total_image_storage_raid5 + migration_data_tb) * nas_redundancy_factor * nas_backup_years) / contract_duration
                        )
                        nas_storage_tb = nas_storage_gb / 1024  # Convert to TB
                    else:
                        nas_storage_gb = round(
                            (total_image_storage_raid5 * nas_redundancy_factor * nas_backup_years) / contract_duration
                        )
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
                total_gpu_memory = 0
                gpu_modules = []
                # Updated GPU allocation per module
                gpu_memory_requirements = {
                    "Organ Segmentator": 16,
                    "Radiomics": 0,
                    "Speech to Text": 10,
                    "Chatbot": 16,
                    "X-ray Nodule & Consolidation": 4
                }

                # Initialize GPU specs
                gpu_specs = ""

                if aidocker_included or ark_included:
                    st.markdown("###  GPU Requirements Summary")

                    total_gpu_memory_u9 = 0
                    total_gpu_memory_ark = 0
                    gpu_modules_u9 = []
                    gpu_modules_ark = []

                    # U9 GPU Requirements
                    if aidocker_included:
                        if organ_segmentator:
                            gpu_modules_u9.append("Organ Segmentator")
                            total_gpu_memory_u9 += gpu_memory_requirements["Organ Segmentator"]
                        if radiomics:
                            gpu_modules_u9.append("Radiomics")
                        if speech_to_text:
                            gpu_modules_u9.append("Speech to Text")
                            total_gpu_memory_u9 += gpu_memory_requirements["Speech to Text"]
                        if chatbot:
                            gpu_modules_u9.append("Chatbot")
                            total_gpu_memory_u9 += gpu_memory_requirements["Chatbot"]
                        if xray_nodule:
                            gpu_modules_u9.append("X-ray Nodule & Consolidation")
                            total_gpu_memory_u9 += gpu_memory_requirements["X-ray Nodule & Consolidation"]

                        gpu_specs += f"""
                ####  U9 Integrated AI Modules  
                - **Total GPU Memory Required**: `{total_gpu_memory_u9} GB`  
                - **Modules Requiring GPU**: `{', '.join(gpu_modules_u9) if gpu_modules_u9 else 'None'}`  
                - **Driver**: `NVIDIA Driver ≥ 450.80.02`, `CUDA ≥ 11.4`
                """

                    # ARK GPU Requirements
                    if ark_included:
                        # Always add base ARK Lab GPU
                        ark_lab_gpu = 24
                        total_gpu_memory_ark += ark_lab_gpu
                        gpu_modules_ark.append("ARK Lab Core (24 GB dedicated)")

                        if not share_segmentation:
                            gpu_modules_ark.append("ARK Segmentator")
                            total_gpu_memory_ark += gpu_memory_requirements["Organ Segmentator"]
                        if ark_chatbot_enabled and not share_chatbot:
                            gpu_modules_ark.append("ARK Chatbot")
                            total_gpu_memory_ark += gpu_memory_requirements["Chatbot"]

                        gpu_specs += f"""
                ####  ARK AI Modules  
                - **Total GPU Memory Required**: `{total_gpu_memory_ark} GB`  
                - **Modules Requiring GPU**: `{', '.join(gpu_modules_ark)}`  
                - **ARK Lab **: Recommended to have a dedicated GPU with at least `24 GB`   
                - **Driver**: `NVIDIA Driver ≥ 450.80.02`, `CUDA ≥ 11.4`
                """

                    st.markdown(gpu_specs)
            if load_balancers_enabled:
                st.subheader("Load Balancer Network Design Note")

                st.info("""
                           Load balancer is enabled. The following setup is recommended:
                           - 🟢 **Layer 4 Load Balancer** (Active-Passive) for: PACS Archive and HL7/WL Brokers
                           - 🔵 **Layer 7 Load Balancer** (Active-Passive) for: Ultima Viewer, RIS, Referring Physician Portal, Patient Portal
                           """)

                # Auto extract required inputs from existing form/session (you can replace with your values or variables)
                study_volume = num_studies  # example: 100000
                working_days = 280
                avg_study_size_mb = 100

                pacs_ccu = pacs_ccu  # viewer concurrent users
                priors_per_study = 3  # from your inputs
                ris_ccu = ris_ccu
                referring_ccu = ref_phys_ccu
                patient_ccu = patient_portal_ccu

                l4 = calculate_layer4_throughput(study_volume, working_days, avg_study_size_mb)
                l7 = calculate_layer7_throughput(
                    pacs_ccu, priors_per_study, ris_ccu,
                    referring_ccu=referring_ccu, patient_ccu=patient_ccu
                )

                # 📌 Display Summary (no UI inputs here, just results)
                st.subheader("Load Balancer Network Design Summary")
                st.markdown("""
                               - **Layer 4 Load Balancer** (Active-Passive): Handles Archive and HL7/WL Brokers  
                               - **Layer 7 Load Balancer** (Active-Passive): Handles Ultima Viewer, RIS, Referring Physician Portal, Patient Portal
                               """)
                st.markdown("###  Bandwidth Recommendations")
                st.write(f"**Layer 4 (Archive + Brokers):** {l4['recommended_l4_mbps']} Mbps")
                st.write(f"**Layer 7 (PACS Viewer, RIS, Portals):** {l7['recommended_l7_mbps']} Mbps")

            # Filter Ubuntu Licenses
            ubuntu_vms_count = len(
                non_total_display_results[non_total_display_results["Operating System"] == "Ubuntu 20.4"])

            # Calculate base Windows and Antivirus license count
            # Base license counts
            windows_base_count = int(len(non_total_display_results) - ubuntu_vms_count)
            antivirus_base_count = windows_base_count

            # Additional licenses for separate host
            additional_windows_license_count = 0
            additional_notes = []

            if additional_vms:
                has_test_vm = any(
                    vm["VM Type"] == "Test Environment VM (Ultima, PACS, Broker)" for vm in additional_vms)
                has_management_vm = any(
                    vm["VM Type"] == "Management VM (Backup, Antivirus, vCenter)" for vm in additional_vms)

                if has_test_vm:
                    additional_windows_license_count += 1
                    additional_notes.append("1x Test")

                if has_management_vm:
                    additional_windows_license_count += 1
                    additional_notes.append("1x Management")

            # Construct final Windows Server License description
            if additional_windows_license_count > 0:
                breakdown = f"{windows_base_count}x for core system + {additional_windows_license_count}x for additional VMs on separate host: {', '.join(additional_notes)}"
            else:
                breakdown = f"{windows_base_count}x for core system"

            windows_license_desc = f"MS Windows Server Standard 2019 or higher ({breakdown})"

            # Build the final DataFrame
            third_party_licenses = pd.DataFrame({
                "Item Description": [
                    sql_license,
                    windows_license_desc,
                    "Antivirus Server License",
                    "Virtualization Platform License(VMware or HyperV) (Supports High Availability)" if high_availability else "Virtualization Platform License",
                    "Backup Software (Compatible with VMs)",
                    "Ubuntu 20.4 Server License" if ubuntu_vms_count > 0 else None
                ],
                "Qty": [
                    2 if sql_always_on else 1,
                    windows_base_count + additional_windows_license_count,
                    antivirus_base_count,
                    1,
                    1,
                    int(ubuntu_vms_count) if ubuntu_vms_count > 0 else None
                ]
            })

            # Remove None rows and cast Qty to int
            third_party_licenses = third_party_licenses.dropna(subset=["Item Description", "Qty"])
            third_party_licenses["Qty"] = third_party_licenses["Qty"].astype(int)

            # Display
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
            - A connection from the VMs to (https://paxaeraultima.net:1435) for PaxeraHealth licensing except the database VM.
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


            ark_feature = "Included" if ark_included else "Not Included"

            # 🧠 Prepare AI-related values for input table (new modules only)
            ai_features = []
            if aidocker_included:
                if organ_segmentator:
                    ai_features.append("Organ Segmentation + Spine Labeling + Rib Counting")
                if radiomics:
                    ai_features.append("Radiomics")
                if speech_to_text:
                    ai_features.append("Speech to Text")
                if chatbot:
                    ai_features.append("Chatbot (All Services + Report Rephrasing)")
                if xray_nodule:
                    ai_features.append("X-ray Nodule & Consolidation")

            # 📋 Initialize the base Input and Value lists
            input_list = []
            value_list = []


            # 🎯 Annual studies
            if num_studies > 0:
                input_list.append("Number of Studies")
                value_list.append(f"{num_studies:,}")
                # ✅ Migration — add this section first
            if legacy_data_migration:
                    input_list.append("Migration")
                    value_list.append("Included")

            if num_locations > 1:
                input_list.append("Number of Locations")
                value_list.append(num_locations)

            if num_machines > 1:
                input_list.append("Number of Machines")
                value_list.append(num_machines)

            input_list.append("Contract Duration (years)")
            value_list.append(contract_duration)

            input_list.append("Study Size (MB)")
            value_list.append(study_size_mb)

            if annual_growth_rate > 0:
                input_list.append("Annual Growth Rate (%)")
                value_list.append(annual_growth_rate)

            if pacs_ccu and pacs_ccu > 0:
                input_list.append("PACS CCU")
                value_list.append(pacs_ccu)

            if ris_ccu and ris_ccu > 0:
                input_list.append("RIS CCU")
                value_list.append(ris_ccu)

            if ref_phys_ccu and ref_phys_ccu > 0:
                input_list.append("Referring Physician CCU")
                value_list.append(ref_phys_ccu)

            # 🔄 Broker logic with cleaned list
            if broker_level and broker_level != "Not Required":
                broker_value = ", ".join(broker_level) if isinstance(broker_level, list) else broker_level
                input_list.append("Broker VM")
                value_list.append(broker_value)
            elif ris_ccu and ris_ccu > 0:
                input_list.append("Broker VM")
                value_list.append("WL")

            # 🤖 AI Features
            if ai_features:
                input_list.append("AI Features")
                value_list.append("\n".join(f"• {item}" for item in ai_features))
            # 🧠 ARKAI
            if ark_included:
                input_list.append("ARKAI")
                value_list.append("Included")

            # 📄 Build DataFrame for Word export
            input_values = pd.DataFrame({
                "Input": input_list,
                "Value": value_list
            })

            # Initialize Workstation Specifications
            diagnostic_specs = None
            viewing_specs = None
            ris_specs = None

            st.subheader("Workstation Specifications (Ultima Viewer)")

            # Diagnostic Workstation
            diagnostic_specs = pd.DataFrame({
                "Item": [
                    "CPU", "Memory", "Disk Space", "External Video Card", "OS", "Additional"
                ],
                "Description": [
                    "64-bit, Min 6 Cores (Recommended: 12 cores, Intel Xeon ≥ 2.2 GHz)",
                    "Min 16 GB (Recommended: 32 GB)",
                    "200 GB SSD",
                    "Required** (see medical monitor specs)",
                    "Windows 10 Pro 64-bit or newer",
                    "MS Word 2019 or MS Word 365"
                ]
            })
            st.markdown("### Diagnostic Workstation")
            st.dataframe(diagnostic_specs)

            # Review / Technologist Workstation
            review_specs = pd.DataFrame({
                "Item": ["CPU", "Memory", "Disk Space", "OS"],
                "Description": [
                    "64-bit Intel Core i7 (≥ 4 cores) or i9 for enhanced performance",
                    "Min 8 GB (Recommended: 16 GB)",
                    "150 GB SSD",
                    "Windows 10 Pro 64-bit or newer"
                ]
            })
            st.markdown("### Review / Technologist Workstation")
            st.dataframe(review_specs)

            # Clinician Workstation
            clinician_specs = pd.DataFrame({
                "Item": ["CPU", "Memory", "Disk Space", "External Video Card", "OS"],
                "Description": [
                    "64-bit Intel Core i7 (≥ 4 cores) or i9 for enhanced performance",
                    "Min 8 GB (Recommended: 16 GB)",
                    "150 GB SSD",
                    "Required** (see medical monitor specs)",
                    "Windows 10 Pro 64-bit or newer"
                ]
            })
            st.markdown("### Clinician Workstation")
            st.dataframe(clinician_specs)

            # RIS Workstation (only if RIS is enabled)
            if ris_enabled:
                ris_specs = pd.DataFrame({
                    "Item": ["CPU", "Memory", "Disk Space", "OS", "Additional"],
                    "Description": [
                        "64-bit Intel Core i5 or higher (≥ 4 cores)",
                        "Min 8 GB",
                        "150 GB SSD",
                        "Windows 10 Pro 64-bit or newer",
                        "MS Office (for scheduling and reporting integration)"
                    ]
                })
                st.markdown("### RIS Workstation")
                st.dataframe(ris_specs)

            # Notes Section
            st.markdown("""
            **📝 Notes:**  
            - **External Video Card:** Required for medical monitor connection (single or dual). Follow vendor specs.  
            - **Medical Monitors:** At least 3MP for general radiology, 5MP for mammography, 2MP for clinical review.  
            - **Network:** Minimum 1× 1Gbps NIC  
            - **Additional Applications:** Verify minimum specs of any 3rd-party applications integrated with Ultima Viewer.
            """)
        workstation_notes = (
            "- **External Video Card:** Required for medical monitor connection (single or dual). Follow vendor specs.\n"
            "- **Medical Monitors:** At least 3MP for general radiology, 5MP for mammography, 2MP for clinical review.\n"
            "- **Network:** Minimum 1× 1Gbps NIC\n"
            "- **Additional Applications:** Verify minimum specs of any 3rd-party applications integrated with Ultima Viewer."
        )
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
            server_specs=server_specs,
            gpu_specs=gpu_specs,
            first_year_storage_raid5=first_year_storage_raid5,
            total_image_storage_raid5=total_image_storage_raid5,
            num_studies=num_studies,
            storage_title=storage_type,
            shared_storage=shared_storage,
            raid_1_storage_tb=raid_1_storage_tb,
            gateway_specs=gateway_specs,
            diagnostic_specs=diagnostic_specs,  # Diagnostic Workstation (new format)
            review_specs=review_specs,  # Technologist / Review Workstation
            clinician_specs=clinician_specs,  # Clinician Workstation
            ris_specs=ris_specs if ris_enabled else None,  # RIS Workstation (only if enabled)
            workstation_notes=workstation_notes,
            project_grade=project_grade,
            storage_table=storage_table,
            physical_design=physical_design_string,
            nas_backup_details=nas_backup_string,
            tier_2_disks=tier_2_disks,
            tier_2_disk_size=tier_2_disk_size,
            tier_3_disks=tier_3_disks,
            tier_3_disk_size=tier_3_disk_size,
            additional_vm_table=additional_vms_table,
            additional_vm_notes=additional_vm_notes,
            general_notes=general_notes,
            additional_servers=additional_servers,
            additional_vms=additional_vms,
            additional_requirements_table=additional_requirements_table
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
