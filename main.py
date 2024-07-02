import os
import time
import logging
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

    threads_to_allocate = min(remaining_threads, max_threads)
    ram_to_allocate = min(remaining_ram, max_ram)

    if threads_to_allocate % 2 != 0:
        threads_to_allocate -= 1

    if threads_to_allocate > 128:
        processors_to_allocate = 'Dual'
        cores_per_processor = min((threads_to_allocate // 4), 64)
        total_cores = cores_per_processor * 2
        total_threads = total_cores * 2
    else:
        if threads_to_allocate > 20:
            processors_to_allocate = 'Dual'
            cores_per_processor = min((threads_to_allocate // 4), 64)
            total_cores = cores_per_processor * 2
            total_threads = total_cores * 2
        else:
            processors_to_allocate = 'Single'
            cores_per_processor = min((threads_to_allocate // 2), 64)
            total_cores = cores_per_processor
            total_threads = total_cores * 2

    servers.append({
        "Processors": processors_to_allocate,
        "Total Cores": total_cores,
        "Total Threads": total_threads,
        "Cores per Processor": cores_per_processor,
        "Threads per Processor": total_threads // (2 if processors_to_allocate == 'Dual' else 1),
        "RAM": ram_to_allocate
    })

    logging.info(
        f"Allocated Server: Processors={processors_to_allocate}, Total Cores={total_cores}, Total Threads={total_threads}, RAM={ram_to_allocate} GB")
    print(
        f"Allocated Server: Processors={processors_to_allocate}, Total Cores={total_cores}, Total Threads={total_threads}, RAM={ram_to_allocate} GB")

    return remaining_threads - total_threads, remaining_ram - ram_to_allocate


def format_server_specs(servers):
    server_specs = ""
    for i, server in enumerate(servers):
        server_specs += f"""
        **Server {i + 1}:**
          - Processors: {server['Processors']}
          - Total CPU: {server['Total Cores']} Cores / {server['Total Threads']} Threads
          - Per Processor: {server['Cores per Processor']} Cores / {server['Threads per Processor']} Threads
          - RAM: {server['RAM']} GB
        """
    return server_specs


def calculate_raid5_disks(usable_storage_tb):
    standard_disk_sizes_tb = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    for disk_size in standard_disk_sizes_tb:
        total_disks = (usable_storage_tb / disk_size) + 1
        if total_disks >= 3 and total_disks.is_integer():
            total_disks = int(total_disks)
            return total_disks, disk_size
    return None, None


def round_to_nearest_divisible_by_two(value):
    return round(value / 2) * 2


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

    with st.expander("Contract and Study Details"):
        breakdown_per_modality = st.radio("Breakdown per Modality?", ["Yes", "No"])
        if breakdown_per_modality == "No":
            num_studies = st.number_input("Number of studies per year:", min_value=0, value=100000)
            modality_cases = {}
        else:
            st.subheader("Modality Breakdown:")
            modality_cases = {
                "CT": st.number_input("CT Cases:", min_value=0),
                "MR": st.number_input("MR Cases:", min_value=0),
                "US": st.number_input("US Cases:", min_value=0),
                "NM": st.number_input("NM Cases:", min_value=0),
                "X-ray": st.number_input("X-ray Cases:", min_value=0),
                "MG": st.number_input("MG Cases:", min_value=0),
                "Cath": st.number_input("Cath Cases:", min_value=0),
            }
            num_studies = sum(modality_cases.values())

        contract_duration = st.number_input("Contract Duration (years):", min_value=1, value=3)
        study_size_mb = st.number_input("Study Size (MB):", min_value=0, value=100)
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

    with st.expander("Location and Machine Details"):
        num_locations = st.number_input("Number of Locations:", min_value=1, value=1)
        location_details = []
        if num_locations > 1:
            st.subheader("Interconnection Details for Additional Locations")
            for i in range(2, num_locations + 1):
                location_type = st.selectbox(
                    f"Select interconnection type for Location {i}",
                    ["Uploader HW Only", "Business Continuity Mini PACS", "Complete PACS Solution", "Isolated"]
                )
                location_details.append(location_type)
        num_machines = st.number_input("Number of Machines (Modalities):", min_value=1, value=1)

    with st.expander("CCU Details"):
        pacs_enabled = st.checkbox("Include PACS")
        if pacs_enabled:
            pacs_ccu = st.number_input("PACS CCU:", min_value=0, value=8)
        else:
            pacs_ccu = 0

        ris_enabled = st.checkbox("Include RIS")
        if ris_enabled:
            ris_ccu = st.number_input("RIS CCU:", min_value=0, value=8)
        else:
            ris_ccu = 0

        ref_phys_enabled = st.checkbox("Include Referring Physician")
        if ref_phys_enabled:
            ref_phys_ccu = st.number_input("Referring Physician CCU:", min_value=0, value=8)
            ref_phys_external_access = st.checkbox("External Access for Referring Physician Portal")
        else:
            ref_phys_ccu = 0
            ref_phys_external_access = False

    with st.expander("Broker Details"):
        broker_required = st.checkbox("Broker VM Required (check if explicitly requested)", value=False)
        if broker_required:
            broker_level = st.radio("Broker Level:", ["WL", "HL7 Unidirectional", "HL7 Bidirectional"], index=0)
        else:
            broker_level = "WL"

    with st.expander("Additional Features"):
        aidocker_included = st.checkbox(
            "Include U9th Integrated AI modules (Auto Segmentation, Spine Labeling, etc.)",
            value=False,
            disabled=(num_studies == 0)
        )

        ark_included = st.checkbox(
            "Include ARKAI",
            value=False,
            disabled=(num_studies == 0)
        )
        high_availability = st.checkbox("High Availability HW Design Required", value=False)
        training_vm_included = st.checkbox("Include Testing/Training VM", value=False)

    if aidocker_included and ark_included:
        aidocker_included = False

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
                num_studies, pacs_ccu, ris_ccu, ref_phys_ccu, project_grade, broker_required, broker_level,
                num_machines,
                contract_duration, study_size_mb, annual_growth_rate, aidocker_included=aidocker_included,
                ark_included=ark_included, ref_phys_external_access=ref_phys_external_access,
                training_vm_included=training_vm_included, high_availability=high_availability, **modality_cases
            )
            logging.info(f"VM Requirements Calculation completed in {time.time() - start_calc_time:.2f} seconds")

            if not results.empty:
                results["Operating System"] = "Windows Server 2019 or Higher"
                results["Other Software"] = ""

                for index in results.index:
                    if "TestVM" in index:
                        results.at[index, "Other Software"] = sql_license
                    if "DBServer" in index:
                        results.at[index, "Other Software"] = sql_license
                    if "AISegmentationDocker" in index:
                        results.at[index, "Operating System"] = "Ubuntu 20.4"
                        results.at[
                            index, "Other Software"] = "Nvidia Driver version 450.80.02 or higher\nNvidia driver to support CUDA version 11.4 or higher"
                    if "AIARKLAB01" in index:
                        results.at[index, "Operating System"] = "Ubuntu 20.4"
                        results.at[index, "Other Software"] = "RTX 4080 / RTX 4090 Video Cards"

            # Add Test Environment VM
            if training_vm_included:
                test_vm_specs = {
                    "VM Type": "Test Environment VM",
                    "vCores": 0,
                    "RAM (GB)": 0,
                    "Storage (GB)": 150
                }
                if sql_license == "SQL Express":
                    test_vm_specs["vCores"] = 8
                    test_vm_specs["RAM (GB)"] = 16
                elif sql_license == "SQL Standard":
                    test_vm_specs["vCores"] = 10
                    test_vm_specs["RAM (GB)"] = 32
                elif sql_license == "SQL Enterprise":
                    test_vm_specs["vCores"] = 12
                    test_vm_specs["RAM (GB)"] = 64

                test_vm_name = "Test Environment VM (Ultima, PACS, Broker)"


            # Add Management VM for High Availability
            if high_availability:
                management_vm_specs = {
                    "VM Type": "Management VM",
                    "vCores": 8,
                    "RAM (GB)": 32 if sql_license in ["SQL Express", "SQL Standard"] else 64,
                    "Storage (GB)": 150
                }
                management_vm_name = "Management VM (Backup, Antivirus, vCenter)"
            display_results = results.drop(["RAID 1 (SSD)", "RAID 5 (HDD)"])

            last_index = display_results.tail(1).index
            display_results.loc[last_index, "Operating System"] = ""
            display_results.loc[last_index, "Other Software"] = ""

            st.subheader("VM Recommendations:")
            st.dataframe(display_results.style.apply(
                lambda x: ['background-color: yellow' if 'Test Environment VM' in x.name else '' for i in x], axis=1).format(precision=2))

            results_grade1, _, first_year_storage_raid5_grade1, total_image_storage_raid5_grade1, total_vcpu_grade1, total_ram_grade1, total_storage_grade1 = calculate_vm_requirements(
                num_studies, pacs_ccu, ris_ccu, ref_phys_ccu, 1, broker_required, broker_level, num_machines,
                contract_duration, study_size_mb, annual_growth_rate, breakdown_per_modality,
                aidocker_included=aidocker_included, ark_included=ark_included,
                ref_phys_external_access=ref_phys_external_access, training_vm_included=training_vm_included,
                **modality_cases
            )

            results_grade3, _, first_year_storage_raid5_grade3, total_image_storage_raid5_grade3, total_vcpu_grade3, total_ram_grade3, total_storage_grade3 = calculate_vm_requirements(
                num_studies, pacs_ccu, ris_ccu, ref_phys_ccu, 3, broker_required, broker_level, num_machines,
                contract_duration, study_size_mb, annual_growth_rate, breakdown_per_modality,
                aidocker_included=aidocker_included, ark_included=ark_included,
                ref_phys_external_access=ref_phys_external_access, training_vm_included=training_vm_included,
                **modality_cases
            )

            raid_1_storage_tb = results.loc["RAID 1 (SSD)", "Storage (GB)"] / 1024
            raid_5_storage_tb = results.loc["RAID 5 (HDD)", "Storage (GB)"] / 1024

            st.subheader("Storage Requirements:")
            st.markdown(f"**RAID 1 Storage:** {raid_1_storage_tb:.2f} TB")
            st.markdown(f"**RAID 5 Storage (First Year):** {first_year_storage_raid5 / 1024:.2f} TB")
            st.markdown(f"**RAID 5 Storage (Full Contract Duration):** {total_image_storage_raid5 / 1024:.2f} TB")

            comparison_data = {
                "Specification": ["Total vCores", "Total RAM (GB)", "RAID 1 (SSD) (TB)", "RAID 5 (HDD) 1 Year (TB)", "RAID 5 (HDD) Full Duration (TB)"],
                "Minimum Specs": [round(total_vcpu_grade1, 2), round(total_ram_grade1, 2), round(raid_1_storage_tb, 2), round(first_year_storage_raid5_grade1 / 1024, 2), round(total_image_storage_raid5_grade1 / 1024, 2)],
                "Recommended Specs": [round(total_vcpu_grade3, 2), round(total_ram_grade3, 2), round(raid_1_storage_tb, 2), round(first_year_storage_raid5_grade3 / 1024, 2), round(total_image_storage_raid5_grade3 / 1024, 2)]
            }
            df_comparison = pd.DataFrame(comparison_data)

            st.subheader("Minimum vs. Recommended Resources:")
            st.dataframe(df_comparison.style.format(precision=2))

            logging.info("Starting Physical Server Allocation")
            total_vcpu = results.loc["Total", "vCores"]
            total_ram = results.loc["Total", "RAM (GB)"]

            max_threads_per_server = 128
            max_ram_per_server = 512

            # Adjust total resources for high availability if needed
            if high_availability:
                logging.info("Calculating high availability resources")
                total_vcpu_ha = int(total_vcpu * 0.75)
                total_ram_ha = int(total_ram * 0.75)

                servers = []
                remaining_threads, remaining_ram = total_vcpu_ha, total_ram_ha
                while remaining_threads > 0 or remaining_ram > 0:
                    if remaining_ram == 0:
                        logging.warning("No remaining RAM to allocate. Exiting loop.")
                        break
                    remaining_threads, remaining_ram = add_server(
                        servers, remaining_threads, remaining_ram, max_threads_per_server, max_ram_per_server
                    )

                # Duplicate the servers for high availability
                ha_servers = servers + servers
                server_specs = format_server_specs(ha_servers)
                st.subheader("High Availability Physical Server Design:")
                st.markdown(server_specs)

                logging.info("High availability server allocation complete")
            else:
                total_vcpu = results.loc["Total", "vCores"]
                total_ram = results.loc["Total", "RAM (GB)"]

                num_servers = (total_vcpu + max_threads_per_server - 1) // max_threads_per_server
                avg_ram_per_server = (total_ram + num_servers - 1) // num_servers

                servers = []
                remaining_threads, remaining_ram = total_vcpu, total_ram
                while remaining_threads > 0 or remaining_ram > 0:
                    if remaining_ram == 0:
                        logging.warning(f"No remaining RAM to allocate. Exiting loop.")
                        break
                    remaining_threads, remaining_ram = add_server(
                        servers, remaining_threads, remaining_ram, max_threads_per_server, avg_ram_per_server
                    )

                server_specs = format_server_specs(servers)
                st.subheader("Physical Server Design:")
                st.markdown(server_specs)

            raid_1_storage = results.loc["RAID 1 (SSD)", "Storage (GB)"]
            usable_storage_tb = round_to_nearest_divisible_by_two(total_image_storage_raid5 / 1024)
            total_disks, disk_size = calculate_raid5_disks(usable_storage_tb)
            hdd_storage_final = f"{total_disks}x {disk_size} TB (RAID 5)"

            shared_storage = f"""
            **Shared DAS/SAN Storage:**
              - SSD: 2x {raid_1_storage / 1024:.2f} TB (RAID 1)
              - HDD: {hdd_storage_final}
            """
            st.markdown(shared_storage)

            non_total_display_results = display_results[display_results.index != "Total"]
            windows_count = len(non_total_display_results)

            third_party_licenses = pd.DataFrame({
                "Item Description": [
                    sql_license,
                    "MS Windows Server Standard 2019 or higher",
                    "Antivirus Server License",
                    "VMware vSphere Essentials KIT" if not high_availability else "VMware vSphere Essentials Plus KIT",
                    "Backup Software"
                ],
                "Qty": [
                    2 if training_vm_included else 1,
                    windows_count,
                    windows_count,
                    1,
                    1
                ]
            })

            st.subheader("Third Party Licenses")
            st.dataframe(third_party_licenses.style.set_properties(subset=["Item Description"], **{'width': '300px'}))

            st.subheader("Sizing Notes:")
            st.markdown("""
            - Provided VM sizing of the Virtual servers will be scaled up horizontally or vertically based on the expected volume of data and number of CCUs.
            - SSD Datastore recommended for all OS Virtual disks of all VMs.
            - SSD recommended for the data drive of the Database Server.
            - It's recommended to have SSD storage for the short Term Storage (STS) that keep last 6 month of data for higher performance in data access.
            """)

            gpu_specs = ""
            if aidocker_included:
                st.subheader("GPU Requirements:")
                gpu_specs = """
                    - GPUs: 1
                    - GPU Memory: 32 GB
                    - Nvidia Tesla V100 or equivalent RTX (Preferred)
                    - Nvidia Driver version 450.80.02 or higher
                    - Nvidia driver to support CUDA version 11.4 or higher
                    """
            elif ark_included:
                st.subheader("GPU Requirements:")
                gpu_specs = """
                - GPUs: 3
                - GPU Memory: 32 GB
                -  2* Nvidia Tesla V100 or equivalent RTX (Preferred)(For Segmentation Dockers)
                - Nvidia Driver version 450.80.02 or higher
                - Nvidia driver to support CUDA version 11.4 or higher
                - RTX 4080 / RTX 4090 Video Cards ( For ARK LAB)
                    """
            st.markdown(gpu_specs)
            st.subheader("Technical Requirements:")
            st.markdown("""
            - Provide remote access to the VMs (all VMs) for SW installation and configurations.
            - In case not using dongles, a connection from the VMs to (https://paxaeraultima.net:1435) for PaxeraHealth licensing except the database VM.
            """)

            st.subheader("Network Requirements (LAN):")
            st.markdown("""
            - LAN bandwidth to be 1GB dedicated.
            - Paxera PACS VMs, Paxera Ultima Viewer VMs, Paxera Broker Integration VMs must be in same vLAN.
            - 1 Gb/s dedicated bandwidth across the vLAN with the maximum number of two network hops.
            - Latency less than 1ms.
            """)

            notes = {
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
                "network_requirements": """
                        - LAN bandwidth to be 1GB dedicated.
                        - Paxera PACS VMs, Paxera Ultima Viewer VMs, Paxera Broker Integration VMs must be in same vLAN.
                        - 1 Gb/s dedicated bandwidth across the vLAN with the maximum number of two network hops.
                        - Latency less than 1ms.
                        """
            }

            input_values = pd.DataFrame({
                "Input": ["PACS CCU", "RIS CCU", "Referring Physician CCU", "Project Grade",
                          "Broker VM Required", "Contract Duration (years)", "Study Size (MB)",
                          "Annual Growth Rate (%)", "Number of Locations", "Number of Machines"],
                "Value": [pacs_ccu, ris_ccu, ref_phys_ccu, project_grade, broker_required,
                          contract_duration, study_size_mb, annual_growth_rate, num_locations, num_machines]
            })

            if not results.empty:
                doc = generate_document_from_template(
                    os.path.join(app_dir, "assets", "templates", "Temp.docx"),
                    results,
                    results_grade1,
                    results_grade3,
                    df_comparison,
                    third_party_licenses,
                    notes, input_values,
                    customer_name=customer_name,
                    high_availability=high_availability,
                    server_specs=server_specs,
                    gpu_specs=gpu_specs,
                    first_year_storage_raid5=first_year_storage_raid5,
                    total_image_storage_raid5=total_image_storage_raid5,
                    num_studies=num_studies
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
