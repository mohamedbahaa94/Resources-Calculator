import pandas as pd

modality_sizes = {
    "CT": 1.0240,
    "MR": 0.100,
    "US": 0.010,
    "NM": 0.010,
    "X-ray": 0.030,
    "MG": 0.16,
    "Cath": 0.300
}

def calculate_referring_physician_resources(ref_phys_ccu):
    ccu_thresholds = [8, 16, 24, 32, 48, 64]
    ram_gb_tiers = [8, 16, 24, 32, 48, 64]
    vcores_tiers = [4, 6, 8, 10, 10, 12]
    max_ccu_per_vm = 64

    if ref_phys_ccu <= max_ccu_per_vm:
        for i, threshold in enumerate(ccu_thresholds):
            if ref_phys_ccu <= threshold:
                return [(1, ram_gb_tiers[i], vcores_tiers[i])]
    else:
        num_full_vms = ref_phys_ccu // max_ccu_per_vm
        remaining_ccu = ref_phys_ccu % max_ccu_per_vm

        vm_configs = []

        for _ in range(num_full_vms):
            vm_configs.append((1, ram_gb_tiers[-1], vcores_tiers[-1]))

        if remaining_ccu > 0:
            for i, threshold in enumerate(ccu_thresholds):
                if remaining_ccu <= threshold:
                    vm_configs.append((1, ram_gb_tiers[i], vcores_tiers[i]))
                    break

        return vm_configs

def calculate_vm_specifications(vm_type, num_vms, pacs_ccu, ris_ccu, ref_phys_ccu, project_grade):
    vm_specs = {
        "PaxeraUltima": {"vcores": 8, "base_ram": 8, "storage_gb": 150},
        "PaxeraPACS": {"vcores": 8, "base_ram": 16, "storage_gb": 150},
        "PaxeraBroker": {"vcores": 8, "base_ram": 16, "storage_gb": 150},
        "PaxeraRIS": {"vcores": 8, "base_ram": 8, "storage_gb": 150},
        "DBServer": {"vcores": 12, "base_ram": 32, "storage_gb": 400},
        "Referring Physician": {"vcores": 8, "base_ram": 8, "storage_gb": 150},
    }

    vm_requirements = {}
    max_ccu_per_vm = 32

    def get_vm_specs(ccu, base_ram, base_vcores):
        if ccu <= 2:
            ram = 8
            vcores = 6
        else:
            ram = base_ram + ((ccu - 2) * 2)
            vcores = base_vcores + ((ccu - 2) // 2) * 2
        return vcores, ram

    if vm_type in ["PaxeraUltima", "PaxeraRIS"]:
        ccu = pacs_ccu if vm_type == "PaxeraUltima" else ris_ccu
        vm_count = (ccu + max_ccu_per_vm - 1) // max_ccu_per_vm
        for i in range(vm_count):
            vm_name = f"{vm_type}0{i + 1}"
            ccu_for_vm = min(max_ccu_per_vm, ccu - i * max_ccu_per_vm)
            vcores, ram = get_vm_specs(ccu_for_vm, vm_specs[vm_type]["base_ram"], vm_specs[vm_type]["vcores"])
            vm_requirements[vm_name] = {
                "VM Type": vm_type,
                "vCores": vcores,
                "RAM (GB)": ram,
                "Storage (GB)": vm_specs[vm_type]["storage_gb"],
            }
    else:
        for i in range(num_vms):
            vm_name = f"{vm_type}0{i + 1}"
            base_ram = vm_specs[vm_type].get("base_ram", 32)
            vm_requirements[vm_name] = {
                "VM Type": vm_type,
                "vCores": vm_specs[vm_type]["vcores"],
                "RAM (GB)": base_ram,
                "Storage (GB)": vm_specs[vm_type]["storage_gb"],
            }

    return vm_requirements

def calculate_dbserver_resources(num_studies, ris_ccu=0):
    max_studies_single_vm = 300000
    db_tiers = {
        5000: (8, 16),
        50000: (10, 32),
        300000: (12, 64)
    }

    # If number of studies is zero, base the calculation on ris_ccu
    if num_studies == 0 and ris_ccu > 0:
        num_studies = ris_ccu * 1250  # Assumption: 1 ris_ccu ~ 1250 studies

    if num_studies <= max_studies_single_vm:
        prev_studies_threshold = 0
        prev_base_vcores, prev_base_ram_gb = 0, 0
        for studies_threshold, (base_vcores, base_ram_gb) in db_tiers.items():
            if num_studies <= studies_threshold:
                if studies_threshold != 5000:
                    scaling_factor = (num_studies - prev_studies_threshold) / (studies_threshold - prev_studies_threshold)
                else:
                    scaling_factor = num_studies / studies_threshold
                ram_gb = prev_base_ram_gb + (base_ram_gb - prev_base_ram_gb) * scaling_factor
                vcores = prev_base_vcores + (base_vcores - prev_base_vcores) * scaling_factor
                return [(1, int(round(ram_gb)), int(round(vcores)))]
            prev_studies_threshold, prev_base_vcores, prev_base_ram_gb = studies_threshold, base_vcores, base_ram_gb
    else:
        vm_configs = []
        remaining_studies = num_studies
        while remaining_studies > 0:
            vm_studies = min(max_studies_single_vm, remaining_studies)
            remaining_studies -= vm_studies

            for studies_threshold, (base_vcores, base_ram_gb) in db_tiers.items():
                if vm_studies <= studies_threshold:
                    config = (1, base_ram_gb, base_vcores)
                    if config not in vm_configs:
                        vm_configs.append(config)
                    break

        return vm_configs

def calculate_paxera_ultima_resources(pacs_ccu):
    ccu_thresholds = [4, 8, 12,16, 24, 32]
    ram_gb_tiers = [8, 16, 24, 32,48, 64]
    vcores_tiers = [4, 6, 8, 10,10, 12]
    max_ccu_per_vm = 32

    if pacs_ccu <= max_ccu_per_vm:
        for i, threshold in enumerate(ccu_thresholds):
            if pacs_ccu <= threshold:
                return [(1, ram_gb_tiers[i], vcores_tiers[i])]
    else:
        num_full_vms = pacs_ccu // max_ccu_per_vm
        remaining_ccu = pacs_ccu % max_ccu_per_vm

        vm_configs = []

        for _ in range(num_full_vms):
            vm_configs.append((1, ram_gb_tiers[-1], vcores_tiers[-1]))

        if remaining_ccu > 0:
            for i, threshold in enumerate(ccu_thresholds):
                if remaining_ccu <= threshold:
                    vm_configs.append((1, ram_gb_tiers[i], vcores_tiers[i]))
                    break

        return vm_configs

def calculate_paxera_pacs_resources(num_studies):
    min_ram_gb = 14
    max_ram_gb = 48
    min_vcores = 8
    max_vcores = 14
    max_studies_per_vm = 200000

    if num_studies <= max_studies_per_vm:
        scaling_factor = num_studies / max_studies_per_vm
        ram_gb = min_ram_gb + (max_ram_gb - min_ram_gb) * scaling_factor
        vcores = min_vcores + (max_vcores - min_vcores) * scaling_factor
        return [(1, int(round(ram_gb)), int(round(vcores)))]
    else:
        num_full_vms = num_studies // max_studies_per_vm
        remaining_studies = num_studies % max_studies_per_vm

        vm_configs = []

        for _ in range(num_full_vms):
            vm_configs.append((1, max_ram_gb, max_vcores))

        if remaining_studies > 0:
            scaling_factor = remaining_studies / max_studies_per_vm
            ram_gb = min_ram_gb + (max_ram_gb - min_ram_gb) * scaling_factor
            vcores = min_vcores + (max_vcores - min_vcores) * scaling_factor
            vm_configs.append((1, int(round(ram_gb)), int(round(vcores))))

        return vm_configs

def calculate_vm_requirements(
    num_studies,
    pacs_ccu,
    ris_ccu,
    ref_phys_ccu,
    project_grade,
    broker_required,
    broker_level,
    num_machines,
    contract_duration,
    study_size_mb,
    annual_growth_rate,
    breakdown_per_modality=False,
    aidocker_included=False,
    ark_included=False,
    u9_ai_features=None,  # Updated: Add dynamic U9.AI feature handling
    ref_phys_external_access=False,
    patient_portal_ccu=0,
    patient_portal_external_access=False,
    training_vm_included=False,
    high_availability=False,
    combine_pacs_ris=False,
    **modality_cases
):
    """
    Calculates the VM requirements for a medical imaging solution.
    Incorporates dynamic handling of AI features and other system components.
    """

    vm_specs = {
        "PaxeraUltima": {"vcores": 8, "base_ram": 8, "storage_gb": 150},
        "PaxeraPACS": {"vcores": 8, "base_ram": 16, "storage_gb": 150},
        "PaxeraRIS": {"vcores": 8, "base_ram": 8, "storage_gb": 150},
        "DBServer": {"vcores": 12, "base_ram": 32, "storage_gb": 400},
        "Referring Physician": {"vcores": 8, "base_ram": 8, "storage_gb": 150},
    }

    vms_needed = {
        "PaxeraUltima": 1 if pacs_ccu > 0 else 0,
        "DBServer": 1,
        "PaxeraBroker": 1 if broker_required else 0,
        "PaxeraRIS": 1 if ris_ccu > 0 else 0,
        "Referring Physician": 1 if ref_phys_ccu > 0 and ref_phys_external_access else 0,
        "Patient Portal": 1 if patient_portal_ccu > 0 and patient_portal_external_access else 0,  # Add this line
    }

    if breakdown_per_modality:
        total_cases = sum(modality_cases.values())
        total_storage = sum(modality_cases[modality] * modality_sizes.get(modality, 0) for modality in modality_cases)
        average_study_size = total_storage / total_cases if total_cases > 0 else 0
        image_storage_raid5_modality = sum(modality_cases[modality] * modality_sizes.get(modality, 0) * contract_duration for modality in modality_cases)
    else:
        total_cases = num_studies
        average_study_size = study_size_mb
        image_storage_raid5_modality = num_studies * average_study_size * contract_duration * (1 + annual_growth_rate / 100)

    vm_requirements = {}
    vm_config_lists = {
        "PaxeraUltima": calculate_paxera_ultima_resources(pacs_ccu) if pacs_ccu > 0 else [],
        "PaxeraRIS": calculate_paxera_ultima_resources(ris_ccu) if ris_ccu > 0 else [],
        "Referring Physician": calculate_referring_physician_resources(
            ref_phys_ccu) if ref_phys_ccu > 0 and ref_phys_external_access else [],
        "Patient Portal": calculate_referring_physician_resources(
            patient_portal_ccu) if patient_portal_ccu > 0 and patient_portal_external_access else []  # Add this line
    }

    for vm_type, config_list in vm_config_lists.items():
        for i, (num_vms, ram_gb, vcores) in enumerate(config_list):
            vm_name = f"{vm_type}{i + 1:02d}"
            vm_requirements[vm_name] = {
                "VM Type": vm_type,
                "vCores": vcores,
                "RAM (GB)": ram_gb,
                "Storage (GB)": 150,
            }

    dbserver_vm_configs = calculate_dbserver_resources(num_studies,ris_ccu)
    _, ram_gb, vcores = dbserver_vm_configs[0]
    vm_requirements["DBServer01"] = {
        "VM Type": "DBServer",
        "vCores": vcores,
        "RAM (GB)": ram_gb,
        "Storage (GB)": 400
    }

    if num_studies > 0:
        pacs_vm_configs = calculate_paxera_pacs_resources(num_studies)
        for i, (num_vms, ram_gb, vcores) in enumerate(pacs_vm_configs):
            vm_name = f"PaxeraPACS{i + 1:02d}"
            vm_requirements[vm_name] = {
                "VM Type": vm_name,
                "vCores": vcores,
                "RAM (GB)": ram_gb,
                "Storage (GB)": 150,
            }
    if combine_pacs_ris and pacs_ccu > 0 and ris_ccu > 0:
        # Filter all PaxeraUltima and PaxeraRIS VMs
        ultima_vms = [vm for vm in vm_requirements if vm.startswith("PaxeraUltima")]
        ris_vms = [vm for vm in vm_requirements if vm.startswith("PaxeraRIS")]

        combined_vms = []
        num_combined_vms = max(len(ultima_vms), len(ris_vms))  # Take the larger number of VMs

        for i in range(num_combined_vms):
            # Handle combining corresponding VMs or use only one if the other is missing
            ultima_vm = vm_requirements.get(ultima_vms[i]) if i < len(ultima_vms) else None
            ris_vm = vm_requirements.get(ris_vms[i]) if i < len(ris_vms) else None

            if ultima_vm and ris_vm:
                # Combine the corresponding VMs, taking the maximum of their specs
                combined_vm = {
                    "VM Type": "PaxeraUltima/PaxeraRIS",
                    "vCores": max(ultima_vm["vCores"], ris_vm["vCores"]),
                    "RAM (GB)": max(ultima_vm["RAM (GB)"], ris_vm["RAM (GB)"]),
                    "Storage (GB)": max(ultima_vm["Storage (GB)"], ris_vm["Storage (GB)"]),
                }
            elif ultima_vm:
                # Use only the Ultima VM specs if RIS VM is missing
                combined_vm = ultima_vm.copy()
                combined_vm["VM Type"] = "PaxeraUltima/PaxeraRIS"
            elif ris_vm:
                # Use only the RIS VM specs if Ultima VM is missing
                combined_vm = ris_vm.copy()
                combined_vm["VM Type"] = "PaxeraUltima/PaxeraRIS"

            combined_vms.append(combined_vm)

        # Remove the original Ultima and RIS VMs
        for vm in ultima_vms + ris_vms:
            vm_requirements.pop(vm, None)

        # Add the combined VMs to the requirements
        for i, combined_vm in enumerate(combined_vms, start=1):
            combined_vm_name = f"PaxeraUltima/RIS{i:02d}"  # Ensure consistent naming with incrementing numbers
            vm_requirements[combined_vm_name] = combined_vm

        # Disable the separate VM types in `vms_needed`
        vms_needed["PaxeraUltima"] = 0
        vms_needed["PaxeraRIS"] = 0

    # Broker VM Logic with Project Grades
    # Broker VM Logic with Project Grades
    if broker_required or (pacs_ccu > 0 and ris_ccu > 0):
        # Determine the number of modalities
        modalities = num_machines if broker_required else num_machines

        # Calculate the number of Broker VMs (1 Broker per 10 modalities)
        broker_vms = (modalities + 9) // 10

        for i in range(broker_vms):
            vm_name = f"PaxeraBroker{i + 1:02d}"

            # Define VM specs based on broker level
            if broker_level == "HL7 Unidirectional":
                vm_type = "PaxeraBroker HL7 Unidirectional"
                vcores = 8
                ram_gb = 16
            elif broker_level == "HL7 Bidirectional":
                vm_type = "PaxeraBroker HL7 Bidirectional"
                vcores = 10
                ram_gb = 24
            else:
                vm_type = "PaxeraBroker WL"
                vcores = 6
                ram_gb = 12

            storage_gb = 150  # Fixed storage per broker

            if project_grade in [1, 2] and i == 0 and "DBServer01" in vm_requirements:
                # Combine first broker with DBServer in Grade 1 & 2
                db_vcores = vm_requirements["DBServer01"]["vCores"]
                db_ram = vm_requirements["DBServer01"]["RAM (GB)"]
                db_storage = vm_requirements["DBServer01"]["Storage (GB)"]

                combined_vcores = min(12, 2 * round((db_vcores + vcores) / 1.2 / 2))
                combined_ram = min(64, 2 * round((db_ram + ram_gb) / 1.2 / 2))
                combined_storage = db_storage + storage_gb

                # Update DBServer to include broker resources with explicit broker type
                vm_requirements["DBServer01"] = {
                    "VM Type": f"DBServer/Broker ({vm_type})",  # Explicitly include the Broker type
                    "vCores": combined_vcores,
                    "RAM (GB)": combined_ram,
                    "Storage (GB)": combined_storage,
                }
            else:
                # Add as standalone Broker VM
                vm_requirements[vm_name] = {
                    "VM Type": vm_type,
                    "vCores": vcores,
                    "RAM (GB)": ram_gb,
                    "Storage (GB)": storage_gb,
                }

    # Referring Physician VM Logic
    if not ref_phys_external_access and ref_phys_ccu > 0:
        ref_phys_vm_resources = calculate_referring_physician_resources(ref_phys_ccu)
        max_vcores_per_vm = 12  # Example threshold for vCores
        max_ram_per_vm = 64  # Example threshold for RAM in GB

        for num_vms, ram_gb, vcores in ref_phys_vm_resources:
            if combine_pacs_ris:
                if "PaxeraUltima/RIS01" in vm_requirements:
                    current_vcores = vm_requirements["PaxeraUltima/RIS01"]["vCores"]
                    current_ram = vm_requirements["PaxeraUltima/RIS01"]["RAM (GB)"]
                    new_vcores = current_vcores + vcores
                    new_ram = current_ram + ram_gb

                    if new_vcores > max_vcores_per_vm or new_ram > max_ram_per_vm:
                        new_vm_name = f"PaxeraUltima/RIS{len(vm_requirements) + 1:02d}"
                        vm_requirements[new_vm_name] = {
                            "VM Type": "PaxeraUltima/RIS",
                            "vCores": vcores,
                            "RAM (GB)": ram_gb,
                            "Storage (GB)": 150,
                        }
                    else:
                        vm_requirements["PaxeraUltima/RIS01"]["vCores"] = new_vcores
                        vm_requirements["PaxeraUltima/RIS01"]["RAM (GB)"] = new_ram
                else:
                    vm_requirements["PaxeraUltima/RIS01"] = {
                        "VM Type": "PaxeraUltima/RIS",
                        "vCores": vcores,
                        "RAM (GB)": ram_gb,
                        "Storage (GB)": 150,
                    }

    # Project Grades Logic
    if project_grade == 1:
        if "PaxeraPACS01" in vm_requirements and "PaxeraUltima01" in vm_requirements:
            combined_vm = {
                "VM Type": "PaxeraPACS/PaxeraUltima",
                "vCores": 2 * round((vm_requirements["PaxeraPACS01"]["vCores"] + vm_requirements["PaxeraUltima01"][
                    "vCores"]) / 1.2 / 2),
                "RAM (GB)": 2 * round((vm_requirements["PaxeraPACS01"]["RAM (GB)"] + vm_requirements["PaxeraUltima01"][
                    "RAM (GB)"]) / 1.2 / 2),
                "Storage (GB)": vm_requirements["PaxeraPACS01"]["Storage (GB)"]
            }
            vm_requirements["PaxeraPACS01"] = combined_vm
            del vm_requirements["PaxeraUltima01"]

    elif project_grade == 2:
        if broker_required and "DBServer01" in vm_requirements:
            combined_vm = {
                "VM Type": f"DBServer/Broker ({vm_type})",
                "vCores": min(12, 2 * round((vm_requirements["DBServer01"]["vCores"] + 8) / 1.5 / 2)),
                "RAM (GB)": min(64, 2 * round((vm_requirements["DBServer01"]["RAM (GB)"] + 16) / 1.5 / 2)),
                "Storage (GB)": vm_requirements["DBServer01"]["Storage (GB)"] + 150,
            }
            vm_requirements["DBServer01"] = combined_vm

    elif project_grade == 3:
        for vm_name, specs in list(vm_requirements.items()):
            if specs["VM Type"] != "PaxeraPACS":
                specs["vCores"] = 2 * round(specs["vCores"] * 1 / 2)
                specs["RAM (GB)"] = 2 * round(specs["RAM (GB)"] * 1 / 2)

        if broker_required:
            for i in range(broker_vms):
                vm_name = f"PaxeraBroker{i + 1:02d}"
                if vm_name in vm_requirements:
                    broker_specs = vm_requirements[vm_name].copy()
                    broker_specs["VM Type"] = vm_type
                    vm_requirements[vm_name] = broker_specs

    df_results = pd.DataFrame()
    total_image_storage_raid5 = 0
    first_year_storage_raid5 = (num_studies * study_size_mb * (1 + annual_growth_rate / 100)) / 1024
    current_studies = num_studies
    for year in range(contract_duration):
        image_storage_raid5 = (current_studies * study_size_mb * (1 + annual_growth_rate / 100)) / 1024
        total_image_storage_raid5 += image_storage_raid5
        current_studies *= (1 + annual_growth_rate / 100)
    total_image_storage_raid5 = round(total_image_storage_raid5, 2)
    first_year_storage_raid5 = round(first_year_storage_raid5, 2)

    total_vcpu = sum([vm_requirements[vm_name]["vCores"] for vm_name in vm_requirements])
    total_ram = sum([vm_requirements[vm_name]["RAM (GB)"] for vm_name in vm_requirements])
    total_storage = sum([vm_requirements[vm_name]["Storage (GB)"] for vm_name in vm_requirements])

    if vm_requirements:
        df_results = pd.DataFrame.from_dict(vm_requirements, orient='index')
        # Initialize AI Docker-specific VMs if U9.AI is included
        if aidocker_included and u9_ai_features:
            for feature in u9_ai_features:
                if feature == "Organ Segmentator":
                    vm_requirements["OrganSegmentator"] = {
                        "VM Type": "Organ Segmentator Docker",
                        "vCores": 12,
                        "RAM (GB)": 48,
                        "Storage (GB)": 100
                    }
                    total_vcpu += 12
                    total_ram += 48
                    total_storage += 100
                elif feature == "Lesion Segmentator 2D":
                    vm_requirements["LesionSegmentator2D"] = {
                        "VM Type": "Lesion Segmentator 2D Docker",
                        "vCores": 4,
                        "RAM (GB)": 8,
                        "Storage (GB)": 150
                    }
                    total_vcpu += 4
                    total_ram += 8
                    total_storage += 50
                elif feature == "Lesion Segmentator 3D":
                    vm_requirements["LesionSegmentator3D"] = {
                        "VM Type": "Lesion Segmentator 3D Docker",
                        "vCores": 6,
                        "RAM (GB)": 8,
                        "Storage (GB)":150
                    }
                    total_vcpu += 6
                    total_ram += 8
                    total_storage += 150
                elif feature == "Speech-to-text Container":
                    vm_requirements["SpeechToText"] = {
                        "VM Type": "Speech-to-text Docker",
                        "vCores": 4,
                        "RAM (GB)": 4,
                        "Storage (GB)": 150
                    }
                    total_vcpu += 4
                    total_ram += 4
                    total_storage += 150

        if ark_included:
            vm_requirements["AIARKManager01"] = {
                "VM Type": "AI ARK Manager",
                "vCores": 12,
                "RAM (GB)": 32,
                "Storage (GB)": 200
            }
            vm_requirements["AIARKLAB01"] = {
                "VM Type": "AI ARK LAB",
                "vCores": 12,
                "RAM (GB)": 32,
                "Storage (GB)": 300
            }
            total_vcpu += 24
            total_ram += 64
            total_storage += 500


            total_vcpu += 12
            total_ram += 32
            total_storage += 100

        df_results = pd.DataFrame.from_dict(vm_requirements, orient='index')

        df_results.loc["Total"] = ["-", total_vcpu, total_ram, total_storage]
        df_results.loc["RAID 1 (SSD)"] = ["-", "-", "-", total_storage]
        df_results.loc["RAID 5 (HDD)"] = ["-", "-", "-", total_image_storage_raid5]

    if num_studies <= 50000:
        sql_license = "SQL Express"
    elif num_studies <= 500000:
        sql_license = "SQL Standard"
    else:
        sql_license = "SQL Enterprise"


    df_results = pd.DataFrame.from_dict(vm_requirements, orient='index')
    df_results.loc["Total"] = ["-", total_vcpu, total_ram, total_storage]
    df_results.loc["RAID 1 (SSD)"] = ["-", "-", "-", total_storage]
    df_results.loc["RAID 5 (HDD)"] = ["-", "-", "-", total_image_storage_raid5]

    return df_results, sql_license, first_year_storage_raid5, total_image_storage_raid5, total_vcpu, total_ram, total_storage
