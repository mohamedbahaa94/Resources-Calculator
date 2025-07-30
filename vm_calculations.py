import pandas as pd

modality_sizes = {
    "CT": 0.5,
    "MR": 0.2,  # Increased from 0.1
    "US": 0.01,
    "NM": 0.05,  # Increased from 0.01
    "X-ray": 0.03,
    "MG": 0.16,
    "Cath": 0.3,
    "PET/CT": 1.0,  # New
    "Mammo Tomo": 0.6  # New
}

def calculate_referring_physician_resources(ref_phys_ccu):
    import math

    # Define CCU thresholds and resource tiers
    ccu_thresholds = [8, 16, 24, 32, 48, 64]
    ram_gb_tiers = [8, 16, 24, 32, 48, 64]
    vcores_tiers = [4, 6, 8, 10, 10, 12]
    max_ccu_per_vm = 64

    # Function to get RAM and vCores based on CCU
    def get_resources_for_ccu(ccu):
        for i, threshold in enumerate(ccu_thresholds):
            if ccu <= threshold:
                return ram_gb_tiers[i], vcores_tiers[i]
        return ram_gb_tiers[-1], vcores_tiers[-1]

    # If within one VM, return directly
    if ref_phys_ccu <= max_ccu_per_vm:
        ram, vcores = get_resources_for_ccu(ref_phys_ccu)
        return [(1, ram, vcores)]

    # Otherwise split into multiple VMs
    num_vms = math.ceil(ref_phys_ccu / max_ccu_per_vm)
    ccu_per_vm = math.ceil(ref_phys_ccu / num_vms)

    ram, vcores = get_resources_for_ccu(ccu_per_vm)
    return [(1, ram, vcores)] * num_vms

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
    if num_studies == 0 and ris_ccu > 0:
        num_studies = ris_ccu * 1250  # Estimate studies if unknown

    if num_studies < 10000:
        return [(1, 4, 2)]  # RAM, vCores
    elif num_studies < 30000:
        return [(1, 8, 4)]
    elif num_studies <= 500000:
        # Linear scaling from 30k → 500k
        min_studies = 30000
        max_studies = 500000
        min_ram = 8
        max_ram = 64
        min_vcores = 4
        max_vcores = 16

        scaling = (num_studies - min_studies) / (max_studies - min_studies)
        ram = min_ram + (max_ram - min_ram) * scaling
        vcores = min_vcores + (max_vcores - min_vcores) * scaling

        return [(1, int(round(ram)), int(round(vcores)))]
    else:
        return [(1, 64, 16)]


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
    ccu_thresholds = [4, 8, 12, 16, 24, 32]
    ram_gb_tiers =   [8, 16, 24, 32, 48, 64]
    vcores_tiers =   [4, 6, 8, 10, 10, 12]
    max_ccu_per_vm = 32

    import math

    if pacs_ccu <= max_ccu_per_vm:
        for i, threshold in enumerate(ccu_thresholds):
            if pacs_ccu <= threshold:
                return [(1, ram_gb_tiers[i], vcores_tiers[i])]

    # If CCU exceeds the max, split evenly across identical VMs
    num_vms = math.ceil(pacs_ccu / max_ccu_per_vm)
    ccu_per_vm = math.ceil(pacs_ccu / num_vms)

    # Find the proper tier for this per-VM CCU
    for i, threshold in enumerate(ccu_thresholds):
        if ccu_per_vm <= threshold:
            ram = ram_gb_tiers[i]
            vcores = vcores_tiers[i]
            break
    else:
        ram = ram_gb_tiers[-1]
        vcores = vcores_tiers[-1]

    return [(1, ram, vcores)] * num_vms

def calculate_paxera_pacs_resources(num_studies):
    import math

    max_studies_per_vm = 500_000

    # Study thresholds for interpolation
    thresholds = [5000, 25000, 50000, 100000, 250000, 500000]
    ram_values = [12, 20, 32, 48, 56, 64]
    core_values = [4, 6, 8, 12, 14, 16]

    def interpolate(thresholds, values, x):
        for i in range(1, len(thresholds)):
            if x <= thresholds[i]:
                x0, x1 = thresholds[i - 1], thresholds[i]
                y0, y1 = values[i - 1], values[i]
                ratio = (x - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return values[-1]

    if num_studies <= max_studies_per_vm:
        ram_gb = interpolate(thresholds, ram_values, num_studies)
        vcores = interpolate(thresholds, core_values, num_studies)
        return [(1, int(round(ram_gb)), int(round(vcores)))]

    # ---------------- Advanced Case: Split Evenly ----------------
    num_vms = math.ceil(num_studies / max_studies_per_vm)
    studies_per_vm = math.ceil(num_studies / num_vms)

    ram_gb = interpolate(thresholds, ram_values, studies_per_vm)
    vcores = interpolate(thresholds, core_values, studies_per_vm)

    return [(1, int(round(ram_gb)), int(round(vcores)))] * num_vms

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
        share_segmentation=False,
        share_chatbot=False,
        ark_chatbot_enabled=False,
        u9_ai_features=None,
        ref_phys_external_access=False,
        patient_portal_ccu=0,
        patient_portal_external_access=False,
        training_vm_included=False,
        high_availability=False,
        combine_pacs_ris=False,
        sql_always_on=False,
        split_pacs=False,
        add_n_plus_one=False,
        load_balancers_enabled=False,
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
        "Referring Physician": 1 if ref_phys_ccu > 0  else 0,
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
    import math

    vm_config_lists = {
        "PaxeraUltima": calculate_paxera_ultima_resources(pacs_ccu) if pacs_ccu > 0 else [],
        "PaxeraRIS": [
            (vm_count, math.ceil(ram_gb ), math.ceil(vcores ))
            for (vm_count, ram_gb, vcores) in calculate_referring_physician_resources(math.ceil(ris_ccu*1.2))
        ] if ris_ccu > 0 else [],
        "Referring Physician": calculate_referring_physician_resources(ref_phys_ccu) if ref_phys_ccu > 0 else [],
        "Patient Portal": calculate_referring_physician_resources(
            patient_portal_ccu) if patient_portal_ccu > 0 and patient_portal_external_access else []
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

    dbserver_vm_configs = calculate_dbserver_resources(num_studies, ris_ccu)
    _, ram_gb, vcores = dbserver_vm_configs[0]

    num_db_vms = 2 if sql_always_on else 1
    for i in range(num_db_vms):
        vm_requirements[f"DBServer0{i + 1}"] = {
            "VM Type": "DBServer (Primary)" if i == 0 else "DBServer (Secondary - Always On)",
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
    if combine_pacs_ris and pacs_ccu > 0 and ris_ccu > 0 and not any(
            vm.startswith("PaxeraUltimaRIS") for vm in vm_requirements):
        import math

        # Step 1: Extract original VMs
        ultima_vms = [vm for vm in vm_requirements if vm.startswith("PaxeraUltima")]
        ris_vms = [vm for vm in vm_requirements if vm.startswith("PaxeraRIS")]

        combined_vms_raw = []
        num_combined_vms = max(len(ultima_vms), len(ris_vms))

        for i in range(num_combined_vms):
            ultima_vm = vm_requirements.get(ultima_vms[i]) if i < len(ultima_vms) else None
            ris_vm = vm_requirements.get(ris_vms[i]) if i < len(ris_vms) else None

            if ultima_vm and ris_vm:
                combined_vm = {
                    "vCores": ultima_vm["vCores"] + ris_vm["vCores"],
                    "RAM (GB)": ultima_vm["RAM (GB)"] + ris_vm["RAM (GB)"],
                    "Storage (GB)": max(ultima_vm["Storage (GB)"], ris_vm["Storage (GB)"]),
                }
            elif ultima_vm:
                combined_vm = {
                    "vCores": ultima_vm["vCores"],
                    "RAM (GB)": ultima_vm["RAM (GB)"],
                    "Storage (GB)": ultima_vm["Storage (GB)"],
                }
            elif ris_vm:
                combined_vm = {
                    "vCores": ris_vm["vCores"],
                    "RAM (GB)": ris_vm["RAM (GB)"],
                    "Storage (GB)": ris_vm["Storage (GB)"],
                }

            combined_vms_raw.append(combined_vm)

        # Step 2: Apply 20% discount
        total_vcores = sum(vm["vCores"] for vm in combined_vms_raw)
        total_ram = sum(vm["RAM (GB)"] for vm in combined_vms_raw)
        max_storage = max(vm["Storage (GB)"] for vm in combined_vms_raw)

        optimized_vcores = math.ceil(total_vcores * 0.8)
        optimized_ram = math.ceil(total_ram * 0.8)

        # Step 3: Split into balanced VMs
        MAX_VCORES = 12
        MAX_RAM = 64

        num_final_vms = max(
            math.ceil(optimized_vcores / MAX_VCORES),
            math.ceil(optimized_ram / MAX_RAM)
        )

        vcores_per_vm = min(math.ceil(optimized_vcores / num_final_vms), MAX_VCORES)
        ram_per_vm = min(math.ceil(optimized_ram / num_final_vms), MAX_RAM)

        # Step 4: Create combined VMs
        combined_vms = []
        for i in range(num_final_vms):
            combined_vms.append({
                "VM Type": "PaxeraUltima/PaxeraRIS",
                "vCores": vcores_per_vm,
                "RAM (GB)": ram_per_vm,
                "Storage (GB)": max_storage
            })

        # Step 5: Clean up originals and insert new VMs
        keys_to_remove = [key for key in vm_requirements if
                          key.startswith("PaxeraUltima") or key.startswith("PaxeraRIS")]
        for key in keys_to_remove:
            del vm_requirements[key]

        for i, vm in enumerate(combined_vms):
            vm_name = f"PaxeraUltimaRIS_{i + 1}"
            vm_requirements[vm_name] = vm

    # Broker VM Logic with Project Grades
    # 🧩 Determine number of modalities for broker sizing
    if broker_required or (pacs_ccu > 0 and ris_ccu > 0):
        modalities = num_machines
        created_hl7 = False
        created_wl = False

        for level in broker_level:
            if level == "HL7":
                # Always create only 1 HL7 VM
                broker_vms = 1
                created_hl7 = True
            elif level == "WL":
                broker_vms = (modalities + 9) // 10
                created_wl = True
            else:
                continue  # Skip unknown levels

            for i in range(broker_vms):
                vm_name = f"PaxeraBroker_{level}_{i + 1:02d}"

                if level == "HL7":
                    vm_type = "PaxeraBroker HL7"
                    vcores = 8
                    ram_gb = 16
                elif level == "WL":
                    vm_type = "PaxeraBroker WL"
                    vcores = 6
                    ram_gb = 12
                else:
                    continue

                storage_gb = 150

                vm_requirements[vm_name] = {
                    "VM Type": vm_type,
                    "vCores": vcores,
                    "RAM (GB)": ram_gb,
                    "Storage (GB)": storage_gb,
                }

        # Combine only the WL Broker with DBServer in project_grade 2 and modalities < 10
        import math

        # ----------- COMBINE PACS + ULTIMA ON GRADE 1 ONLY -----------
        if project_grade == 1 and combine_pacs_ris==0:
            ultima_keys = [k for k in vm_requirements if k.startswith("PaxeraUltima")]
            pacs_keys = [k for k in vm_requirements if k.startswith("PaxeraPACS")]

            if ultima_keys and pacs_keys:
                total_vcores = sum(vm_requirements[k]["vCores"] for k in ultima_keys + pacs_keys)
                total_ram = sum(vm_requirements[k]["RAM (GB)"] for k in ultima_keys + pacs_keys)
                max_storage = max(vm_requirements[k]["Storage (GB)"] for k in ultima_keys + pacs_keys)

                # Apply 20% optimization
                total_vcores = min(12, math.ceil(total_vcores * 0.8))
                total_ram = min(64, math.ceil(total_ram * 0.8))

                # Create new combined VM
                vm_requirements["PaxeraPACS/Ultima"] = {
                    "VM Type": "PaxeraPACS / PaxeraUltima",
                    "vCores": total_vcores,
                    "RAM (GB)": total_ram,
                    "Storage (GB)": max_storage,
                }

                for k in ultima_keys + pacs_keys:
                    del vm_requirements[k]

        # ----------- COMBINE DB + BROKER (WL preferred, HL7 fallback) -----------
        if project_grade in [1, 2] and modalities < 10 and "DBServer01" in vm_requirements and not sql_always_on:
            wl_keys = [k for k in vm_requirements if "PaxeraBroker_WL" in k]
            hl7_keys = [k for k in vm_requirements if "PaxeraBroker_HL7" in k]

            broker_to_combine = None
            broker_label = ""

            if wl_keys:
                broker_to_combine = wl_keys
                broker_label = "Broker WL"
            elif hl7_keys:
                broker_to_combine = hl7_keys
                broker_label = "Broker HL7"

            if broker_to_combine:
                db_vm = vm_requirements["DBServer01"]
                total_vcores = db_vm["vCores"] + sum(vm_requirements[k]["vCores"] for k in broker_to_combine)
                total_ram = db_vm["RAM (GB)"] + sum(vm_requirements[k]["RAM (GB)"] for k in broker_to_combine)
                max_storage = max(
                    [db_vm["Storage (GB)"]] + [vm_requirements[k]["Storage (GB)"] for k in broker_to_combine])

                # Apply 20% optimization
                total_vcores = min(12, math.ceil(total_vcores * 0.8))
                total_ram = min(64, math.ceil(total_ram * 0.8))

                # Update DBServer01
                vm_requirements["DBServer01"] = {
                    "VM Type": f"DBServer / {broker_label}",
                    "vCores": total_vcores,
                    "RAM (GB)": total_ram,
                    "Storage (GB)": max_storage
                }

                for k in broker_to_combine:
                    del vm_requirements[k]

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

        # 🧠 U9 AI VMs
        if aidocker_included and u9_ai_features:
            for feature in u9_ai_features:
                if feature == "Organ Segmentation + Spine Labeling + Rib Counting":
                    vm_requirements["OrganSegmentationDocker"] = {
                        "VM Type": "Organ Segmentation + Spine + Rib Docker",
                        "vCores": 8,
                        "RAM (GB)": 48,
                        "Storage (GB)": 300
                    }
                    total_vcpu += 8
                    total_ram += 48
                    total_storage += 300

                elif feature == "Radiomics":
                    vm_requirements["RadiomicsDocker"] = {
                        "VM Type": "Radiomics Docker",
                        "vCores": 4,
                        "RAM (GB)": 32,
                        "Storage (GB)": 300
                    }
                    total_vcpu += 4
                    total_ram += 32
                    total_storage += 300

                elif feature == "Speech to Text":
                    vm_requirements["SpeechToTextDocker"] = {
                        "VM Type": "Speech-to-Text Docker",
                        "vCores": 6,
                        "RAM (GB)": 12,
                        "Storage (GB)": 300
                    }
                    total_vcpu += 6
                    total_ram += 12
                    total_storage += 300

                elif feature == "Chatbot (All Services + Report Rephrasing)":
                    vm_requirements["ChatbotDocker"] = {
                        "VM Type": "Chatbot Docker",
                        "vCores": 6,
                        "RAM (GB)": 32,
                        "Storage (GB)": 300
                    }
                    total_vcpu += 6
                    total_ram += 32
                    total_storage += 300

                elif feature == "X-ray Nodule & Consolidation":
                    vm_requirements["XrayNoduleDocker"] = {
                        "VM Type": "X-ray Nodule & Consolidation Docker",
                        "vCores": 4,
                        "RAM (GB)": 16,
                        "Storage (GB)": 300
                    }
                    total_vcpu += 4
                    total_ram += 16
                    total_storage += 300

        # 🤖 ARK AI VMs
        if ark_included:
            # ARK Manager and Lab
            vm_requirements["AIARKManager01"] = {
                "VM Type": "AI ARK Manager",
                "vCores": 12,
                "RAM (GB)": 32,
                "Storage (GB)": 300
            }
            vm_requirements["AIARKLAB01"] = {
                "VM Type": "AI ARK LAB",
                "vCores": 16,
                "RAM (GB)": 64,
                "Storage (GB)": 300
            }
            total_vcpu += 28
            total_ram += 96
            total_storage += 600

            # 🧾 ARK DB
            vm_requirements["ARKDatabase"] = {
                "VM Type": "ARK Dedicated Database",
                "vCores": 8,
                "RAM (GB)": 16,
                "Storage (GB)": 400
            }
            total_vcpu += 8
            total_ram += 16
            total_storage += 400

            # 🧠 Shared or Separate Segmentation Docker
            if share_segmentation:
                pass  # Shared with U9, nothing added
            else:
                vm_requirements["ARK_SegmentationDocker"] = {
                    "VM Type": "ARK Segmentation Docker",
                    "vCores": 8,
                    "RAM (GB)": 48,
                    "Storage (GB)": 300
                }
                total_vcpu += 8
                total_ram += 48
                total_storage += 300

            # 🤖 Shared or Separate Chatbot Docker
            if ark_chatbot_enabled:
                if share_chatbot:
                    pass  # Shared with U9, nothing added
                else:
                    vm_requirements["ARK_ChatbotDocker"] = {
                        "VM Type": "ARK Chatbot Docker",
                        "vCores": 6,
                        "RAM (GB)": 32,
                        "Storage (GB)": 300
                    }
                    total_vcpu += 6
                    total_ram += 32
                    total_storage += 300
        # ----------------- Optional Forced Split of PaxeraPACS -----------------
        if split_pacs and not add_n_plus_one:
            pacs_keys = [key for key in vm_requirements if key.startswith("PaxeraPACS")]
            is_combined = any(
                key.startswith("PaxeraUltimaRIS_") or key.startswith("PaxeraPACS/Ultima")
                for key in vm_requirements
            )

            # Apply only if 1 PACS VM exists and not combined
            if len(pacs_keys) == 1 and not is_combined:
                original_key = pacs_keys[0]
                pacs_vm = vm_requirements[original_key]

                total_vcores = pacs_vm["vCores"]
                total_ram = pacs_vm["RAM (GB)"]
                storage_gb = pacs_vm["Storage (GB)"]

                del vm_requirements[original_key]

                PACS_MAX_VCORES = 16
                PACS_MAX_RAM = 64

                splits_vcores = -(-total_vcores // PACS_MAX_VCORES)
                splits_ram = -(-total_ram // PACS_MAX_RAM)
                num_instances = max(2, splits_vcores, splits_ram)

                vcores_per = max(2, -(-total_vcores // num_instances))
                ram_per = max(4, -(-total_ram // num_instances))

                for i in range(num_instances):
                    vm_requirements[f"PaxeraPACS{i + 1:02d}"] = {
                        "VM Type": "PaxeraPACS",
                        "vCores": vcores_per,
                        "RAM (GB)": ram_per,
                        "Storage (GB)": storage_gb,
                    }

        # --------------------- Optional N+1 VM Addition ---------------------
        if add_n_plus_one:
            ha_tags = ("PaxeraPACS", "PaxeraRIS", "PaxeraUltima")
            n_plus_one_vms = {}

            for key, config in vm_requirements.items():
                base_name = config["VM Type"]
                if any(base_name.startswith(tag) for tag in ha_tags):
                    clone_key = f"{key}_HA"
                    if clone_key not in vm_requirements and clone_key not in n_plus_one_vms:
                        n_plus_one_vms[clone_key] = config.copy()

            vm_requirements.update(n_plus_one_vms)

            reordered = {}
            for key in list(vm_requirements.keys()):
                reordered[key] = vm_requirements[key]
                ha_key = f"{key}_HA"
                if ha_key in vm_requirements:
                    reordered[ha_key] = vm_requirements[ha_key]

            vm_requirements.clear()
            vm_requirements.update(reordered)

        # --------------------- Optional Load Balancer Split ---------------------
        # 🧩 Load Balancer Splitting or Duplication Logic
        if load_balancers_enabled:
            lb_tags = (
                "PaxeraPACS",
                "PaxeraUltima",
                "PaxeraRIS",
                "Referring",
                "PatientPortal",
                "PaxeraBroker WL",
                "PaxeraBroker HL7",
            )

            reordered = {}
            keys_to_process = list(vm_requirements.keys())

            for key in keys_to_process:
                config = vm_requirements[key]
                base_name = config["VM Type"]

                if any(base_name.startswith(tag) for tag in lb_tags):
                    same_type_vms = [k for k in vm_requirements if
                                     vm_requirements[k]["VM Type"] == base_name and not k.endswith("_HA")]

                    if base_name.startswith("PaxeraBroker"):
                        # Duplicate instead of splitting
                        for i in range(2):
                            new_key = f"{key}_LB{i + 1}"
                            reordered[new_key] = config.copy()

                        del vm_requirements[key]
                    else:
                        # Split if only one instance
                        if len(same_type_vms) == 1:
                            original_key = same_type_vms[0]
                            original_vm = vm_requirements[original_key]
                            total_vcores = original_vm["vCores"]
                            total_ram = original_vm["RAM (GB)"]
                            storage_gb = original_vm["Storage (GB)"]

                            del vm_requirements[original_key]

                            vcores_per = max(2, total_vcores // 2)
                            ram_per = max(4, total_ram // 2)

                            for i in range(2):
                                new_key = f"{base_name}{i + 1:02d}"
                                reordered[new_key] = {
                                    "VM Type": base_name,
                                    "vCores": vcores_per,
                                    "RAM (GB)": ram_per,
                                    "Storage (GB)": storage_gb,
                                }

            vm_requirements.update(reordered)

        # ✅ At the end of your VM generation logic, add this block to ensure accurate totals:

        total_vcpu = sum(vm.get("vCores", 0) for vm in vm_requirements.values())
        total_ram = sum(vm.get("RAM (GB)", 0) for vm in vm_requirements.values())
        total_storage = sum(vm.get("Storage (GB)", 0) for vm in vm_requirements.values())

        # Reconstruct the final DataFrame

        df_results = pd.DataFrame.from_dict(vm_requirements, orient='index')
        df_results.loc["Total"] = ["-", total_vcpu, total_ram, total_storage]
        df_results.loc["RAID 1 (SSD)"] = ["-", "-", "-", total_storage]
        df_results.loc["RAID 5 (HDD)"] = ["-", "-", "-", total_image_storage_raid5]
        # Already extracted above:
        _, ram_gb, vcores = dbserver_vm_configs[0]

    if sql_always_on:
        sql_license = f"SQL Enterprise  (Always On Enabled) – {vcores} Cores per License"
    elif num_studies < 10000:
        sql_license = "SQL Express"
    elif num_studies < 30000:
        sql_license = "SQL Express (Recommended: SQL Standard)"
    elif num_studies <= 500000:
        sql_license = f"SQL Standard – {vcores} Cores"
    else:
        sql_license = f"SQL Enterprise – {vcores} Cores"

    df_results = pd.DataFrame.from_dict(vm_requirements, orient='index')
    df_results.loc["Total"] = ["-", total_vcpu, total_ram, total_storage]
    df_results.loc["RAID 1 (SSD)"] = ["-", "-", "-", total_storage]
    df_results.loc["RAID 5 (HDD)"] = ["-", "-", "-", total_image_storage_raid5]

    return df_results, sql_license, first_year_storage_raid5, total_image_storage_raid5, total_vcpu, total_ram, total_storage
