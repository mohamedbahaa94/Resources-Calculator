import streamlit as st

# ---------- Calculation Functions ----------
def calculate_layer4_throughput(study_volume, working_days, avg_study_size_mb, hl7_buffer_mbps=2, safety_factor=3.5):
    daily_volume = study_volume / working_days
    daily_data_gb = daily_volume * avg_study_size_mb / 1024
    hourly_data_gb = daily_data_gb / 8
    pacs_throughput_mbps = hourly_data_gb * 8 * 1024 / 3600
    total_l4 = round((pacs_throughput_mbps + hl7_buffer_mbps) * safety_factor)
    return {
        "daily_volume": daily_volume,
        "daily_data_gb": daily_data_gb,
        "pacs_mbps": pacs_throughput_mbps,
        "hl7_wl_buffer": hl7_buffer_mbps,
        "recommended_l4_mbps": total_l4
    }

def calculate_layer7_throughput(
    pacs_ccu,
    priors_per_study,
    ris_ccu,
    referring_ccu=0,
    patient_ccu=0,
    base_viewing_rate_gb=2.0,
    ris_per_user_mbps=0.2,
    safety_factor=1.25
):
    # ✅ Viewing rate for diagnostic users (includes priors)
    diagnostic_viewing_rate = base_viewing_rate_gb + priors_per_study * 0.33
    diagnostic_total_gb_per_hr = pacs_ccu * diagnostic_viewing_rate

    # ✅ Viewing rate for referring and patient portal (no priors)
    referring_total_gb_per_hr = referring_ccu * base_viewing_rate_gb
    patient_total_gb_per_hr = patient_ccu * base_viewing_rate_gb

    # Total viewing load
    total_gb_per_hr = diagnostic_total_gb_per_hr + referring_total_gb_per_hr + patient_total_gb_per_hr
    pacs_throughput_mbps = total_gb_per_hr * 8 * 1024 / 3600

    # RIS Load
    ris_throughput_mbps = ris_ccu * ris_per_user_mbps

    total_l7 = round((pacs_throughput_mbps + ris_throughput_mbps) * safety_factor)

    return {
        "diagnostic_users": pacs_ccu,
        "referring_users": referring_ccu,
        "patient_users": patient_ccu,
        "effective_viewing_diagnostic": diagnostic_viewing_rate,
        "pacs_throughput_mbps": pacs_throughput_mbps,
        "ris_throughput_mbps": ris_throughput_mbps,
        "recommended_l7_mbps": total_l7
    }

def calculate_total_site_throughput(l4_info, l7_info):
    return l4_info["recommended_l4_mbps"] + l7_info["recommended_l7_mbps"]
if __name__ == "__main__":

    # ---------- Streamlit App UI ----------
    st.title("PaxeraHealth Load Balancer Throughput Estimator")

    # Layer 4 Inputs
    st.header("Layer 4 (Archive, HL7, WL Broker)")
    study_volume = st.number_input("Annual Study Volume", min_value=1, value=100000)
    working_days = st.number_input("Working Days per Year", min_value=1, value=280)
    avg_study_size = st.number_input("Average Study Size (MB)", min_value=1, value=100)

    if st.button("Calculate Layer 4 Throughput"):
        l4 = calculate_layer4_throughput(study_volume, working_days, avg_study_size)
        st.session_state["l4_info"] = l4  # Save result
        st.subheader("Layer 4 Results")
        st.write(f"Daily Volume: {l4['daily_volume']:.1f} studies/day")
        st.write(f"Daily Data: {l4['daily_data_gb']:.2f} GB/day")
        st.write(f"Estimated Archive Bandwidth: {l4['pacs_mbps']:.2f} Mbps")
        st.write(f"HL7/WL Buffer: {l4['hl7_wl_buffer']} Mbps")
        st.success(f"**Recommended Layer 4 Bandwidth: {l4['recommended_l4_mbps']} Mbps**")

    st.divider()

    # Layer 7 Inputs
    st.header("Layer 7 (PACS Viewer, RIS, Portals)")
    pacs_ccu = st.number_input("PACS Viewer CCUs", min_value=0, value=60)
    priors = st.slider("Priors per Study", 0, 6, 3)
    ris_ccu = st.number_input("RIS CCUs", min_value=0, value=6)
    referring_ccu = st.number_input("Referring Physician CCUs", min_value=0, value=10)
    patient_ccu = st.number_input("Patient Portal CCUs", min_value=0, value=5)

    if st.button("Calculate Layer 7 Throughput"):
        l7 = calculate_layer7_throughput(
            pacs_ccu, priors, ris_ccu,
            referring_ccu=referring_ccu,
            patient_ccu=patient_ccu
        )
        st.session_state["l7_info"] = l7  # Save result
        st.subheader("Layer 7 Results")
        st.write(
            f"Diagnostic PACS Users: {l7['diagnostic_users']} → {l7['effective_viewing_diagnostic']:.2f} GB/hr (with priors)")
        st.write(f"Referring Users: {l7['referring_users']} → 2.00 GB/hr (no priors)")
        st.write(f"Patient Portal Users: {l7['patient_users']} → 2.00 GB/hr (no priors)")

        st.write(f"PACS Throughput: {l7['pacs_throughput_mbps']:.2f} Mbps")
        st.write(f"RIS Users: {ris_ccu} → {l7['ris_throughput_mbps']:.2f} Mbps")
        st.success(f"**Recommended Layer 7 Bandwidth: {l7['recommended_l7_mbps']} Mbps**")

    st.divider()

    # Final Combined Calculation
    st.header("Total Site Bandwidth")
    if st.button("Calculate Total Site Bandwidth"):
        if "l4_info" not in st.session_state or "l7_info" not in st.session_state:
            st.error("Please calculate both Layer 4 and Layer 7 first.")
        else:
            total = calculate_total_site_throughput(
                st.session_state["l4_info"],
                st.session_state["l7_info"]
            )
            st.success(f"**Total Recommended Site Bandwidth: {total} Mbps**")
