import streamlit as st

st.title("PaxeraHealth Remote Site Bandwidth Estimator")

# Input fields
study_volume = st.number_input("Annual Study Volume", min_value=1, value=100000)
working_days = st.number_input("Working Days per Year", min_value=1, max_value=366, value=280)
concurrent_users = st.number_input("Number of Concurrent Users", min_value=1, value=60)
avg_study_size = st.number_input("Average Study Size (MB)", min_value=1, value=100)
viewing_rate_per_user = st.slider("Average Viewing Rate per User (GB/hour)", 0.5, 5.0, 2.0)
priors_per_study = st.slider("Average Number of Priors per Study", 0, 6, 3)

if st.button("Calculate Bandwidth"):
    # Calculations
    daily_volume = study_volume / working_days
    daily_data_volume_gb = daily_volume * avg_study_size / 1024  # convert MB to GB

    # Image transfer bandwidth estimate
    transfer_bandwidth_gb_per_hour = daily_data_volume_gb / 12
    transfer_bandwidth_mbps = transfer_bandwidth_gb_per_hour * 8 * 1024 / 3600  # GB to Mbps
    recommended_transfer_bandwidth = max(15, round(transfer_bandwidth_mbps * 1.5))

    # Viewing bandwidth estimate including priors
    total_viewing_rate = viewing_rate_per_user + priors_per_study * 0.33  # Approx +1 GB/hr for 3 priors
    viewing_total_gb_per_hour = concurrent_users * total_viewing_rate
    viewing_bandwidth_mbps = viewing_total_gb_per_hour * 8 * 1024 / 3600
    recommended_viewing_bandwidth = round(viewing_bandwidth_mbps * 1.25)

    total_site_bandwidth = recommended_transfer_bandwidth + recommended_viewing_bandwidth

    # Display Results
    st.subheader("Results")
    st.write(f"**Daily Study Volume:** {daily_volume:.1f} studies/day")
    st.write(f"**Daily Data Volume:** {daily_data_volume_gb:.2f} GB/day")
    st.write(f"**Estimated Image Transfer Bandwidth:** {transfer_bandwidth_mbps:.2f} Mbps")
    st.write(f"**Recommended Uplink Bandwidth (Transfer):** {recommended_transfer_bandwidth} Mbps")
    st.write(f"**Effective Viewing Rate (including {priors_per_study} priors — estimated as {priors_per_study * 0.33:.2f} GB/hr added):** {total_viewing_rate:.2f} GB/hour per user")
    st.write(f"**Estimated Peak Viewing Bandwidth:** {viewing_bandwidth_mbps:.2f} Mbps")
    st.write(f"**Recommended Bandwidth for Viewing Performance:** {recommended_viewing_bandwidth} Mbps")
    st.markdown(f"### **Total Recommended Bandwidth for Site: {total_site_bandwidth} Mbps**")

    st.info("Use QoS and VPN with compression to ensure prioritized, efficient medical data flow.")