import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import os # Keep OS for checking logo path
import traceback # Keep for catching potential errors

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

def create_hourly_solar_profile(total_daily_solar_kwh_6_to_16):
    """Creates a 24-hour solar profile array (in Wh) based on total kWh."""
    hourly_solar_generation_wh = np.zeros(24)
    solar_shape_factors = np.array([0.3, 0.6, 0.8, 0.9, 1.0, 1.0, 0.8, 0.6, 0.3, 0.2]) # Indices 0-9 correspond to hours 6-15
    if total_daily_solar_kwh_6_to_16 <= 0: return hourly_solar_generation_wh
    shape_sum = np.sum(solar_shape_factors)
    if shape_sum == 0: return hourly_solar_generation_wh
    total_daily_solar_wh = total_daily_solar_kwh_6_to_16 * 1000.0
    wh_per_factor_unit = total_daily_solar_wh / shape_sum
    for i in range(len(solar_shape_factors)):
        hour_index = 6 + i
        if hour_index < 24:
             hourly_solar_generation_wh[hour_index] = solar_shape_factors[i] * wh_per_factor_unit
    return hourly_solar_generation_wh


def run_simulation(hourly_load_demand_ac_wh, hourly_solar_generation_wh,
                   battery_voltage_v, total_nominal_capacity_ah, max_dod,
                   inverter_efficiency, initial_soc_percent,
                   max_charging_power_kw, max_battery_discharge_power_kw,
                   battery_discharge_start_hour, battery_discharge_end_hour):
    """
    Runs grid-tied simulation with battery discharge restricted to a time window.
    LOAD = SOLAR + BATTERY (within window) + GRID.
    Expects HOURLY inputs in Wh.
    Returns results including grid import.
    """
    # --- Start of run_simulation ---
    total_nominal_capacity_wh = total_nominal_capacity_ah * battery_voltage_v
    if total_nominal_capacity_wh <= 0:
        st.warning("Battery Capacity cannot be zero. Simulation stopped.")
        return None

    min_soc_percent = (1.0 - max_dod) * 100.0
    min_soc_wh = total_nominal_capacity_wh * (1.0 - max_dod)
    current_soc_wh = total_nominal_capacity_wh * (initial_soc_percent / 100.0)
    current_soc_wh = max(min_soc_wh, min(total_nominal_capacity_wh, current_soc_wh)) # Ensure initial SoC is valid

    max_charging_power_w = max_charging_power_kw * 1000.0
    max_battery_discharge_power_w = max_battery_discharge_power_kw * 1000.0

    # Initialize tracking lists/variables
    soc_over_time_percent = []
    energy_charged_hourly_wh = []
    energy_discharged_for_load_hourly_wh = []
    grid_import_hourly_wh = []
    load_met_by_solar_hourly_wh = []
    load_met_by_battery_hourly_wh = []
    actual_load_ac_hourly_wh = []
    charging_hours_count = 0

    for i in range(24): # Loop through hours 0 to 23
        # Get hourly values
        load_ac_wh = hourly_load_demand_ac_wh[i]
        actual_load_ac_hourly_wh.append(load_ac_wh)
        load_ac_at_dc_wh = load_ac_wh / inverter_efficiency if inverter_efficiency > 0 else load_ac_wh
        total_load_at_dc_wh = load_ac_at_dc_wh
        solar_gen_wh = hourly_solar_generation_wh[i]

        # Reset hourly calculation variables
        discharge_for_load_wh = 0
        grid_import_wh = 0
        load_met_solar_dc = 0
        load_met_battery_dc = 0 # Tracks battery energy used *for load*
        is_charging_this_hour = False

        # --- Simulation Flow ---
        # 1. Meet load from solar first
        solar_direct_to_load = min(solar_gen_wh, total_load_at_dc_wh)
        load_met_solar_dc = solar_direct_to_load
        remaining_load_after_solar = total_load_at_dc_wh - solar_direct_to_load
        excess_solar = solar_gen_wh - solar_direct_to_load

        # 2. Meet remaining load from battery (if any) - Check Discharge Window
        remaining_load_after_battery = remaining_load_after_solar # Initialize
        is_within_discharge_window = (i >= battery_discharge_start_hour and i < battery_discharge_end_hour)

        if remaining_load_after_solar > 0.1 and is_within_discharge_window:
            available_battery_wh = max(0, current_soc_wh - min_soc_wh)
            discharge_for_load_wh = min(remaining_load_after_solar,
                                        available_battery_wh,
                                        max_battery_discharge_power_w)

            if discharge_for_load_wh > 0.1:
                current_soc_wh -= discharge_for_load_wh
                load_met_battery_dc = discharge_for_load_wh
            remaining_load_after_battery = remaining_load_after_solar - discharge_for_load_wh

        # 3. Meet remaining load from Grid
        if remaining_load_after_battery > 0.1:
            grid_import_wh = remaining_load_after_battery

        # 4. Charge battery from excess solar (if any)
        if excess_solar > 0.1:
            solar_available_for_charge = excess_solar
            charge_potential_wh = total_nominal_capacity_wh - current_soc_wh
            actual_charge = min(solar_available_for_charge, charge_potential_wh, max_charging_power_w)

            if actual_charge > 0.1:
                current_soc_wh += actual_charge
                energy_charged_hourly_wh.append(actual_charge)
                is_charging_this_hour = True
            else:
                energy_charged_hourly_wh.append(0)
        else:
            energy_charged_hourly_wh.append(0)

        # --- Record hourly results ---
        soc_percent = (current_soc_wh / total_nominal_capacity_wh) * 100.0 if total_nominal_capacity_wh > 0 else 0
        soc_over_time_percent.append(soc_percent)
        energy_discharged_for_load_hourly_wh.append(discharge_for_load_wh) # Record actual discharge
        grid_import_hourly_wh.append(grid_import_wh)

        if is_charging_this_hour:
            charging_hours_count += 1

        # --- Convert load met sources ---
        original_total_load_wh = load_ac_wh # Only AC load now
        fraction_ac = 1.0 # Always 100% AC

        load_met_solar_ac = (load_met_solar_dc * fraction_ac) * inverter_efficiency
        load_met_by_solar_hourly_wh.append(load_met_solar_ac) # Only AC part

        load_met_battery_ac = (load_met_battery_dc * fraction_ac) * inverter_efficiency
        load_met_by_battery_hourly_wh.append(load_met_battery_ac) # Only AC part

    # --- End of simulation loop ---

    results = {
        "soc_over_time_percent": soc_over_time_percent,
        "energy_charged_hourly_wh": energy_charged_hourly_wh,
        "energy_discharged_for_load_hourly_wh": energy_discharged_for_load_hourly_wh,
        "grid_import_hourly_wh": grid_import_hourly_wh,
        "load_met_by_solar_hourly_wh": load_met_by_solar_hourly_wh,
        "load_met_by_battery_hourly_wh": load_met_by_battery_hourly_wh,
        "final_soc_wh": current_soc_wh,
        "total_nominal_capacity_wh": total_nominal_capacity_wh,
        "actual_load_ac_hourly_wh": actual_load_ac_hourly_wh,
        "hourly_solar_generation_wh": hourly_solar_generation_wh,
        "min_soc_percent": min_soc_percent,
        "total_charging_hours": charging_hours_count,
    }
    return results


def create_line_graph_plotly(simulation_results, min_soc_percent):
    """Creates the interactive Plotly line graph including Grid Import."""
    hours = np.arange(24); hour_labels = [f"{h:02d}:00" for h in hours]
    soc_over_time_percent = simulation_results["soc_over_time_percent"]
    total_hourly_load_demand_wh = np.array(simulation_results["actual_load_ac_hourly_wh"])
    hourly_solar_generation_wh = np.array(simulation_results["hourly_solar_generation_wh"])
    hourly_grid_import_wh = np.array(simulation_results["grid_import_hourly_wh"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=hour_labels, y=soc_over_time_percent, name='Battery SoC (%)', mode='lines', line=dict(color='green', width=2), hovertemplate='SoC: %{y:.1f}%<extra></extra>'), secondary_y=False)
    fig.add_trace(go.Scatter(x=hour_labels, y=total_hourly_load_demand_wh, name='Hourly Load (Wh)', mode='lines', line=dict(color='red', dash='dot'), opacity=0.8, hovertemplate='Load: %{y:.0f} Wh<extra></extra>'), secondary_y=True)
    fig.add_trace(go.Scatter(x=hour_labels, y=hourly_solar_generation_wh, name='Hourly Solar (Wh)', mode='lines', line=dict(color='orange'), opacity=0.8, hovertemplate='Solar: %{y:.0f} Wh<extra></extra>'), secondary_y=True)
    fig.add_trace(go.Scatter(x=hour_labels, y=hourly_grid_import_wh, name='Hourly Grid Import (Wh)', mode='lines', line=dict(color='blue'), opacity=0.7, hovertemplate='Grid Import: %{y:.0f} Wh<extra></extra>'), secondary_y=True)
    fig.add_hline(y=min_soc_percent, line_dash="dash", line_color="red", annotation_text=f"Min SoC ({min_soc_percent:.0f}%)", annotation_position="bottom right", secondary_y=False)
    fig.update_layout(title_text='System Performance Over 24 Hours (Grid-Tied)', xaxis_title='Hour of Day', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.6)'), hovermode="x unified")
    fig.update_yaxes(title_text="<b>State of Charge (%)</b>", secondary_y=False, range=[0, 105], tickformat=".0%")
    fig.update_yaxes(title_text="<b>Energy (Wh)</b>", secondary_y=True, rangemode='tozero')
    return fig

def create_donut_chart_plotly(simulation_results):
    """Creates the interactive Plotly donut chart showing load sources (Solar, Battery, Grid)."""
    total_load_met_by_solar = sum(simulation_results["load_met_by_solar_hourly_wh"])
    total_load_met_by_battery = sum(simulation_results["load_met_by_battery_hourly_wh"])
    total_grid_import_wh = sum(simulation_results["grid_import_hourly_wh"])
    total_original_load_wh = sum(simulation_results["actual_load_ac_hourly_wh"])
    fraction_ac_total = 1.0 if total_original_load_wh > 0 else 0; inverter_eff = 0.9
    load_met_grid_approx_ac = (total_grid_import_wh * fraction_ac_total) * inverter_eff
    total_load_met_by_grid_approx = load_met_grid_approx_ac
    labels, values, colors, hover_text = [], [], [], []
    if total_load_met_by_solar > 0.1: labels.append('Load Met by Solar'); values.append(total_load_met_by_solar); colors.append('red'); hover_text.append(f'{total_load_met_by_solar / 1000.0:.2f} kWh')
    if total_load_met_by_battery > 0.1: labels.append('Load Met by Battery'); values.append(total_load_met_by_battery); colors.append('green'); hover_text.append(f'{total_load_met_by_battery / 1000.0:.2f} kWh')
    if total_load_met_by_grid_approx > 0.1: labels.append('Load Met by Grid'); values.append(total_load_met_by_grid_approx); colors.append('blue'); hover_text.append(f'{total_load_met_by_grid_approx / 1000.0:.2f} kWh')
    if sum(values) > 0:
        trace = go.Pie(labels=labels, values=values, hole=.4, marker_colors=colors, textinfo='percent', hoverinfo='label+text', hovertext=hover_text, insidetextorientation='radial', sort=False)
        center_text = f"Total Load:<br>{total_original_load_wh / 1000.0:.2f} kWh"
        layout = go.Layout(title_text='Daily Load Coverage by Source', annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=12, showarrow=False)], showlegend=True, legend_title_text='Source', legend=dict(traceorder='reversed'))
        fig = go.Figure(data=[trace], layout=layout)
    else:
        layout = go.Layout(title_text='Daily Load Coverage by Source', xaxis={'visible': False}, yaxis={'visible': False}, annotations=[dict(text="No Load or Activity", x=0.5, y=0.5, showarrow=False, font_size=14)])
        fig = go.Figure(layout=layout)
    return fig

def create_6hr_input_grid(profile_name, default_values_kwh_block, key_prefix):
    """Creates a grid of 4 number inputs for 6-hour block totals in kWh."""
    st.subheader(f"{profile_name} (Total kWh per 6-Hour Block)")
    block_kwh_values = []; block_labels = ["00:00 - 05:59", "06:00 - 11:59", "12:00 - 17:59", "18:00 - 23:59"]
    cols = st.columns(4)
    for i in range(4):
        with cols[i]: value = st.number_input( label=block_labels[i], min_value=0.0, value=default_values_kwh_block[i], step=0.1, format="%.2f", key=f"{key_prefix}_block_{i}" ); block_kwh_values.append(value)
    return np.array(block_kwh_values)

def distribute_block_to_hourly(block_values_kwh):
    """Converts 4 block kWh totals into 24 hourly average kWh values."""
    hourly_kwh = []
    for block_total_kwh in block_values_kwh: avg_hourly_kwh = block_total_kwh / 6.0 if block_total_kwh > 0 else 0.0; hourly_kwh.extend([avg_hourly_kwh] * 6)
    return np.array(hourly_kwh)

def aggregate_to_6hr_kwh(hourly_wh_array):
    """Aggregates hourly Wh array to 4 block kWh totals."""
    block_totals_kwh = []
    for i in range(0, 24, 6): block_sum_wh = np.sum(hourly_wh_array[i:i+6]); block_totals_kwh.append(block_sum_wh / 1000.0)
    return np.array(block_totals_kwh)

# ----------------------------------------------------
# Streamlit App UI
# ----------------------------------------------------

st.set_page_config(page_title="Solar Battery Sim 1", layout="wide")

# --- Logo Display ---
# Define logo path here so it can be checked
# Center the image using HTML within st.markdown.
st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="https://raw.githubusercontent.com/Azrael1ai5/seistrackpowerapp/main/LOGO.png" width="150">
        </div>
        """, unsafe_allow_html=True
    )
st.title("☀️🔋🔌 SEISTRACK (Solar + Battery + Grid ) Simulator")
st.markdown("Created By: Eric G. Bundalian")
st.markdown("""
Simulate daily energy flow for a grid-tied system with battery storage (AC Loads Only).
Estimate required PV system size and Return on Investment (ROI) based on inputs.
""")

# --- Input Parameters (Sidebar) ---
st.sidebar.header("System Configuration")
st.sidebar.subheader("🔋 Battery")
bat_volt_opts = [12.0, 24.0, 48.0, 51.2]
battery_voltage_v = st.sidebar.selectbox("Nominal Voltage (V)", options=bat_volt_opts, index=3) # Default 51.2V
total_nominal_capacity_ah = st.sidebar.number_input("Total Capacity (Ah)", min_value=1.0, value=200.0, step=10.0)
max_dod_percent = st.sidebar.slider("Max Depth of Discharge (%)", min_value=10, max_value=100, value=80, step=5) # Default 80%
max_dod = max_dod_percent / 100.0
initial_soc_percent = st.sidebar.slider("Initial State of Charge (%)", min_value=0, max_value=100, value=20, step=5) # Default 20%
max_charging_power_kw = st.sidebar.number_input("🔋Max Battery Charging Power (kW)", min_value=0.1, value=3.0, step=0.1, help="Maximum power used to charge the battery (from excess solar).") # Default 3.0 kW
max_battery_discharge_power_kw = st.sidebar.number_input("🪫Max Battery Discharge Power (kW)", min_value=0.1, value=2.0, step=0.1, help="Maximum power the battery can supply to the load.") # Default 2.0 kW

st.sidebar.subheader("🪫Battery Discharge Window")
battery_discharge_start_hour = st.sidebar.number_input("🪫➡️💡Allow Discharge From Hour (0-23)", min_value=0, max_value=23, value=0, step=1, help="Hour when battery can START discharging for load (inclusive).")
battery_discharge_end_hour = st.sidebar.number_input("Allow Discharge Until Hour (1-24)", min_value=1, max_value=24, value=24, step=1, help="Hour when battery MUST STOP discharging for load (exclusive, e.g., 22 means ends at 21:59).")

st.sidebar.subheader("System & Location")
inverter_efficiency_percent = st.sidebar.slider("Inverter Efficiency (%)", min_value=50, max_value=100, value=90, step=1)
inverter_efficiency = inverter_efficiency_percent / 100.0
peak_sun_hours = st.sidebar.number_input("Peak Sun Hours (PSH)", min_value=1.0, max_value=8.0, value=4.5, step=0.1, help="Equivalent hours of peak sunlight per day for your location (average).")
system_loss_percent = st.sidebar.number_input("PV System Losses (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0, help="Overall losses (temp, wiring, dirt, etc.) excluding inverter eff. for load.")

# --- Load Input Profiles (Main Area - REMOVED DC Load) ---
st.header("💡Load Profiles Input (AC Loads Only)")
default_ac_load_wh_hourly = np.array([50, 50, 50, 50, 80, 150, 250, 200, 100, 80, 70, 70, 90, 100, 120, 150, 300, 400, 350, 250, 150, 100, 50, 50])
default_ac_load_kwh_block = aggregate_to_6hr_kwh(default_ac_load_wh_hourly)
with st.expander("Edit AC Load Profile (Total kWh per 6-Hour Block)", expanded=True):
    block_ac_load_kwh = create_6hr_input_grid("AC Load Demand", default_ac_load_kwh_block, "ac_load")
hourly_ac_load_kwh = distribute_block_to_hourly(block_ac_load_kwh)
hourly_load_demand_ac_wh = hourly_ac_load_kwh * 1000.0

# --- Solar Generation Input (Main Area - Single Value) ---
st.header("🌞Solar Generation Input")
total_daily_solar_kwh_input = st.number_input("Target Daily Solar Generation (6 AM - 4 PM) (kWh)", min_value=0.0, value=15.0, step=0.5, format="%.1f", help="The amount of usable solar energy you want the system to generate per day.")
hourly_solar_generation_wh = create_hourly_solar_profile(total_daily_solar_kwh_input)


# --- Simulation & Display ---
st.header("📊Simulation Results")

# Input validation checks
run_sim = True
if total_nominal_capacity_ah <= 0 or battery_voltage_v <= 0: st.error("Battery Voltage and Capacity must be greater than zero."); run_sim = False
elif max_charging_power_kw <= 0: st.error("Max Charging Power must be greater than zero."); run_sim = False
elif max_battery_discharge_power_kw <=0: st.error("Max Battery Discharge Power must be greater than zero."); run_sim = False
elif battery_discharge_start_hour < 0 or battery_discharge_start_hour > 23: st.error("Battery Discharge Start Hour must be between 0 and 23."); run_sim = False
elif battery_discharge_end_hour < 1 or battery_discharge_end_hour > 24: st.error("Battery Discharge End Hour must be between 1 and 24."); run_sim = False
elif battery_discharge_end_hour <= battery_discharge_start_hour: st.error("Battery Discharge End Hour must be after Discharge Start Hour."); run_sim = False
elif peak_sun_hours <= 0: st.error("Peak Sun Hours must be greater than zero."); run_sim = False
elif system_loss_percent < 0 or system_loss_percent >= 100: st.error("System Losses must be between 0% and 99%."); run_sim = False

if run_sim:
    # Run Simulation
    simulation_results_data = run_simulation(
        hourly_load_demand_ac_wh, hourly_solar_generation_wh,
        battery_voltage_v, total_nominal_capacity_ah, max_dod,
        inverter_efficiency, initial_soc_percent,
        max_charging_power_kw, max_battery_discharge_power_kw,
        battery_discharge_start_hour, battery_discharge_end_hour
    )

    if simulation_results_data:
        st.subheader("Visualizations")
        col1, col2 = st.columns([3, 2])
        min_soc_p_for_graph = simulation_results_data["min_soc_percent"]
        fig_line = create_line_graph_plotly(simulation_results_data, min_soc_p_for_graph)
        fig_donut = create_donut_chart_plotly(simulation_results_data)
        with col1: st.markdown("**Performance Over Time**"); st.plotly_chart(fig_line, use_container_width=True)
        with col2: st.markdown("**Daily Load Coverage by Source**"); st.plotly_chart(fig_donut, use_container_width=True)

        # Prepare Summary Stats Data
        st.subheader("Summary Statistics (Daily Totals) - Simulation")
        final_soc_percent = simulation_results_data["soc_over_time_percent"][-1]
        final_soc_wh = simulation_results_data['final_soc_wh']
        sim_total_charged_wh = sum(simulation_results_data["energy_charged_hourly_wh"])
        sim_total_discharged_for_load_wh = sum(simulation_results_data["energy_discharged_for_load_hourly_wh"])
        sim_total_load_met_by_solar_wh = sum(simulation_results_data["load_met_by_solar_hourly_wh"])
        sim_total_load_met_by_battery_wh = sum(simulation_results_data["load_met_by_battery_hourly_wh"])
        sim_total_grid_import_wh = sum(simulation_results_data["grid_import_hourly_wh"])
        sim_total_daily_load_kwh = sum(block_ac_load_kwh)
        sim_total_solar_gen_kwh = total_daily_solar_kwh_input
        sim_total_charging_hours = simulation_results_data["total_charging_hours"]

        sim_summary_dict_display = { # Dictionary for display
            "Final SoC": f"{final_soc_percent:.1f}%", "Final SoC Wh": f"{final_soc_wh:.0f} Wh",
            "Load Met by Battery": f"{sim_total_load_met_by_battery_wh / 1000.0:.2f} kWh", "Battery Discharged": f"{sim_total_discharged_for_load_wh / 1000.0:.2f} kWh Discharged",
            "Load Met by Solar": f"{sim_total_load_met_by_solar_wh / 1000.0:.2f} kWh", "Solar Generated": f"{sim_total_solar_gen_kwh:.2f} kWh Generated Target",
            "Total Battery Charged": f"{sim_total_charged_wh / 1000.0:.2f} kWh", "Charging Hours": f"{sim_total_charging_hours} Charging Hrs",
            "Total Daily Load": f"{sim_total_daily_load_kwh:.2f} kWh", "Total Grid Import": f"{sim_total_grid_import_wh / 1000.0:.2f} kWh"
        }

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Final SoC", sim_summary_dict_display["Final SoC"], sim_summary_dict_display["Final SoC Wh"])
        col_b.metric("(Sim) Load Met by Battery", sim_summary_dict_display["Load Met by Battery"], sim_summary_dict_display["Battery Discharged"])
        col_c.metric("(Sim) Load Met by Solar", sim_summary_dict_display["Load Met by Solar"], sim_summary_dict_display["Solar Generated"])
        col_d, col_e, col_f = st.columns(3)
        col_d.metric("(Sim) Total Battery Charged", sim_summary_dict_display["Total Battery Charged"], sim_summary_dict_display["Charging Hours"])
        col_e.metric("(Sim) Total Daily Load", sim_summary_dict_display["Total Daily Load"])
        col_f.metric("(Sim) Total Grid Import", sim_summary_dict_display["Total Grid Import"])
        st.markdown("---")
        
        # --- Predicted PV System Size ---
        st.subheader("🌞⚡PV System Size Estimation")
        predicted_system_size_kwp, total_pv_system_cost_php = 0.0, 0.0
        price_per_kwp_php = st.number_input( "PV System Price per kWp (PHP)", min_value=0.0, value=25000.0, step=1000.0, format="%.0f", key="price_kwp_pv", help="Estimated cost per kWp for the PV panels and installation." ) # Default 25000
        system_efficiency_factor = (1 - system_loss_percent / 100.0)
        if peak_sun_hours > 0 and system_efficiency_factor > 0 and total_daily_solar_kwh_input > 0:
            predicted_system_size_kwp = total_daily_solar_kwh_input / (peak_sun_hours * system_efficiency_factor)
            total_pv_system_cost_php = predicted_system_size_kwp * price_per_kwp_php
            col_pv_1, col_pv_2 = st.columns(2)
            with col_pv_1: st.metric("Predicted PV System Size (kWp)", f"{predicted_system_size_kwp:.2f} kWp", help=f"Estimated size needed to meet the target {total_daily_solar_kwh_input:.1f} kWh/day generation, considering {peak_sun_hours:.1f} PSH and {system_loss_percent:.0f}% system losses.")
            with col_pv_2: st.metric("Estimated PV System Cost (PHP)", f"PHP {total_pv_system_cost_php:,.2f}")
        else:
            st.metric("Predicted PV System Size (kWp)", "N/A", help="Calculation requires non-zero Target Solar Gen, PSH and losses < 100%.")
            st.metric("Estimated PV System Cost (PHP)", "N/A")
        pv_results_dict = { 'predicted_system_size_kwp': predicted_system_size_kwp if predicted_system_size_kwp > 0 else "N/A", 'total_pv_system_cost_php': total_pv_system_cost_php if total_pv_system_cost_php > 0 else "N/A" }
        st.markdown("---")

        # --- Correction Factor Section ---
        st.subheader("Correction Factor Calculation (Compare Simulation with Actuals)")
        st.markdown("Enter your actual measured daily totals (kWh) from your energy monitor for the same period.")
        col_act_1, col_act_2, col_act_3 = st.columns(3)
        with col_act_1: actual_solar_kwh = st.number_input("Actual Solar Generated (kWh)", min_value=0.0, value=sim_total_solar_gen_kwh, step=0.1, format="%.2f", key="actual_solar")
        with col_act_2: actual_batt_discharged_kwh = st.number_input("Actual Battery Discharged (kWh)", min_value=0.0, value=sim_total_discharged_for_load_wh/1000.0, step=0.1, format="%.2f", key="actual_batt_dis", help="Total energy that came *out* of the battery.")
        with col_act_3: actual_grid_import_kwh = st.number_input("Actual Grid Import (kWh)", min_value=0.0, value=sim_total_grid_import_wh/1000.0, step=0.1, format="%.2f", key="actual_grid")
        st.markdown("**Calculated Correction Factors (Actual / Simulated):**")
        col_fac_1, col_fac_2, col_fac_3 = st.columns(3)
        sim_solar_kwh_for_factor = sim_total_solar_gen_kwh
        solar_factor, batt_dis_factor, grid_factor = "N/A", "N/A", "N/A"
        with col_fac_1:
            if sim_solar_kwh_for_factor > 0.01: solar_factor = actual_solar_kwh / sim_solar_kwh_for_factor; st.metric("Solar Gen. Factor", f"{solar_factor:.2f}")
            else: st.metric("Solar Gen. Factor", "N/A", help="Simulated solar was zero.")
        sim_batt_discharge_kwh = sim_total_discharged_for_load_wh / 1000.0
        with col_fac_2:
            if sim_batt_discharge_kwh > 0.01: batt_dis_factor = actual_batt_discharged_kwh / sim_batt_discharge_kwh; st.metric("Battery Discharge Factor", f"{batt_dis_factor:.2f}")
            else: st.metric("Battery Discharge Factor", "N/A", help="Simulated battery discharge was zero.")
        sim_grid_import_kwh = sim_total_grid_import_wh / 1000.0
        with col_fac_3:
            if sim_grid_import_kwh > 0.01: grid_factor = actual_grid_import_kwh / sim_grid_import_kwh; st.metric("Grid Import Factor", f"{grid_factor:.2f}")
            else: st.metric("Grid Import Factor", "N/A", help="Simulated grid import was zero.")
        st.markdown("---")

        # --- ROI Section ---
        st.subheader("Return on Investment (ROI) Estimations")
        st.markdown("Enter cost information to estimate payback period and savings.")
        col_roi_gen_1, col_roi_gen_2, col_roi_gen_3 = st.columns(3)
        with col_roi_gen_1: price_kwh_php = st.number_input("Grid Electricity Price (PHP/kWh)", min_value=0.1, value=10.0, step=0.1, format="%.2f", key="price_kwh")
        with col_roi_gen_2: price_increase_percent = st.number_input("Annual Price Increase (%)", min_value=0.0, value=3.0, step=0.1, format="%.1f", key="price_inc")
        with col_roi_gen_3: pass

        st.markdown("#### Battery System ROI")
        col_roi_batt_1, col_roi_batt_2 = st.columns(2)
        with col_roi_batt_1: battery_system_cost_php = st.number_input("Battery System Cost (PHP)", min_value=0.0, value=70000.0, step=1000.0, format="%.0f", key="system_cost_batt") # Default 70000
        with col_roi_batt_2: battery_degradation_percent = st.number_input("Annual Battery Deterioration (% Capacity Loss)", min_value=0.0, max_value=10.0, value=1.5, step=0.1, format="%.1f", key="batt_degrade", help="Estimated percentage loss of battery effectiveness per year.")
        sim_battery_used_kwh = sim_total_discharged_for_load_wh / 1000.0
        daily_savings_batt_php_yr1 = sim_battery_used_kwh * price_kwh_php
        annual_savings_batt_php_yr1 = daily_savings_batt_php_yr1 * 365
        payback_years_batt, cumulative_savings_batt_php, max_years_sim, annual_savings_batt_list, battery_life_years = -1, 0.0, 30, [], -1
        degradation_rate_batt = battery_degradation_percent / 100.0
        price_increase_rate = price_increase_percent / 100.0
        if annual_savings_batt_php_yr1 > 0 and battery_system_cost_php > 0:
            for year in range(1, max_years_sim + 1):
                current_price_kwh = price_kwh_php * ((1 + price_increase_rate) ** (year - 1))
                battery_effectiveness_factor = (1 - degradation_rate_batt) ** (year - 1)
                degraded_battery_used_kwh = sim_battery_used_kwh * battery_effectiveness_factor
                annual_savings_this_year = degraded_battery_used_kwh * current_price_kwh * 365
                annual_savings_batt_list.append(annual_savings_this_year)
                cumulative_savings_batt_php += annual_savings_this_year
                if cumulative_savings_batt_php >= battery_system_cost_php and payback_years_batt == -1:
                    savings_needed_this_year = battery_system_cost_php - (cumulative_savings_batt_php - annual_savings_this_year)
                    fraction_of_year = savings_needed_this_year / annual_savings_this_year if annual_savings_this_year > 0.01 else 0
                    payback_years_batt = (year - 1) + fraction_of_year
                if battery_effectiveness_factor <= 0.20 and battery_life_years == -1 and degradation_rate_batt > 0: battery_life_years = year
            if payback_years_batt == -1: payback_years_batt = float('inf')
            if battery_life_years == -1 and degradation_rate_batt > 0: battery_life_years = max_years_sim
            elif degradation_rate_batt == 0: battery_life_years = float('inf')
        total_savings_batt_during_life = 0
        if battery_life_years != float('inf') and battery_life_years > 0:
            life_years_int = min(int(math.ceil(battery_life_years)), len(annual_savings_batt_list))
            total_savings_batt_during_life = sum(annual_savings_batt_list[:life_years_int])
        elif battery_life_years == float('inf') and len(annual_savings_batt_list)>0: total_savings_batt_during_life = sum(annual_savings_batt_list)
        elif len(annual_savings_batt_list) == 0: total_savings_batt_during_life = 0.0

        col_roi_res_batt_1, col_roi_res_batt_2, col_roi_res_batt_3 = st.columns(3)
        with col_roi_res_batt_1:
            if payback_years_batt == float('inf'): st.metric("Battery Payback Period", "Never", help="Savings from battery alone may be too low or cost too high.")
            elif payback_years_batt != -1: st.metric("Battery Payback Period", f"{payback_years_batt:.1f} Years")
            else: st.metric("Battery Payback Period", "N/A", help="Requires non-zero cost and savings.")
        with col_roi_res_batt_2:
             if battery_life_years == float('inf'): st.metric("Est. Battery Life", "> 30 Years", help="Based on low degradation rate.")
             elif battery_life_years != -1: st.metric("Est. Battery Life", f"~ {battery_life_years:.0f} Years", help="Based on annual deterioration hitting 80% capacity loss.")
             else: st.metric("Est. Battery Life", "N/A")
        with col_roi_res_batt_3:
             if battery_life_years == float('inf') or np.isnan(total_savings_batt_during_life): st.metric(f"Battery Savings (Est. Life)", "Very High / N/A")
             elif battery_life_years != -1 : st.metric(f"Battery Savings (Est. Life)", f"PHP {total_savings_batt_during_life:,.2f}")
             else: st.metric(f"Battery Savings (Est. Life)", "N/A")
        st.caption("_Battery ROI considers savings only from battery discharging to avoid grid import. Life estimated to 80% capacity loss._")
        st.markdown("---")

        st.markdown("#### PV System ROI")
        pv_deterioration_percent = st.number_input("Annual PV System Deterioration (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.1, format="%.1f", key="pv_degrade", help="Estimated percentage loss of PV panel output per year.")
        sim_solar_gen_kwh_yr1 = sim_total_solar_gen_kwh
        daily_savings_pv_php_yr1 = sim_solar_gen_kwh_yr1 * price_kwh_php
        annual_savings_pv_php_yr1 = daily_savings_pv_php_yr1 * 365
        payback_years_pv, cumulative_savings_pv_php, annual_savings_pv_list = -1, 0.0, []
        pv_system_life_years = max_years_sim
        pv_deterioration_rate = pv_deterioration_percent / 100.0
        if annual_savings_pv_php_yr1 > 0 and total_pv_system_cost_php > 0:
             for year in range(1, max_years_sim + 1):
                current_price_kwh = price_kwh_php * ((1 + price_increase_rate) ** (year - 1))
                pv_effectiveness_factor = (1 - pv_deterioration_rate) ** (year - 1)
                degraded_solar_gen_kwh = sim_solar_gen_kwh_yr1 * pv_effectiveness_factor
                annual_savings_this_year = degraded_solar_gen_kwh * current_price_kwh * 365
                annual_savings_pv_list.append(annual_savings_this_year)
                cumulative_savings_pv_php += annual_savings_this_year
                if cumulative_savings_pv_php >= total_pv_system_cost_php and payback_years_pv == -1:
                    savings_needed_this_year = total_pv_system_cost_php - (cumulative_savings_pv_php - annual_savings_this_year)
                    fraction_of_year = savings_needed_this_year / annual_savings_this_year if annual_savings_this_year > 0.01 else 0
                    payback_years_pv = (year - 1) + fraction_of_year
             if payback_years_pv == -1: payback_years_pv = float('inf')
        total_savings_pv_during_life = sum(annual_savings_pv_list)

        col_roi_res_pv_1, col_roi_res_pv_2, col_roi_res_pv_3 = st.columns(3)
        with col_roi_res_pv_1:
             if payback_years_pv == float('inf'): st.metric("PV Payback Period", "Never", help="Savings may be too low or cost too high.")
             elif payback_years_pv != -1: st.metric("PV Payback Period", f"{payback_years_pv:.1f} Years")
             else: st.metric("PV Payback Period", "N/A", help="Requires non-zero cost and savings.")
        with col_roi_res_pv_2: st.metric("PV System Life Assumed", f"{pv_system_life_years} Years", help="Used for Total Savings calculation.")
        with col_roi_res_pv_3: st.metric(f"PV Savings ({pv_system_life_years} Years)", f"PHP {total_savings_pv_during_life:,.2f}")
        st.caption("_PV ROI considers savings from all generated solar energy offsetting grid costs. Does not include maintenance, inverter replacement, etc._")

        st.markdown("---")
        total_system_cost = battery_system_cost_php + total_pv_system_cost_php
        st.metric("Total Estimated System Cost (PV + Battery)", f"PHP {total_system_cost:,.2f}")
        # --- End ROI Sections ---


        # --- Client Summary Section ---
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown("---")
        st.subheader("📋 Simulation Summary for Client")

        pv_size_for_summary = pv_results_dict.get('predicted_system_size_kwp', 'N/A')
        pv_size_str = f"{pv_size_for_summary:.2f}" if isinstance(pv_size_for_summary, (int, float)) else pv_size_for_summary

        st.write(f"Based on your inputs, here's an estimate of how a {pv_size_str} kWp PV system combined with a {total_nominal_capacity_ah:.0f} Ah battery might perform:")

        total_load_met_kwh = sim_total_daily_load_kwh
        if total_load_met_kwh > 0.01:
            percent_solar = (sim_total_load_met_by_solar_wh / 1000.0) / total_load_met_kwh * 100
            percent_battery = (sim_total_load_met_by_battery_wh / 1000.0) / total_load_met_kwh * 100
            percent_grid = (sim_total_grid_import_wh / 1000.0) / total_load_met_kwh * 100
            total_percent = percent_solar + percent_battery + percent_grid
            if total_percent > 100.1:
                 scale_factor = 100 / total_percent
                 percent_solar *= scale_factor; percent_battery *= scale_factor; percent_grid *= scale_factor
        else: percent_solar, percent_battery, percent_grid = 0, 0, 0

        st.subheader(f"""
        **On a typical day, your energy sources would be approximately:**
        *   ☀️ **Solar:** Provided **{sim_total_load_met_by_solar_wh / 1000.0:.2f} kWh** (covering ~{percent_solar:.0f}% of your load directly).
        *   🔋 **Battery:** Supplied **{sim_total_load_met_by_battery_wh / 1000.0:.2f} kWh** (covering ~{percent_battery:.0f}% of your load, using stored solar).
        *   🔌 **Grid:** Imported **{sim_total_grid_import_wh / 1000.0:.2f} kWh** (covering ~{percent_grid:.0f}% of your load).

        *This means the system potentially covers **{min(100, percent_solar + percent_battery):.0f}%** of your daily load with solar energy (directly or stored).*
        """)

        payback_pv_display = f"{payback_years_pv:.1f} years" if isinstance(payback_years_pv, (int,float)) and payback_years_pv != float('inf') else "Never / N/A"
        payback_batt_display = f"{payback_years_batt:.1f} years" if isinstance(payback_years_batt, (int,float)) and payback_years_batt != float('inf') else "Never / N/A"

        st.subheader(f"""
        **Estimated Financials:**
        *   **Total Upfront Cost (PV + Battery):** PHP {total_system_cost:,.2f}
        *   **PV System Payback:** Approximately **{payback_pv_display}** (based on offsetting grid costs with *all* solar generation).
        *   **Battery Payback:** Approximately **{payback_batt_display}** (based *only* on savings from battery offsetting grid costs).
        *   **Total PV Savings (30 Years):** Est. PHP {total_savings_pv_during_life:,.2f}
        *   **Total Battery Savings (Est. {battery_life_years:.0f} Yr Life):** Est. PHP {total_savings_batt_during_life:,.2f}
        """)

        usable_batt_kwh = (total_nominal_capacity_ah * battery_voltage_v * max_dod)/1000.0

        st.subheader(f"""
        **System Notes:**
        *   Recommended PV System Size: **{pv_size_str} kWp**
        *   Battery Capacity: **{total_nominal_capacity_ah:.0f} Ah** at **{battery_voltage_v} V** (~{usable_batt_kwh :.1f} kWh usable)
        *   Battery Discharge allowed between **{battery_discharge_start_hour:02d}:00** and **{battery_discharge_end_hour:02d}:00**.
        """)

        st.caption("Please Note: These are estimations based on the provided inputs and simplified models. Actual performance and savings can vary due to weather, actual usage patterns, equipment degradation differing from estimates, future electricity price changes, and potential maintenance costs not included here.")
        # --- End Client Summary Section ---
        
        # --- Configuration Summary ---
        st.markdown("---")
        st.write(f"**Detailed Configuration Used (for reference):**")
        st.write(f"- Battery: {total_nominal_capacity_ah:.0f} Ah @ {battery_voltage_v:.0f}V ({simulation_results_data['total_nominal_capacity_wh']:.0f} Wh nominal)")
        st.write(f"- Settings: {max_dod_percent:.0f}% Max DoD (Min SoC: {min_soc_p_for_graph:.0f}%), Initial SoC: {initial_soc_percent:.0f}%")
        st.write(f"- Max Battery Charging Power: {max_charging_power_kw:.1f} kW")
        st.write(f"- Max Battery Discharge Power: {max_battery_discharge_power_kw:.1f} kW")
        st.write(f"- Battery Discharge Window: {battery_discharge_start_hour:02d}:00 to {battery_discharge_end_hour:02d}:00")
        st.write(f"- Inverter Efficiency: {inverter_efficiency_percent:.0f}%")
        st.write(f"- Target Daily Solar (6AM-4PM): {sim_total_solar_gen_kwh:.2f} kWh")
        st.write(f"- Location/System: {peak_sun_hours:.1f} Peak Sun Hours, {system_loss_percent:.0f}% PV System Losses")

    else: # Simulation failed
        st.warning("Simulation did not produce results (likely due to zero battery capacity).")

else: # run_sim is False
    st.info("Adjust parameters in the sidebar and ensure no errors are shown above to run the simulation.")


st.sidebar.markdown("---")
st.sidebar.info("Adjust parameters, load profiles, solar generation, PSH, losses, and battery discharge window.")
