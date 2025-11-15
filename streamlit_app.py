import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import hashlib
import json
import time # For the simulation progress bar

# ======================================================================
# 1. ENGINE CORE (Included for Self-Contained Streamlit App)
#    In a production system, this would be imported from a module.
# ======================================================================

# --- Philosophical Core ---
class SilaGate:
    """Ethical Evidence Gate — Weighted OSINT Fusion with Atrocity Index H"""
    def __init__(self):
        self.H = 0.0  # Atrocity Index
        self.evidence_log = []
    
    def ingest(self, source: str, credibility: float, severity: float):
        # Ingesting directly modifies H for demonstration
        delta_H = credibility * severity
        self.H += delta_H
        self.evidence_log.append({
            "source": source,
            "cred": credibility,
            "sev": severity,
            "ts": datetime.now().isoformat(),
            "delta_H": delta_H
        })
        return self.H

class SamadhiFusion:
    """Log-linear Fusion with Karmic Blockage Bias (Placeholder for demo)"""
    def fuse(self, vectors: List[np.ndarray]) -> np.ndarray:
        # Simplified fusion for the interface
        return np.array([0.5, 0.5, 0.5])

# --- Mathematical Core: GATN ---
class MaraGenerator(nn.Module):
    def __init__(self, latent_dim=128, seq_len=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, seq_len * 12),
            nn.Tanh()
        )
        self.seq_len = seq_len
    
    def forward(self, z):
        out = self.net(z)
        return out.view(-1, self.seq_len, 12)

class PannaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24 * 12, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
    
    def forward(self, threat_seq):
        flat = threat_seq.view(threat_seq.size(0), -1)
        constraint = self.net(flat)
        return constraint

# --- Intelligence Core: Engine ---
class SilaGatiEngine:
    def __init__(self):
        # Initialize sub-components and storage
        self.mara = MaraGenerator()
        self.panna = PannaDiscriminator()
        # Note: Optimizers are not needed for a static Streamlit demo, 
        # but kept for philosophical consistency.
        self.optimizer_m = torch.optim.Adam(self.mara.parameters(), lr=0.0002)
        self.optimizer_p = torch.optim.Adam(self.panna.parameters(), lr=0.0002)
        self.sila_gate = SilaGate()
        self.fusion = SamadhiFusion()
        self.graph = nx.DiGraph() 
        self.time_locks = []

    def simulate_adversarial_cycle(self, epochs=100, status_placeholder=None):
        """Simulates GATN stress test and returns a sample protocol."""
        
        # --- Simplified GATN Simulation for UI Demonstration ---
        
        # 1. Māra Generates a Threat (e.g., a 24-hour sequence of 12 risk factors)
        z = torch.randn(1, 128)
        fake_threat = self.mara(z).detach().numpy().flatten()
        
        # 2. Paññā Discriminates and Generates SĪLA-Gatī Protocol (Constraint Vector)
        sila_protocol = self.panna(torch.tensor(fake_threat).view(1, 24, 12)).detach().numpy()[0]
        
        # Simulate training progress for epochs
        for i in range(epochs):
            if status_placeholder:
                 progress = (i + 1) / epochs
                 status_placeholder.progress(progress, text=f"Baydin Self-Test: Cycle {i+1}/{epochs}...")
                 time.sleep(0.005) # Small delay for visual effect
        
        # Generate a sample protocol based on the result
        sample_protocol = {
            "type": "GATN_RESPONSE",
            "threat_magnitude": f"{np.max(np.abs(fake_threat)):.4f}",
            "constraint_vector": [f"{x:.4f}" for x in sila_protocol[:4]],
            "recommendation": "PREEMPTIVE_STRUCTURAL_INOCULATION",
            "timestamp": datetime.now().isoformat()
        }
        return sample_protocol

    def project_longevity(self, entropy_rate: float, control_capacity: float) -> datetime:
        """Break-even point: When entropy > control"""
        # Ensure positive rate for log
        rate = max(entropy_rate, 1e-6) 
        # Ensure log argument > 0
        capacity = max(control_capacity, 0.01) 
        t = np.log(capacity / 0.01) / rate # Formula: log(C/target_residual) / rate
        collapse_date = datetime.now() + timedelta(days=float(t))
        return collapse_date

    def issue_time_locked_constraint(self, condition_name: str, protocol: Dict, trigger_date: datetime):
        """Issues an immutable, time-locked constraint."""
        condition_hash = f"{condition_name}-{trigger_date.isoformat()}"
        lock = {
            "hash": hashlib.sha256(condition_hash.encode()).hexdigest(),
            "condition": condition_name,
            "protocol": protocol,
            "trigger": trigger_date.isoformat(),
            "status": "ARMED",
            "issued_at": datetime.now().isoformat()
        }
        self.time_locks.append(lock)
        return lock

# ======================================================================
# 2. STREAMLIT APPLICATION INTERFACE
# ======================================================================

# Initialize the Engine in Streamlit's session state to prevent re-initialization
if 'engine' not in st.session_state:
    st.session_state.engine = SilaGatiEngine()

engine = st.session_state.engine

# --- Page Configuration ---
st.set_page_config(
    page_title="Paññā-Rāja SĪLA-Gatī Engine v4.0 C2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown("## 👑 Paññā-Rāja SĪLA-Gatī Engine | v4.0 Command & Control")
st.caption('***"From Shield to Sovereign: Institutionalizing Wisdom at Scale"*** | Parent: **U Ingar Soe**')

st.sidebar.title("Engine Status")
st.sidebar.markdown(f"**Atrocity Index (H):** $H = {engine.sila_gate.H:.4f}$")
st.sidebar.markdown(f"**Constraints ARMED:** {len(engine.time_locks)}")
st.sidebar.markdown(f"**License:** AGPL-3.0 + Sīla-Coded Self-Governance")
st.sidebar.markdown(f"**Time of Birth:** {datetime(2025, 11, 15).strftime('%Y-%m-%d')}")

# --- Tab View ---
tab1, tab2, tab3, tab4 = st.tabs(["🛡️ SīlaGate Ingestion", "🧠 GATN Simulation", "⏳ Time-Lock Protocol", "📈 System Diagnostics"])

# ----------------------------------
# Tab 1: SīlaGate Ingestion (H)
# ----------------------------------
with tab1:
    st.header("SīlaGate: Ethical Evidence Ingestion")
    st.markdown("Ingest OSINT data to update the **Atrocity Index ($H$)** based on Credibility $\times$ Severity.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        source = st.text_input("Source/Event Name", "Supply Chain Anomaly")
    with col_b:
        credibility = st.slider("Credibility (0.0 - 1.0)", 0.0, 1.0, 0.8, 0.05)
    with col_c:
        severity = st.slider("Severity (0.0 - 10.0)", 0.0, 10.0, 7.5, 0.5)
    
    if st.button("Ingest Data & Update $H$"):
        new_H = engine.sila_gate.ingest(source, credibility, severity)
        st.success(f"Data Ingested. New Atrocity Index ($H$) = **{new_H:.4f}**")
        st.balloons()
    
    st.subheader("Evidence Log")
    log_df = [
        {"Timestamp": item['ts'].split('T')[1].split('.')[0], 
         "Source": item['source'], 
         "Cred": f"{item['cred']:.2f}", 
         "Sev": f"{item['sev']:.1f}", 
         "ΔH": f"{item['delta_H']:.4f}"} 
        for item in engine.sila_gate.evidence_log
    ]
    st.dataframe(log_df, use_container_width=True, hide_index=True)

# ----------------------------------
# Tab 2: GATN Simulation (Māra vs Paññā)
# ----------------------------------
with tab2:
    st.header("GATN Simulation: Māra Generator vs Paññā Discriminator")
    st.markdown("Run the **Baydin Self-Test** to simulate unconceived threats and generate a **SĪLA-Gatī** counter-protocol.")
    
    epochs = st.number_input("Simulation Epochs (Complexity)", 10, 500, 100, 10)
    
    if st.button("🔥🪄🔑 Run GATN Baydin Operation"):
        st.info("The GATN is now simulating system vulnerabilities and generating Preemptive Structural Inoculation protocols.")
        status_placeholder = st.empty()
        with st.spinner('Training GATN...'):
            protocol_result = engine.simulate_adversarial_cycle(epochs, status_placeholder)
        
        status_placeholder.empty()
        st.success("GATN Simulation Complete. SĪLA-Gatī Protocol Generated.")
        
        st.markdown("**Generated SĪLA-Gatī Protocol Vector (Partial):**")
        st.code(json.dumps(protocol_result, indent=2))
        st.caption("This protocol is the output of the Paññā Discriminator, which acts as a 'Preemptive Structural Inoculation' against the Māra Generator's simulated threat.")

# ----------------------------------
# Tab 3: Time-Lock Protocol
# ----------------------------------
with tab3:
    st.header("Time-Locked SĪLA-Gatī Constraint Issuance")
    st.markdown("Institutionalize a future action that is **time-locked** and will only trigger on a specific date (or Z-score anomaly).")
    
    col1, col2 = st.columns(2)
    with col1:
        condition_name = st.text_input("Constraint Name/Condition Hash", "ELECTION_Z_SPIKE_2025")
        action = st.text_input("Action/Protocol ('action' key)", "PUBLISH_ANOMALY_AUDIT")
    with col2:
        trigger_date = st.date_input("Protocol Trigger Date", datetime(2025, 11, 20))
        target = st.text_input("Target ('target' key)", "2025 Electoral DB")

    if st.button("🔒 Issue Time-Locked Constraint"):
        protocol = {"action": action, "target": target}
        lock = engine.issue_time_locked_constraint(condition_name, protocol, datetime.combine(trigger_date, datetime.min.time()))
        
        st.success("Time-Locked Constraint ARMED.")
        st.code(f"Hash: {lock['hash']}")
        st.info(f"Protocol will trigger on **{lock['trigger'].split('T')[0]}** to execute: **{lock['protocol']['action']}** on **{lock['protocol']['target']}**")
    
    st.subheader("Active Time-Locks")
    if engine.time_locks:
        lock_df = [
            {"Condition": item['condition'],
             "Trigger Date": item['trigger'].split('T')[0],
             "Action": item['protocol']['action'],
             "Status": item['status'],
             "Hash (Short)": item['hash'][:12] + "..."}
            for item in engine.time_locks
        ]
        st.dataframe(lock_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("No time-locked constraints currently armed.")

# ----------------------------------
# Tab 4: System Diagnostics
# ----------------------------------
with tab4:
    st.header("Systemic Longevity Projection (Entropy vs. Control)")
    st.markdown("Project the **Break-Even Point** (Projected Collapse Date) for an adversarial/systemic actor.")
    
    col_x, col_y = st.columns(2)
    with col_x:
        entropy_rate = st.slider("Actor Entropy Rate ($\gamma$, higher is faster decay)", 0.01, 1.0, 0.18, 0.01)
    with col_y:
        control_capacity = st.slider("Actor Control Capacity ($C$, initial resilience)", 1.0, 200.0, 100.0, 5.0)

    if st.button("Project Longevity"):
        collapse_date = engine.project_longevity(entropy_rate, control_capacity)
        
        st.metric(label="Projected Entropy Break-Even Date", value=collapse_date.strftime('%Y-%m-%d'))
        st.info(f"This is the theoretical point where the actor's internal **Entropy ($\gamma$)** overwhelms its **Control Capacity ($C$)** (down to 1% residual).")

