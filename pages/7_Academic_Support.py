import streamlit as st

st.set_page_config(page_title="Personalized Academic Support System", page_icon="📘", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .tier-card {
        background: white;
        border-radius: 12px;
        padding: 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    
    .tier-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 12px;
    }
    
    .tag-red { background: #fee2e2; color: #b91c1c; }
    .tag-yellow { background: #fef3c7; color: #b45309; }
    .tag-green { background: #dcfce7; color: #15803d; }
    
    .action-item {
        padding: 8px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 14px;
        color: #334155;
    }
    .action-item:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

st.title("📘 Personalized Academic Support Framework")
st.caption("Actionable, multi-tiered educational interventions based on predicted performance bands.")

subject_tab = st.selectbox(
    "Select Subject for Support Roadmap:",
    ["Mathematics Problem Solving", "Reading Comprehension", "Writing & Language Arts"]
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="tier-card" style="border-top: 4px solid #ef4444;">
        <span class="tier-tag tag-red">Tier 1: Intensive Support (&lt; 50)</span>
        <h4 style="margin: 0 0 12px 0; color: #0f172a;">Core Foundations</h4>
        <div class="action-item">📌 <strong>Foundational Diagnostic:</strong> Target prerequisite gaps.</div>
        <div class="action-item">👥 <strong>Structured 1-on-1 Tutoring:</strong> 2 hours weekly guided problem-solving.</div>
        <div class="action-item">📝 <strong>Step-by-Step Worked Examples:</strong> Reduce cognitive load via scaffolded questions.</div>
        <div class="action-item">⏰ <strong>Test Preparation Course:</strong> Mandatory enrollment in preparation module.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tier-card" style="border-top: 4px solid #f59e0b;">
        <span class="tier-tag tag-yellow">Tier 2: Developing (50 – 69)</span>
        <h4 style="margin: 0 0 12px 0; color: #0f172a;">Application & Consistency</h4>
        <div class="action-item">🎯 <strong>Targeted Practice Sets:</strong> Focus on recurring error patterns.</div>
        <div class="action-item">🤝 <strong>Peer Study Groups:</strong> Collaborative discussions and concept clarification.</div>
        <div class="action-item">⏱️ <strong>Timed Quiz Simulation:</strong> Build exam stamina and time management.</div>
        <div class="action-item">📈 <strong>Bi-weekly Milestone Tracking:</strong> Check mastery of difficult sub-topics.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="tier-card" style="border-top: 4px solid #10b981;">
        <span class="tier-tag tag-green">Tier 3: Advanced Mastery (70 – 100)</span>
        <h4 style="margin: 0 0 12px 0; color: #0f172a;">Enrichment & Leadership</h4>
        <div class="action-item">🚀 <strong>Advanced Synthesis:</strong> Multi-step challenge problem sets.</div>
        <div class="action-item">🧑‍🏫 <strong>Peer Tutoring Mentorship:</strong> Teach peers to solidify deep conceptual mastery.</div>
        <div class="action-item">💡 <strong>Independent Research/Projects:</strong> Real-world applications and inquiry.</div>
        <div class="action-item">🌟 <strong>Honors & Olympiad Preparation:</strong> Extended curriculum enrichment.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("💡 **Implementation Recommendation:** Interventions are most impactful when deployed in the first 4 weeks of the academic semester.")
