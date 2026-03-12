import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

#Page Theme 
st.set_page_config(page_title="AI Dependency Awareness", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    
    /* Shiny Glowing Titles with Blue Border */
    h1, h2, h3 { 
        color: #00d4ff !important; 
        font-weight: 800; 
        border-bottom: 3px solid #00d4ff;
        box-shadow: 0 5px 15px -5px rgba(0, 212, 255, 0.6);
        padding-bottom: 8px;
        margin-bottom: 25px !important;
        text-shadow: 0 0 12px rgba(0, 212, 255, 0.4);
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
    }
    /* Insight Cards */
    .insight-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #7a00ff;
        padding: 15px;
        margin-top: 5px;
        margin-bottom: 25px;
        border-radius: 0 12px 12px 0;
        font-size: 0.95rem;
    }
    /* Luxury Compact Table Styling */
    .styled-table-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        max-width: 95%;
        margin: auto;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #f0f9ff; /* Light Pearl/Ice Blue */
        color: #1e3a8a; /* Deep Navy Blue */
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem; /* Smaller Font */
    }
    .custom-table thead th {
        background-color: #00d4ff;
        color: white;
        text-align: left;
        padding: 8px 10px; /* Reduced padding */
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .custom-table tbody td {
        padding: 6px 10px; /* Reduced padding */
        border-bottom: 1px solid #e2e8f0;
        font-weight: 700; /* Bold Font */
    }
    .custom-table tbody tr:hover {
        background-color: #e0f2fe;
    }
    .fun-note {
        color: #00ffcc;
        font-style: italic;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean():
    df = pd.read_csv("ai_dependency_awareness_dataset.csv")
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        'primary_task_type':'task_type',
        'ai_usage_frequency_per_day':'ai_usage_day',
        'task_completion_without_ai_score': 'score_no_ai',
        'task_completion_with_ai_score': 'score_with_ai',
        'self_perceived_ai_dependency': 'ai_dep_score',
        'confidence_without_ai': 'conf_no_ai',
        'confidence_with_ai': 'conf_with_ai',
        'awareness_of_ai_dependency': 'awareness_ai'
    }, inplace=True)
    
    df['ai_impact'] = df['score_with_ai'] - df['score_no_ai']
    df['confidence_boost'] = df['conf_with_ai'] - df['conf_no_ai']
    df['risk_score'] = ((df['ai_usage_day']/14)*30 + (df['ai_dep_score']/10)*40 + ((10-df['score_no_ai'])/10)*30)
    
    avg_usage = df['ai_usage_day'].mean()
    def classify_users(row):
        if row['awareness_ai'] == 'High': return 'Strategic'
        elif row['ai_usage_day'] > avg_usage: return 'Blind Reliance'
        else: return 'Casual'
    df['Final_Profile'] = df.apply(classify_users, axis=1)
    return df

df = load_and_clean()

# side bar
with st.sidebar:
    st.title("Project Overview")
    st.markdown("""**Dataset Description:** This dataset analyzes the relationship between AI usage frequency, task performance, and human confidence, with a special focus on users’ awareness of their own AI dependency.""")
    st.divider()
    st.markdown("### Filters")
    task_filter = st.multiselect("Select Task Type", options=df['task_type'].unique(), default=df['task_type'].unique())
    mindset_filter = st.selectbox("Select User Profile", options=["All Mindsets"] + list(df['Final_Profile'].unique()))
    usage_slider = st.slider("Daily AI Usage (Hours)", 0.0, 15.0, (0.0, 15.0))

filtered_df = df[(df['task_type'].isin(task_filter)) & (df['ai_usage_day'].between(usage_slider[0], usage_slider[1]))]
if mindset_filter != "All Mindsets":
    filtered_df = filtered_df[filtered_df['Final_Profile'] == mindset_filter]

# main page
st.title("🤖 AI DEPENDENCY AWARENESS")
st.markdown("#### Welcome to my project! Explore the evolving relationship between human skills and AI, and how AI tools are reshaping the way we work and think.")

# the dataset perview
with st.expander("✨ Click here to preview the dataset!"):
    st.markdown("### Participant Data Overview")
    
    html_table = f"""
    <div class="styled-table-container">
        <table class="custom-table">
            <thead>
                <tr>
                    {"".join([f"<th>{col}</th>" for col in filtered_df.columns])}
                </tr>
            </thead>
            <tbody>
                {"".join([
                    f"<tr>{''.join([f'<td>{val}</td>' for val in row])}</tr>" 
                    for row in filtered_df.head(10).values
                ])}
            </tbody>
        </table>
    </div>
    """
    st.write(html_table, unsafe_allow_html=True)
    st.caption("Displaying top 10 records with optimized compact UI.")

#Summary Statistics
st.header("⭐ Summary Statistics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("AI Dependency Risk", f"{filtered_df['risk_score'].mean():.1f}%")
m2.metric("AI Score Improvement", f"+{filtered_df['ai_impact'].mean():.2f}")
m3.metric("Dependency Level", f"{filtered_df['ai_dep_score'].mean():.1f} / 10")
m4.metric("Confidence Gain from AI", f"+{filtered_df['confidence_boost'].mean():.2f}")

st.divider()

def render_insight(text):
    st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)

cold_neons = ['#00d4ff', '#7a00ff', '#00ffcc', '#2563eb', '#a855f7', '#10b981']

#VISUALIZATIONS 
c1, c2 = st.columns(2)
with c1:
    st.header("⭐ Performance Comparison")
    st.markdown('<p class="fun-note">Ever wondered if AI actually makes a difference in the performance? Let’s look at the numbers!</p>', unsafe_allow_html=True)
    avg_scores = [filtered_df['score_no_ai'].mean(), filtered_df['score_with_ai'].mean()]
    fig1 = go.Figure(data=[go.Bar(x=['Without AI', 'With AI'], y=avg_scores, marker_color=['#334155', '#00d4ff'])])
    fig1.update_layout(template="plotly_dark", yaxis_title="Average Score", height=300)
    st.plotly_chart(fig1, use_container_width=True)
    render_insight("The Insight💡: AI assistance increased the average task completion score from 5.70 to 7.59, highlighting its effectiveness in improving performance.")

with c2:
    st.header("⭐ Different Roles Using AI")
    st.markdown('<p class="fun-note">Which group relies on AI the most? Let’s look at the data to find out.</p>', unsafe_allow_html=True)
    usage_stats = filtered_df.groupby('role')['ai_usage_day'].mean()
    fig2 = px.pie(values=usage_stats, names=usage_stats.index, hole=0.5, color_discrete_sequence=cold_neons)
    fig2.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig2, use_container_width=True)
    render_insight("The Insight💡: Students have the highest AI usage frequency, indicating that AI plays a major role in their academic tasks.")

c3, c4 = st.columns(2)
with c3:
    st.header("⭐ Age Distribution of AI Users")
    st.markdown('<p class="fun-note">Is AI just for the young generation? Let’s check the age spread of our 500 participants.</p>', unsafe_allow_html=True)
    fig3 = px.histogram(filtered_df, x="age", nbins=20, color_discrete_sequence=['#7a00ff'])
    fig3.update_layout(template="plotly_dark", height=300, yaxis_title="Number of Users")
    st.plotly_chart(fig3, use_container_width=True)
    render_insight("The Insight💡: The age distribution shows that participants from a wide range of age groups are actively using AI.")

with c4:
    st.header("⭐ Confidence Distribution in AI")
    st.markdown('<p class="fun-note">How much do people actually trust their work with AI? Here is the spread of confidence across roles.</p>', unsafe_allow_html=True)
    fig4 = px.box(filtered_df, x='role', y='conf_with_ai', color='role', template="plotly_dark", color_discrete_sequence=cold_neons)
    fig4.update_layout(height=300, yaxis_title="Average Score", showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    render_insight("The Insight💡: Confidence levels are high across all roles when AI is used, regardless of background.")

c5, c6 = st.columns(2)
with c5:
    st.header("⭐ Skill Improvement vs Confidence Boost")
    st.markdown('<p class="fun-note">Does AI actually improve work, or just add a fake confidence boost?</p>', unsafe_allow_html=True)
    skill_conf = filtered_df.groupby('task_type')[['ai_impact', 'confidence_boost']].mean().reset_index()
    fig5 = px.bar(skill_conf, x='task_type', y=['ai_impact', 'confidence_boost'], barmode='group', color_discrete_sequence=['#00d4ff', '#7a00ff'])
    fig5.update_layout(template="plotly_dark", height=300, yaxis_title="Average Increase")
    st.plotly_chart(fig5, use_container_width=True)
    render_insight("The Insight💡: AI enhances performance and confidence in learning tasks. But for writing tasks, it mostly boosts confidence.")

with c6:
    st.header("⭐ Dependency Level")
    st.markdown('<p class="fun-note">Which types of tasks make us feel the most "stuck" without AI?</p>', unsafe_allow_html=True)
    task_dep = filtered_df.groupby('task_type')['ai_dep_score'].mean().reset_index()
    fig9 = px.bar(task_dep, x='task_type', y='ai_dep_score', color='task_type', color_discrete_sequence=px.colors.sequential.Blues_r)
    fig9.update_layout(template="plotly_dark", height=300, yaxis_title="Average Score", showlegend=False)
    st.plotly_chart(fig9, use_container_width=True)
    render_insight("The Insight💡: High reliance scores across all task types prove that AI dependency is not limited to complex tasks.")

st.header("⭐ AI Users Profiles")
st.markdown("""
**Categorizing users into three distinct profiles by intersecting usage frequency with risk awareness:**
* **Strategic:** Moderate AI usage combined with high awareness of potential risks.
* **Casual:** Low AI usage with low awareness of dependency risks.
* **Blind Reliance:** High AI usage with low awareness of dependency risks.
""")
fig6 = px.histogram(filtered_df, x='Final_Profile', color='Final_Profile', color_discrete_map={'Strategic':'#10b981', 'Casual':'#3b82f6', 'Blind Reliance':'#7a00ff'})
fig6.update_layout(template="plotly_dark", height=350, yaxis_title="Number of Users")
st.plotly_chart(fig6, use_container_width=True)
render_insight("The Insight💡: The Blind Reliance group represents the largest portion of users, highlighting the need for responsible usage.")

c7, c8 = st.columns(2)
with c7:
    st.header("⭐ AI Efficiency")
    st.markdown('<p class="fun-note">Where does AI truly shine? This heatmap shows the biggest boost per role.</p>', unsafe_allow_html=True)
    heatmap_data = filtered_df.pivot_table(index='role', columns='task_type', values='ai_impact', aggfunc='mean')
    fig7 = px.imshow(heatmap_data, text_auto=".2f", color_continuous_scale='Blues')
    fig7.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig7, use_container_width=True)
    render_insight("The Insight💡: AI acts as a 'Tutor' for Freelancers, delivering its peak impact in their Learning journey.")

with c8:
    st.header("⭐ AI Dependency Awareness")
    st.markdown('<p class="fun-note">Does knowing the risks help you use AI less? (The Awareness Paradox)</p>', unsafe_allow_html=True)
    fig8 = px.box(filtered_df, x='awareness_ai', y='ai_dep_score', color='awareness_ai', color_discrete_sequence=['#3b82f6', '#00d4ff', '#10b981'])
    fig8.update_layout(template="plotly_dark", height=350, yaxis_title="Average Score", showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)
    render_insight("The Insight💡: High awareness does not guarantee lower dependency; knowing the risk doesn’t always change the behavior.")

# PREDICTOR 
st.divider()
st.header("⭐ AI Risk Predictor")
st.markdown("_Fill in your details below to see how reliant you might be on AI._")

df['risk_score'] = (
    (df['ai_usage_day']/14)*30 + 
    (df['ai_dep_score']/10)*40 + 
    ((10-df['score_no_ai'])/10)*30
)

features = ['ai_usage_day', 'ai_dep_score', 'score_no_ai', 'conf_no_ai']
X = df[features]
y = df['risk_score']

model = LinearRegression().fit(X, y)

with st.container():
    p1, p2 = st.columns(2)
    with p1:
        
        u_usage = st.slider("Daily AI Usage (Hours)", 0.0, 24.0, 4.0)
        u_dep = st.slider("Perceived Dependency (1-10)", 1, 10, 5)
    with p2:
        u_conf = st.slider("Confidence WITHOUT AI (1-10)", 1, 10, 5)
        u_score = st.slider("Predicted Score WITH AI (1-10)", 1, 10, 5)

    prediction = np.clip(model.predict([[u_usage, u_dep, u_score_no, u_conf]])[0], 0, 100)
    
    st.markdown(f"### Your Risk Level is: `{prediction:.2f}%`")
    st.progress(prediction / 100)
    
    if prediction < 40:
        st.success("🟢 **Low Dependency Status**")
        st.info("You are maintaining a balanced relationship with AI tools. Continue using AI to enhance productivity while preserving your independent thinking and creativity when solving problems.")
    elif 40 <= prediction <= 70:
        st.warning("🟡 **Moderate Dependency Status**")
        st.info("AI is becoming an important part of your workflow. While this can improve efficiency, it is essential to carefully review and question AI-generated results instead of assuming they are always correct.")
    else:
        st.error("🔴 **High Dependency Status**")
        st.info("Your workflow shows a strong reliance on AI tools, which may make your skills vulnerable. It is recommended to regularly practice working independently without AI assistance to maintain and strengthen your abilities.")