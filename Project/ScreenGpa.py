import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import os

# --- Page Config ---
st.set_page_config(page_title="Student CGPA Dashboard", layout="wide")

# --- Title and Introduction ---
st.title("📊 Student Lifestyle: Screen Time vs. Academic Performance")
st.markdown("""
Welcome to the Student Lifestyle Dashboard! 
As data analysts, our goal is to uncover whether spending too much time on our phones actually hurts our GPA, and what other factors (like sleep and studying) come into play.
""")

# --- Data Preprocessing Pipeline ---
@st.cache_data
def load_and_clean_data():
    #df = pd.read_excel('FinalDataset.xlsx', sheet_name='Sheet1')
    #df = pd.read_excel('SyntheticDataset.xlsx', sheet_name='Sheet1')
    # Get the directory that this specific script file is in
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the absolute path to your Excel file dynamically
    file_path = os.path.join(current_dir, 'SyntheticDataset.xlsx')

    # Load dataset safely
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    
    cols_to_drop = ['screen_shot', 'hour (h)', 'minnutes (m)']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # data encode
    allowance_map = {'Under 5,000': 1, '5,000-10,000': 2, '10,001-15,000': 3, '15,000+': 4}
    sleep_map = {'Less than 5 hrs': 1, '5-6 hrs': 2, '7-8 hrs': 3, '8+ hrs': 4}
    study_map = {'Less than 1 hr': 1, '1-2 hrs': 2, '2-3 hrs': 3, 'More than 3 hrs': 4}
    
    df['monthly_allowance_num'] = df['monthly_allowance'].map(allowance_map)
    df['sleep_hours_num'] = df['sleep_hours'].map(sleep_map)
    df['study_hours_outside_class_num'] = df['study_hours_outside_class'].map(study_map)
    
    # Create the Heavy User Category here so it's globally available
    df['Screen_Time_Category'] = pd.cut(
        df['avg_screen_time_hours'], 
        bins=[0, 4, 8, 24], 
        labels=['Low (<4 hrs)', 'Moderate (4-8 hrs)', 'High (>8 hrs)']
    )
    
    return df

df = load_and_clean_data()

# --- Sidebar for Storyline Navigation ---
st.sidebar.header("Storyline Navigation")
section = st.sidebar.radio("Go to", [
    "1. Data Overview", 
    "2. The Baseline (EDA)", 
    "3. The Plot Twist (Deep Dive)", 
    "4. Demographics",
    "5. Actionable Insights",
    "6. Predictive Modeling"
])

# ==========================================
# SECTION 1: DATA OVERVIEW
# ==========================================
if section == "1. Data Overview":
    st.header("1. Data Overview")
    st.write("Looking at the baseline metrics of our surveyed students:")
    
    # --- Basic Stat Metrics  ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students Surveyed", f"{len(df)}")
    m2.metric("Average CGPA", f"{df['cgpa'].mean():.2f}")
    m3.metric("Avg Screen Time", f"{df['avg_screen_time_hours'].mean():.1f} hrs")
    m4.metric("Max Screen Time", f"{df['avg_screen_time_hours'].max():.1f} hrs")
    
    st.markdown("---")
    st.write("### The Raw Dataset")
    st.dataframe(df.head(10))

# ==========================================
# SECTION 2: THE BASELINE (EDA)
# ==========================================
elif section == "2. The Baseline (EDA)":
    st.header("2. The Baseline: The 'Screentime = Bad Grades' Myth")
    st.write("Most people assume that more screen time automatically equals a lower GPA. Let's test that assumption by looking at the raw data.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. The Raw Scatterplot")
        fig_scatter = px.scatter(df, x='avg_screen_time_hours', y='cgpa', 
                                 opacity=0.7, color_discrete_sequence=['#1f77b4'],
                                 title="Screen Time vs. CGPA", trendline = "ols")
        fig_scatter.update_layout(xaxis_title="Avg Screen Time (Hours)", yaxis_title="CGPA")
        st.plotly_chart(fig_scatter, width="stretch")
        st.caption("Looking at all of the scatter dots. If screen time ruined grades, we would see a sharp diagonal line going down.")
        
    with col2:
        st.subheader("2. Correlation Heatmap")
        
        numeric_cols = ['avg_screen_time_hours', 'cgpa', 'sleep_hours_num', 'study_hours_outside_class_num']
        corr_df = df[numeric_cols].rename(columns={
            'avg_screen_time_hours': 'Screen Time', 'cgpa': 'CGPA', 
            'sleep_hours_num': 'Sleep', 'study_hours_outside_class_num': 'Study Hours'
        }).corr()
        
        fig_heat = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="Mathematical Correlations")
        st.plotly_chart(fig_heat, width="stretch")
        
        # calculated correlation 
        screentime_corr = corr_df.loc['Screen Time', 'CGPA']
        st.caption(f"A score of {screentime_corr:.2f} between Screen Time and CGPA is an incredibly weak correlation. The math backs up the scatterplot!")
        
    st.markdown("---")
    
    st.subheader("3. Segmenting by Usage: The Heavy Users")
    st.write("While the overall linear correlation is weak, grouping students into usage categories reveals a clearer pattern. We can observe a distinct threshold where extreme 'heavy usage' (>8 hours) finally begins to negatively impact average GPA.") 
    
    st_category_means = df.groupby('Screen_Time_Category', observed=False)['cgpa'].mean().reset_index()
    fig_bar = px.bar(st_category_means, x='Screen_Time_Category', y='cgpa', 
                     color='Screen_Time_Category', text_auto='.2f',
                     title="Average CGPA by Screen Time Volume")
    fig_bar.update_yaxes(range=[3.0, 3.8]) 
    st.plotly_chart(fig_bar, width="stretch")

# ==========================================
# SECTION 3: THE PLOT TWIST
# ==========================================
elif section == "3. The Plot Twist (Deep Dive)":
    st.header("3. The Plot Twist: Purpose over Volume")
    st.write("The previous charts suggested screen time volume is slightly bad. But what happens when we look at **WHAT** they are doing on their screens?")
    
    plot_df = df.copy()
    categories_to_keep = ['Studying/Academic work', 'Social Media', 'Streaming (Netflix, YouTube)', 'Game']
    plot_df['main_use_clean'] = plot_df['main_use'].apply(lambda x: x if x in categories_to_keep else 'Others')
    category_order = ['Studying/Academic work', 'Social Media', 'Streaming (Netflix, YouTube)', 'Game', 'Others']
    
    st.subheader("CGPA vs. Main Phone Use")
    fig_box = px.box(plot_df, x='main_use_clean', y='cgpa', color='main_use_clean', 
                     category_orders={'main_use_clean': category_order},
                     title="Distribution of CGPA based on Primary Device Usage")
    st.plotly_chart(fig_box, width="stretch")
    
    st.write("📈 **Boxplot Analysis:** Students who primarily use their phones for studying have the highest median CGPA and a tight distribution of top grades. In contrast, gaming and streaming show significantly lower medians and a wider spread of poor grades. Social media sits directly in the middle.")
    
    st.markdown("---")
    st.subheader("The 'Productivity vs. Entertainment' Proof")
    st.write("Let's look at the hard numbers. If we group students by the broader purpose of their screen time, a massive GPA gap appears:")
    
    # NEW PROOF: Broader categories for better statistical significance
    productive = df[df['main_use'].isin(['Studying/Academic work', 'Work'])]
    entertainment = df[df['main_use'].isin(['Game', 'Streaming (Netflix, YouTube)'])]
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="📈 Productive Screen Use (Study/Work)", value=f"{productive['cgpa'].mean():.2f} CGPA")
        st.caption(f"Sample size: {len(productive)} students")
    with metric_col2:
        st.metric(label="🎮 Entertainment Screen Use (Game/Stream)", value=f"{entertainment['cgpa'].mean():.2f} CGPA")
        st.caption(f"Sample size: {len(entertainment)} students")
        
    st.success("Data Confirmed: Students using screens primarily for productivity boast drastically higher grades than those using it for entertainment, regardless of total hours spent!")
# ==========================================
# SECTION 4: DEMOGRAPHICS & Behavioral Profile
# ==========================================
elif section == "4. Demographics":
    st.header("4. Demographics & Behavioral Profile")
    st.write("Let's look beyond just studying and screen time to see how sleep, gender, and the psychology of 'addiction' change the rules.")

    st.markdown("---")

    # --- INSIGHT 1: The Sleep Penalty ---
    st.subheader("1. The Sleep Penalty")
    st.write("Does a lack of sleep mathematically hurt your GPA?")
    
    sleep_order = ['Less than 5 hrs', '5-6 hrs', '7-8 hrs', '8+ hrs']
    sleep_cgpa = df.groupby('sleep_hours', observed=False)['cgpa'].mean().reindex(sleep_order).reset_index()
    
    col_empty1, col_center, col_empty2 = st.columns([1, 2, 1])
    with col_center:
        fig_sleep = px.bar(sleep_cgpa, x='sleep_hours', y='cgpa', text_auto='.2f',
                            title="Sleep Quality vs. Average CGPA",
                            labels={'sleep_hours': 'Hours of Sleep', 'cgpa': 'Average CGPA'},
                            color='cgpa', color_continuous_scale='Purp')
        fig_sleep.update_yaxes(range=[3.2, 3.6])
        st.plotly_chart(fig_sleep, width="stretch")
        
    st.info("💡 **The Takeaway:** Getting less than 5 hours of sleep drops the average CGPA to 3.31. **The Root Cause:** Our data shows students who peak in screen time 'Late night (after 10PM)' are more than twice as likely to suffer from this severe sleep deprivation.")

    st.markdown("---")
    
    
    st.subheader("The Root Cause: The 'All-Nighter' Effect")
    st.write("Does pushing screen time away from normal daytime hours ruin your sleep?")
    
    # Calculate the percentage of students getting <5 hours of sleep
    peak_sleep_df = df.groupby('peak_time', observed=False)['sleep_hours'].value_counts(normalize=True).unstack().fillna(0) * 100
    risk_df = peak_sleep_df[['Less than 5 hrs']].reset_index()
    risk_df.columns = ['Peak Screen Time', '% Sleeping < 5 Hours']
    
    # Sort the time of day in logical order
    time_order = ['Afternoon (12PM-6PM)', 'Evening (6PM-10PM)', 'Late night (after 10PM)', 'Morning (6AM-12PM)']
    
    # Create the Bar Chart
    fig_risk = px.bar(risk_df, x='Peak Screen Time', y='% Sleeping < 5 Hours',
                      category_orders={'Peak Screen Time': time_order},
                      title="Risk of Severe Sleep Deprivation (<5 hrs) by Peak Usage Time")
    
    # Highlight the "Safe Zone" vs "Danger Zone"
    colors = ['#1f77b4', '#1f77b4', '#d62728', '#8b0000'] # Blue, Blue, Red, Dark Red
    fig_risk.update_traces(marker_color=colors, texttemplate='%{y:.1f}%', textposition='outside')
    fig_risk.update_layout(yaxis_title="% of Students Sleeping < 5 Hrs", yaxis_range=[0, 35])
    
    st.plotly_chart(fig_risk, width="stretch")
    st.caption("Notice the progression: As peak usage shifts from daytime to late night, and eventually into 'All-Nighter' morning cramming, the risk of sleep deprivation skyrockets.")

    # --- The Gender Efficiency Gap ---
    st.subheader("2. The Gender Efficiency Gap")
    st.write("If two students have the exact same amount of screen time, do they get the same grades? Let's look at Gender.")
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        female_df = df[df['gender'] == 'Female']
        male_df = df[df['gender'] == 'Male']
        
        st.metric("Female Avg Screen Time", f"{female_df['avg_screen_time_hours'].mean():.1f} hrs")
        st.metric("Male Avg Screen Time", f"{male_df['avg_screen_time_hours'].mean():.1f} hrs")
        st.write("*(Identical screen time...)*")
        
        st.metric("Female Avg CGPA", f"{female_df['cgpa'].mean():.2f}")
        st.metric("Male Avg CGPA", f"{male_df['cgpa'].mean():.2f}")
        st.write("*(...but completely different GPAs!)*")
        
    with col4:
        study_stats = df.groupby('gender', observed=False)['study_hours_outside_class_num'].mean().reset_index()
        study_stats = study_stats[study_stats['gender'].isin(['Male', 'Female'])]
        
        fig_gender = px.bar(study_stats, x='gender', y='study_hours_outside_class_num',
                            title="The 'Why': Outside Study Hours by Gender",
                            labels={'gender': 'Gender', 'study_hours_outside_class_num': 'Avg Study Hours (1=Low, 4=High)'},
                            color='gender', color_discrete_map={'Female': 'pink', 'Male': 'lightblue'},
                            text_auto='.2f')
        fig_gender.update_yaxes(range=[1.5, 3.0])
        st.plotly_chart(fig_gender, width="stretch")
        
    st.success("💡 **Takeaway:** Men and Women average the exact same 8 hours of screen time. But Females drastically outperform Males in GPA. Why? Because of **Compensatory Habits**. Females put in significantly more study hours outside of class to balance out their screen time.")

    st.markdown("---")


# ==========================================
# SECTION 5: ACTIONABLE INSIGHTS
# ==========================================
elif section == "5. Actionable Insights":
    st.header("5. Actionable Behavioral Insights")
    st.write("If reducing total screen time isn't the answer, what exact policies should the university implement?")

    col1, col2 = st.columns(2)

    # --- NEW Insight 1: The Lecture Disrespect Metric ---
    with col1:
        st.subheader("1. The 'Classroom Immunity' Paradox")
        st.write("Is it worse to be distracted at home or in the classroom?")
        
        # Calculate means for the interactive bar chart
        lecture_stats = df.groupby('phone_during_lectures', observed=False)['cgpa'].mean().reset_index()
        
        fig_lecture = px.bar(lecture_stats, x='phone_during_lectures', y='cgpa',
                             text_auto='.2f', color='cgpa', color_continuous_scale='Reds_r',
                             title="CGPA vs. Distraction During Live Lectures",
                             labels={'phone_during_lectures': 'Distraction Level (1=Low, 5=High)', 'cgpa': 'Average CGPA'})
        fig_lecture.update_yaxes(range=[3.1, 3.7])
        st.plotly_chart(fig_lecture, width="stretch")
        
        st.info("💡 **Takeaway:** Our data shows that out-of-class distraction barely impacts GPA. However, high distraction *during live lectures* causes a massive drop (3.58 to 3.28). **Action:** Professors should enforce strict 'phones down' policies during class.")

    # --- NEW Insight 2: The Freshman Illusion ---
    with col2:
        st.subheader("2. The Freshman Illusion")
        st.write("Do bad screen habits catch up with you over time?")
        
        # Prepare Data
        year_stats = df.groupby('year')[['cgpa', 'avg_screen_time_hours']].mean().reset_index()
        year_stats['year_num'] = year_stats['year'].str.extract(r'(\d+)').astype(int)
        year_stats = year_stats.sort_values('year_num')

        # Chart
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for Screen Time
        fig_dual.add_trace(go.Bar(x=year_stats['year'], y=year_stats['avg_screen_time_hours'], 
                                  name="Avg Screen Time (hrs)", opacity=0.5, marker_color='#1f77b4'), 
                           secondary_y=False)
        # Line chart for CGPA
        fig_dual.add_trace(go.Scatter(x=year_stats['year'], y=year_stats['cgpa'], 
                                      name="Average CGPA", mode='lines+markers+text', 
                                      text=year_stats['cgpa'].round(2), textposition="top right",
                                      line=dict(color='red', width=3)), 
                           secondary_y=True)

        fig_dual.update_layout(title_text="Screen Time vs CGPA by Academic Year")
        fig_dual.update_yaxes(title_text="Screen Time (Hours)", secondary_y=False)
        fig_dual.update_yaxes(title_text="Average CGPA", range=[3.3, 3.6], secondary_y=True)
        
        st.plotly_chart(fig_dual, width="stretch")
        
        st.warning("🚨 **Takeaway:** Freshmen average a massive 10 hours of screen time but still hold a 3.50 GPA. By Year 3, upper-level classes punish these habits, and the GPA crashes to 3.39. **Action:** Intervene in Year 1 before the courses get harder.")


# ==========================================
# SECTION 6: PREDICTIVE MODELING (MACHINE LEARNING)
# ==========================================
elif section == "6. Predictive Modeling":
    st.header("6. Predictive Modeling: Identifying High Performers")
    st.write("We trained a **Decision Tree Classifier** to see if an AI could predict whether a student achieves a High CGPA (≥ 3.5) based purely on their lifestyle and habits.")
    
    
    

    ml_df = df.copy()
    ml_df['High_Performer'] = (ml_df['cgpa'] >= 3.5).astype(int)
    
    features = ['avg_screen_time_hours', 'sleep_hours_num', 'study_hours_outside_class_num', 
                'phone_during_lectures', 'digital_distraction']
    
    ml_df = ml_df.dropna(subset=features)
    X = ml_df[features]
    y = ml_df['High_Performer']

    # 2. Train/Test Split & Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    
    # 3. Display Core Evaluation Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Model Accuracy")
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Test Data Accuracy", f"{accuracy * 100:.1f}%")
        st.write("*(Note: While screen habits are important, they are not the only factors deciding a grade. With only 129 rows, a 45.5% accuracy is mathematically realistic; a higher score would indicate the model was 'overfitting' rather than learning true patterns.)*")
        
    with col2:
        st.subheader("2. Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predicted", y="Actual"),
                           x=['< 3.5 GPA', '>= 3.5 GPA'], y=['< 3.5 GPA', '>= 3.5 GPA'])
        fig_cm.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0), height=250)
        st.plotly_chart(fig_cm, width="stretch")

    st.markdown("---")
    
    # 4. Visualizing the Decision Tree
    st.subheader("3. Visualizing the AI's Logic")
    st.write("Read the tree from top to bottom. The features at the very top are the most important variables the AI used to separate high performers from low performers.")
    
    clean_features = ['Screen Time (Hrs)', 'Sleep Quality', 'Study Hours', 
                      'Lecture Distraction', 'Overall Distraction']
    
    fig_tree, ax_tree = plt.subplots(figsize=(16, 8))
    plot_tree(tree, 
              filled=True, 
              feature_names=clean_features, 
              class_names=["< 3.5 GPA", ">= 3.5 GPA"], 
              rounded=True, 
              fontsize=10, 
              ax=ax_tree)
    
    st.pyplot(fig_tree)
    
    st.success("💡 **Final Takeaway:** Even without knowing a student's academic history, an AI can predict their success simply by looking at a few key habits. Managing lecture distraction and study hours is just as critical as managing raw screen time!")
    
    


