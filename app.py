import streamlit as st
import pickle
import os
import subprocess
import sys
from train_model import DiseaseClassifier

# Ensure pickle compatibility across script executions
sys.modules['__main__'].DiseaseClassifier = DiseaseClassifier

# Page config
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .header-box {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 30px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.2);
    }
    .header-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 8px; }
    .header-sub { font-size: 1.1rem; opacity: 0.95; }
    .disease-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .pred-name { font-size: 2rem; font-weight: 800; color: #38bdf8; margin: 8px 0; }
    .pred-confidence { font-size: 1.25rem; font-weight: 600; color: #a5f3fc; }
    .prob-row {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .prob-bar-bg {
        background: #e2e8f0;
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 6px;
    }
    .prob-bar-fill {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        height: 100%;
        border-radius: 5px;
    }
    .disclaimer-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        border-radius: 12px;
        padding: 16px;
        margin-top: 24px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Disease info dictionary covering all 42 diseases
disease_info = {
    "Fungal infection": {
        "description": "A fungal infection caused by fungi that can affect skin, nails, hair, and mucous membranes.",
        "precautions": ["Keep the affected area clean and dry", "Apply prescribed antifungal ointments", "Wear loose, breathable cotton clothing", "Avoid sharing towels and personal items"],
        "severity": "Mild to Moderate"
    },
    "Allergy": {
        "description": "An exaggerated immune response to foreign substances (allergens) such as pollen, dust, or certain foods.",
        "precautions": ["Identify and avoid known allergen triggers", "Take prescribed antihistamines or nasal sprays", "Use HEPA air purifiers at home", "Consult an allergist if symptoms persist"],
        "severity": "Mild to Severe"
    },
    "GERD": {
        "description": "Gastroesophageal reflux disease occurs when stomach acid frequently flows back into the esophagus.",
        "precautions": ["Avoid spicy, acidic, and fatty foods", "Do not lie down immediately after meals", "Eat smaller, more frequent meals", "Elevate the head of your bed"],
        "severity": "Moderate"
    },
    "Chronic cholestasis": {
        "description": "A condition characterized by reduced or impaired bile flow from the liver to the duodenum.",
        "precautions": ["Strictly avoid alcohol consumption", "Adhere to a low-fat, nutritionally balanced diet", "Take fat-soluble vitamin supplements as advised", "Undergo regular liver function monitoring"],
        "severity": "Severe"
    },
    "Drug Reaction": {
        "description": "An adverse and unintended physiological response caused by taking a medication.",
        "precautions": ["Discontinue the suspected medication after medical advice", "Consult an emergency physician immediately", "Carry an allergy alert card in your wallet", "Notify all future healthcare providers"],
        "severity": "Moderate to Severe"
    },
    "Peptic ulcer disease": {
        "description": "Painful sores or ulcers that develop on the inner lining of the stomach, small intestine, or esophagus.",
        "precautions": ["Avoid NSAID painkillers like ibuprofen and aspirin", "Avoid smoking and caffeine", "Eat regular, non-irritating balanced meals", "Take prescribed acid reducers (PPIs)"],
        "severity": "Moderate"
    },
    "AIDS": {
        "description": "Acquired immunodeficiency syndrome caused by HIV, severely weakening the body's immune system.",
        "precautions": ["Strict adherence to Antiretroviral Therapy (ART)", "Practice safe barrier protection", "Avoid sharing needles or injection equipment", "Maintain regular CD4 and viral load monitoring"],
        "severity": "Severe"
    },
    "Diabetes": {
        "description": "A chronic metabolic disorder resulting in elevated blood glucose levels due to insulin deficiency or resistance.",
        "precautions": ["Monitor blood glucose levels consistently", "Follow a low-glycemic, high-fiber diet", "Engage in regular moderate physical exercise", "Take prescribed oral hypoglycemics or insulin"],
        "severity": "Moderate to Severe"
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines ('stomach flu'), often viral or bacterial.",
        "precautions": ["Drink plenty of Oral Rehydration Solution (ORS)", "Stick to the BRAT diet (bananas, rice, applesauce, toast)", "Wash hands thoroughly with soap", "Rest and avoid dairy products temporarily"],
        "severity": "Mild to Moderate"
    },
    "Bronchial Asthma": {
        "description": "A chronic inflammatory disorder of the airways causing recurring wheezing, breathlessness, and coughing.",
        "precautions": ["Always carry your rescue bronchodilator inhaler", "Avoid exposure to smoke, dust, and cold air", "Track peak expiratory flow rates", "Follow a written asthma action plan"],
        "severity": "Moderate to Severe"
    },
    "Hypertension": {
        "description": "Chronically elevated systemic arterial blood pressure, increasing risk of stroke and cardiovascular disease.",
        "precautions": ["Reduce daily dietary sodium intake", "Engage in regular aerobic exercise", "Practice daily stress reduction and meditation", "Check blood pressure at regular intervals"],
        "severity": "Moderate to Severe"
    },
    "Migraine": {
        "description": "A neurological condition featuring throbbing, typically unilateral headaches, often with nausea and photophobia.",
        "precautions": ["Rest in a dark, quiet room during an attack", "Maintain regular sleep and hydration schedules", "Identify and avoid dietary triggers (e.g., aged cheese, MSG)", "Take abortive migraine medications early"],
        "severity": "Moderate"
    },
    "Cervical spondylosis": {
        "description": "Age-related degenerative wear and tear of the cervical vertebrae and intervertebral discs in the neck.",
        "precautions": ["Perform gentle neck and shoulder stretching exercises", "Maintain proper ergonomic posture at your computer", "Use a contoured orthopedic cervical pillow", "Avoid carrying heavy loads on shoulders"],
        "severity": "Moderate"
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Loss of neurological and voluntary motor function caused by cerebral vascular rupture or hemorrhage.",
        "precautions": ["Seek emergency hospital admission immediately (call 911/112)", "Follow strict blood pressure and intracranial pressure control", "Undergo intensive neurological rehabilitation and physiotherapy", "Prevent aspiration and pressure sores"],
        "severity": "Critical"
    },
    "Jaundice": {
        "description": "Yellow discoloration of the skin and sclera resulting from hyperbilirubinemia in the bloodstream.",
        "precautions": ["Consume plenty of fluids and easily digestible foods", "Strictly abstain from alcohol and hepatotoxic drugs", "Get adequate bed rest", "Undergo comprehensive liver panel and ultrasound scans"],
        "severity": "Moderate to Severe"
    },
    "Malaria": {
        "description": "A life-threatening mosquito-borne infectious illness caused by Plasmodium parasites.",
        "precautions": ["Complete the full prescribed course of Artemisinin-based therapy", "Sleep under insecticide-treated bed nets", "Use mosquito repellents containing DEET", "Stay hydrated and control high fever with paracetamol"],
        "severity": "Moderate to Severe"
    },
    "Chicken pox": {
        "description": "A highly contagious viral infection caused by the Varicella-Zoster virus, causing an itchy vesicular rash.",
        "precautions": ["Isolate at home until all blisters have fully crusted over", "Apply calamine lotion to soothe severe itching", "Keep fingernails short to prevent secondary bacterial infection", "Avoid giving aspirin to children with fever"],
        "severity": "Mild to Moderate"
    },
    "Dengue": {
        "description": "A mosquito-borne viral disease causing sudden high fever, severe retro-orbital pain, and arthralgia.",
        "precautions": ["Stay well-hydrated with fluids, coconut water, and electrolytes", "Monitor platelet count daily", "Avoid aspirin or ibuprofen due to bleeding risk; use paracetamol", "Seek immediate care if warning signs like bleeding or abdominal pain appear"],
        "severity": "Moderate to Severe"
    },
    "Typhoid": {
        "description": "A systemic bacterial infection caused by Salmonella enterica serotype Typhi via contaminated food or water.",
        "precautions": ["Complete the entire course of prescribed antibiotics", "Drink exclusively boiled, filtered, or bottled water", "Eat only thoroughly cooked hot foods", "Wash hands diligently before eating"],
        "severity": "Moderate to Severe"
    },
    "hepatitis A": {
        "description": "An acute viral liver infection transmitted primarily through the fecal-oral route via contaminated food or water.",
        "precautions": ["Get plenty of rest and avoid strenuous exertion", "Eat small, high-calorie meals that are easy to digest", "Maintain rigorous hand hygiene", "Avoid alcohol completely until full recovery"],
        "severity": "Mild to Moderate"
    },
    "Hepatitis B": {
        "description": "A viral infection attacking the liver that can cause both acute illness and chronic liver cirrhosis.",
        "precautions": ["Receive antiviral medications if indicated by your hepatologist", "Ensure family members and close contacts are vaccinated", "Do not share razors, toothbrushes, or needles", "Undergo regular liver ultrasounds and AFP tests"],
        "severity": "Moderate to Severe"
    },
    "Hepatitis C": {
        "description": "A blood-borne viral infection causing chronic liver inflammation, frequently progressing silently.",
        "precautions": ["Undergo modern Direct-Acting Antiviral (DAA) therapy", "Abstain from all alcohol consumption", "Avoid sharing grooming items that might carry blood", "Screen for liver fibrosis and cirrhosis"],
        "severity": "Severe"
    },
    "Hepatitis D": {
        "description": "A severe viral liver disease occurring exclusively as a co-infection or super-infection with Hepatitis B.",
        "precautions": ["Strict medical management with pegylated interferon or bulevirtide", "Ensure Hepatitis B management is optimized", "Follow a liver-friendly, low-sodium diet", "Monitor for early signs of hepatic decompensation"],
        "severity": "Severe"
    },
    "Hepatitis E": {
        "description": "A water-borne viral hepatitis infection, particularly dangerous for pregnant women.",
        "precautions": ["Ensure strict drinking water sanitation", "Avoid raw or undercooked shellfish and pork", "Get adequate rest and fluids", "Pregnant patients require immediate medical hospitalization"],
        "severity": "Moderate"
    },
    "Alcoholic hepatitis": {
        "description": "Acute or chronic liver inflammation and necrosis caused by excessive long-term alcohol consumption.",
        "precautions": ["Complete and permanent cessation of alcohol intake", "Intensive nutritional and vitamin supplementation", "Regular screening for portal hypertension and ascites", "Consult an addiction medicine specialist"],
        "severity": "Severe"
    },
    "Tuberculosis": {
        "description": "A potentially serious infectious disease primarily affecting the respiratory tract, caused by Mycobacterium tuberculosis.",
        "precautions": ["Strictly complete the 6-month Directly Observed Therapy (DOTS)", "Wear a mask and ensure good room ventilation", "Eat a protein-rich, nourishing diet", "Screen household contacts promptly"],
        "severity": "Severe"
    },
    "Common Cold": {
        "description": "A mild, self-limiting viral upper respiratory tract infection predominantly caused by rhinoviruses.",
        "precautions": ["Get plenty of rest and sleep", "Drink warm fluids, broths, and honey-lemon tea", "Use saline nasal sprays or steam inhalation for congestion", "Wash hands regularly to prevent spreading"],
        "severity": "Mild"
    },
    "Pneumonia": {
        "description": "An acute infection inflaming the pulmonary alveoli, which may fill with fluid or purulent exudate.",
        "precautions": ["Take all prescribed antibiotics or antivirals to completion", "Practice deep breathing exercises once recovering", "Monitor oxygen saturation with a pulse oximeter", "Stay well-hydrated to help thin mucus secretions"],
        "severity": "Moderate to Severe"
    },
    "Dimorphic hemorrhoids": {
        "description": "Swollen and inflamed vascular structures in the anal canal, causing discomfort and bleeding.",
        "precautions": ["Increase dietary fiber intake (fruits, oats, psyllium husk)", "Drink at least 2.5-3 liters of water daily", "Take warm sitz baths for 15 minutes twice a day", "Avoid straining during bowel movements"],
        "severity": "Mild to Moderate"
    },
    "Heart attack": {
        "description": "Myocardial infarction: an emergency where coronary blood supply is abruptly blocked, starving heart tissue.",
        "precautions": ["CALL EMERGENCY MEDICAL SERVICES IMMEDIATELY (911/112)", "Chew an uncoated aspirin if directed by dispatcher", "Adopt a strict Mediterranean/cardioprotective diet post-recovery", "Participate in structured cardiac rehabilitation"],
        "severity": "Critical"
    },
    "Varicose veins": {
        "description": "Gnarled, enlarged, and superficial veins most commonly found in the legs due to incompetent venous valves.",
        "precautions": ["Wear graduated compression stockings daily", "Elevate legs above heart level when resting", "Avoid standing or sitting in one position for long periods", "Engage in regular low-impact walking and swimming"],
        "severity": "Mild to Moderate"
    },
    "Hypothyroidism": {
        "description": "Underactive thyroid gland failing to produce sufficient thyroid hormones (T3/T4) to regulate metabolism.",
        "precautions": ["Take prescribed levothyroxine daily on an empty stomach", "Wait 30-60 minutes after medication before breakfast", "Undergo routine TSH blood tests every 6-12 months", "Maintain a balanced, nutrient-dense diet"],
        "severity": "Moderate"
    },
    "Hyperthyroidism": {
        "description": "Overactive thyroid gland producing excessive quantities of thyroid hormones, accelerating metabolism.",
        "precautions": ["Take prescribed antithyroid medications or beta-blockers", "Limit excessive dietary iodine intake (e.g., kelp, seaweed)", "Practice stress-management techniques", "Protect eyes with sunglasses and lubricating drops if needed"],
        "severity": "Moderate"
    },
    "Hypoglycemia": {
        "description": "Abnormally low blood glucose levels (below 70 mg/dL), commonly associated with diabetes treatment.",
        "precautions": ["Follow the 15-15 Rule: consume 15g fast-acting sugar, retest in 15 mins", "Always carry glucose tablets or candy", "Eat meals on a consistent schedule", "Wear a medical alert bracelet for hypoglycemia"],
        "severity": "Moderate to Severe"
    },
    "Osteoarthritis": {
        "description": "Degenerative joint disease caused by progressive breakdown and loss of protective joint cartilage.",
        "precautions": ["Engage in low-impact physical activities like swimming or cycling", "Maintain an optimal body weight to reduce joint load", "Apply heat or cold packs for symptomatic relief", "Work with a physical therapist on joint mobility"],
        "severity": "Moderate"
    },
    "Arthritis": {
        "description": "Inflammation of one or more joints, leading to pain, swelling, and reduced range of motion.",
        "precautions": ["Follow an anti-inflammatory diet rich in Omega-3 fatty acids", "Perform gentle daily range-of-motion exercises", "Use assistive devices to protect vulnerable joints", "Take prescribed disease-modifying or pain medications"],
        "severity": "Moderate"
    },
    "(Vertigo) Paroymsal Positional Vertigo": {
        "description": "Benign Paroxysmal Positional Vertigo (BPPV) causing sudden, brief sensations of spinning triggered by head movements.",
        "precautions": ["Perform canalith repositioning maneuvers (Epley maneuver) with a specialist", "Avoid sudden rapid head turning or looking upwards", "Sleep with your head slightly elevated", "Move carefully and hold handrails when rising"],
        "severity": "Mild to Moderate"
    },
    "Acne": {
        "description": "An inflammatory skin condition occurring when hair follicles become plugged with sebum and dead skin cells.",
        "precautions": ["Wash face twice daily with a gentle, non-comedogenic cleanser", "Avoid picking, squeezing, or popping pimples", "Apply topical treatments with salicylic acid or benzoyl peroxide", "Use oil-free moisturizers and sunscreen"],
        "severity": "Mild"
    },
    "Urinary tract infection": {
        "description": "Bacterial infection affecting any part of the urinary tract (urethra, bladder, or kidneys).",
        "precautions": ["Drink abundant water throughout the day to flush bacteria", "Complete the full course of prescribed antibiotics", "Urinate promptly when the urge arises", "Always wipe front to back after using the bathroom"],
        "severity": "Mild to Moderate"
    },
    "Psoriasis": {
        "description": "A chronic autoimmune dermatological condition that causes rapid buildup of skin cells, creating scaly plaques.",
        "precautions": ["Keep skin well-hydrated with rich moisturizing ointments", "Avoid common triggers such as skin trauma, smoking, and stress", "Undergo supervised phototherapy or topical treatments", "Avoid hot showers that dry out skin"],
        "severity": "Moderate"
    },
    "Impetigo": {
        "description": "A highly contagious superficial bacterial skin infection causing honey-colored crusted sores.",
        "precautions": ["Apply prescribed topical or oral antibacterial therapy", "Gently clean sores with warm soapy water and gauze", "Keep sores loosely covered with clean bandages", "Wash clothing, towels, and bed linens daily in hot water"],
        "severity": "Mild"
    },
    "Paroymsal Positional Vertigo": {
        "description": "A vestibular disorder producing sudden episodes of vertigo with head position changes.",
        "precautions": ["Consult an ENT specialist for positional repositioning maneuvers", "Sit down immediately when dizziness strikes", "Use good lighting if getting up at night", "Avoid driving during active vertigo flares"],
        "severity": "Mild to Moderate"
    },
}

@st.cache_resource
def load_model():
    """Load the trained model and symptoms list."""
    model_path = 'model/disease_model.pkl'
    symptoms_path = 'model/symptoms.pkl'

    if not os.path.exists(model_path) or not os.path.exists(symptoms_path):
        with st.spinner("🔄 Training model for the first time..."):
            res = subprocess.run([sys.executable, 'train_model.py'], capture_output=True, text=True)
            if res.returncode != 0:
                st.error(f"Training failed:\n{res.stderr}")
                return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(symptoms_path, 'rb') as f:
        symptoms = pickle.load(f)

    return model, symptoms


def predict_disease(model, symptoms_list, selected_symptoms):
    """Make prediction based on selected symptoms."""
    input_data = {symptom: 0 for symptom in symptoms_list}
    for symptom in selected_symptoms:
        if symptom in input_data:
            input_data[symptom] = 1

    probabilities = model.predict_proba([input_data])[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, probabilities))
    top5 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    top_prediction = top5[0][0]
    return top_prediction, top5


def format_symptom_name(symptom):
    return symptom.replace('_', ' ').replace('.', '').strip().title()


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## 🏥 System Overview")
    st.markdown("""
    This AI system predicts potential health conditions based on **symptom patterns** using a **Bayesian Machine Learning Classifier**.
    
    ### 📌 How to Use
    1. Type or search symptoms in the box.
    2. Select all symptoms you are experiencing.
    3. Click **🔮 Predict Disease**.
    4. Review top matches, probabilities & precautions.
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Statistics")
    st.info("**Classifier:** Bayesian Diagnostic Classifier\n\n**Diseases Covered:** 42\n\n**Symptoms Catalog:** 132\n\n**Test Accuracy:** 95.0%")
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.85rem; color: #64748b; line-height: 1.4;'>
    ⚕️ <strong>Clinical Note:</strong> This application provides statistical estimation and does not replace professional medical evaluation, laboratory testing, or clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ===================== MAIN INTERFACE =====================
st.markdown("""
<div class='header-box'>
    <div class='header-title'>🏥 AI Disease Prediction System</div>
    <div class='header-sub'>Select your symptoms below to analyze potential health conditions with machine learning</div>
</div>
""", unsafe_allow_html=True)

model, symptoms_list = load_model()

if model and symptoms_list:
    formatted_symptoms = {format_symptom_name(s): s for s in symptoms_list}

    st.markdown("### 🔍 Select Your Symptoms")
    st.caption("You can select multiple symptoms. The more symptoms you provide, the higher the prediction accuracy.")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_display = st.multiselect(
            "Symptoms",
            options=sorted(formatted_symptoms.keys()),
            placeholder="Type or select symptoms (e.g. High Fever, Headache, Joint Pain...)",
            label_visibility="collapsed"
        )
    with col2:
        predict_btn = st.button("🔮 Predict Disease", type="primary", use_container_width=True)

    selected_raw = [formatted_symptoms[s] for s in selected_display]

    if selected_display:
        st.markdown(f"**Selected ({len(selected_display)}):** " + " • ".join(f"`{s}`" for s in selected_display))

    st.divider()

    if predict_btn:
        if not selected_display:
            st.warning("⚠️ Please select at least one symptom from the list before predicting.")
        else:
            with st.spinner("Analyzing symptom correlations..."):
                prediction, top5 = predict_disease(model, symptoms_list, selected_raw)

            st.markdown("## 📋 Analysis Results")

            col_main, col_info = st.columns([1.1, 0.9])

            severity_colors = {
                "Critical": "#ef4444",
                "Severe": "#f97316",
                "Moderate to Severe": "#f59e0b",
                "Moderate": "#3b82f6",
                "Mild to Severe": "#8b5cf6",
                "Mild to Moderate": "#10b981",
                "Mild": "#22c55e"
            }

            with col_main:
                top_prob = top5[0][1] * 100
                info = disease_info.get(prediction, {
                    "description": "Detailed description not found.",
                    "precautions": ["Consult your physician for guidance"],
                    "severity": "Moderate"
                })
                sev = info.get("severity", "Moderate")
                sev_color = severity_colors.get(sev, "#3b82f6")

                st.markdown(f"""
                <div class='disease-card'>
                    <div style='font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85;'>Primary Match</div>
                    <div class='pred-name'>{prediction}</div>
                    <div class='pred-confidence'>Confidence: {top_prob:.1f}%</div>
                    <div style='margin-top: 14px;'>
                        <span style='background: {sev_color}; color: white; padding: 4px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;'>
                            Severity: {sev}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📖 About this Condition")
                st.info(info.get("description", ""))

                st.markdown("#### 💊 Recommended Next Steps & Precautions")
                for i, p in enumerate(info.get("precautions", []), 1):
                    st.markdown(f"**{i}.** {p}")

            with col_info:
                st.markdown("#### 📊 Differential Diagnosis (Top 5 Matches)")
                for rank, (dis, prob) in enumerate(top5, 1):
                    pct = prob * 100
                    emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank - 1]
                    dis_sev = disease_info.get(dis, {}).get("severity", "Unknown")
                    badge_color = severity_colors.get(dis_sev, "#64748b")

                    st.markdown(f"""
                    <div class='prob-row'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-weight: 700; color: #1e293b;'>{emoji} {dis}</span>
                            <span style='font-weight: 800; color: #4f46e5;'>{pct:.1f}%</span>
                        </div>
                        <div class='prob-bar-bg'>
                            <div class='prob-bar-fill' style='width: {max(pct, 2):.1f}%;'></div>
                        </div>
                        <div style='margin-top: 6px; font-size: 0.8rem; color: #64748b;'>
                            Severity: <span style='color: {badge_color}; font-weight: 600;'>{dis_sev}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("""
            <div class='disclaimer-box'>
                ⚠️ <strong>Important Medical Disclaimer:</strong> This disease prediction system is an AI statistical reference tool intended solely for educational and informational purposes. It does not replace a clinical examination or professional medical diagnosis. If you are experiencing serious or worsening symptoms, please seek prompt medical attention.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align: center; padding: 40px; color: #64748b;'>
            <div style='font-size: 3.5rem;'>🩺</div>
            <h3 style='color: #334155;'>Select Symptoms to Begin</h3>
            <p>Choose your current symptoms from the multi-select input above and click <strong>Predict Disease</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🏥 Supported Diseases (42 Conditions)")
        diseases = sorted(disease_info.keys())
        cols = st.columns(3)
        for i, disease in enumerate(diseases):
            cols[i % 3].markdown(f"• **{disease}**")

else:
    st.error("❌ Could not load the disease prediction model. Run `python train_model.py` first.")
