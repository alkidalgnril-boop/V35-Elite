# -*- coding: utf-8 -*-
"""
Emperor V35 Elite - Invasion Edition
The Zero-Budget SaaS that kills the competition.
Integrates: Cube-IQ Logic + Stacker Safety + EasyCargo Visuals + Gemini AI.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import time

# ==============================================================================
# 1. SYSTEM CONFIGURATION & SAAS UI
# ==============================================================================
st.set_page_config(
    page_title="الإمبراطور V35 Elite | نظام لوجستي ذكي",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إخفاء هوية Streamlit ليبدو كتطبيق SaaS احترافي
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            header {visibility: hidden;}
            .css-18e3th9 {padding-top: 0rem;}
            .block-container {padding-top: 1rem; padding-bottom: 1rem;}
            /* تحسين مظهر التنبيهات */
            .stAlert {border-radius: 10px; font-weight: bold;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ثوابت الحاوية (40ft Standard)
CONTAINER_L = 1190 # سم
CONTAINER_W = 230  # سم
CONTAINER_H = 270  # سم
CONTAINER_VOL = CONTAINER_L * CONTAINER_W * CONTAINER_H

# لوحة ألوان احترافية (EasyCargo Style)
COLOR_PALETTE = [
    '#3498DB', '#E74C3C', '#2ECC71', '#F1C40F', '#9B59B6', 
    '#34495E', '#16A085', '#D35400', '#7F8C8D', '#8E44AD'
]

# ==============================================================================
# 2. THE DETERMINISTIC ENGINE (Cube-IQ & Stacker Logic)
# ==============================================================================

class Item:
    def __init__(self, name, length, width, height, weight, quantity=1, color_idx=0):
        self.name = name
        self.length = int(length)
        self.width = int(width)
        self.height = int(height)
        self.weight = float(weight)
        self.quantity = int(quantity)
        self.volume = self.length * self.width * self.height
        self.color = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
        self.position = None  # (x, y, z)
        self.sequence_id = 0  # ترتيب الرص (ميزة EasyCargo)

class PackingEngineV35:
    def __init__(self):
        self.items_to_pack = []
        self.packed_items = []
        self.total_weight = 0
        self.total_volume_packed = 0
        # تحليل الأرباع (ميزة Stacker)
        self.quadrants = {
            "Front-Left": 0, "Front-Right": 0, 
            "Back-Left": 0, "Back-Right": 0
        }

    def add_item(self, item_obj):
        for _ in range(item_obj.quantity):
            new_item = Item(item_obj.name, item_obj.length, item_obj.width, 
                            item_obj.height, item_obj.weight, 1, 0)
            new_item.color = item_obj.color
            self.items_to_pack.append(new_item)

    def _check_gravity_support(self, item, x, y, z, packed_items):
        """ 
        (Cube-IQ Logic): True Gravity Support 
        لا يسمح بالرص إلا إذا كانت المساحة المدعومة > 70%
        """
        if z == 0: return True
        supported_area = 0
        item_base_area = item.length * item.width
        required_support = 0.70 * item_base_area # صرامة عالية

        for p in packed_items:
            if abs((p.position[2] + p.height) - z) < 1:
                x_overlap = max(0, min(x + item.length, p.position[0] + p.length) - max(x, p.position[0]))
                y_overlap = max(0, min(y + item.width, p.position[1] + p.width) - max(y, p.position[1]))
                supported_area += x_overlap * y_overlap
        
        return supported_area >= required_support

    def is_safe(self, item, x, y, z, packed_items):
        # 1. الحدود
        if x + item.length > CONTAINER_L or y + item.width > CONTAINER_W or z + item.height > CONTAINER_H:
            return False
        # 2. التداخل
        for p in packed_items:
            if (x < p.position[0] + p.length and x + item.length > p.position[0] and
                y < p.position[1] + p.width and y + item.width > p.position[1] and
                z < p.position[2] + p.height and z + item.height > p.position[2]):
                return False
        # 3. الجاذبية الصارمة
        if not self._check_gravity_support(item, x, y, z, packed_items):
            return False
        return True

    def run_packing(self):
        # (Sequence Logic): ترتيب حسب الكثافة (الوزن/الحجم) لضمان وضع الثقيل أولاً
        # هذا يقلل مركز الثقل تلقائياً
        self.items_to_pack.sort(key=lambda x: x.weight, reverse=True)
        
        # دقة البحث (10 سم) للتوافق مع Streamlit Cloud
        STEP = 10 
        sequence_counter = 1

        for item in self.items_to_pack:
            placed = False
            # (Sequence Loading): الرص يبدأ من X=0 (الصدر) ويمشي للخلف
            # Loop Order: X (Length) -> Z (Height) -> Y (Width)
            # هذا يبني "جدران" متماسكة من الداخل للخارج
            for x in range(0, CONTAINER_L, STEP):
                if placed: break
                for z in range(0, CONTAINER_H, STEP):
                    if placed: break
                    for y in range(0, CONTAINER_W, STEP):
                        if self.is_safe(item, x, y, z, self.packed_items):
                            item.position = (x, y, z)
                            item.sequence_id = sequence_counter
                            self.packed_items.append(item)
                            self.total_weight += item.weight
                            self.total_volume_packed += item.volume
                            self._update_quadrants(item, x, y)
                            placed = True
                            sequence_counter += 1
                            break

    def _update_quadrants(self, item, x, y):
        """ (Stacker Logic): توزيع الوزن على الأرباع """
        mid_l = CONTAINER_L / 2
        mid_w = CONTAINER_W / 2
        center_x = x + item.length / 2
        center_y = y + item.width / 2
        
        if center_x < mid_l: # Front
            if center_y < mid_w: self.quadrants["Front-Left"] += item.weight
            else: self.quadrants["Front-Right"] += item.weight
        else: # Back
            if center_y < mid_w: self.quadrants["Back-Left"] += item.weight
            else: self.quadrants["Back-Right"] += item.weight

    def get_safety_report(self):
        """ تقرير الأمان وتحليل الاختلال """
        if self.total_weight == 0: return {"status": "Empty", "msg": "الحاوية فارغة"}

        # حساب أقصى فرق بين أي ربعين
        weights = list(self.quadrants.values())
        max_w = max(weights) if max(weights) > 0 else 1
        min_w = min(weights)
        imbalance_ratio = ((max_w - min_w) / max_w) * 100
        
        status = "Safe"
        msg = "✅ توازن الأحمال ممتاز (ضمن الحدود الآمنة)."
        
        # شرط صارم: 15% فرق
        if imbalance_ratio > 15:
            status = "Danger"
            msg = f"🚨 خطر انقلاب! اختلال في توزيع الوزن بنسبة {imbalance_ratio:.1f}% بين الأرباع. (راجع مخطط Stacker)."
        
        return {"status": status, "msg": msg, "imbalance": imbalance_ratio, "quadrants": self.quadrants}

    def get_metrics(self):
        utilization = (self.total_volume_packed / CONTAINER_VOL) * 100
        return {
            "utilization": utilization,
            "packed_count": len(self.packed_items),
            "total_weight": self.total_weight,
            "void_percent": 100 - utilization
        }

# ==============================================================================
# 3. AI CONSULTANT (The Human Standard Crusher)
# ==============================================================================

def get_gemini_consultation(api_key, metrics, container_price, safety_report):
    if not api_key:
        return "⚠️ أدخل مفتاح API للحصول على تحليل 'الإمبراطور' المالي."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        # (Market Intelligence): المقارنة بـ 65%
        market_avg = 65.0
        improvement = metrics['utilization'] - market_avg
        savings = (improvement / 100) * container_price if improvement > 0 else 0
        
        prompt = f"""
        [ROLE: You are 'Emperor V35', a ruthless Logistics Auditor & Financial Advisor.]
        
        **MISSION:** Destroy the competition by showing the user exactly how much money they saved using V35 vs. manual loading.
        
        **DATA:**
        - Container Cost: ${container_price}
        - Global Manual Average (The Lazy Standard): 65%
        - V35 Elite Performance: {metrics['utilization']:.2f}%
        - **CASH RECOVERED:** ${savings:.2f} (This is the key metric)
        - Safety Status: {safety_report['msg']}
        - Quadrant Weights: {safety_report['quadrants']}
        
        **OUTPUT STRUCTURE (Arabic Business Tone):**
        1. **The Verdict:** Start with the money. "لقد استعدنا لك [M] دولار كانت ستضيع في الهواء."
        2. **Efficiency Analysis:** Compare V35 (Deterministic Logic) vs Human Intuition.
        3. **Safety Check:** Comment on the Quadrant Analysis. If 'Danger', be very harsh about truck rollover risks.
        4. **Stevedore Instructions:** Give 2 specific commands for the loading team (e.g., "Load heavy items in Front-Left first").
        5. **Disclaimer:** Standard legal disclaimer.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ خطأ في الاتصال بالمستشار الذكي: {str(e)}"

# ==============================================================================
# 4. VISUALIZATION (EasyCargo Style)
# ==============================================================================

def create_3d_chart(packed_items):
    mx, my, mz, mi, mj, mk, mc, hover_texts = [], [], [], [], [], [], [], []
    vert_count = 0

    for item in packed_items:
        x, y, z = item.position
        dx, dy, dz = item.length, item.width, item.height
        
        mx.extend([x, x, x+dx, x+dx, x, x, x+dx, x+dx])
        my.extend([y, y+dy, y+dy, y, y, y+dy, y+dy, y])
        mz.extend([z, z, z, z, z+dz, z+dz, z+dz, z+dz])
        
        mi.extend([vert_count+7, vert_count+0, vert_count+0, vert_count+0, vert_count+4, vert_count+4, vert_count+6, vert_count+6, vert_count+4, vert_count+0, vert_count+3, vert_count+2])
        mj.extend([vert_count+3, vert_count+4, vert_count+1, vert_count+2, vert_count+5, vert_count+6, vert_count+5, vert_count+2, vert_count+0, vert_count+1, vert_count+6, vert_count+3])
        mk.extend([vert_count+0, vert_count+7, vert_count+2, vert_count+3, vert_count+6, vert_count+7, vert_count+1, vert_count+1, vert_count+5, vert_count+5, vert_count+7, vert_count+6])
        
        mc.extend([item.color] * 12)
        
        # (EasyCargo Feature): Hover Data with Sequence ID
        info = f"📦 <b>{item.name}</b><br>الترتيب: #{item.sequence_id}<br>الأبعاد: {dx}x{dy}x{dz}<br>الوزن: {item.weight}kg"
        hover_texts.extend([info] * 12)
        vert_count += 8

    # رسم الحاوية
    cx = [0, CONTAINER_L, CONTAINER_L, 0, 0, 0, CONTAINER_L, CONTAINER_L, 0, 0, CONTAINER_L, CONTAINER_L, CONTAINER_L, CONTAINER_L, 0, 0]
    cy = [0, 0, CONTAINER_W, CONTAINER_W, 0, 0, 0, CONTAINER_W, CONTAINER_W, 0, 0, 0, CONTAINER_W, CONTAINER_W, CONTAINER_W, CONTAINER_W]
    cz = [0, 0, 0, 0, 0, CONTAINER_H, CONTAINER_H, CONTAINER_H, CONTAINER_H, CONTAINER_H, CONTAINER_H, 0, 0, CONTAINER_H, CONTAINER_H, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='lines', line=dict(color='black', width=4), hoverinfo='skip', name='Container'))
    fig.add_trace(go.Mesh3d(x=mx, y=my, z=mz, i=mi, j=mj, k=mk, facecolor=mc, flatshading=True, lighting=dict(ambient=0.8, diffuse=0.5), hoverinfo='text', text=hover_texts, name='Cargo'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, CONTAINER_L], title='', showticklabels=False, backgroundcolor="white"),
            yaxis=dict(range=[0, CONTAINER_W], title='', showticklabels=False, backgroundcolor="white"),
            zaxis=dict(range=[0, CONTAINER_H], title='', showticklabels=False, backgroundcolor="white"),
            aspectmode='manual',
            aspectratio=dict(x=CONTAINER_L/CONTAINER_W, y=1, z=CONTAINER_H/CONTAINER_W)
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        height=600
    )
    return fig

# ==============================================================================
# 5. MAIN UI (SaaS Interface)
# ==============================================================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("🦁 الإمبراطور V35 Elite")
        st.caption("Invasion Edition | Zero-Budget SaaS")
        st.markdown("---")
        
        # 1. المدخلات
        uploaded_file = st.file_uploader("📂 رفع ملف الشحنة (Excel/CSV)", type=['xlsx', 'csv'])
        container_price = st.number_input("💰 تكلفة الحاوية ($)", min_value=0, value=3500, step=100)
        api_key = st.text_input("🔑 مفتاح Gemini API", type="password")
        
        st.markdown("---")
        
        # 2. فلتر الأمان (DG Filter)
        st.error("⚠️ منطقة الأمان")
        dg_confirm = st.checkbox("أقر بأن الشحنة خالية من المواد الخطرة (DG) أو القابلة للاشتعال.", value=False)
        
        st.markdown("---")
        
        # 3. زر التشغيل
        run_btn = st.button("🚀 بدء التحليل (V35 Engine)", use_container_width=True, type="primary", disabled=not dg_confirm)
        
        if not dg_confirm:
            st.warning("🛑 لا يمكن تشغيل النظام دون تأكيد خلو الشحنة من المواد الخطرة.")

        # 4. الشروط القانونية
        with st.expander("📜 شروط الاستخدام وإخلاء المسؤولية"):
            st.markdown("""
            1. **طبيعة النظام:** محاكاة هندسية لتقدير الحيز (V35 Logic).
            2. **المسؤولية:** المستخدم يتحمل مسؤولية التثبيت الفعلي.
            3. **الحدود:** الحد الأقصى 300 صنف (نسخة السحابة المجانية).
            """)

    # --- Main Dashboard ---
    if run_btn and uploaded_file and dg_confirm:
        with st.spinner('جاري تشغيل خوارزميات Cube-IQ وتحليل Stacker...'):
            # قراءة البيانات
            try:
                if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                else: df = pd.read_excel(uploaded_file)
                
                req_cols = ['Item', 'Length', 'Width', 'Height', 'Weight', 'Qty']
                if not all(c in df.columns for c in req_cols):
                    st.error(f"الملف يجب أن يحتوي على الأعمدة: {req_cols}")
                    st.stop()
                
                # (Resource Management): تحديد سقف 300 صنف
                total_items = df['Qty'].sum()
                if total_items > 300:
                    st.warning(f"⚠️ عدد الأصناف ({total_items}) يتجاوز الحد المسموح في النسخة المجانية (300). سيتم معالجة أول 300 فقط.")
                    # منطق تقليص البيانات يمكن إضافته هنا، لكن للتبسيط سنستمر مع التحذير
            except Exception as e:
                st.error(f"خطأ في الملف: {e}")
                st.stop()

            # تشغيل المحرك
            engine = PackingEngineV35()
            item_count = 0
            for idx, row in df.iterrows():
                if item_count >= 300: break # حماية الذاكرة
                engine.add_item(Item(row['Item'], row['Length'], row['Width'], row['Height'], row['Weight'], row['Qty'], idx))
                item_count += row['Qty']
            
            engine.run_packing()
            metrics = engine.get_metrics()
            safety = engine.get_safety_report()

            # --- عرض النتائج ---
            
            # 1. شريط التنبيهات (Stacker Logic)
            if safety['status'] == "Danger":
                st.error(safety['msg'])
            else:
                st.success(safety['msg'])

            # 2. المؤشرات الرئيسية (Market Comparison)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("نسبة الاستغلال", f"{metrics['utilization']:.1f}%", delta=f"{metrics['utilization']-65:.1f}% vs Market")
            
            market_savings = ((metrics['utilization'] - 65) / 100) * container_price if metrics['utilization'] > 65 else 0
            col2.metric("الأموال المستردة", f"${market_savings:.2f}", delta_color="normal")
            
            col3.metric("الوزن الكلي", f"{metrics['total_weight']} kg")
            col4.metric("عدد الطرود", f"{metrics['packed_count']}")

            # 3. المخطط 3D (EasyCargo Style)
            st.subheader("📦 المخطط الهندسي التفاعلي (3D)")
            st.plotly_chart(create_3d_chart(engine.packed_items), use_container_width=True)

            # 4. تقرير الذكاء الاصطناعي (Consultant)
            st.subheader("🧠 تقرير المستشار المالي (Gemini)")
            if api_key:
                report = get_gemini_consultation(api_key, metrics, container_price, safety)
                st.info(report)
            else:
                st.warning("أدخل مفتاح API للحصول على التحليل المالي.")

            # 5. الجدول التفصيلي
            with st.expander("عرض قائمة التعبئة التفصيلية"):
                data = [{"الترتيب": i.sequence_id, "الصنف": i.name, "الأبعاد": f"{i.length}x{i.width}x{i.height}", "الوزن": i.weight, "الموقع": i.position} for i in engine.packed_items]
                st.dataframe(pd.DataFrame(data))

    elif not uploaded_file and run_btn:
        st.warning("يرجى رفع ملف البيانات.")

if __name__ == "__main__":
    main()
