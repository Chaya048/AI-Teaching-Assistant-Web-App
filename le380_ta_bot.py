import streamlit as st
import pandas as pd
from google import genai
import os
import json
from datetime import datetime
from streamlit_float import *
import re

# --- 1. CONFIG & SETUP (ต้องอยู่บรรทัดแรกสุด) ---
st.set_page_config(page_title="LE380 TA ชื่อ AI", layout="wide")

# กำหนดสถานะ
DEFAULT_MODEL = "gemini-2.5-flash" 
DEFAULT_SUBJECT = "เครื่องมือวัดและการวัดทางไฟฟ้า"
ROLE_PROMPTS = {
    "Exam Coach": f'''
        คุณคือผู้ช่วยสอนวิชา{DEFAULT_SUBJECT} ที่มีข้อมูลจากข้อสอบเก่า 
        บทบาทของคุณคือการให้คะแนนตามเกณฑ์
        บอกคะแนนโดยมี "คะแนน: " คะแนนเต็ม 5 คะแนน และจะบอกผ่านเมื่อมีคะแนน 3 คะแนนขึ้นไป พร้อมบอกข้อนั้นที่ผ่านด้วยเสมอ และชี้จุดบกพร่องของคำตอบนักศึกษา
        โดยจะไม่ตอบคำถามที่นอกเหนือจากเนื้อหาและข้อสอบเก่าวิชานี้
        ''',
    "Mock Examiner" : f'''
        คุณคือผู้ช่วยสอนวิชา{DEFAULT_SUBJECT} ที่มีข้อมูลจากข้อสอบเก่า 
        บทบาทของคุณคือการตั้งคำถามซ้ำที่ยากขึ้น หรือคำถามต่อเนื่องจากคำตอบแรกของนักศึกษา เพื่อทดสอบความรู้เชิงลึก 
        บอกคะแนนโดยมี "คะแนน: " คะแนนเต็ม 5 คะแนน และจะบอกผ่านเมื่อมีคะแนน 3 คะแนนขึ้นไป พร้อมบอกข้อนั้นที่ผ่านด้วยเสมอ
        โดยจะไม่ตอบคำถามที่นอกเหนือจากเนื้อหาและข้อสอบเก่าวิชานี้''',
    "Socratic Tutor": f'''
        คุณคือผู้ช่วยสอนวิชา{DEFAULT_SUBJECT} ที่มีข้อมูลจากข้อสอบเก่า 
        บทบาทของคุณคือการติวแบบ Socratic โดยไม่ให้คำตอบทันที แต่จะแนะนำการเรียนรู้ด้วยการป้อนคำถามทีละขั้น เช็คคำตอบและให้ข้อแนะนำสั้นๆหากจำเป็น จะบอกผ่านเมื่อนักศึกษาสามารถตอบคำถามได้ถูกต้องตามเกณฑ์
        พร้อมบอกข้อนั้นที่ผ่านด้วยเสมอ โดยจะไม่ตอบคำถามที่นอกเหนือจากเนื้อหาและข้อสอบเก่าวิชานี้''',
    "Lecture": f'''
        คุณคือผู้ช่วยสอนวิชา{DEFAULT_SUBJECT} ที่มีข้อมูลจากข้อสอบเก่า 
        บทบาทของคุณคือการบรรยายเนื้อหาที่เกี่ยวข้องกับคำถามของนักศึกษา 
        เป้าหมายของคุณคือการทำให้นักศึกษาเข้าใจหลักการวัดทางไฟฟ้าอย่างลึกซึ้ง
        เนื้อหาที่นำมาสอนสามรถใช้ได้ทั้งจากข้อสอบเก่าและความรู้ทั่วไปในวิชานี้{DEFAULT_SUBJECT}
        โดยใช้ภาษาที่เป็นทางการแต่เข้าใจง่าย เน้นย้ำความละเอียดรอบคอบแบบวิศวกรวัดคุมและจะไม่ตอบคำถามที่นอกเหนือจากเนื้อหาและข้อสอบเก่าวิชานี้
        และบอกผ่านเมื่อนักศึกษาสามารถตอบคำถามได้ถูกต้องตามเกณฑ์ พร้อมบอกข้อนั้นที่ผ่านด้วยเสมอ''',
}

# --- 2. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
# เพิ่มตัวแปรเก็บสถานะคะแนน (ตัวอย่าง)
if "user_score" not in st.session_state:
    st.session_state["user_score"] = 0
if "questions_done" not in st.session_state:
    st.session_state["questions_done"] = []
if "selected_file" not in st.session_state:
    st.session_state["selected_file"] = ""

# --- 3. HELPER FUNCTIONS ---
def list_json_files():
    # ตรวจสอบว่ามีไฟล์หรือไม่ ถ้าไม่มีให้ return list ว่างป้องกัน error
    try:
        flist = os.listdir('.') 
        json_files = [f for f in flist if f.endswith(".json")]
        return json_files
    except:
        return []

def load_exam_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f) # โหลดเป็น JSON Object เพื่อให้นับจำนวนข้อได้
            text = json.dumps(data, ensure_ascii=False) # แปลงกลับเป็น String เพื่อส่งให้ AI
            return data, text
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return [], ""
    
def load_css():
    with open("static/styles.css", "r", encoding="utf-8") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

load_css()

# --- 4. SIDEBAR ---
with st.sidebar:
    logo = 'https://ece.engr.tu.ac.th/assets/img/logo/ECE-department-logo.svg'
    st.logo(logo, icon_image=logo, size="large")
    st.title("⚙️ Settings")
    ta_model = st.selectbox("เลือกผู้ช่วยสอน (Role)", options=ROLE_PROMPTS.keys())
    
    file_options = list_json_files()
    if file_options:
        data_file = st.selectbox("เลือกไฟล์ข้อสอบ", options=file_options)
        if st.session_state["selected_file"] != data_file:
            st.session_state["selected_file"] = data_file
            st.session_state.messages = []  # ล้างประวัติการสนทนาเมื่อเปลี่ยนไฟล์
            st.session_state["user_score"] = 0
            st.session_state["questions_done"] = []
    else:
        st.warning("ไม่พบไฟล์ .json ในโฟลเดอร์")
        data_file = None

    st.session_state["api_key"] = st.text_input("Gemini API KEY", type="password")
    
    if st.button("ล้างประวัติการสนทนา"):
        st.session_state.messages = []
        st.session_state["user_score"] = 0
        st.session_state["questions_done"] = []
        st.rerun()

# --- 5. MAIN CONTENT ---
st.title(f"🤖 LE380 AI Teaching Assistant")

# ตรวจสอบ API Key ก่อนเริ่ม
if not st.session_state["api_key"]:
    st.info("👈 กรุณาใส่ API Key ที่ Sidebar เพื่อเริ่มต้นใช้งาน")
    st.stop()
else:
    try:
        ai_client = genai.Client(api_key=st.session_state["api_key"])
    except Exception as e:
        st.error(f"API Key Error: {e}")
        st.stop()

# โหลดข้อมูลข้อสอบ
exam_data_list = []
exam_text = ""
if data_file:
    exam_data_list, exam_text = load_exam_data(data_file)

# สร้าง Tabs
tab1, tab2 = st.tabs(["💬 Chat กับ TA", "📊 Status การฝึก"])

# ==========================================
# TAB 1: Chat Interface
# ==========================================
float_init(theme=True, include_unstable_primary=False)

df = pd.DataFrame(exam_data_list)
for q in df.question_id:
    df.loc[df.question_id == q, 'question_id'] = f"{q[8:11]}"

with tab1:
    # แสดงประวัติการแชท
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                # User: จัดขวา (row-reverse) + สี user-bubble
                div = f"""
                    <div class="row-reverse">
                        <div class="chat-bubble user-bubble">{message['content']}</div>
                    </div>
                    """
            else:
                # AI: จัดซ้าย (chat-row) + สี ai-bubble
                div = f"""
                    <div class="chat-row">
                        <img class="chat-icon" src="app/static/ai_2.png" width=32 height=32>
                        <div class="chat-bubble ai-bubble">
                            {message['content']}
                        </div>
                    </div>
                    """
            st.markdown(div, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* 🔥 สำคัญ: ลบ Margin ด้านล่างของตัว Input เอง เพื่อไม่ให้ดันกล่องลอย */
    [data-testid="stBottom"] {
        padding-bottom: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # Input Box
    inputbox_container = st.container()
    with inputbox_container:
        user_input = st.chat_input("ยินดีต้อนรับสู่ระบบ TA แบบ AI ของภาคไฟฟ้าและคอมพิวเตอร์ ม.ธรรมศาสตร์ มาเริ่มกันเลยไหมครับ ?")
        button_css = float_css_helper(width="78%", bottom="-150px", transition=0)

        # 🔥 เพิ่มบรรทัดนี้: ใส่สีดำโปร่งแสง + เอฟเฟกต์เบลอ
        box_design_css = """
            background-color: rgba(14, 17, 23, 1); 
            
            box-shadow: 0px -15px 40px rgba(14,17,23,1);
            
            padding-left: 20px;         /* จัดระยะห่างภายใน */
            padding-right: 20px;
            padding-top: 0px;
            padding-bottom: 0px;

            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            border-bottom-left-radius: 0px;
            border-bottom-right-radius: 0px;

            margin-bottom: 0px !important;
        """
        float_parent(css=button_css + box_design_css)

    if user_input :
        # 1. แสดงข้อความ User
        st.session_state.messages.append({"role": "user", "content": user_input})
        div_user = f"""
            <div class="row-reverse">
                <div class="chat-bubble user-bubble">{user_input}</div>
            </div>
            """
        st.markdown(div_user, unsafe_allow_html=True)
            
        # 2. เตรียม Prompt
        SYSTEM_PROMPT = ROLE_PROMPTS[ta_model]
        full_prompt = SYSTEM_PROMPT + "\n"
        full_prompt += f"นี่คือข้อมูลข้อสอบทั้งหมด (JSON): {exam_text}\n"
        full_prompt += "History:\n"
            
        # ใส่ History ย้อนหลัง (เพื่อไม่ให้ Token เต็มเร็วเกินไป อาจจะ limit ไว้ที่ 10 ข้อความล่าสุด)
        for msg in st.session_state.messages[-10:]:
            full_prompt += f"{msg['role']}: {msg['content']}\n"
            
        # 3. เรียก Gemini AI
        try:
            with st.spinner("TA กำลังคิดคำตอบ..."):
                response = ai_client.models.generate_content(
                    model=DEFAULT_MODEL, 
                    contents=full_prompt
                )
                reply_text = response.text
                
            div_ai = f"""
                <div class="chat-row">
                    <img class="chat-icon" src="app/static/ai_2.png" width=32 height=32>
                    <div class="chat-bubble ai-bubble">{reply_text}</div>
                </div>
            """
            st.markdown(div_ai, unsafe_allow_html=True)

            # 4. บันทึกตอบกลับ
            st.session_state.messages.append({"role": "assistant", "content": reply_text})
                
            # (Optional) อัพเดทจำนวนข้อเพื่อเอาไปคำนวณ Status แบบคร่าวๆ
            keywords = ["Correct", "Incorrect", "ผ่าน", "ไม่ผ่าน"]
            current_ID = [q for q in df['question_id'] if q in user_input]
            is_pass_keyword = any(kw in reply_text for kw in keywords)
            if current_ID and is_pass_keyword:
                current_ID = current_ID[0]
                if current_ID not in st.session_state["questions_done"]:
                    st.session_state["questions_done"].append(current_ID)

            # คะแนน:\s* -> หาคำว่า "คะแนน:" (และ \s* คือเผื่อมีเว้นวรรค)
            # (\d+)      -> กลุ่มที่ 1: หาตัวเลข (คะแนนที่ได้)
            # /          -> หาเครื่องหมาย /51
            # (\d+)      -> กลุ่มที่ 2: หาตัวเลข (คะแนนเต็ม)
            pattern = r"คะแนน:\s*(\d+(?:\.\d+)?)"
            score = re.search(pattern, reply_text)
            if score is not None:
                score = float(score.group(1))
                st.session_state["user_score"] += score
            else:
                st.session_state["user_score"] += 0

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ AI: {e}")

# ==========================================
# TAB 2: Status Dashboard
# ==========================================
with tab2:
    st.header("📈 ผลการเรียนรู้และการฝึกฝน")
    
    if not exam_data_list:
        st.warning("กรุณาเลือกไฟล์ข้อสอบเพื่อดูสถิติ")
    else:   
        total_questions = len(exam_data_list)
        questions_done = len(st.session_state["questions_done"])
        
        #estimated_progress = min(questions_done / total_questions * 100, 100) 
        estimated_progress = 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="จำนวนข้อสอบทั้งหมด", value=f"{total_questions} ข้อ")
        with col2:
            st.metric(label="จำนวนข้อที่ผ่าน", value=f"{questions_done} ข้อ")
        with col3:
            st.metric(label=f"คะแนนสะสม (เต็ม {total_questions * 5} คะแนน)", value=f"{st.session_state['user_score']} คะแนน")

        st.subheader("ความคืบหน้า (Estimated Progress)")
        st.progress(int(estimated_progress))
        st.write(f"**{estimated_progress:.1f}%** ของเนื้อหาในชุดข้อสอบนี้")
        if estimated_progress == 100:
            st.balloons()
            st.success("🎉 ยินดีด้วย! คุณทำข้อสอบครบทุกข้อแล้ว สุดยอดมากครับ!", icon="🏆")
            st.toast("ภารกิจสำเร็จ! ครบ 100% แล้ว", icon="🎓")

        st.divider()
        
        st.subheader("📋 รายการข้อสอบในชุดนี้")
        # แสดงรายการข้อสอบแบบ Table ย่อๆ
        df.set_index(df['question_id'], inplace=True)
        df.drop(columns=['question_id', 'related_CLO', 'estimated_time_sec'], inplace=True)

        # สร้าง Column ใหม่ชื่อ "สถานะ"
        # ถ้า id ของแถวนั้น อยู่ใน session_state ให้ใส่ '✅' ถ้าไม่ ให้ขีด '-'
        df['Status'] = df.index.to_series().apply(
            lambda x: '✅' if str(x) in st.session_state["questions_done"] else '-'
        )

        if not df.empty:
            # สมมติว่า JSON มี field 'id' กับ 'question'
            # ปรับ column ตาม structure จริงของ json คุณ
            st.dataframe(df, use_container_width=True)
        else:
            st.text("รูปแบบ JSON ไม่ตรงกับตาราง หรือไม่มีข้อมูล")