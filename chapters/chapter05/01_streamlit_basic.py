import streamlit as st

# 페이지 기본 설정 - 브라우저 탭 제목, 아이콘, 레이아웃을 지정합니다
st.set_page_config(
    page_title="Chapter 5 - Streamlit 기본 실습",
    page_icon="🧪",
    layout="wide"
)

# 화면 상단에 크게 표시되는 제목
st.title("Chapter 5 - Streamlit 기본 실습")

# 일반 텍스트 설명 문구
st.write("이 앱은 Streamlit의 기본 구성 요소를 연습하기 위한 실습입니다.")
st.write("아직 Claude API는 연결하지 않고, 입력창과 버튼 동작만 확인합니다.")

# 텍스트 입력창에 미리 채워 둘 예시 문의 내용
sample_message = """체중계가 앱이랑 연결이 안 됩니다.
AS 문의를 남겼는데 답변이 늦어서 답답합니다."""

# 여러 줄 텍스트 입력창 - value로 기본값, height로 높이를 지정합니다
customer_message = st.text_area(
    "고객 문의 입력",
    value=sample_message,
    height=160
)

# 버튼 - type="primary"로 강조 색상을 적용합니다
# 버튼을 클릭하면 if 블록 안의 코드가 실행됩니다
if st.button("입력 내용 확인하기", type="primary"):
    # strip()으로 공백만 있는 경우도 빈 입력으로 처리합니다
    if customer_message.strip():
        # 초록색 성공 메시지 박스를 표시합니다
        st.success("입력 내용이 정상적으로 확인되었습니다.")

        st.subheader("입력한 고객 문의")
        st.write(customer_message)

        st.subheader("간단한 정보")
        st.write(f"글자 수: {len(customer_message)}자")
    else:
        # 노란색 경고 메시지 박스를 표시합니다
        st.warning("고객 문의를 먼저 입력해주세요.")
