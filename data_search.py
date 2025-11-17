import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("CSV파일 분석")

st.sidebar.header("파일 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=['csv']) 
# 파일 업로드 CSV만 받음

# --- 헤더 설정 입력 필드 추가 ---
st.sidebar.markdown("---")
st.sidebar.header("데이터 구조 설정")
# 사용자가 헤더 행 번호를 입력합니다. (1부터 시작하는 행 번호)
# CSV 파일의 첫 번째 행이 1입니다.
header_row_input = st.sidebar.number_input(
    "헤더(컬럼명)가 위치한 행 번호 (1부터 시작)", 
    min_value=0, 
    value=1,  # 기본값: 1행 (CSV의 첫 행)
    step=1, 
    help="0을 입력하면 헤더가 없는 것으로 간주됩니다. 7을 입력하면 7번째 줄을 헤더로 사용합니다."
)
# Pandas는 0부터 시작하는 인덱스를 사용하므로, 사용자 입력 (N)을 N-1로 변환
header_index = header_row_input - 1 if header_row_input >= 1 else None

def load_csv_with_encoding_check(uploaded_file, header_index): #인코딩 순차적으로 도는 함수. gemini 구현
    """
    업로드된 CSV 파일을 여러 인코딩을 시도하여 읽어들이는 함수
    """
    # 일반적으로 많이 사용되는 인코딩 목록을 우선순위 순으로 정의합니다.
    # 1. UTF-8 (가장 일반적이고 권장됨)
    # 2. CP949 (한국어 Windows 환경에서 자주 사용됨)
    # 3. EUC-KR (또 다른 한국어 표준)
    # 4. Latin-1 (대부분의 문자를 읽을 수 있는Fallback 인코딩)
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
    
    # 파일을 메모리 버퍼로 읽음
    file_bytes = uploaded_file.getvalue()

    for encoding in encodings:
        try:
            # StringIO를 사용하여 버퍼에서 파일처럼 읽어들입니다.
            # pd.read_csv()는 파일 경로뿐만 아니라 파일과 유사한 객체도 받습니다.
            df = pd.read_csv(
                pd.io.common.BytesIO(file_bytes), # 바이트 데이터를 파일 객체처럼 전달
                encoding=encoding,
                header=header_index # 사용자가 입력한 헤더 위치 적용
            )
            st.success(f"파일이 **{encoding}** 인코딩으로 성공적으로 로드되었습니다! 🎉")
            return df
        except UnicodeDecodeError:
            # 해당 인코딩으로 디코딩 실패 시 다음 인코딩 시도
            continue
        except Exception as e:
            # 기타 예외 처리
            st.warning(f"인코딩 시도 중 기타 오류 발생 ({encoding}): {e}")
            return None
        
    with st.expander("⚠️ 인코딩 자동 감지 실패! 수동으로 입력해주세요.", expanded=True):
        st.warning("자동 감지된 인코딩을 찾을 수 없습니다. 자주 사용되는 인코딩을 직접 입력하고 [재시도] 버튼을 눌러주세요.")
        
        # st.text_input을 통해 사용자에게 인코딩 입력받기
        user_encoding = st.text_input("인코딩 입력 (예: cp949, utf-8, euc-kr):", value="cp949", key="user_enc")
        
        # 버튼을 누르면 인코딩을 세션 상태에 저장하고 앱을 재실행
        if st.button("수동 인코딩으로 재시도"):
            try:
                # 사용자가 입력한 인코딩으로 재시도
                df = pd.read_csv(
                    pd.io.common.BytesIO(file_bytes),
                    encoding=user_encoding
                )
                st.success(f"파일이 **{user_encoding}** 인코딩으로 성공적으로 로드되었습니다! 🎉")
                return df
            except UnicodeDecodeError:
                st.error(f"입력하신 인코딩 (**{user_encoding}**)으로도 파일을 읽을 수 없습니다.")
            except Exception as e:
                st.error(f"재시도 중 기타 오류가 발생했습니다: {e}")
            
    # 최종적으로 재시도 실패 또는 입력 대기 상태
    return None

def show_fileInfo(df):
    with st.expander("데이터 기본 정보 보기"): #체크박스들을 하나로 묶기
        # 🌟 변경된 부분: 체크박스 대신 숫자 입력 필드를 사용
        # 🌟 1. 기능을 켜고 끄는 체크박스
        st.subheader("데이터 원본 보기")
        show_data = st.checkbox(
            "데이터 원본 (상위 N개) 보기", 
            value=True, # 기본값을 True로 설정하여 처음부터 보이도록 할 수 있습니다.
            key='show_data_checkbox'
        )

        # 🌟 2. 체크박스가 선택된 경우에만 숫자 입력 필드와 데이터프레임을 표시
        if show_data:
            # 사용자에게 상위 N개 행을 입력받습니다.
            num_rows = st.number_input(
                "확인할 상위 행 개수 (N)", 
                min_value=1, 
                max_value=len(df),
                value=5,           # 기본값은 5
                step=1,
                key='head_num'
            )
            
            # 입력된 값(num_rows)을 df.head()에 적용하여 출력
            st.dataframe(df.head(num_rows))
        
        if st.checkbox("데이터 Shape (행/열)"):
            st.write(df.shape)
            st.write("데이터는", df.shape[0], "개의 행과", df.shape[1], "개의 열로 이루어져 있습니다.")

        if st.checkbox("데이터 기본 정보 (df.info)"):
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.code(s) #text도 되고 code도 됨
        #st.write(df.info()) 이렇게 쓰면 안됨

        if st.checkbox("기초 통계량 (df.describe)"):
            st.dataframe(df.describe())
        
def show_menu2(df):
    column_list = df.columns.tolist()
    selected_column = st.selectbox("분석할 컬럼을 선택하세요", column_list)

    if selected_column:
        st.subheader(f"'{selected_column}' 컬럼 분석")

        # 1. 숫자형(numerical) 컬럼일 경우
        # (여기서는 간단히 int나 float 타입인지 체크)
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.write("선택한 컬럼은 숫자형입니다.")
            st.write("데이터 분포 (히스토그램)")
            fig, ax = plt.subplots() #
            ax.hist(df[selected_column].dropna(), bins=30, edgecolor='k')
            # 결측치가 있으면 오류남 .fillna()
            st.pyplot(fig) # st.pyplot()으로 Matplotlib 차트 표시

        # 2. 문자형/범주형(categorical) 컬럼일 경우
        else:
            st.write("선택한 컬럼은 범주형입니다.")
            st.write("데이터 빈도 (막대 그래프)")
        
            # 값(value)들의 개수를 세서 막대 그래프로 그리기
            value_counts = df[selected_column].fillna("NaN").value_counts() #문자형 컬럼은 NaN도 하나의 정보라 실제 NaN을 문자열 NaN으로 바꿔 채우기
        
            # Streamlit의 내장 막대 차트 사용
            st.bar_chart(value_counts)

# --- 3. 사용자 정의 그룹화 및 집계 분석 함수 (새로 추가) ---
def show_menu3(df):
    st.header("사용자 정의 그룹화 및 집계 분석")
    st.markdown("사용자가 직접 그룹 기준 컬럼, 집계 대상 컬럼, 집계 방식을 선택하여 데이터를 요약합니다.")
    
    # 1. 컬럼 목록 준비 (생략)
    column_list = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        st.warning("데이터프레임에 집계(sum, mean 등)를 수행할 수 있는 수치형 컬럼이 없습니다.")
        return

    # 2. 사용자 입력 받기 (생략)
    col1, col2, col3 = st.columns(3) 

    with col1:
        input1 = st.multiselect(
            "그룹 기준 컬럼 선택 (GroupBy)",
            column_list,
            default=column_list[0:1] if column_list else [], 
            key='group_col_3', # 🌟 키가 'group_col' 이었다면 'group_col_3' 등으로 변경하여 충돌 방지
            help="하나 이상의 컬럼을 선택하여 계층적으로 그룹화할 수 있습니다."
        )
    with col2:
        input2 = st.selectbox(
            "집계 대상 컬럼 선택 (Target)",
            numeric_cols,
            key='target_col_3',
            help="sum, mean 등의 연산을 적용할 수 있는 수치형 컬럼을 선택합니다."
        )
    with col3:
        input3 = st.selectbox(
            "집계 방식 선택 (Aggregation)",
            ["mean", "sum", "count", "max", "min", "std"],
            key='agg_func_3',
            help="선택된 대상 컬럼에 적용할 통계 함수를 고릅니다."
        )

    st.markdown("---")

    # 3. 데이터 집계 및 계산
    if input1 and input2 and input3:
        try:
            # groupby()와 agg()를 사용하여 집계 수행
            result4 = df.groupby(input1, as_index=False)\
                        .agg(value = (input2, input3))
            
            # 🌟 결과 데이터프레임 표시
            st.subheader(f"결과: {', '.join(input1)} 별 {input2}의 {input3}")
            st.dataframe(result4)
            
            # --- 🌟 차트 부분 수정 시작 ---
            
            # Multiselect로 두 개 이상 선택 시, 인덱스 충돌을 피하기 위해 새로운 인덱스 컬럼 생성
            if len(input1) > 1:
                # 선택된 그룹 컬럼들의 값을 하나의 문자열로 결합하여 새로운 인덱스 컬럼을 만듭니다.
                result4['Chart_Index'] = result4[input1].astype(str).agg(' - '.join, axis=1)
                chart_df = result4.set_index('Chart_Index')['value']
            else:
                # 컬럼이 하나만 선택된 경우, 기존 컬럼을 인덱스로 사용
                chart_df = result4.set_index(input1[0])['value'] 

            st.subheader("결과 차트")
            # Pandas Series를 bar_chart에 전달
            st.bar_chart(chart_df)

            # --- 차트 부분 수정 끝 ---

        except Exception as e:
            # 🌟 오류 메시지를 명확히 분리하여, MultiIndex 관련 오류를 명확히 안내
            if 'index — streamlit-generated' in str(e) or 'MultiIndex' in str(e):
                st.error("데이터 집계 중 차트 오류가 발생했습니다. 두 개 이상의 그룹 컬럼을 선택했을 때 차트 라이브러리 충돌이 발생할 수 있습니다. (Pandas MultiIndex 문제)")
            else:
                 st.error(f"데이터 집계 중 오류가 발생했습니다. 선택한 컬럼과 함수 조합을 확인해주세요. (오류: {e})")
    
    else:
        st.warning("그룹 기준 컬럼, 집계 대상, 집계 방식을 모두 선택해야 분석을 시작할 수 있습니다.")
            
if uploaded_file is not None:
    # **변경:** 파일 읽기 로직을 함수로 대체
    df = load_csv_with_encoding_check(uploaded_file, header_index)
    
    if df is not None:
        st.header("1. 데이터 기본 정보")
        show_fileInfo(df) #함수 내부에서 전역변수를 참조하는것보다 인수로 받아서 처리하는게 더 안전
        st.markdown("---")
        st.header("2. 컬럼 상세 분석")
        show_menu2(df)
        st.markdown("---")
        show_menu3(df)

    # (실패 시) df가 None일 경우 실행되는 부분
    else: 
        # 파일 업로드/인코딩 처리가 완전히 실패했음을 사용자에게 알림
        # (load_csv_with_encoding_check 함수에서 이미 오류 메시지가 출력되었을 가능성이 높음)
        st.error("☝️ CSV 파일 분석을 진행할 수 없습니다. 위의 인코딩 오류 메시지 또는 파일 형식을 확인해주세요.")

else:
    st.info("사이드바에서 CSV 파일을 업로드해주세요.")