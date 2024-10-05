import os #listdir을 위한 모듈
# import os.path #isdir, isfile을 위한 모듈
import pandas as pd
import numpy as np
import re #sub(regex, replacement, str) 등의 Regex를 위한 모듈
import chardet #encoding 자동감지 라이브러리
import datetime #생년월일 다룰때 필요
from fuzzywuzzy import fuzz #연락처가 1~2개 차이로 다른경우 단순기입오류일듯한데 다르게 취급되고 있어서 문자열 유사성 detect을 위한 모듈
                            #-> 성능향상을 위해 'python-Levenshtein' 라이브러리를 선택적으로 사용할 수 있는데 해당 라이브러리가 설치되지 않은 경우,
                            #기본적으로 순수 파이썬으로 작성된 'Sequencematcher'를 사용하며, 이때 경고 메시지가 표시됨.
#->    pip install python-Levenshtein 로 해결
import zipfile #try except 구문에서 오류 잡기위해 import        
import io #StringIO 쓰기위한 라이브러리 (문자열을 파일 객체처럼 만드는)   
# pip install --upgrade openpyxl 필요
import impyute as impy #MICE 결측치대체 라이브러리
from impyute.imputation.cs import mice #MICE 결측치대체 라이브러리
from sklearn.experimental import enable_iterative_imputer #MICE 결측치대체 라이브러리
from sklearn.impute import IterativeImputer #MICE 결측치대체 라이브러리

#그래프 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl

#MICE
from impyute.imputation.cs import mice
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#multiple linear regression
import statsmodels.api as sm
from statsmodels.formula.api import ols
# from sklearn.metrics import mean_squared_error   

#메모리사용량 확인
import psutil             

#호출 인수 받기
import sys          

#pickle로 객체 불러오기
import pickle  

# FutureWarning 무시
import warnings


                    
# '''폴더 주소 적는 코드'''
dir = input('본 파이썬 파일을 필요한 상위폴더와 같은 위치에 저장하고, 해당 상위폴더 이름을 정확히 기재해주세요(예; RAW ):')
# dir = 'RAW'
# 현재 스크립트의 경로 가져오기
# script_dir = os.path.dirname(os.path.abspath(__file__))
# 해당 경로로 이동
# os.chdir(script_dir)

root_dir = os.path.abspath(dir)
# root_dir = "C:/Users/RexSoft/Desktop/Project/2024년 광주 도시문제 해결 AI솔루션 제작 지원사업_by 20241130/데이터/데이터_확인용/RAW-20240813/RAW"

path_list_patient_info=[] #1) 예약현황파일 파일 경로를 리스트에 저장
path_list_balance=[] #2) 균형능력측정및훈련시스템' 파일 경로를 리스트에 저장
path_list_muscle=[] #3) 근기능평가세트 파일 경로를 리스트에 저장  
path_list_urine=[] #4) 뇨분석기 파일 경로를 리스트에 저장 
path_list_autonomic=[] #5) 자율신경측정시스템 파일 경로를 리스트에 저장 
path_list_gait=[] #6) 전신반응분석시스템(txt) 파일 경로를 리스트에 저장 
path_list_cognition=[] #7) 종합신경인지검사시스템 파일 경로를 리스트에 저장 
path_list_inbody=[] #8) 체성분분석기 파일 경로를 리스트에 저장
path_list_scoliosis=[] #9) 체형굴곡측정시스템(txt,csv) 파일 경로를 리스트에 저장

def files_in_dir(i):#, prefix):
    files = os.listdir(i)
    for file in files:
        path = os.path.join(i, file)
        #'메뉴얼' '설명서' '양식' 등의 필요없는 폴더 제외. '피부진단시스템'&'체성신경'&'무선근전도분석기'&'지엔아이티씨VR'&'카운트마인드' DB 필요없음. 
        if os.path.isdir(path) and not any(k in path for k in ['기타', '모니터링', '메뉴얼', '설명서', '양식', '피부진단', '체성신경','무선근전도','지엔아이','인바디','혈압','카운트마인드','체형굴곡측정기']): 
            #폴더라면 Recursive하게 다시 함수호출하여 재탐색!
            files_in_dir(path)#, prefix + "   "): 이렇게 하면 출력값이 시각적으로 계층구조를 가지도록 하는 tip!
            '''각 필요한 파일들의 경로를 DB별로 리스트에 저장!!'''
            #'취소','장비거부' 등 필요없는 파일 제외.
            '''1) 예약현황파일 -> '성함','연락처','생년월일','성별' 4가지 데이터 추출필요''' #이런 주석은 indentation 안지키면 오류뜸!
        elif '예약현황' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']) and not any(o in path for o in ['예약취소', '취소']): 
            path_list_patient_info.append(path)
            '''2) 균형능력측정및훈련시스템 -> 'Romberg Quitient'변수와 'Test Name: 김귀 2021-09-23 - 1', 'Person Name: 귀옥 김' 3가지 데이터 추출필요  '''
        elif '균형능력' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']):
            path_list_balance.append(path)
            '''3) 근기능평가세트 -> '날짜(2024-07-31형태임)','이름','나이','Goniometer Gubun'=='손목'쪽에 가까운 'Left Avg','Right Avg'변수값 필요함 '''
        elif '근기능평가' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']) and not '새 텍스트' in path:
            path_list_muscle.append(path)   
            '''4) 뇨분석기 -> 'classifiy(오타아니고 실제로 이렇게 되어있음)','DATE(06월 17일 형식)', 'KET','PRO','GLU','SG' '''
        elif '뇨분석' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']) and not '양식' in path:
            path_list_urine.append(path)  
            '''5) 자율신경측정시스템 -> 'Name','Date(2024-08-06 09.44.37 형식)','ECG1_SDNN' '''
        elif '자율신경측정' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']) and not '전신반응분석시스템' in path :
            path_list_autonomic.append(path)              
            '''6) 전신반응분석시스템(txt) -> 'Last Name(성문)','First Name(전)','Birthdate(1992-05-15)','Velocity' '''
        elif '전신반응' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']) and not '자율신경측정' in path:
            path_list_gait.append(path)          
            '''7) 종합신경인지검사시스템 ->'이름'우측값, '검사일'우측값,주의력sheet:'집중력 유지시간', 기억력sheet:'시각단기기억검사->반응시간',감각및운동협응sheet:'시청각반응시간검사->반응시간',문제해결력sheet:'논리적사고력검사->반응시간'  '''
        elif '종합신경인지' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']) :
            path_list_cognition.append(path)          
            '''8) 체성분분석기 ->혈압폴더와 인바디폴더에 따로 저장된뒤 merge되는듯! 일단 merge된 파일을 기준으로 보자
            인바디sheet:'신장','생년월일','14. 검사일시(2023.12.06. 09:55:04형식)','체중','45. BMI','155. 제지방량지수','156. 체지방량지수,'172. 50kHz-Whole Body Phase Angle','175. 복부 바깥둘레',
            '205. SMI (Skeletal Muscle Index)' '''
        elif '체성분분석' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']) :
            path_list_inbody.append(path)
            '''9) 체형굴곡측정시스템(txt,csv,xlsx)-> '성','이름','생일','lateral_deviation_(surface)_VPDM_(rms)_[mm]','kyphotic_angle_ICT-ITL_(max)_[°]','lordotic_angle_ITL-ILS_(max)_[°]'  '''
        elif '체형굴곡측정시스템' in path and any(path.endswith(ext) for ext in ['.txt', '.csv', '.xlsx', '.xls']) and not '장비거부' in path:
            path_list_scoliosis.append(path)                                                             
        else:
            pass

files_in_dir(root_dir) # files_in_dir() 함수실행!
# print(path_list_inbody)
#성함에 이상한이름들이 들어가있음
def process_name(name): #'성함'에 
    # 문자열을 확인하고 숫자만 있는 경우와 길이가 2 이하인 경우를 NaN으로 변환
    name = str(name).strip()  # 문자열 앞뒤 공백 제거
    if pd.isna(name) or name in ['', '광주여대']:
        return np.nan
    if not re.fullmatch(r'[가-힣]{2,5}', name): #앞패턴이 최소 2번, 최대4번 반복해서 나타나는경우
        return np.nan
    if re.fullmatch(r'\d+', name): # 숫자만 포함된 경우
        return np.nan
    if len(name) < 2:# 글자 수가 2 미만인 경우
        return np.nan
    return str(name)

# 전화번호 포맷을 통일하는 함수 정의
def format_phone_number(phone):
    # 숫자만 추출
    if pd.isna(str(phone).strip()) or (str(phone).strip() == ''): #isna()는 None, NaN 및 NA를 걸러냄
        return np.nan #nan으로 채워서 일관성을 유지
    digits = re.sub(r'\D', '', phone) #r은 "raw" 문자열 리터럴을 의미. 
    #\D: 정규식에서 \D는 "숫자가 아닌 문자"를 의미. 즉, 숫자를 제외한 문자를 찾는 패턴. ->digits은 숫자만 남음.
    # 10자리나 11자리 전화번호로 포맷
    if len(digits) == 10 and digits[1]=='0': #1025012793 or 00-2528-9438 같은형태들
        return f"010-{digits[2:6]}-{digits[6:]}"
    elif len(digits) == 11: #(010)25012793 or 010-2501-2793 or 010.2501.2793 
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    elif len(digits) == 10 and digits[1]!='0': #062-523-0057
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    else:
        return np.nan  # 나머지 값들은 na처리

# 생년월일 포맷을 변경/통일 하는 함수 정의
#datetime.datetime.strptime: 문자열로 된 날짜 및 시간을 datetime 객체로 변환하는데 사용
#datetime.datetime.strftime: 지정한 포맷에 맞게 문자열로 변환
#datetime.datetime.date: datetime 객체에서 시간 정보를 제외한 date 객체를 반환
def format_date(date): #생년월일, 검사일시를 db1에 맞게 바꾸기 위한 함수 정의
    if pd.isna(str(date)) or str(date).strip() == '': 
        return np.nan
    date=str(date).strip() #문자열 앞 뒤 공백 제거
    for fmt in ('%Y-%m-%d %H:%M:%S','%Y.%m.%d. %H:%M:%S', '%Y-%m-%d', '%Y%m%d', '%Y-%m-%d %I:%M:%S %p','%Y.%m.%d.','%Y.%m.%d', '%Y-%m-%d %p %I:%M:%S','%Y-%m-%d %p %i:%M:%S'\
        , '%Y.%m.%d.', '%Y.%m.%d', '%Y-%m-%d %H.%M.%S'): #%p는 AM, PM이런거임
        try:
            return datetime.datetime.strptime(date, fmt).strftime('%Y%m%d') #문자열로 반환
        except ValueError:
            continue
    return np.nan

#한글 추출 함수
def extract_korean(text):
    match = re.findall(r'[가-힣]+', str(text))
    return ''.join(match) if match else np.nan  # 한글이 있는 경우만 추출하여 반환, 없으면 원래 값 반환


def process_to_nan(x): 
    # 문자열을 확인하고 숫자만 있는 경우와 길이가 2 이하인 경우를 NaN으로 변환
    if pd.isna(str(x).strip()) or (str(x).strip() in ['', '광주여대','-','test','nan']) : #엑셀이라 '-' 추가! #문자형으로 'nan'이 적힌 행도 있었음
        return np.nan
    return str(x)

'''★★★★df_inbody_before '''
df_inbody_before=pd.DataFrame() #빈 데이터프레임 생성
# for j,name in enumerate(path_list_inbody):
# 		if '송모세' in name:
# 			print(name)

# if '송모세' in path_list_inbody: #sys.argv[2] 성함을 적을것임 (기본적으로 헬스기기에서 저장되는 파일명에는 이름이 꼭 들어갈 것임.)sys.argv[2]
#     # print(path_list_inbody)
# for j,name in enumerate(path_list_inbody):
# 	if '송모세' in name:
for j,name in enumerate(path_list_inbody):
	if sys.argv[1] in name:
		try:
			if name.endswith('.xlsx') or name.endswith('.xls'): 
				df_dict1=pd.read_excel(name, sheet_name=None, header=0, index_col=None, engine='openpyxl', dtype=str)  
				a = list(df_dict1.keys())
				if len(a)>=2: #혈압 sheet까지 반드시 있는 대상자만 merge -> '인바디'sheet에는 이름, 연락처가 masking되어있고 혈압sheet는 이름과, 검사일시가 나와있음
					df_dict1[a[0]].rename(columns={'14. 검사일시':'3. 검사일시'}, inplace=True) 
					df_dict1_0 = df_dict1[a[0]].map(lambda x: str(x).strip() ) #원래 applymap
					df_dict1_1 = df_dict1[a[1]].map(lambda x: str(x).strip() ) 
					df_excel=df_dict1_0[['1. 성명','3. 신장','4. 생년월일','5. 성별','3. 검사일시','15. 체중','45. BMI','155. 제지방량지수','156. 체지방량지수','172. 50kHz-Whole Body Phase Angle',\
						'175. 복부 바깥둘레','205. SMI (Skeletal Muscle Index)']].merge(df_dict1_1[['1. 성명','3. 검사일시','4. 최고혈압','5. 최저혈압']],\
							on=['1. 성명','3. 검사일시'], how='outer')#,'7. 평균혈압'
					df_excel = df_excel.map(lambda x: str(x) if not pd.isna(x) else np.nan)
					# print(df_excel)
					# if len(df_excel) > 1:                                                                            
					df_excel2=df_excel.copy()
					k=df_excel[['1. 성명','4. 최고혈압','5. 최저혈압',]].iloc[1]
					df_excel2.loc[0, ['1. 성명','4. 최고혈압','5. 최저혈압']] = k #SettingWithCopyWarning 피하려고.
					df_inbody_before=pd.concat([df_inbody_before, df_excel2.iloc[[0]]], ignore_index=True, join='outer',axis=0) #axis=0이 세로 결합
		except UnicodeDecodeError as e:
			print(f"warning; 인코딩이 적절치 않습니다(xlsx혹은 xls파일이 아님): {name}, Error: {e}")
		except pd.errors.EmptyDataError as e:
			print(f"warning; 빈 파일입니다: {name}, Error: {e}")
		except FileNotFoundError as e:
			print(f"warning; 파일을 찾을 수 없습니다: {name}, Error: {e}")
		except ImportError as e:
			print(f"warning; 모듈을 불러올 수 없습니다: {name}, Error: {e}")
		except Exception as e:
			print(f"warning; 파일열기 실패 등의 기타오류 발생: {name}, Error: {e}")
# 변수 값이 NA이면 대체값 적용
# for i, name in enumerate(df_inbody_before.columns):
# 	if pd.isna(df_inbody_before[name]).any(): # A열에 NA가 하나라도 있으면 True
# 		df_inbody_before[name] = 0 #연령별 성별 평균 대체값

df_inbody_before.rename(columns={'1. 성명':'성함','3. 신장':'신장','4. 생년월일':'생년월일','5. 성별':'성별','3. 검사일시':'검사일시','15. 체중':'체중','45. BMI':'BMI',\
 '155. 제지방량지수':'FFMI','156. 체지방량지수':'BFMI','172. 50kHz-Whole Body Phase Angle':'wholeBPA','175. 복부 바깥둘레':'WC',\
         '205. SMI (Skeletal Muscle Index)':'SMI','4. 최고혈압':'SBP','5. 최저혈압':'DBP'},inplace=True) #,'7. 평균혈압':'meanBP'
df_inbody_before['생년월일']=df_inbody_before['생년월일'].apply(format_date) 
# df_inbody_before
df_inbody_before['검사일시']=df_inbody_before['검사일시'].apply(format_date) #인바디 sheet와 혈압 sheet의 검사일시는 동일하기에 merge하고 format 맞춰줘도 괜찮음.

# '전체' 열에 대해 처리 적용 -> 원래 applymap을 썼었는데 FurueWarning뜸. 곧 map으로 그 기능이 통합되는듯
df_inbody_before = df_inbody_before.map(lambda x: process_to_nan(x))  #'성함'에 '-','test','test1','홍길동185' 등이 있음. #apply(,axis=1) 옵션은 적용할 축은 columns란 의미 -> 각 행series를 전달함(0: index가 디폴트인듯)
df_inbody_before['성함'] = df_inbody_before['성함'].apply(lambda x: str(x).strip()) # process_to_nan 함수에서 np.nan으로 처리된 값들로 인해 모든처리가 끝나면 float로 바뀌어서 오류가 뜨는듯
df_inbody_before['성함'] = df_inbody_before['성함'].apply(process_name)  #np.nan으로 처리 잘되었음. 
# df_inbody_before

'''db8 inbody: 데이터에서 [성함, 생년월일, 검사일시] 중에 하나라도 na가 있는 행을 제외하고, 그 세가지 변수가 같은행을 unique하게 남긴 데이터 :하루에 2번 검사한 사람이면 젤 마지막 검사로 덮어쓰기 '''
df_inbody_before.dropna(subset=['성함','생년월일','검사일시'], how='any',inplace=True)#how='any'(default)
db8_inbody=df_inbody_before.drop_duplicates(['성함','생년월일','검사일시'], keep='last', ignore_index=True) #3개의 변수를 unique행으로 남겨놔야 나중에 '성함 검사일시'로 merge도 가능
db8_inbody=db8_inbody.sort_values(['성함','생년월일','검사일시'], na_position='first', ignore_index=True) #오름차순 정렬
#'연령' 생성!
db8_inbody['연령']=db8_inbody.apply(lambda row: str(int(row['검사일시'][0:4]) - int(row['생년월일'][0:4]) + 1) \
    if pd.notna(row['생년월일']) and pd.notna(row['검사일시']) and len(row['검사일시']) == 8 and len(row['생년월일']) == 8\
        else np.nan , axis=1)  #'연령' 생성 #axis=1이 있어야 각 행단위로 수행됨 (axis=0이 default임)
# db8_inbody.to_csv('db8_inbody.csv', index=False, encoding='euc-kr', na_rep=np.nan,chunksize=100)

db8_inbody2=db8_inbody[['성함', '연령', '성별', '생년월일',  '검사일시','신장' , '체중', 'BMI', 'FFMI', 'BFMI', 'wholeBPA', 'WC', 'SMI', 'SBP', 'DBP']]  #성별
db8_inbody2.replace({'M':'남','F':'여'},inplace=True)
db8_inbody2.drop_duplicates(inplace=True, keep='last')
db8_inbody2.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True) 

# 생년월일 확인으로 동명이인 시스템적으로 확인하기
# if format_date(sys.argv[3]) == db8_inbody2['생년월일']:
# 	pass
# else:
# 	raise ValueError(f"warning; db8_inbody2 생년월일이 다릅니다")

'''★★★★ db4_urine : 뇨분석기 '''
def extract_and_convert_date(row):
    try:
        # 문자열에서 첫 6자리 추출
        date_parted = str(row).split('_')[0]
        # 6자리 날짜를 '%y%m%d' 형식으로 변환하고, 이를 '%Y%m%d'로 변환
        try:
            a= pd.to_datetime(date_parted, format='%y%m%d', errors='coerce') #'ignore'로 하면 오류남. nan으로 인해 float처리되어 그런듯?
            # 변환된 날짜가 NaT일 경우 NaN 반환
            if pd.isna(a):
                return np.nan
        except:
            return np.nan
        a_str = datetime.datetime.strftime(a,'%Y%m%d')
        return a_str
    except ValueError:
        return np.nan   # 변환 실패 시 NaN 반환

db4_urine=pd.DataFrame() #빈 df 생성
# if  in path_list_urine:#sys.argv[2]
# for i, name in enumerate(path_list_urine):
# 	if '송모세' in name:
for i, name in enumerate(path_list_urine):
	if sys.argv[1] in name:
		# print(name)
		try:
			urine=pd.read_excel(name, header=0, index_col=None, engine='openpyxl', dtype=str) #sheet_name=None
			urine.rename(columns={'classifiy':'성함', 'DATE':'검사일시'},inplace=True)
			# 함수 적용
			urine['검사일시']=urine['성함'].apply(extract_and_convert_date)
			urine['성함'] = urine['성함'].apply(extract_korean)
			db4_urine=pd.concat([db4_urine, urine[['성함','검사일시','KET','PRO','GLU','SG']]], sort=True, ignore_index=True, join='outer')
		except UnicodeDecodeError as e:
			print(f"warning; 인코딩이 적절치 않습니다(xlsx 파일이 아님): {name}, Error: {e}")
		except pd.errors.EmptyDataError as e:
			print(f"warning; 빈 파일입니다: {name}, Error: {e}")
		except FileNotFoundError as e:
			print(f"warning; 파일을 찾을 수 없습니다: {name}, Error: {e}")
		except ImportError as e:
			print(f"warning; 모듈을 불러올 수 없습니다: {name}, Error: {e}")
		except Exception as e:
			print(f"warning; 파일열기 실패 등의 기타오류 발생: {name}, Error: {e}")
# 변수 값이 NA이면 대체값 적용
for i, name in enumerate(db4_urine.columns):
	if pd.isna(db4_urine[name]).any(): # A열에 NA가 하나라도 있으면 True
		db4_urine[name] = 0 #연령별 성별 평균 대체값

db4_urine['성함']=db4_urine['성함'].apply(extract_korean)
db4_urine['성함']=db4_urine['성함'].apply(process_name)
db4_urine=db4_urine[['성함','검사일시','KET','PRO','GLU','SG']]
db4_urine.dropna(subset=['성함','검사일시'],inplace=True,ignore_index=True) #how='any'(default)
db4_urine.sort_values(['성함','검사일시'], na_position='first', ignore_index=True, inplace=True)
# db4_urine.to_csv('db4_urine.csv',index=False, na_rep=np.nan, encoding='euc-kr',chunksize=100)

# if format_date(sys.argv[3]) == db4_urine['생년월일']:
#     pass
# else:
#     raise ValueError(f"warning; db4_urine 생년월일이 다릅니다")
'''db_148'''
db_148=db8_inbody2.merge(db4_urine, on=['성함','검사일시'], how='left')
db_148.drop_duplicates(['성함','생년월일','검사일시'],inplace=True,ignore_index=True)
db_148.dropna(subset=['성함','생년월일','검사일시'],ignore_index=True,inplace=True)
db_148.sort_values(['성함','검사일시'],na_position='first',ignore_index=True)



'''db2_balance'''

def extract_name(person_info):
    if isinstance(person_info, str) and person_info.startswith('Person Name:'): #해당 행에서 'Person Name: 으로 시작하는 요소일 경우:
        a=person_info.split(': ')[1].strip() # 'Person Name: ' 이후의 이름을 추출
        try:
            if len(a)==3: #예) '공 유'
                return a[2]+a[0:2]
            elif len(a)==4: #예) '길동 홍'
                return a[3]+a[0:3]
            elif len(a)==5: #예) '에스더 이' 
                return a[4]+a[0:4]
        except IndexError:
            return np.nan
    

def extract_date(date_info):
    if isinstance(date_info, str) and date_info.startswith('Created:'):
        a=date_info.split(':')[1].strip()[0:10] # 'Person Name: ' 이후의 이름을 추출
        return datetime.datetime.strptime(a, '%d.%m.%Y').strftime('%Y%m%d') #문자열로 반환
    return np.nan

def extract_romberg_from_tab(date_info):
    if isinstance(date_info, str) and date_info.startswith('Eyes Open'):
        a=date_info.split('\t')[6].strip()[0:3] # \t를 중심으로 나눠서 5번째 값이 romberg
        return a #문자열로 반환
    return np.nan

def find_index(df, keyword):
    # 각 열을 문자열로 변환하여 'keyword'를 포함하는지 확인
    detects = df.apply(lambda x: x.str.contains(keyword, na=False))
    # 'keyword'를 포함하는 행의 인덱스를 가져오기
    row_numbers = df[detects.any(axis=1)].index
    return row_numbers

def find_columns(df, keyword):
    # columns_with_keyword = []
    for i, col in enumerate(df.columns):
        if df[col].astype(str).str.contains(keyword).any(axis=0):  # '정확도'를 포함하는 값이 있는지 확인
            a=i
    return a

#함수: 파일의 인코딩을 자동으로 감지
def read_file_with_detected_encoding_csv_comma(file_path):
   
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    # 감지된 인코딩으로 파일 읽기
    return pd.read_csv(file_path, header=0, index_col=None, dtype=str, encoding=encoding)

def read_file_with_detected_encoding_csv_tab(file_path):
   
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    # 감지된 인코딩으로 파일 읽기
    return pd.read_csv(file_path, sep='\t',header=0, index_col=None, dtype=str, encoding=encoding)


df_balance=pd.DataFrame() #빈 df_balance 생성
# a=[]
#'''.xls .csv 가 혼재해 있지만, .xls도 실제로 .csv이고 확장자만 .xls이기에 read_csv로 불러오자'''
# if sys.argv[2] in path_list_balance:
# for i,name in enumerate(path_list_balance):
# 	if '송모세' in name:
for i,name in enumerate(path_list_balance):
	if sys.argv[1] in name:
		try:
			if name.endswith('.xls'): #or name.endswith('.csv'): csv파일은 이상함;;
				try:
					balance=pd.read_csv(name, header=0, index_col=None, dtype=str, encoding='euc-kr')#, sep='\t')
				except UnicodeDecodeError:
					try:
						balance = read_file_with_detected_encoding_csv_comma(name)
					except UnicodeDecodeError as e:
						print(f"warning; 인코딩이 적절치 않습니다(xls 파일이 아님): {name}, Error: {e}")
			balance['성함'] = balance.apply(lambda x: extract_name(x.iat[0]), axis=1)[0] 
			balance['검사일시'] = balance.apply(lambda x: extract_date(x.iat[0]), axis=1)[3]
			balance2 = balance.copy()
			romberg_index_loc = find_index(balance2, 'Eyes Open').tolist()[0] #romberg값은 eyesopen이라 적힌 행에 존재함
			romberg_columns_loc = find_columns(balance2, 'Velocity')+3 #한칸 오른쪽에 Romberg quotient 적혀있음
			balance2['romberg']=balance2.apply(lambda x: extract_romberg_from_tab(x.iat[0]), axis=1)[9]
			balance2['성함']=balance2['성함'].apply(lambda x: str(x))
			balance2['검사일시']=balance2['검사일시'].apply(format_date)
			balance2['검사일시']=balance2['검사일시'].apply(lambda x: str(x))
			a = balance2[['성함', 'romberg', '검사일시']].iloc[[0]]
			df_balance=pd.concat([df_balance, a], ignore_index=True, join='outer', axis=0)
			# balance2['성함']=balance2['성함'].apply(lambda x: str(x))
			# balance2['검사일시']=balance2['검사일시'].apply(format_date)
			# balance2['검사일시']=balance2['검사일시'].apply(lambda x: str(x))
		# except UnicodeDecodeError as e:
		# 	print(f"warning; 인코딩이 적절치 않습니다(.xls 파일이 아님): {i}, Error: {e}")
		except pd.errors.EmptyDataError as e:
			print(f"warning; 빈 파일입니다: {name}, Error: {e}")
		except FileNotFoundError as e:
			print(f"warning; 파일을 찾을 수 없습니다: {name}, Error: {e}")
		except ImportError as e:
			print(f"warning; 모듈을 불러올 수 없습니다: {name}, Error: {e}")
		except Exception as e:
			print(f"warning; 파일열기 실패 등의 기타오류 발생: {name}, Error: {e}")
# df_balance
# # 변수 값이 NA이면 대체값 적용
# for i, name in enumerate(df_balance.columns):
# 	if pd.isna(df_balance[name]).any(): # A열에 NA가 하나라도 있으면 True
# 		df_balance[name] = 0 #연령별 성별 평균 대체값


df_balance.dropna(subset=['성함','검사일시'],how='any',inplace=True)
db2_balance=df_balance.drop_duplicates(['성함','검사일시'],keep='last',ignore_index=True)
db2_balance=db2_balance[['성함','검사일시','romberg']]
db2_balance['성함'] = db2_balance['성함'].apply(process_name)
db2_balance['검사일시'] = db2_balance['검사일시'].apply(lambda x: str(x).strip())
db2_balance.sort_values(['성함','검사일시'], na_position='first', ignore_index=True, inplace=True)
# db2_balance.to_csv('db2_balance.csv',na_rep=np.nan, encoding='euc-kr',index=False, chunksize=100)#,errors='ignore') #아래와 같은 에러때문에 errors='ignore'

# if format_date(sys.argv[3]) == db2_balance['생년월일']:
#     pass
# else:
#     raise ValueError(f"warning; db2_balance 생년월일이 다릅니다")


'''db_1248 merge'''
db_1248=db_148.merge(db2_balance, on=['성함','검사일시'], how='left')
db_1248.drop_duplicates(['성함','생년월일','검사일시'],ignore_index=True,inplace=True)
db_1248.dropna(subset=['성함','생년월일','검사일시'],ignore_index=True,inplace=True)
# db_1248.to_csv('db_1248.csv',na_rep=np.nan, encoding='euc-kr',index=False, chunksize=100) 

'''★★★★db3_muscle'''
df_muscle=[] #빈 df_muscle 생성

def extract_from_kg(value):
    # NaN 이거나 빈 문자열인 경우 np.nan 반환
    if pd.isna(value) or value == '':
        return np.nan
    # 문자열이 아닌 경우 문자열로 변환
    if not isinstance(value, str):
        value = str(value)
    # 정규식을 사용하여 숫자와 소수점을 추출
    number = re.findall(r'\d+\.?\d*', value)
    if number:
        return number[0]  # 숫자만 반환
    return np.nan

#위에도 쓰던 자동 인코딩감지 라이브러리
def read_file_with_detected_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    # 감지된 인코딩으로 파일 읽기
    return pd.read_csv(file_path, header=0, index_col=None, dtype=str, encoding=encoding)


# for i,name in enumerate(path_list_muscle):
# 	if '송모세' in name:
for i,name in enumerate(path_list_muscle):
	if sys.argv[1] in name:
		try:
			if name.endswith('.csv'):
				try:
					muscle = pd.read_csv(name, header=0, index_col=None, dtype=str, encoding='euc-kr')
				except UnicodeDecodeError as e:
					print(f'warning; 인코딩이 적절치 않습니다(.csv)인코딩문제 {i}')
					try:
						muscle = read_file_with_detected_encoding(i)
					except UnicodeDecodeError as e:
						print(f'warning; 인코딩이 적절치 않습니다(detect.csv)인코딩문제 {i}')

			elif name.endswith('.xlsx'):
				try:
					muscle = pd.read_excel(name, header=0, index_col=None, engine='openpyxl', dtype=str)
				except UnicodeDecodeError as e:
					print(f'warning; 인코딩이 적절치 않습니다(excel)인코딩문제 {i}')
				except zipfile.BadZipFile: 
					print(f"warning; (openpyxl)압축파일이 아닙니다: {i} ")
			muscle.rename(columns={'이름':'성함','날짜':'검사일시','Left Avg.1':'왼손악력','Right Avg.1':'오른손악력'}, inplace=True)
			muscle['왼손악력']=muscle['왼손악력'].apply(extract_from_kg) #lambda x: extract_from_kg(x.iloc[0],axis=1)
			muscle['오른손악력']=muscle['오른손악력'].apply(extract_from_kg)
			muscle['검사일시']=muscle['검사일시'].apply(format_date)
			muscle['성별']=muscle['성별'].map({'Male':'남', 'Female':'여'})
			muscle['성함']=muscle['성함'].apply(extract_korean)
			muscle['성함']=muscle['성함'].apply(process_name)
			a=muscle[['성함','검사일시','성별','왼손악력','오른손악력']].iloc[[0]]
			df_muscle.append(a) 
		# except UnicodeDecodeError as e:
		# 	print(f"warning; 인코딩이 적절치 않습니다(.xls 파일이 아님): {i}, Error: {e}")
		except pd.errors.EmptyDataError as e:
			print(f"warning; 빈 파일입니다: {name}, Error: {e}")
		except FileNotFoundError as e:
			print(f"warning; 파일을 찾을 수 없습니다: {name}, Error: {e}")
		except ImportError as e:
			print(f"warning; 모듈을 불러올 수 없습니다: {name}, Error: {e}")
		except Exception as e:
			print(f"warning; 파일열기 실패 등의 기타오류 발생: {name}, Error: {e}")

 
# # 변수 값이 NA이면 대체값 적용
# for i, name in enumerate(df_muscle.columns):
# 	if pd.isna(df_muscle[name]).any(): # A열에 NA가 하나라도 있으면 True
# 		df_muscle[name] = 0 #연령별 성별 평균 대체값
	
db3_muscle=pd.concat(df_muscle, sort=False, ignore_index=True, join='outer').drop_duplicates(subset=['성함','검사일시'],keep='last',ignore_index=True)
db3_muscle=db3_muscle.drop_duplicates(subset=['성함','검사일시'])
db3_muscle.dropna(subset=['성함','검사일시'],how='any',axis=0,inplace=True)
db3_muscle1=db3_muscle.sort_values(['성함','검사일시'],na_position='first',  ignore_index=True)
# db3_muscle1.to_csv('db3_muscle.csv', index=False, na_rep=np.nan, encoding='euc-kr', chunksize=100) 



'''db_12348 merge'''
db_12348=db_1248.merge(db3_muscle1, on=['성함','검사일시'], how='left').drop_duplicates().sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True)
# #db3에 있는 성별을 db1248의 na에 덮어씌우는 작업
def cover_na_sex(row):
    if pd.isna(row['성별_x']):
        return row['성별_y']
    else:
        return row['성별_x']
    
# db_12348.rename(columns={'성별_x':'성별'}, inplace=True)
db_12348['성별']=db_12348.apply(cover_na_sex, axis=1)
db_12348.drop(['성별_y','성별_x'],axis=1, inplace=True)
db_12348=db_12348.sort_values(['성함','검사일시'],na_position='first',  ignore_index=True)
# db_12348.to_csv('db_12348.csv', index=False, na_rep=np.nan, encoding='euc-kr', chunksize=100) 

'''★★★★db5_autonomic'''
df_autonomic=pd.DataFrame() #빈 df_autonomic 생성
# if sys.argv[2] in path_list_autonomic:
# for i,name in enumerate(path_list_autonomic):
# 	if '송모세' in name:
for i,name in enumerate(path_list_autonomic):
	if sys.argv[1] in name:
	# for i in path_list_autonomic:
		try:

			autonomic = read_file_with_detected_encoding_csv_comma(name)
			if 'Date' not in autonomic.columns:
				# print(f"(sep=',' )Date column missing in {name}, skipping this file.")
				try:
					autonomic = read_file_with_detected_encoding_csv_tab(name)
					if 'Date' not in autonomic.columns:		
						autonomic = pd.read_csv(name, header=0, index_col=None, dtype=str, encoding='euc-kr', sep=',')
						if 'Date' not in autonomic.columns:		
							print(f"(sep='\\t' )Date column missing in {name}, skipping this file.")
				except UnicodeDecodeError as e:
					print(f"warning; 인코딩이 적절치 않습니다(csv가 아님): {name}, Error: {e}")
			autonomic2= autonomic.rename(columns={'Name':'성함','Gender':'성별','Date':'검사일시','ECG1_SDNN':'SDNN'})
			autonomic3=autonomic2.copy()
			autonomic3['검사일시']=autonomic2['검사일시'].apply(format_date)
			autonomic3['성별'] = autonomic3['성별'].map({'Male': '남', 'Female': '여'})
			autonomic3['성함']=autonomic3['성함'].apply(extract_korean)
			autonomic3['성함']=autonomic3['성함'].apply(process_name)
			# print(df5_auto3.head())
			autonomic4=autonomic3[['성함','성별','검사일시','SDNN']]
			df_autonomic=pd.concat([df_autonomic,autonomic4], ignore_index=True, join='outer')
		# except UnicodeDecodeError as e:
		# 	print(f"warning; 인코딩이 적절치 않습니다(.xls 파일이 아님): {i}, Error: {e}")
		except pd.errors.EmptyDataError as e:
			print(f"warning; 빈 파일입니다: {name}, Error: {e}")
		except FileNotFoundError as e:
			print(f"warning; 파일을 찾을 수 없습니다: {name}, Error: {e}")
		except ImportError as e:
			print(f"warning; 모듈을 불러올 수 없습니다: {name}, Error: {e}")
		except Exception as e:
			print(f"warning; 파일열기 실패 등의 기타오류 발생: {name}, Error: {e}")
 
# 변수 값이 NA이면 대체값 적용
# for i, name in enumerate(df_autonomic.columns):
# 	if pd.isna(df_autonomic[name]).any(): # A열에 NA가 하나라도 있으면 True
# 		df_autonomic[name] = 0 #연령별 성별 평균 대체값
  

df_autonomic = df_autonomic.drop_duplicates(subset=['성함','검사일시'],keep='last')
df_autonomic.dropna(subset=['성함','검사일시'], how='any',inplace=True)
db5_autonomic=df_autonomic.sort_values(['성함','검사일시'],na_position='first',ignore_index=True)   
# try:
# 	db5_autonomic.to_csv('db5_autonomic.csv', na_rep=np.nan, encoding='euc-kr',index=False, chunksize=100 )
# except UnicodeEncodeError as e:
# 	print(f'아마 한자때문이었던듯? {e}')    

# if format_date(sys.argv[3]) == db5_autonomic['생년월일']:
#     pass
# else:
#     raise ValueError(f"warning; db5_autonomic 생년월일이 다릅니다")


'''db123458 merge'''
# db_12348=pd.read_csv("db_12348.csv",header=0, index_col=None, dtype=str, encoding='euc-kr' )
db_123458=db_12348.merge(db5_autonomic, on=['성함','검사일시'], how='left')
db_123458.drop_duplicates(subset=['성함','생년월일','검사일시'],keep='last',ignore_index=True,inplace=True)
db_123458.dropna(subset=['성함','생년월일','검사일시'],inplace=True)
db_123458['성별']=db_123458.apply(cover_na_sex, axis=1)
db_123458.drop(['성별_y','성별_x'],axis=1, inplace=True)
db_123458=db_123458.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True)      
# db_123458.to_csv('db_123458.csv', na_rep=np.nan, chunksize=100, index=False,encoding='euc-kr')

'''★★★★db6_gait'''
    
##########시간이 왜 오류 뜨는지 모르겠음; AM PM
def format_date_specific2(date): #생년월일, 검사일시를 db1에 맞게 바꾸기 위한 함수 정의
    if pd.isna(str(date)) or str(date).strip() == '' or len(str(date))<10 or str(date)=='nan': 
        return np.nan
    date=str(date).strip() #문자열 앞 뒤 공백 제거
    date = date.replace('-','')
    return str(date[:9])

df_gait=pd.DataFrame() #빈 df_gait 생성
# if sys.argv[2] in path_list_gait:
# for i,name in enumerate(path_list_gait):
# 	if '송모세' in name:
for i,name in enumerate(path_list_gait):
	if sys.argv[1] in name:
	# for i in path_list_gait:
		try:
			try:
				if name.endswith('.csv'):
					gait=pd.read_csv(name,header=0, index_col=None, dtype=str, encoding='euc-kr')
				elif name.endswith('.txt'):
					gait=read_file_with_detected_encoding_csv_tab(name) #errors='ignore'
				# gait.columns=gait.columns.str.replace(' ', '', regex=False)
				# gait.rename(columns={'Date/TimeofTest':'검사일시','Birthdate':'생년월일'}, inplace=True)
				# print(gait['검사일시'])
			except UnicodeDecodeError as e:
				print(f"warning; 인코딩이 적절치 않습니다(csv나 txt가 아님): 파일경로:{name}, {e}")
			gait.columns=gait.columns.str.replace(' ', '', regex=False)
			if 'FirstName' in gait.columns:
				gait['성함']=gait['FirstName']+gait['LastName']
				gait['성함']=gait['성함'].apply(process_name)
				gait['성함']=gait['성함'].apply(lambda x: str(x).strip())
				gait.rename(columns={'Date/TimeofTest':'검사일시','Birthdate':'생년월일'}, inplace=True)
				if '검사일시' in gait.columns:
					gait['검사일시'] = gait['검사일시'].apply(format_date_specific2)
					gait['검사일시'] = gait['검사일시'].apply(lambda x: str(x).strip()) #nan 때문에 한번더 str해주자 혹시 오류때문에.
					gait['생년월일'] = gait['생년월일'].apply(format_date)
					gait['생년월일'] = gait['생년월일'].apply(lambda x: str(x).strip())
					# gait['연락처']=gait['연락처'].apply(format_phone_number)
					a=gait[['성함','생년월일','검사일시','Velocity']].iloc[[0]] #연락처
					# print(a.head())
					# df_gait.append(a)
					df_gait = pd.concat([df_gait, a], ignore_index=True, join='outer').drop_duplicates()
		# except UnicodeDecodeError as e:
		# 	print(f"warning; 인코딩이 적절치 않습니다(.xls 파일이 아님): {i}, Error: {e}")
		except pd.errors.EmptyDataError as e:
			print(f"warning; 빈 파일입니다: {name}, Error: {e}")
		except FileNotFoundError as e:
			print(f"warning; 파일을 찾을 수 없습니다: {name}, Error: {e}")
		except ImportError as e:
			print(f"warning; 모듈을 불러올 수 없습니다: {name}, Error: {e}")
		except Exception as e:
			print(f"warning; 파일열기 실패 등의 기타오류 발생: {name}, Error: {e}")
 
# 변수 값이 NA이면 대체값 적용
# for i, name in enumerate(df_gait.columns):
# 	if pd.isna(df_gait[name]).any(): # A열에 NA가 하나라도 있으면 True
# 		df_gait[name] = 0 #연령별 성별 평균 대체값
  
  
df_gait['성함']=df_gait['성함'].apply(lambda x: str(x).strip())
df_gait['검사일시']=df_gait['검사일시'].apply(lambda x: str(x).strip())
df_gait['생년월일']=df_gait['생년월일'].apply(lambda x: str(x).strip())
df_gait=df_gait.dropna(subset=['성함','생년월일','검사일시'], how='any')
df_gait=df_gait.sort_values(['성함','생년월일','검사일시'], na_position='first',ignore_index=True)
df_gait2=df_gait.copy()
df_gait2=df_gait2.dropna(subset=['성함','생년월일','검사일시'], how='any')
df_gait2.drop_duplicates(subset=['성함','생년월일','검사일시'], inplace=True, keep='last', ignore_index=True)
db6_gait=df_gait2.sort_values(['성함','생년월일','검사일시'], na_position='first',ignore_index=True)
# db6_gait.to_csv('db6_gait.csv', na_rep=np.nan, chunksize=100, index=False, encoding='euc-kr')#, errors='ignore')

# if format_date(sys.argv[3]) == db6_gait['생년월일']:
#     pass
# else:
#     raise ValueError(f"warning; db6_gait 생년월일이 다릅니다")

'''db_1234568 merge'''
db_1234568=db_123458.merge(db6_gait, on=['성함','생년월일','검사일시'], how='left')  
db_1234568.drop_duplicates(subset=['성함','생년월일','검사일시'], keep='last', ignore_index=True, inplace=True)
db_1234568.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True)
# db_1234568.to_csv('db_1234568.csv', na_rep=np.nan, chunksize=100, index=False, encoding='euc-kr') #

'''★★★★ db9_scoliosis'''
def read_file_with_detected_encoding_excel(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    # 감지된 인코딩으로 파일 읽기
    return pd.read_excel(file_path, header=0, index_col=None, engine='openpyxl', dtype=str)

df_scoliosis=pd.DataFrame() #빈 DF 생성
# if sys.argv[2] in path_list_scoliosis:
# 	for i in path_list_scoliosis:
# for i,name in enumerate(path_list_scoliosis):
# 	if '송모세' in name:
for i,name in enumerate(path_list_scoliosis):
	if sys.argv[1] in name:
		try:
			try:
				if name.endswith('.csv'):
					scoliosis=read_file_with_detected_encoding_csv_comma(name)
				elif name.endswith('.txt'):
					scoliosis=read_file_with_detected_encoding_csv_tab(name)
				elif name.endswith('.xlsx'):
					scoliosis=read_file_with_detected_encoding_excel(name)   
			except UnicodeDecodeError as e:
				print(f'warning; 인코딩이 적절치 않습니다(csv가 아님) {name}, {e}')
			scoliosis.columns=scoliosis.columns.str.replace(' ', '', regex=False)
			scoliosis['성함']=scoliosis['성'].apply(lambda x: str(x).strip()) + scoliosis['이름'].apply(lambda x: str(x).strip())
			scoliosis['성함']=scoliosis['성함'].apply(process_name)
			scoliosis['성함']=scoliosis['성함'].str.strip()
			scoliosis.rename(columns={'Examination_date':'검사일시','생일':'생년월일','kyphotic_angle_ICT-ITL_(max)_[°]':'kyphotic_angle',\
						'lordotic_angle_ITL-ILS_(max)_[°]':'lordotic_angle', 'lateral_deviation_(surface)_VPDM_(rms)_[mm]':'lateral_deviation'}, inplace=True)
			scoliosis['검사일시'] = scoliosis['검사일시'].apply(format_date_specific2)
			scoliosis['검사일시'] = scoliosis['검사일시'].str.strip()
			scoliosis['생년월일'] = scoliosis['생년월일'].apply(format_date)
			scoliosis['생년월일'] = scoliosis['생년월일'].str.strip()
			a=scoliosis[['성함','생년월일','검사일시','kyphotic_angle','lordotic_angle','lateral_deviation']].iloc[[0]]
			df_scoliosis = pd.concat([df_scoliosis, a], ignore_index=True, join='outer')
		# except UnicodeDecodeError as e:
		# 	print(f"warning; 인코딩이 적절치 않습니다(.xls 파일이 아님): {i}, Error: {e}")
		except pd.errors.EmptyDataError as e:
			print(f"warning; 빈 파일입니다: {name}, Error: {e}")
		except FileNotFoundError as e:
			print(f"warning; 파일을 찾을 수 없습니다: {name}, Error: {e}")
		except ImportError as e:
			print(f"warning; 모듈을 불러올 수 없습니다: {name}, Error: {e}")
		except Exception as e:
			print(f"warning; 파일열기 실패 등의 기타오류 발생: {name}, Error: {e}")
 
# 변수 값이 NA이면 대체값 적용
# for i, name in enumerate(df_scoliosis.columns):
# 	if pd.isna(df_scoliosis[name]).any(): # A열에 NA가 하나라도 있으면 True
# 		df_scoliosis[name] = 0 #연령별 성별 평균 대체값

df_scoliosis.drop_duplicates(['성함','생년월일','검사일시'], keep='last',ignore_index=True, inplace=True)
db9_scoliosis=df_scoliosis.dropna(subset=['성함','생년월일','검사일시'], how='any', ignore_index=True)
db9_scoliosis=db9_scoliosis.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True)
# db9_scoliosis.to_csv('db9_scoliosis.csv',na_rep=np.nan, chunksize=100,  index=False, encoding='euc-kr') #

# if format_date(sys.argv[3]) == db9_scoliosis['생년월일']:
#     pass
# else:
#     raise ValueError(f"warning; db9_scoliosis 생년월일이 다릅니다")



'''db_12345689 merge'''
db_12345689=db_1234568.merge(db9_scoliosis, on=['성함','생년월일','검사일시'], how='left')
db_12345689.drop_duplicates(['성함','생년월일','검사일시'], keep='last',ignore_index=True, inplace=True)
db_12345689.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True)
# db_12345689.to_csv('db_12345689.csv',na_rep=np.nan, chunksize=100,  index=False, encoding='euc-kr') #
# db_12345689
'''db_model1'''
db_model1=db_148.copy()  # (성별도 8개 빼고 채워 넣은 데이터)
db_model1.drop_duplicates(['성함','생년월일','검사일시'], inplace=True)
db_model1.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True) 


'''db_model2'''
db_92=db9_scoliosis.merge(db2_balance, on=['성함','검사일시'], how='left')
db_92.drop_duplicates(['성함','생년월일','검사일시'], keep='last',ignore_index=True, inplace=True)
db_92.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True)
db_923=db_92.merge(db3_muscle, on=['성함','검사일시'], how='left')
db_923.drop_duplicates(['성함','생년월일','검사일시'], keep='last',ignore_index=True, inplace=True)
db_923.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True)
db_9235=db_923.merge(db5_autonomic.drop(columns=['성별'],axis=1), on=['성함','검사일시'], how='left')
db_9235.drop_duplicates(['성함','생년월일','검사일시'], keep='last',ignore_index=True, inplace=True)
db_9235.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True)
db_92356=db_9235.merge(db6_gait, on=['성함','생년월일','검사일시'], how='left')
db_92356.drop_duplicates(['성함','생년월일','검사일시'], keep='last',ignore_index=True, inplace=True)
db_92356.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True)
db_923568=db_92356.merge(db8_inbody[['성함','생년월일','검사일시','SMI','SBP','DBP','wholeBPA','BMI']], on=['성함','생년월일','검사일시'], how='left')
db_923568.drop_duplicates(['성함','생년월일','검사일시'], keep='last',ignore_index=True, inplace=True)
db_923568.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True,inplace=True)
#연령생성
db_923568['연령']=db_923568.apply(lambda row: str(int(row['검사일시'][0:4]) - int(row['생년월일'][0:4]) + 1) \
    if pd.notna(row['생년월일']) and pd.notna(row['검사일시']) and len(row['검사일시']) == 8 and len(row['생년월일']) == 8\
        else np.nan , axis=1)  #'연령' 생성 #axis=1이 있어야 각 행단위로 수행됨 (axis=0이 default임)

# 성별 na인 사람 최대한 채우기
db_model2_before=db_923568.merge(db_1234568[['성함','성별','생년월일','검사일시']], on=['성함','생년월일','검사일시'], how='left')
# print(db_model2_before)
# db_model2_before.rename(columns={'성별_x':'성별'}, inplace=True)
db_model2_before['성별']=db_model2_before.apply(cover_na_sex, axis=1)
db_model2_before.drop(['성별_y','성별_x'],axis=1, inplace=True)
db_model2_before=db_model2_before.drop_duplicates(['성함','생년월일','검사일시'],ignore_index=True,keep='last')   
db_model2_before=db_model2_before.sort_values(['성함','생년월일','검사일시'],na_position='first',ignore_index=True)
db_model2=db_model2_before[['성함','연령','성별','생년월일','검사일시','kyphotic_angle','lordotic_angle','lateral_deviation','romberg','왼손악력','오른손악력','SDNN','Velocity',\
    'SMI','SBP','DBP','wholeBPA','BMI']]
# db_model1
# db_model2
def categorize_age(age):
    if age < 40:
        return '10~30대'
    elif age < 60:
        return '40~50대'
    elif age < 65:
        return '60~65'
    elif age < 70:
        return '65~70'
    elif age < 75:
        return '70~75'
    elif age < 80:
        return '75~80'
    elif age < 85:
        return '80~85'
    else:
        return '85+'

def format_date2(date): #생년월일, 검사일시를 db1에 맞게 바꾸기 위한 함수 정의
    if pd.isna(str(date)) or str(date).strip() == '': 
        return np.nan
    date=str(date).strip() #문자열 앞 뒤 공백 제거
    for fmt in ('%Y-%m-%d %H:%M:%S','%Y.%m.%d. %H:%M:%S', '%Y-%m-%d', '%Y%m%d', '%Y-%m-%d %I:%M:%S %p','%Y.%m.%d.','%Y.%m.%d', '%Y-%m-%d %p %I:%M:%S','%Y-%m-%d %p %i:%M:%S'\
        , '%Y.%m.%d.', '%Y.%m.%d', '%Y-%m-%d %H.%M.%S'): #%p는 AM, PM이런거임
        try:
            return datetime.datetime.strptime(date, fmt).strftime('%Y-%m-%d') #문자열로 반환
        except ValueError as e:
            continue
    return np.nan

#db_model11
db_model11=db_model1.apply(lambda x: x.astype(float, errors='ignore'))
db_model11.loc[db_model11['KET'] == 'neg', 'KET'] = 0
db_model11.loc[db_model11['GLU'] == 'neg', 'GLU'] = 0
db_model11.loc[db_model11['PRO'] == 'neg', 'PRO'] = 0
db_model11.loc[db_model11['SG'] == 'neg', 'SG'] = 0
db_model11[['생년월일','검사일시']] = db_model11[['생년월일','검사일시']].astype(int).astype(str, errors='ignore')
db_model11['검사일시'] = db_model11['검사일시'].apply(format_date2)  #'%Y-%m-%d'매핑대로 이렇게 바꿈
db_model11['생년월일'] = db_model11['생년월일'].apply(format_date2)  #'%Y-%m-%d'매핑대로 이렇게 바꿈
db_model11['연령그룹'] = db_model11['연령'].apply(categorize_age) 
db_model11.loc[:,'성별'] = db_model11['성별'].map({'남': '남성', '여': '여성'})
db_model11.sort_values(['성별','생년월일','검사일시'],na_position='first',inplace=True)
db_model11.drop_duplicates(subset=['성함','생년월일'], keep='last',inplace=True) ##########가장 최근일자만 남기기


################model1 score
inpath = os.path.join(root_dir, 'inpath')

bp_percent_path = os.path.join(inpath, 'bp_percent.pickle')

# pickle 파일에서 객체 불러오기
with open(bp_percent_path, 'rb') as f:
    loaded_objects = pickle.load(f)

# 불러온 객체를 개별 변수로 할당
for object_name, obj in loaded_objects.items():
    globals()[object_name] = obj
    # print(f"{object_name} 객체가 복원되었습니다.")
    
# pickle_file_path = os.path.join(inpath, 'tref_objects.pkl')

# # pickle 파일에서 객체 불러오기
# with open(pickle_file_path, 'rb') as f:
#     loaded_objects = pickle.load(f)

# # 불러온 객체를 개별 변수로 할당
# for object_name, obj in loaded_objects.items():
#     globals()[object_name] = obj
#     print(f"{object_name} 객체가 복원되었습니다.")


# 1) BP score 계산 ----
def calculate_bp_score(row, bp_percent):
    if row['SBP'] < 100 and row['DBP'] < 80:
        return 0
    elif row['SBP'] >= 160 or row['DBP'] >= 100:
        return 2
    else:
        bp_pc = -( (-0.7071068) * ((row['SBP'] - 132.1) / 18.23501) + (-0.7071068) * ((row['DBP'] - 71.91) / 11.92403) )
        if bp_pc < min(bp_percent['bp_pc']):
            bin_index = 1
        elif bp_pc >= max(bp_percent['bp_pc']):
            bin_index = len(bp_percent['bp_pc'])
        else:
            bin_index = np.digitize(bp_pc, bp_percent['bp_pc'], right=True)
        return bp_percent['bp_score'][bin_index - 1]

db_model11['bp_score'] = db_model11.apply(lambda row: calculate_bp_score(row, loaded_objects), axis=1) #bp_percent
db_model11['bp_score'] = db_model11['bp_score'].clip(lower=0, upper=2)


# 2) UA score : 요검사 4종 ----
db_model11['ket2'] = np.select([db_model11['KET'].values == 0, db_model11['KET'].values <= 30, db_model11['KET'].values <= 100, db_model11['KET'].values <= 300, db_model11['KET'].values <= 1000],  [0, 1, 2, 3, 4], default=5)
db_model11['pro2'] = np.select([db_model11['PRO'].values == 0, db_model11['PRO'].values <= 30, db_model11['PRO'].values <= 100, db_model11['PRO'].values <= 300, db_model11['PRO'].values <= 1000],  [0, 1, 2, 3, 4], default=5)
db_model11['glu2'] = np.select([db_model11['GLU'].values == 0, db_model11['GLU'].values <= 30, db_model11['GLU'].values <= 100, db_model11['GLU'].values <= 300, db_model11['GLU'].values <= 1000],  [0, 1, 2, 3, 4], default=5)
db_model11['sg2'] = np.where((db_model11['SG'].values < 1.003) | (db_model11['SG'].values > 1.030), 1, 0)

db_model11['ua_sum'] = db_model11['ket2'] + db_model11['pro2'] + db_model11['glu2'] + db_model11['sg2']
db_model11['ua_sumsc'] = (db_model11['ua_sum'] - 0.8682) / 1.588944

db_model11['ua_score'] = np.where((db_model11['SG'].values > 1.025) | (db_model11['ket2'].values == 5) | (db_model11['pro2'].values == 5) | (db_model11['glu2'].values == 5), 2, db_model11['ua_sumsc'].values * 0.45 + 0.246)
db_model11['ua_score'] = db_model11['ua_score'].clip(lower=0, upper=2)

# 3) WC score : 허리둘레 ----
def calculate_wc_score(row):
    if row['성별'] == "남성":
        if row['WC'] > 103.9:
            return 2
        sc_wc = (row['WC'] - 84.85) / 9.248993
        if row['WC'] < 90:
            return (sc_wc * 0.45 + 1.05) * 0.77
        elif 90 <= row['WC'] < 102:
            return (sc_wc * 0.45 + 1.05) * 0.89 - 0.16
        elif row['WC'] >= 102:
            return (sc_wc * 0.45 + 1.05) * 5.5 - 8.89
    elif row['성별'] == "여성":
        if row['WC'] > 98.7:
            return 2
        sc_wc = (row['WC'] - 81.66) / 7.669229
        if row['WC'] < 80:
            return (sc_wc * 0.41 + 1.08) * 1.02 - 0.007
        elif 80 <= row['WC'] < 88:
            return (sc_wc * 0.41 + 1.08) * 1.15 - 0.14
        elif row['WC'] >= 88:
            return (sc_wc * 0.41 + 1.08) * 0.87 + 0.265
    return np.nan

db_model11['wc_score'] = db_model11.apply(calculate_wc_score, axis=1)
db_model11['wc_score'] = db_model11['wc_score'].clip(lower=0, upper=2)

# 4) BFMI score ----
def calculate_bfmi_score(row):
    if row['성별'] == "남성":
        if row['BFMI'] > 10.8:
            return 2
        sc_bfmi = (row['BFMI'] - 6.097) / 2.055156
        if row['BFMI'] <= 5.2:
            return (sc_bfmi * 0.432 + 1.009) * 1.2
        elif 5.2 < row['BFMI'] < 7:
            return (sc_bfmi * 0.432 + 1.009) * 1.47 - 0.237
        elif row['BFMI'] >= 7:
            return (sc_bfmi * 0.432 + 1.009) * 0.62 + 0.76
    elif row['성별'] == "여성":
        if row['BFMI'] > 13.2:
            return 2
        sc_bfmi = (row['BFMI'] - 7.952) / 2.2838
        if row['BFMI'] <= 8.2:
            return (sc_bfmi * 0.415 + 1.04523) * 0.91
        elif 8.2 < row['BFMI'] < 10.8:
            return (sc_bfmi * 0.415 + 1.04523) * 1.14 - 0.264
        elif row['BFMI'] >= 10.8:
            return (sc_bfmi * 0.415 + 1.04523) * 1.13 - 0.267
    return np.nan

db_model11['bfmi_score'] = db_model11.apply(calculate_bfmi_score, axis=1)
db_model11['bfmi_score'] = db_model11['bfmi_score'].clip(lower=0, upper=2)

# 5) FFMI score ----
def calculate_ffmi_score(row):
    if row['성별'] == "남성":
        if row['FFMI'] < 14.0:
            return 2
        sc_ffmi = (row['FFMI'] - 18.16) / 1.608909
        if row['FFMI'] >= 19.8:
            return (-sc_ffmi * 0.3 + 1.22) * 1.09
        elif 18.1 < row['FFMI'] < 19.8:
            return (-sc_ffmi * 0.3 + 1.22) * 1.7 - 0.58
        elif row['FFMI'] <= 18.1:
            return (-sc_ffmi * 0.3 + 1.22) * 0.65 + 0.7
    elif row['성별'] == "여성":
        if row['FFMI'] < 13.0:
            return 2
        sc_ffmi = (row['FFMI'] - 15.91) / 1.200285
        if row['FFMI'] >= 16.8:
            return (-sc_ffmi * 0.299 + 1.273) * 0.95
        elif 15.2 < row['FFMI'] < 16.8:
            return (-sc_ffmi * 0.299 + 1.273) * 1.4 - 0.5
        elif row['FFMI'] <= 15.2:
            return (-sc_ffmi * 0.299 + 1.273) * 0.9 + 0.195
    return np.nan

db_model11['ffmi_score'] = db_model11.apply(calculate_ffmi_score, axis=1)
db_model11['ffmi_score'] = db_model11['ffmi_score'].clip(lower=0, upper=2)

# 종합 위험도 및 등급 계산 ----
db_model11['total_score'] = db_model11['bp_score'] + db_model11['ua_score'] + db_model11['wc_score'] + db_model11['bfmi_score'] + db_model11['ffmi_score']

# Calculate ttsc (100-point score, higher is better)
db_model11['ttsc'] = 100 - np.ceil(db_model11['total_score'] * 10)

# Assign risk grade
db_model11['ttgrade'] = np.where(db_model11['ttsc'] > 75, "낮음", np.where(db_model11['ttsc'] > 44, "보통", "높음"))



#db_model22
db_model22=db_model2.apply(lambda x: x.astype(float, errors='ignore'))
db_model22[['생년월일','검사일시']] = db_model22[['생년월일','검사일시']].astype(int).astype(str, errors='ignore')
db_model22['검사일시'] = db_model22['검사일시'].apply(format_date2)  #'%Y-%m-%d'매핑대로 이렇게 바꿈
db_model22['생년월일'] = db_model22['생년월일'].apply(format_date2)  #'%Y-%m-%d'매핑대로 이렇게 바꿈
db_model22['연령그룹'] = db_model22['연령'].apply(categorize_age)
db_model22.loc[:,'성별'] = db_model22['성별'].map({'남': '남성', '여': '여성'})
db_model22.sort_values(['성별','생년월일','검사일시'],na_position='first',inplace=True)
db_model22.drop_duplicates(subset=['성함','생년월일'], keep='last',inplace=True) ##########가장 최근일자만 남기기

# if (format_date2(sys.argv[2]) == db_model22['생년월일'].values) or (format_date2(sys.argv[2]) == db_model22['생년월일'].values): #sys.argv[2]
#     pass
# else:
#     raise ValueError(f"warning; 생년월일이 다릅니다, 동명이인여부 or 오타를 확인하세요")

# db_model22.to_csv('db_model22.csv',encoding='euc-kr')

'''model2  변수 결측일때 성별/연령별 평균으로 대체'''
# 성별 및 연령 그룹에 따른 kyphotic_angle 값 대체
for i, name in enumerate(db_model22.columns):
	if 'kyphotic' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  # kyphotic_angle이 np.nan일 경우
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 45.00036585
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 49.4290625
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 47.934
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 50.72709677
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 51.11525
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 51.35734375
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 53.68916084
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 54.08382353
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 41.03516854
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 46.33927273
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 47.38403509
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 48.65023622
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 49.22390582
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 51.66234719
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 51.25116279
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 54.08382353

	elif 'lordotic' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 32.98134146
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 30.0846875
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 29.69181818
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 28.76832258
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 28.851375
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 27.92359375
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 27.09041958
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 27.055
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 36.7247191
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 38.10227273
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 34.40254386
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 33.58708661
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 31.0831856
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 30.30276284
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 28.45948837
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 28.36416667
      
	elif 'lateral' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 3.920731707
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 4.273125
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 6.272363636
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 5.837096774
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 6.0665
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 6.8603125
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 7.128776224
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 7.867941176
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 4.770224719
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 5.116
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 5.886578947
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 5.227637795
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 5.937645429
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 6.666136919
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 6.430744186
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 8.3275
      
	elif 'romberg' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 146.7307692
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] =128.3928571
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 125.673913
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 147.9112903
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 166.077381
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] =162.6723549
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 168.280543
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 187.0588235
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] =140.9625
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 153.5142857
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] =129.97
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] =145.5045455
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] =149.5813149
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 159.2691218
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 164.9569892
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 140.030303
      
	elif '왼손악력' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 40.20632911
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 36.31290323
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 30.39444444
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 30.00457516
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 29.79617021
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 27.9712766
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 25.92608696
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 22.82376238
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 22.04578313
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 21.60185185
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 20.0990991
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 19.45587045
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 18.92079772
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 17.98585859
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 16.67971014
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 15.74516129

	elif '오른손악력' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 41.4835443
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 37.07741935
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 33.28301887
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 32.082
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 32.38318584
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 29.72513812
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 27.6709434
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 26.16702128
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 23.50963855
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 23.15
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 21.53271028
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 20.50826446
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 20.29415205
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 18.91044386
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 17.30197044
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 16.53448276

	elif 'SDNN' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 69.82956211
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 87.82902798
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 116.5353351
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 112.5554807
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 117.1979055
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 135.5213148
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 132.5804231
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 136.18733
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 63.10516027
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 96.15869435
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 117.2615165
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 138.753362
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 123.7027748
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 128.3951003
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 134.3315611
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 152.5024959
      
	elif 'Velocity' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 107.6223684
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 105.7206897
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 106.8693878
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 100.4496503
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 98.83181818
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 95.73049853
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 87.28995984
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 79.72111111
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 108.3794872
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 109.1838095
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 103.4152381
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 98.95810811
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 97.13015873
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 90.14571429
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 82.73812155
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 71.58571429
      
	elif 'SMI' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 8.286111111
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 8.079166667
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 7.951612903
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 7.848235294
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 7.595275591
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 7.632057416
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 7.384
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 7.3125
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 6.117808219
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 6.374117647
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 6.280519481
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 6.24516129
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 6.293243243
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 6.193454545
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 6.221276596
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 6.075
      
	elif 'SBP' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 131.875
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 140.9583333
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 130.8387097
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 132.6704545
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 132.2519084
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 134.2604651
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 132.3028571
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 133.2121212
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 117.6712329
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 126.2470588
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 128.3636364
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 129.7261146
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 130.3214286
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 133.807971
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 133.6971831
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 139.8928571
      
	elif 'DBP' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 81.25
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 88.75
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 78.77419355
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 76.79545455
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 71.76335878
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 70.39069767
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 68.24571429
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 65.13636364
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 72.32876712
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 77.83529412
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 76.05194805
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 73.1910828
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 70.70089286
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 70.19202899
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 67.32394366
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 67

	elif 'wholeBPA' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 6.573611111
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 5.979166667
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 5.74516129
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 5.431764706
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 5.138582677
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 5.010526316
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 4.641714286
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 4.328125
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 5.010958904
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 5.074117647
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 4.880519481
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 4.768387097
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 4.613963964
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 4.452363636
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 4.317730496
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 4.067857143
      
	elif 'BMI' in name:
		for index, row in db_model22.iterrows():
			if pd.isna(row[name]):  
				if row['성별'] == '남성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 25.76388889
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 26.12916667
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 25.1516129
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 24.67529412
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 23.88188976
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 24.01626794
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 23.73714286
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 23.5296875
				elif row['성별'] == '여성':
					if row['연령그룹'] == '10~30대':
						db_model22.at[index, name] = 22.58082192
					elif row['연령그룹'] == '40~50대':
						db_model22.at[index, name] = 23.98588235
					elif row['연령그룹'] == '60~65':
						db_model22.at[index, name] = 23.42727273
					elif row['연령그룹'] == '65~70':
						db_model22.at[index, name] = 23.56645161
					elif row['연령그룹'] == '70~75':
						db_model22.at[index, name] = 23.81936937
					elif row['연령그룹'] == '75~80':
						db_model22.at[index, name] = 24.03636364
					elif row['연령그룹'] == '80~85':
						db_model22.at[index, name] = 24.42695035
					elif row['연령그룹'] == '85+':
						db_model22.at[index, name] = 23.9
      

####### 정상군 지정 ####### db_model23; imputation된 raw값을 갖는 데이터 #4306명 -> 나중에 한명당 한줄로 만들어도됨

# axis1 Muscle weakness
db_model22['normalgroup_axis1'] = 'abnormal'  # 기본값을 abnormal로 설정
db_model22.loc[(db_model22['성별'] == '남') & ((db_model22['왼손악력'] >= 28) | (db_model22['오른손악력'] >= 28) | (db_model22['SMI'] >= 7.0)) & db_model22['왼손악력'].notna()\
    & db_model22['오른손악력'].notna() & db_model22['SMI'].notna(), 'normalgroup_axis1'] = 'normal'
db_model22.loc[(db_model22['성별'] == '여') & ((db_model22['왼손악력'] >= 18) | (db_model22['오른손악력'] >= 18) | (db_model22['SMI'] >= 5.7)) & db_model22['왼손악력'].notna() \
    & db_model22['오른손악력'].notna() & db_model22['SMI'].notna(), 'normalgroup_axis1'] = 'normal'

# axis2 Malnutrition
db_model22['normalgroup_axis2'] = 'abnormal'
# db_model22.loc[db_model22['normalgroup_axis2'].isna()]['normalgroup_axis2'] = np.nan
db_model22.loc[(db_model22['연령'] < 70) & (db_model22['BMI'] >= 18.5) & (db_model22['wholeBPA'] >= 4.17) & db_model22['BMI'].notna() & db_model22['wholeBPA'].notna(),\
     'normalgroup_axis2'] = 'normal'
db_model22.loc[(db_model22['연령'] >= 70) & (db_model22['BMI'] >= 20) & (db_model22['wholeBPA'] >= 4.17) & db_model22['BMI'].notna() & db_model22['wholeBPA'].notna(),\
     'normalgroup_axis2'] = 'normal'
    
# axis3 Risk for falls
db_model22['normalgroup_axis3'] = 'abnormal'
db_model22.loc[(db_model22['SBP'] >= 100) & (db_model22['romberg'] <= 130) & db_model22['SBP'].notna() & db_model22['romberg'].notna(), 'normalgroup_axis3'] = 'normal'
#romberg 1.3*100

# axis4 impaired physical mobility
condition1 = db_model22['kyphotic_angle'] <= 40
condition2 = (db_model22['lordotic_angle'] >= 22) & (db_model22['lordotic_angle'] <= 50)
condition3 = db_model22['lateral_deviation'] <= 7.58
at_least_two_true = (condition1.astype(int) + condition2.astype(int) + condition3.astype(int)) >= 2
db_model22['normalgroup_axis4'] = 'abnormal'
db_model22.loc[at_least_two_true, 'normalgroup_axis4'] = 'normal'

# axis5 Risk for depressive disorder
db_model22['normalgroup_axis5'] = 'normal'
db_model22.loc[db_model22['SDNN'] < 50, 'normalgroup_axis5'] = 'abnormal'

# 각 행에 대해 모든 조건이 'normal'인지 확인 -> 모두 normal이면 전체 total점수에서 normal인 사람
db_model22['normalgroup_total'] = db_model22[['normalgroup_axis1', 'normalgroup_axis2', 'normalgroup_axis3', 'normalgroup_axis4', 'normalgroup_axis5']]\
    .apply(lambda x: 'normal' if all(x == 'normal') else 'abnormal', axis=1)
# db_model22.isnull().sum()
# db_model22

'''PCA'''
db_model22_pca=db_model22.copy()
db_model22_pca.loc[:,'성별'] = db_model22_pca['성별'].map({'남성': 0, '여성': 1}) #남성 = 0, 여성=1
# db_model22_pca2=db_model22_pca.drop(['성함', '생년월일', '검사일시','연령그룹','normalgroup_axis1','normalgroup_axis2','normalgroup_axis3','normalgroup_axis4','normalgroup_axis5',\
    # 'normalgroup_total'], axis=1) #object 열 제거

import pickle
# 경로 지정
# inpath = os.path.abspath()'C:/Users/soojin/Desktop/AICA/inpath/'
inpath = os.path.join(root_dir, 'inpath')
# print(inpath)

# "C:/Users/RexSoft/Desktop/Project/python_mosi/Gwangju/db_model2_nodup2.pickle"
# 불러올 pickle 파일의 경로
# pickle_file_path = "C:/Users/RexSoft/Desktop/Project/python_mosi/Gwangju/db_model2_nodup2.pickle"
pickle_file_path = os.path.join(inpath, "agegroup.pickle")

# pickle 파일에서 객체 불러오기
with open(pickle_file_path, 'rb') as f:
    agegroup = pickle.load(f)

# 불러온 객체를 개별 변수로 할당
for object_name, obj in agegroup.items():
    globals()[object_name] = obj
    
db_model22_pca2 = db_model22_pca.copy()

#직접 StandardScaler하기
for i, name in enumerate(db_model22_pca2.columns):
    if not name in ['성함', '성별','생년월일', '검사일시','연령그룹','normalgroup_axis1','normalgroup_axis2','normalgroup_axis3','normalgroup_axis4','normalgroup_axis5', 'normalgroup_total',\
        ]:
    	db_model22_pca2[name] = (db_model22_pca[name] - agegroup.loc[:, (name,'mean')].mean()) / agegroup.loc[:, (name,'std')].std()

# #왼손악력, 오른손악력, SMI
# pc1_men = pca_men_axis1.components_  # [[0.58285178 0.58284292 0.56619602]]
# pc1_women = pca_women_axis1.components_ #[[0.61469645 0.6097796  0.50031701]]


#pc1_axis1
if db_model22_pca2['성별'].values == 0:
	db_model22_pca2['pc1_axis1'] = (db_model22_pca2['왼손악력']*0.58285178) + (db_model22_pca2['오른손악력']*0.58284292) + (db_model22_pca2['SMI']* 0.56619602)
else:
	db_model22_pca2['pc1_axis1'] = (db_model22_pca2['왼손악력']*0.61469645) + (db_model22_pca2['오른손악력']*0.6097796) + (db_model22_pca['SMI']*0.50031701)

#SBP DBP
# pc1_axis3_men = pca_men_axis3.components_ [[0.70710678 0.70710678]]
# pc1_axis3_women = pca_women_axis3.components_ [[0.70710678 0.70710678]]
#pc1_axis3
if db_model22_pca2['성별'].values == 0:
	db_model22_pca2['pc1_axis3'] = (db_model22_pca2['SBP']*0.6112982) + (db_model22_pca2['DBP']*0.79140034)
else:
	db_model22_pca2['pc1_axis3'] = (db_model22_pca2['SBP']*0.72248172) + (db_model22_pca2['DBP']*0.69139002)

#변수위치 바꾸기 + #sbp dbp smi 왼손악력, 오른손악력 제거
db_model22_pca2 = db_model22_pca2[['성함', '연령', '성별', '생년월일', '검사일시', 'kyphotic_angle', 'lordotic_angle',\
       'lateral_deviation', 'romberg', 'SDNN', 'Velocity',\
       'wholeBPA', 'BMI','pc1_axis1', 'pc1_axis3', 'normalgroup_axis1','normalgroup_axis2','normalgroup_axis3','normalgroup_axis4','normalgroup_axis5', 'normalgroup_total','연령그룹' ]]

# #pc1 만들고 minmax한번 적용
# for i, name in enumerate(db_model22_pca2.columns):
# 	if name in ['pc1_axis1', 'pc1_axis3']:
# 		db_model22_pca2[name] = (db_model22_pca2[name] - agegroup.loc[:, (name,'mean')].mean()) / agegroup.loc[:, (name,'std')].std()
	# db_model22_pca2[name] = np.clip(db_model22_pca2[name], 0, 2)

for i, name in enumerate(db_model22_pca2):
	if not name in ['성함',  '성별','생년월일', '검사일시','연령그룹','normalgroup_axis1','normalgroup_axis2','normalgroup_axis3','normalgroup_axis4','normalgroup_axis5', 'normalgroup_total',\
        ]:
		db_model22_pca2[name] = (db_model22_pca2[name] - agegroup.loc[:, (name,'min')].min()) / ((agegroup.loc[:, (name,'max')].max())- agegroup.loc[:, (name,'min')].min())

# db_model22_pca2


  
# '''점수화'''
# ###남자 score
db_model22_pca3 = db_model22_pca2.copy()
if db_model22_pca3['성별'].values==0:
	db_model22_pca3['kyphotic_angle'] = db_model22_pca2['kyphotic_angle']*0.1045
	db_model22_pca3['lordotic_angle'] = db_model22_pca2['lordotic_angle']*0.0569
	db_model22_pca3['lateral_deviation'] =  db_model22_pca2['lateral_deviation']*0.0357
	db_model22_pca3['romberg'] = db_model22_pca2['romberg']*0.0496
	db_model22_pca3['SDNN'] = db_model22_pca2['SDNN']*-0.0475
	db_model22_pca3['Velocity'] = db_model22_pca2['Velocity']*0.0233
	db_model22_pca3['wholeBPA'] = db_model22_pca2['wholeBPA']*1.0784
	db_model22_pca3['BMI_score'] = db_model22_pca2['BMI']*0.1091
	db_model22_pca3['pc1_axis1'] = db_model22_pca2['pc1_axis1']*0.0517
	db_model22_pca3['pc1_axis3'] = db_model22_pca2['pc1_axis3']*0.0470

	#각 축끼리 묶기
	#axis1 = Muscle weakness
	db_model22_pca3['axis1_model2_score'] = db_model22_pca3['Velocity']+db_model22_pca3['pc1_axis1']
	#axis2 = Nutrition
	db_model22_pca3['axis2_model2_score'] = db_model22_pca3['wholeBPA']+db_model22_pca3['BMI_score']
	#axis3 = Risk for falls
	db_model22_pca3['axis3_model2_score'] = db_model22_pca3['pc1_axis3']+db_model22_pca3['romberg']
	#axis4 = Impaired physical mobility
	db_model22_pca3['axis4_model2_score'] = db_model22_pca3['kyphotic_angle']+db_model22_pca3['lordotic_angle']+db_model22_pca3['lateral_deviation']
	#axis5 = Depressive disorder
	db_model22_pca3['axis5_model2_score'] = db_model22_pca3['SDNN']

elif db_model22_pca3['성별'].values==1:
#####여자 score
	db_model22_pca3['kyphotic_angle'] = db_model22_pca2['kyphotic_angle']*(0.2425)
	db_model22_pca3['lordotic_angle'] = db_model22_pca2['lordotic_angle']*(0.2039)
	db_model22_pca3['lateral_deviation'] = db_model22_pca2['lateral_deviation']*(0.0030)
	db_model22_pca3['romberg'] = db_model22_pca2['romberg']*(0.0701)
	db_model22_pca3['SDNN'] = db_model22_pca2['SDNN']*(-0.1089)
	db_model22_pca3['Velocity'] = db_model22_pca2['Velocity']*(0.1942)
	db_model22_pca3['wholeBPA'] = db_model22_pca2['wholeBPA']*(0.5592)
	db_model22_pca3['BMI'] = db_model22_pca2['BMI']*(0.1761)
	db_model22_pca3['pc1_axis1'] = db_model22_pca2['pc1_axis1']*(0.0021)
	db_model22_pca3['pc1_axis3'] = db_model22_pca2['pc1_axis3']*(0.0810)


scoring_model2 = db_model22_pca3.copy()


# scoring_model22
	#각 축끼리 묶기
#axis1 = Muscle weakness
scoring_model2['axis1_model2_score'] = scoring_model2['Velocity']+scoring_model2['pc1_axis1']
#axis2 = Nutrition
scoring_model2['axis2_model2_score'] = scoring_model2['wholeBPA']+scoring_model2['BMI']
#axis3 = Risk for falls
scoring_model2['axis3_model2_score'] = scoring_model2['pc1_axis3']+scoring_model2['romberg']
#axis4 = Impaired physical mobility
scoring_model2['axis4_model2_score'] = scoring_model2['kyphotic_angle']+scoring_model2['lordotic_angle']+scoring_model2['lateral_deviation']
#axis5 = Depressive disorder
scoring_model2['axis5_model2_score'] = scoring_model2['SDNN']

#10점만점
scoring_model2['model2_total_score'] = scoring_model2['axis1_model2_score'] +scoring_model2['axis2_model2_score']+scoring_model2['axis3_model2_score']+scoring_model2['axis4_model2_score']+\
    scoring_model2['axis5_model2_score']

#100점만점환산
scoring_model2['model2_total_score'] = scoring_model2['model2_total_score']*10

if scoring_model2['model2_total_score'].values <0:
    scoring_model2['model2_total_score']= -scoring_model2['model2_total_score']
    

'''등급화'''
# def create_quantile_groups(df, column):
#     # qcut으로 4개의 그룹(사분위수) 생성
#     return pd.qcut(df[column], q=[0, 0.1, 0.9, 1], labels=['낮음','보통','높음'], duplicates='drop') #, duplicates='drop' 중복된 bin edge  허용; duplicates='drop' 옵션을 추가하면 중복된 구간이 제거됨

# def create_quantile_groups2(df, column):
#     # qcut으로 2개의 그룹 생성
#     return pd.qcut(df[column], q=[0, 0.1, 0.9, 1], labels=['낮음','높음'], duplicates='drop')

# scoring_model2['tertile'] = create_quantile_groups(scoring_model2, 'model2_total_score')
# scoring_model2['tertile'] = create_quantile_groups2(scoring_model2, 'model2_total_score')

if any(scoring_model2['model2_total_score'] >= 7):
	scoring_model2['tertile'] = '높음'
elif any(scoring_model2['model2_total_score'] >= 3):
	scoring_model2['tertile'] = '중간'    
else:
	scoring_model2['tertile'] = '낮음'
print(scoring_model2['tertile'])


'''Radar chart -model2'''
# minus를 대체할 수 있도록 설정
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family']='Malgun Gothic' #한글깨짐 방지

axis1 = agegroup.loc[scoring_model2['연령그룹'], ('axis1_model2_score','min')]
axis2 = agegroup.loc[scoring_model2['연령그룹'], ('axis2_model2_score','min')]
axis3 = agegroup.loc[scoring_model2['연령그룹'], ('axis3_model2_score','min')]
axis4 = agegroup.loc[scoring_model2['연령그룹'], ('axis4_model2_score','min')]
axis5 = agegroup.loc[scoring_model2['연령그룹'], ('axis5_model2_score','min')]
# axis_total = agegroup.loc[scoring_model2['연령그룹'].value, ('model2_score','min')]

data = pd.DataFrame({
    'Muscle weakness': [scoring_model2['axis1_model2_score'][0], axis1],
    'Malnutrition': [scoring_model2['axis2_model2_score'][0], axis2],
    'Risk for falls': [scoring_model2['axis3_model2_score'][0], axis3],
    'Impaired physical mobility': [scoring_model2['axis4_model2_score'][0], axis4],
    'Depressive disorder': [scoring_model2['axis5_model2_score'][0], axis5]
})


# 데이터 딕셔너리를 생성

#대상자
dict1 = {'근육약화': data.iloc[0, 0], '영양': data.iloc[0, 1], '낙상위험': data.iloc[0, 2], 
         '기동성장애': data.iloc[0, 3], '우울장애': data.iloc[0, 4]}
#정상군
dict2 = {'근육약화': data.iloc[1, 0], '영양': data.iloc[1, 1], '낙상위험': data.iloc[1, 2], 
         '기동성장애': data.iloc[1, 3], '우울장애': data.iloc[1, 4]}


categories1 = list(dict1.keys())
categories1 = [*categories1, categories1[0]]  # 360도로 구부리기 때문에 시작과 마지막을 동일하게(폐곡선이기에 마지막에 시작점을 다시 추가하여 끝점과 시작점을 동일하게 만듦)

numbers1 = list(dict1.values())
numbers1 = [*numbers1, numbers1[0]]  # 360도로 구부리기 때문에 시작과 마지막을 동일하게

numbers2 = list(dict2.values())
numbers2 = [*numbers2, numbers2[0]]  # 360도로 구부리기 때문에 시작과 마지막을 동일하게

# 레이더 차트를 그리기 위한 각도 설정
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(numbers1)) #np.linespace는 지정된 범위내에서 일정한 간격으로 num개의 숫자를 생성함.
#2파이(라디안)=360도


plt.figure(figsize=(480 / 25.4, 460 / 25.4))
ax = plt.subplot(polar=True)
plt.xticks(label_loc, labels=categories1, fontsize=20)

ax.plot(label_loc, numbers1, label='person A', linestyle='dashed', color='darkorange')
ax.fill(label_loc, numbers1, color='darkorange', alpha=0.2)

ax.plot(label_loc, numbers2, label='평균', linestyle='dashed', color='royalblue')
ax.fill(label_loc, numbers2, color='royalblue', alpha=0.4)

ax.legend()
# plt.show()

plt.savefig('graph_5_1.svg',format='svg', bbox_inches='tight')
plt.close()


####################################################################################################################################################################################
###################################################################################################################################################################################

tref_path = os.path.join(inpath, "tref.pickle")
# pickle 파일에서 객체 불러오기
with open(tref_path, 'rb') as f:
    tref = pickle.load(f)
# 불러온 객체를 개별 변수로 할당
for object_name, obj in tref.items():
    globals()[object_name] = obj

'''Radar chart -model2'''
# minus를 대체할 수 있도록 설정
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family']='Malgun Gothic' #한글깨짐 방지

axis1 = tref.loc[tref[(tref['agegr']==db_model11['연령그룹'].iloc[0])&(tref['sex']==db_model11['성별'].iloc[0])].index, ['bp_score']].values.flatten() #values; 1차원 배열로 반환
axis2 = tref.loc[tref[(tref['agegr']==db_model11['연령그룹'].iloc[0])&(tref['sex']==db_model11['성별'].iloc[0])].index, ['wc_score']].values.flatten()
axis3 = tref.loc[tref[(tref['agegr']==db_model11['연령그룹'].iloc[0])&(tref['sex']==db_model11['성별'].iloc[0])].index, ['bfmi_score']].values.flatten()
axis4 = tref.loc[tref[(tref['agegr']==db_model11['연령그룹'].iloc[0])&(tref['sex']==db_model11['성별'].iloc[0])].index, ['ffmi_score']].values.flatten()
axis5 = tref.loc[tref[(tref['agegr']==db_model11['연령그룹'].iloc[0])&(tref['sex']==db_model11['성별'].iloc[0])].index, ['ua_score']].values.flatten()
# axis_total = agegroup.loc[scoring_model2['연령그룹'].value, ('model2_score','min')]

data2 = pd.DataFrame( {
    '혈압': [db_model11['bp_score'][0], axis1], #mean BP로
    '허리둘레': [db_model11['wc_score'][0], axis2],
    '체지방량지수': [db_model11['bfmi_score'][0], axis3],
    '제지방량지수': [db_model11['ffmi_score'][0], axis4],
    '당 대사지수': [db_model11['ua_score'][0], axis5]
})

# 데이터 딕셔너리를 생성
dict1 = {'혈압': data2.iloc[0, 0], '허리둘레': data2.iloc[0, 1], '체지방량지수': data2.iloc[0, 2], 
         '제지방량지수': data2.iloc[0, 3], '당 대사지수': data2.iloc[0, 4]}
dict2 = {'혈압': data2.iloc[1, 0], '허리둘레': data2.iloc[1, 1], '체지방량지수': data2.iloc[1, 2], 
         '제지방량지수': data2.iloc[1, 3], '당 대사지수': data2.iloc[1, 4]}


categories1 = list(dict1.keys())
categories1 = [*categories1, categories1[0]]  # 360도로 구부리기 때문에 시작과 마지막을 동일하게(폐곡선이기에 마지막에 시작점을 다시 추가하여 끝점과 시작점을 동일하게 만듦)

numbers1 = list(dict1.values())
numbers1 = [*numbers1, numbers1[0]]  # 360도로 구부리기 때문에 시작과 마지막을 동일하게

numbers2 = list(dict2.values())
numbers2 = [*numbers2, numbers2[0]]  # 360도로 구부리기 때문에 시작과 마지막을 동일하게

# 레이더 차트를 그리기 위한 각도 설정
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(numbers1)) #np.linespace는 지정된 범위내에서 일정한 간격으로 num개의 숫자를 생성함.
#2파이(라디안)=360도


plt.figure(figsize=(480 / 25.4, 460 / 25.4))
ax = plt.subplot(polar=True)
plt.xticks(label_loc, labels=categories1, fontsize=20)

ax.plot(label_loc, numbers1, label='person A', linestyle='dashed', color='darkorange')
ax.fill(label_loc, numbers1, color='darkorange', alpha=0.2)

ax.plot(label_loc, numbers2, label='평균', linestyle='dashed', color='royalblue')
ax.fill(label_loc, numbers2, color='royalblue', alpha=0.4)

ax.legend()
# plt.show()

plt.savefig('graph_4_1.svg',format='svg', bbox_inches='tight')
plt.close()

'''mapping'''
#매핑 출력되는 output
# a={
# 	'1|기본정보|검사일시:' : db_model22['검사일시'],
# 	'1|기본정보|검사기관:' : db_model22['검사기관'],
# 	'1|기본정보|성명:' : db_model22['성함'],
# 	'1|기본정보|성별:' : db_model22['성별'],
# 	'1|기본정보|생년월일:' : db_model22['생년월일'],
	
# }


print( db_model22['검사일시'])#검사일시',
db_model22['검사기관'] = 'AI 헬스케어실증센터'
print(db_model22['검사기관'])#'검사기관',
print(db_model22['성함']) #'성명'
print(db_model22['성별']) #'성별'
print(db_model22['생년월일']) #'생년월일'

if any(db_model22['BMI']<=18.5):
	bmi=0
	bmi_tag='저체중'
elif any((db_model22['BMI']<23) & (db_model22['BMI'] >18.5)):
	bmi=1
	bmi_tag='정상'
elif any((db_model22['BMI']<25) & (db_model22['BMI'] >=23)):
	bmi=2
	bmi_tag='과체중'
elif any(db_model22['BMI']>=25):
	bmi=3
	bmi_tag='비만'
print(bmi) #BMI4
print(f'비만 정도를 확인할 수 있는 체질량지수인 BMI를 분석한 결과 {bmi_tag}에 해당합니다.\t적절하게 체중을 유지할 수 잇는 식단 관리가 필요합니다.') #1|식단|요약
print(bmi)#BMI4

if any(db_model11['GLU'])==0:
    glu='-'
else:
    glu='+'
print(glu) #요당

if any(db_model11['KET'])==0:
    ket='-'
else:
    ket='+'
print(ket) #케톤

if any(db_model11['PRO'])==0:
    pro='-'
else:
    pro='+'
print(pro) #요단백

# if any(db_model11['SG'])==0:
#     sg='-'
# else:
#     sg='+'
# print(sg) #요비중

if any((db_model11['성별']=='여성') & (db_model11['SMI']>=5.7)):
    smi=0
    smi_tag='정상범위'
elif any((db_model11['성별']=='여성') & (db_model11['SMI']<5.7)):
    smi=1
    smi_tag='위험범위'
elif any((db_model11['성별']=='남성') & (db_model11['SMI']>=7.0)):
    smi=0
    smi_tag='정상범위'
elif any((db_model11['성별']=='남성') & (db_model11['SMI']<7.0)):
    smi=1
    smi_tag='위험범위'
print(bmi)#BMI4
print(smi) #SMI

print(f'비만 정도를 확인할 수 있는 체질량지수인 BMI를 분석한 결과 {bmi_tag}에 해당합니다.\t몸에 무리가 가지 않게 체중을 조절할 수 있는 맞춤 운동이 필요합니다.') #BMI설명
print(f'근육의 상대적인 비율을 파악할 수 있는 골격근지수(SMI)가 {smi_tag}에 해당합니다.\t현재 상태를 종합하여 쉽게 따라하실 수 있는 2주차 운동을 추천드려요!') #SMI설명

print(db_model11['ttgrade']) #대사 위험도

meta = ""
#대사 설명
if any(db_model11['ttgrade']=='높음'):
    meta=print(f'대사 증후군 위험도가 높기 때문에 평균 수치 대비 수치가 높아 위험한 지표를 확인하고, 해당 지표를 중점으로 꾸준한 건강 관리를 진행해주세요!')
elif any(db_model11['ttgrade']=='보통'):
    meta=print(f'대사 증후군 위험도가 보통이나 평균 수치 대비 수치가 높거나 비슷한 지표를 확인하고, 해당 지표를 중점으로 꾸준한 건강 관리를 진행해주세요!')
elif any(db_model11['ttgrade']=='낮음'):
   meta= print(f'대사 증후군 위험도가 낮으나 질병에 영향을 미치는 건강 지표를 확인하고, 해당 지표를 중점으로 꾸준한 건강 관리 및 예방을 진행해주세요!')

elder = ""
#노인 설명
if '높음'in scoring_model2['tertile']:
	elder ='노인 증후군 위험도가 높기 때문에 평균 수치 대비 수치가 높아 위험한 지표를 확인하고, 해당 지표를 중점으로 꾸준한 건강 관리를 진행해주세요!'
elif '보통'in scoring_model2['tertile']:
	elder ='노인 증후군 위험도가 보통이나 평균 수치 대비 수치가 높거나 비슷한 지표를 확인하고, 해당 지표를 중점으로 꾸준한 건강 관리를 진행해주세요!'
elif '낮음'in scoring_model2['tertile']:
	elder ='노인 증후군 위험도가 낮으나 질병에 영향을 미치는 건강 지표를 확인하고, 해당 지표를 중점으로 꾸준한 건강 관리 및 예방을 진행해주세요!'

print(scoring_model2['model2_total_score']) #노인 위험도

#매핑 최종결과물
mapping = pd.DataFrame({
	'Pg' : [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5],
 	'#' : [1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,9,8,9,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8],
 	'Key' : ['검사일시','검사기관','성명','성별','생년월일','BMI_tag','식단|요약|설명','BMI_tag','검사일시','검사기관','성명','성별','생년월일','당뇨','케톤뇨','단백뇨',\
     '검사일시','검사기관','성명','성별','생년월일','BMI_tag','SMI_tag','bmi설명','smi설명','1주차운동','2주차운동','검사일시','검사기관','성명','성별','생년월일','대사|위험도','graph_4_1.svg','대사|설명',\
        '검사일시','검사기관','성명','성별','생년월일','노인위험도','graph_5_1.svg','노인|설명' ],
	'디헬(V)' : [db_model22['검사일시'],db_model22['검사기관'],db_model22['성함'],db_model22['성별'],db_model22['생년월일'], bmi, f'비만 정도를 확인할 수 있는 체질량지수인 BMI를 분석한 결과 {bmi_tag}에 해당합니다.\t적절하게 체중을 유지할 수 잇는 식단 관리가 필요합니다.',\
     bmi,db_model22['검사일시'],db_model22['검사기관'],db_model22['성함'],db_model22['성별'],db_model22['생년월일'],glu, ket,pro, db_model22['검사일시'],db_model22['검사기관'],db_model22['성함'],db_model22['성별'],db_model22['생년월일'],bmi,smi,\
        f'비만 정도를 확인할 수 있는 체질량지수인 BMI를 분석한 결과 {bmi_tag}에 해당합니다.\t몸에 무리가 가지 않게 체중을 조절할 수 있는 맞춤 운동이 필요합니다.' ,  f'근육의 상대적인 비율을 파악할 수 있는 골격근지수(SMI)가 {smi_tag}에 해당합니다.\t현재 상태를 종합하여 쉽게 따라하실 수 있는 2주차 운동을 추천드려요!',\
       '1주차 운동','2주차 운동', db_model22['검사일시'],db_model22['검사기관'],db_model22['성함'],db_model22['성별'],db_model22['생년월일'],db_model11['ttgrade'], 'graph_4_1.svg', \
         meta, db_model22['검사일시'],db_model22['검사기관'],db_model22['성함'],db_model22['성별'],db_model22['생년월일'] ,  scoring_model2['model2_total_score'],'graph_5_1.svg', elder ],
	'인프라' : ['v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v','v']
 })


# mapping
a=db_model22['성함']
b=db_model22['생년월일']
b=b.apply(format_date)
mapping.to_csv(f'{a[0]}'+'_'+f'{b[0]}.csv',encoding='euc-kr')
print(os.path.abspath(f'{a[0]}'+'_'+f'{b[0]}.csv'))
