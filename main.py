import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import math

st.title("Analysis Sat")
st.header("수능을 분석한 결과를 보여주는 사이트")
st.info("분석 수능 년도 : 2017년 ~ 2020년, 분석 수능 과목 : 수학, 분석 수능 시험 갯수 : 16개")

st.markdown('''---''')

problem_len = 21
choice_len = 6
exam_answer = {
    "2017-1" : np.array([5,2,5,4,3,5,1,1,2,3,4,4,3,1,4,2,2,3,1,5,4]),
    "2017-2" : np.array([5,2,5,2,3,1,1,5,4,3,4,4,3,1,4,2,2,3,1,5,4]),
    "2017-3" : np.array([2,3,1,5,2,5,4,3,2,5,1,1,3,4,1,4,3,4,2,5,4]),
    "2017-4" : np.array([2,3,1,5,2,1,4,3,4,5,5,1,3,4,1,2,3,4,2,5,4]),
    "2018-1" : np.array([5,4,3,3,2,2,4,1,2,5,3,1,3,5,4,4,3,2,1,5,4]),
    "2018-2" : np.array([5,4,3,3,2,2,2,1,2,5,3,5,3,1,4,4,3,4,1,5,4]),
    "2018-3" : np.array([2,4,1,1,3,5,5,2,1,3,4,5,2,1,5,3,5,4,2,3,2]),
    "2018-4" : np.array([2,4,5,1,3,1,5,2,1,3,4,1,2,5,5,3,5,4,2,3,2]),
    "2019-1" : np.array([5,3,4,2,3,1,3,1,5,4,4,2,1,4,5,2,1,2,3,5,4]),
    "2019-2" : np.array([5,3,4,2,3,1,3,1,5,2,4,2,5,4,1,4,1,2,3,5,4]),
    "2019-3" : np.array([2,5,3,3,1,2,4,2,5,4,3,2,1,5,1,4,4,3,1,5,1]),
    "2019-4" : np.array([2,5,3,3,1,4,2,2,5,2,3,4,1,5,1,4,4,3,1,5,1]),
    "2020-1" : np.array([5,3,2,5,4,3,2,5,3,4,4,2,1,4,2,3,5,1,5,1,5]),
    "2020-2" : np.array([5,3,2,5,4,3,4,1,3,2,4,2,5,4,2,3,5,1,5,1,5]),
    "2020-3" : np.array([2,4,3,5,2,4,3,1,1,3,4,1,5,3,4,4,1,5,1,2,4]),
    "2020-4" : np.array([2,4,3,5,4,4,3,1,1,3,4,1,5,3,4,2,1,1,5,2,4])
}
exam_answer_keys = np.array(list(exam_answer.keys()))
exam_answer_values = np.array(list(exam_answer.values()))
exam_len = exam_answer_keys.shape[0]
problem_answer = np.array([np.array([exam_answer_values[j, i] for j in range(exam_len)]) for i in range(problem_len)])
problem_answer_per = np.array([np.bincount(i).argmax() for i in problem_answer])

st.write("년도별 수능 답 번호")
df_ea = pd.DataFrame(exam_answer_values, index=exam_answer_keys, columns=np.arange(1, problem_len+1))
st.dataframe(df_ea)

st.write("문제마다 가장 많이 나오는 답 번호")
df_pap = pd.DataFrame([problem_answer_per], index=["->"], columns=np.arange(1, problem_len+1))
st.dataframe(df_pap)

st.write("문제마다 이번 수능에서 찍으면 확률적으로 좋은 답 번호")
for i in range(problem_len):
    line_fitter = LinearRegression()
    line_fitter.fit(np.arange(1, exam_len+1).reshape(-1, 1), problem_answer[i])

    line = np.arange(exam_len*2)
    line1 = problem_answer[i]
    line2 = np.array([line_fitter.predict([[i]]) for i in range(exam_len)])
    way = "일정"

    alpha = line_fitter.predict([[exam_len-1]]) - line_fitter.predict([[0]])
    if alpha > 0:
        way = "상승"
    elif alpha < 0:
        way = "하강"

    ppr = math.floor(line_fitter.predict([[exam_len]]))
    if ppr < 1:
        ppr = 1
    elif ppr > choice_len:
        ppr = choice_len
        
    a, b = 0, 0
    for j in range(exam_len*2):
        if j % 2 == 0:
            line[j] = line1[a]
            a += 1
        else:
            line[j] = line2[b]
            b += 1

    chart_data = pd.DataFrame(
        line.reshape(exam_len, 2),
        columns=['problem answer', 'predict line']
        )
    
    st.text(str(i+1) + "번 문제 : " + "답 " + str(ppr) + "번 추천 (이유 : " + "예측 선이 " + str(ppr) + "번 쪽으로 " + way + "하기 때문이다)")
    st.line_chart(chart_data)