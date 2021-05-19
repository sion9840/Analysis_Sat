import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd

st.title("Analysis Sat")
st.header("수능을 분석한 결과를 보여주는 사이트")
st.info("분석 수능 과목 : 수학")

st.markdown('''---''')

problem_len = 30
choice_len = 6
exam_answer = {
    "2015" : np.random.randint(low=1, high=choice_len+1, size=problem_len),
    "2016" : np.random.randint(low=1, high=choice_len+1, size=problem_len),
    "2017" : np.random.randint(low=1, high=choice_len+1, size=problem_len),
    "2018" : np.random.randint(low=1, high=choice_len+1, size=problem_len),
    "2019" : np.random.randint(low=1, high=choice_len+1, size=problem_len),
    "2020" : np.random.randint(low=1, high=choice_len+1, size=problem_len),
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

st.write("문제마다 이번 수능에 찍으면 좋은 번호")
for i in range(problem_len):
    line_fitter = LinearRegression()
    line_fitter.fit(np.arange(1, exam_len+1).reshape(-1, 1), problem_answer[i])

    line = np.arange(exam_len*2)
    line1 = problem_answer[i]
    line2 = np.array([line_fitter.predict([[i]]) for i in range(exam_len)])

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
   
    st.text(str(i+1) + "번 문제")
    st.line_chart(chart_data)