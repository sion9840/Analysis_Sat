import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd

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

h,w = 6,6

print(exam_answer_values)
print(problem_answer)
print(problem_answer_per)

for i in range(1, problem_len):
    ax = fig.add_subplot(h, w, i+1)
    ax.plot(exam_answer_keys, problem_answer[i], 'ro')
    ax.axis([0, exam_len+1, 0, choice_len+1])
    ax.set_title('analysis problem predict')
    ax.set_xlabel('exam year')
    ax.set_ylabel('answer num')

    line_fitter = LinearRegression()
    line_fitter.fit(np.arange(1, exam_len+1).reshape(-1, 1), problem_answer[i])
    ax.plot([0, exam_len+1], [line_fitter.predict([[0]]), line_fitter.predict([[exam_len+1]])])

st.title("Analysis Sat")
st.header("수능을 분석한 결과를 보여주는 사이트")
st.info("분석 수능 과목 : 수학")

st.markdown('''---''')

st.write("년도별 수능 답 번호")
df_ea = pd.DataFrame(exam_answer_values, index=exam_answer_keys, columns=np.arange(1, problem_len+1))
st.dataframe(df_ea)

st.write("문제마다 가장 많이 나오는 답 번호")
df_pap = pd.DataFrame(problem_answer_per, columns=np.arange(1, problem_len+1))
st.dataframe(df_pap)

#plt.show()