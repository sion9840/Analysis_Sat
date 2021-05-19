import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

fig = plt.figure()
ax = fig.add_subplot(h, w, 1)
ax.plot(np.arange(1, problem_len+1), problem_answer_per, 'ro')
ax.axis([0, problem_len+1, 0, choice_len+1])
ax.set_title('analysis problem answer per')
ax.set_xlabel('problem num')
ax.set_ylabel('answer num')

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

plt.show()