import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
students = ["Ahmed", "Sara", "Karim", "Ali", "Mohamed", "Lina", "Lamis", "Dina", "Khaled", "Youssef", 
            "Omar", "Hoda", "Mona", "Ziad", "Nour", "Layla", "Amr", "Samy", "Reem", "Tamer",
            "Nada", "Fady", "Hassan", "Salma", "Mariam", "Ibrahim", "Nadia", "Wael", "Ola", "Adel"]

math = [85, 92, 78, 65, 90, 88, 99, 74, 80, 67, 95, 82, 70, 60, 88, 94, 76, 55, 91, 83, 72, 68, 89, 93, 81, 77, 62, 84, 79, 96]
science = [82, 89, 76, 70, 94, 86, 95, 73, 79, 68, 92, 85, 68, 58, 85, 91, 74, 52, 88, 80, 70, 65, 87, 90, 79, 75, 60, 81, 77, 93]
english = [90, 91, 77, 66, 88, 85, 96, 78, 81, 69, 94, 88, 72, 62, 86, 92, 75, 58, 89, 82, 74, 69, 85, 91, 80, 78, 64, 83, 75, 94]
behavior = ["good", "bad", "good", "bad", "good", "good", "good", "bad", "good", "bad", "good", "good", "bad", "bad", "good", "good", "good", "bad", "good", "good", "bad", "bad", "good", "good", "good", "good", "bad", "good", "bad", "good"]
absences = [2, 3, 4, 0, 5, 2, 1, 6, 3, 4, 1, 2, 7, 8, 2, 1, 3, 9, 2, 2, 6, 5, 1, 1, 2, 3, 7, 2, 6, 1]
problem = [0, 2, 1, 3, 1, 0, 0, 2, 1, 1, 0, 1, 3, 4, 0, 0, 1, 5, 0, 1, 3, 2, 0, 0, 1, 1, 4, 1, 3, 0]

df=pd.DataFrame(
    {
        "students":students,
        "math":math,
        "science":science,
        "English":english ,
        "behavior":behavior,
        "absences":absences,
        "problem":problem

    }
)
print(df)
df.to_csv("students_Data.csv",index=False)
#change to binomial
le=LabelEncoder()
df["bahevior_binomial"]=le.fit_transform(df["behavior"])
x=df[["math","science","English", "absences" ,"problem"]]
y=df["bahevior_binomial"]
x_train,x_test,y_train,_y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
new_student ={
'math': 70,
'science': 50,
'English': 69,
'absences': 0,
'problem': 10,
}
new_student=pd.DataFrame([new_student])
new_student_pred=model.predict(new_student)
convert=le.inverse_transform(new_student_pred)
print(f"behaviof for new student is -----> {convert[0]}")

behavior_counts = df['behavior'].value_counts()
plt.bar(behavior_counts.index, behavior_counts.values, color=['green', 'red'])
plt.title('Distribution of Student Behavior')
plt.xlabel('Behavior Type')
plt.ylabel('Number of Students')
plt.show
plt.figure(figsize=(12, 6))
x = df['students']
plt.bar(x, df['math'], label='Math', alpha=0.6)
plt.bar(x, df['science'], label='Science', alpha=0.6)
plt.bar(x, df['English'], label='English', alpha=0.6)
plt.title('Students Grades Comparison')
plt.xlabel('Students')
plt.ylabel('Grades')
plt.xticks(rotation=45) 
plt.legend()
plt.tight_layout()
plt.legend()
plt.show()
