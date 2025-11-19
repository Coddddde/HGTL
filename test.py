# a = 'I love China'
# print(a[::-1])

import random


class student:
    def __init__(self, id, score):
        self.id = id
        self.score = score

    def get_score(self):
        return self.score

    def get_id(self):
        return self.id


stu_list = []
for i in range(10):
    id = random.randint(1, 100)
    score = random.random()
    stu = student(id, score)
    stu_list.append(stu)


stud_list = sorted(stu_list, key=lambda x: (x.get_score(), x.get_id()))
for stu in stud_list:
    print("id: {}, score: {}".format(stu.get_id(), stu.get_score()))