#!/usr/bin/python3
'''a = "print('Hello world')"
my_file = open("my_file_next.py", "w")
my_file.write("#!/usr/bin/python3\n")
my_file.write(a)
my_file.close()
import os
myCmd = 'sudo chmod +x my_file_next.py'
os.system (myCmd)
start_next = './my_file_next.py'
os.system (start_next)'''

#формируем список из n уникальных айдишников
'''
import random
list_program_id = []
caunt_programs = 10
for i in range(caunt_programs):
	r = random.randint(100, 999)
	if r not in list_program_id: list_program_id.append(r)
print(list_program_id)
'''

#формируем список из n айдишников (от 0 с шагом 1)
list_program_id = []
count_programs = 10
for i in range(count_programs):	
	list_program_id.append(i)
print("список айдишников программ")
print(list_program_id)

import os
import time

for i in range(count_programs):
	program_name = "program_" + str(list_program_id[i]) + ".py"			#создаём имя программы
	new_program_file = open(program_name, "w")					#создаём программу
	
	new_program_file.write("#!/usr/bin/python3\n")				#пишем в ней python
	new_program_file.write("a = " + str(i) + "\n")				#пишем переменную "a"
	new_program_file.write("b = a * a \n")					#пишем посчитай "b"
	
	answer_name = "ans_file_" + str(list_program_id[i]) + ".txt"			#создаём имя файла ответа	
	new_program_file.write("new_ans_file = open('" + answer_name + "', 'w') \n") #пишем создание файла ответа		
	new_program_file.write("new_ans_file.write(str(b)) \n")			#записываем "b" в файл ответа
	new_program_file.write("new_ans_file.close() \n")				#закрываем файл ответа
	
	new_program_file.close()							#закрываем программу
	
	print(program_name + " was creatad")						#выводим что программа успешно создана
	
	myCmd = "sudo chmod +x " + program_name
	os.system (myCmd)								#ставим созданной программе свойство исполняемости
	start_current_program = "./" + program_name
	os.system (start_current_program)						#запускаем на выполнение
	
	print(program_name + " was started")						#выводим что программа запущена
	time.sleep(0.25)								#для красивой печати
	print("\033[A                       \033[A")
	print("\033[A                       \033[A")
		
print("all programs was creatad")
print("all programs was started")
time.sleep(0.5)


#соберем результаты
s=0
for i in range(count_programs):
	answer_name = "ans_file_" + str(list_program_id[i]) + ".txt"			#еще раз создаём имя файла ответа		
	resault_program = open(answer_name, 'r')					#открываем результат для чтения
	s = s + int(resault_program.read())						#суммируем результат
print ("Сумма квадратов чисел от 0 до " + str(count_programs) + " =", s)		#выводим итоговвый результат
