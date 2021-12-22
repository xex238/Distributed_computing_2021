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
caunt_programs = 10
for i in range(caunt_programs):	
	list_program_id.append(i)
print('список айдишников программ')
print(list_program_id)

import os
import time

for i in range(caunt_programs):
	program_name = 'program_' + str(list_program_id[i]) 			#создаём имя программы
	new_program_file = open(program_name+".py", "w")			#создаём программу
	new_program_file.write("#!/usr/bin/python3\n")			#пишем в ней python
	new_program_file.write("a = " + str(i) + "\n")			#пишем переменную а
	new_program_file.write("b = a * a \n")				#пишем пощитай б
	local_ans = "open('ans_file_" + str(i) + ".txt', 'w')"  		#создаем файл ответа
	new_program_file.write("ans_file = " + str(local_ans) +"\n") 	#пишем создание файла ответа
	new_program_file.write("ans_file.write(str(b)) \n")			#записываем б в файл ответа
	new_program_file.write("ans_file.close() \n")				#закрываем файл ответа
	new_program_file.close()						#закрываем программу
	
	print(program_name + " was creatad")
	#time.sleep(0.1)
	
	myCmd = "sudo chmod +x " + program_name+".py"
	os.system (myCmd)
	start_current_program = "./" + program_name + ".py"
	os.system (start_current_program)
	
	print(program_name + " was started")
	time.sleep(0.25)
	print("\033[A                       \033[A")
	print("\033[A                       \033[A")
		
print("all programs was creatad")
print("all programs was started")
time.sleep(0.4)
	
#соберем результаты
s=0
for i in range(caunt_programs):	
	resault_program = open('ans_file_' + str(i) + '.txt', 'r')
	my_ans = int(resault_program.read())
	s = s + my_ans
print ('s = ', s)
