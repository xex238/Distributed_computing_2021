#!/usr/bin/python3
import os
import time
caunt_programs = 10
for i in range(caunt_programs):
				
	program_name = 'program_' + str(i) + '.py'
	os.remove(program_name)
	print(program_name + " was deleted")
	
	
	resault_program = 'ans_file_' + str(i) + '.txt'
	os.remove(resault_program)
	print(resault_program + " was deleted")
	time.sleep(0.25)
	print("\033[A                            \033[A")
	print("\033[A                            \033[A")

print("all programs was deleted")
print("all resault was deleted")
