# Student_Admission_Predictor
 
# output: This the target name. This is what you use when you type >> make target_name
# The object name is what you use to execute the program ./project
# The structure is 
# gcc -o (output_file) (filename.c) -lm
# make output
all: project.c
	gcc -o project project.c -lm
	./project student_data.txt

# This is used to remove the created object whenever you type 
# make clear 
clear:
	rm project
	rm CostFunctionHistory.txt
