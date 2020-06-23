import os
input_files= os.listdir('Instances')

for file_name in input_files:
    output_file_name='Output\\'+file_name
    command = 'py .\solve.py .\Instances\\'+ file_name+' '+output_file_name+' 16 6666'
    os.system(command)