

while (True): 
    msg = input('Type here: ')
    numbers = msg.split('\t')
    result_text  = ' & '.join(numbers)

    result_text = " & " + result_text + '\\\\\\' + "hline"

    print('\n')
    print(result_text)