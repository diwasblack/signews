def generate_table_content(input_list, group_size=5, file_name="latex_output.txt"):

    str_input_list = [str(x) for x in input_list]

    str_input_list = [
        x if type(x) == str else str(round(x, 2))
        for x in input_list
    ]

    groups = [
        str_input_list[i: i + group_size]
        for i in range(0, len(str_input_list), group_size)
    ]

    with open(file_name, "w") as file:
        for group in groups:
            joined_str = "&".join(group)
            new_str = f"{joined_str}\\\\ \hline \n"
            file.write(new_str)
