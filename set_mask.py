import random

mask_num = 4
def replace_letters(input_str, indices, target_letter):
    output_sequence = ''
    for i, char in enumerate(input_str):
        if i in indices:
            output_sequence += target_letter
        else:
            output_sequence += char
    return output_sequence

def generate_random_indices(length):
    random_indices = random.sample([x for x in range(length) if x != length//2], mask_num)
    return random_indices


# 打开输入文件
with open("dataset/positive_training_set1(1038).txt", "r") as f:
    input_lines = f.readlines()

# 打开输出文件
with open("dataset/positive_training_set_mask4(1038).txt", "w") as f:
    # 处理每一行
    for i, line in enumerate(input_lines):
        if i % 2 == 1:  # 仅处理偶数行
            parts = line.split('\t')
            random_indices = generate_random_indices(len(parts[1]) - 1)
            modified_str = replace_letters(parts[1], random_indices, 'X')
            output_str = f"{parts[0]}\t{modified_str}"
        else:
            output_str = line
        # 写入到输出文件
        f.write(output_str)

print("处理完成！")
