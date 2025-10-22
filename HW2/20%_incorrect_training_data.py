import random
import csv

output_file = "./data/arithmetic_train_20pct_wrong.csv"
num_samples = 263251  # 你的 eval / 小訓練集大小

operators = ['+', '-', '*']
digits = list(range(10, 100))  # 二位數

def generate_complex_expression():
    # 隨機決定運算式長度 (2~3 個數字)
    num_numbers = random.randint(2, 3)
    numbers = [str(random.choice(digits)) for _ in range(num_numbers)]
    ops = [random.choice(operators) for _ in range(num_numbers-1)]

    # 基礎運算式
    expr = numbers[0]
    for i in range(num_numbers-1):
        expr += ops[i] + numbers[i+1]

    # 隨機加入括號
    if num_numbers > 2 and random.random() < 0.7:
        start = random.randint(0, len(numbers)-2)
        end = random.randint(start+1, len(numbers)-1)
        char_start = sum(len(n)+1 for n in numbers[:start])
        char_end = sum(len(n)+1 for n in numbers[:end]) + len(numbers[end-1])
        expr = expr[:char_start] + '(' + expr[char_start:char_end] + ')' + expr[char_end:]

    # 計算正確答案
    try:
        correct_result = eval(expr)
    except ZeroDivisionError:
        correct_result = 0

    return expr + "=", correct_result

# 生成 CSV
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['src', 'tgt'])  # 標題行

    for _ in range(num_samples):
        src, tgt_correct = generate_complex_expression()

        # 20% 機率生成錯誤答案
        if random.random() < 0.2:
            tgt = str(tgt_correct + random.choice(range(-20, 21)))
            while tgt == str(tgt_correct):  # 確保真的錯誤
                tgt = str(tgt_correct + random.choice(range(-20, 21)))
        else:
            tgt = str(tgt_correct)

        writer.writerow([src, tgt])

print(f"已生成 {output_file}，共 {num_samples} 筆二位數運算式資料，其中約 20% 為 incorrect answers。")
