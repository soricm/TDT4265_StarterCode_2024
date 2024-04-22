import os


def generate_labels(folder):
    if not os.path.exists(f"{folder}/labels"):
        os.makedirs(f"{folder}/labels")

    old_file = f'{folder}/gt/gt.txt'
    try:
        f = open(old_file, 'r')
        for line in f:
            elements = line.strip().split(",")
            number = int(elements[0])
            file_path = f"{folder}/labels/{number:06d}.txt"
            with open(file_path, 'a') as file:
                new_line = ",".join(elements[1:])
                file.write(new_line+"\n")
            file.close()
    finally:
        f.close()
    print(f"Files labels have been created for {folder}")


if __name__ == "__main__":
    generate_labels('datasets/1_train-val_1min_aalesund_from_start')
    generate_labels('datasets/2_train-val_1min_after_goal')
    generate_labels('datasets/3_test_1min_hamkam_from_start')
