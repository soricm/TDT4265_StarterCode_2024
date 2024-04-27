import os


def create_directories(root_dir):
    # Create root directory if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Create images and labels directories
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Create train, val, and test directories inside images and labels directories
    for directory in ['train', 'val', 'test']:
        images_subdir = os.path.join(images_dir, directory)
        labels_subdir = os.path.join(labels_dir, directory)
        os.makedirs(images_subdir, exist_ok=True)
        os.makedirs(labels_subdir, exist_ok=True)
        print(f"Created directories for {directory} set.")


def mot_to_yolov8(input_file, output_dir):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(',')
        frame_id, id, bb_left, bb_top, bb_width, bb_height, _, class_id, _ = map(float, parts)

        # Convert MOT format to YOLOv5 format
        class_id = int(class_id-1)
        x_center = (bb_left + bb_width / 2) / 1920  # Normalizing by image width
        y_center = (bb_top + bb_height / 2) / 1080  # Normalizing by image height
        width = bb_width / 1920  # Normalizing by image width
        height = bb_height / 1080  # Normalizing by image height

        # Write YOLOv5 format to output file 
        output_file = os.path.join(output_dir, f'labels/{int(frame_id):06d}.txt')
        with open(output_file, 'a') as f_out:
            f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    root_dir = "./true_dataset"  # Change this to the desired root directory
    create_directories(root_dir)

    mot_to_yolov8("datasets/1_train-val_1min_aalesund_from_start/gt/gt.txt",
                  "datasets/true_dataset/labels/train")
    mot_to_yolov8("datasets/2_train-val_1min_after_goal/gt/gt.txt",
                  "datasets/true_dataset/labels/val")
    mot_to_yolov8("datasets/3_test_1min_hamkam_from_start/gt/gt.txt",
                  "datasets/true_dataset/labels/test")
