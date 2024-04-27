from ultralytics import YOLO
import cv2


def main():
    # load yolov8 model
    model = YOLO('yolov8s.pt')
    print("=============================\nModel imported\n=============================")

    # detect objects
    # track objects
    for i in range(1, 100):
        frame = f"datasets/4_annotate_1min_bodo_start/img1/{i:06d}.jpg"
        results = model.track(frame, persist=True, conf=.1, iou=.5, tracker="bytetrack.yaml")

        # plot result
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
