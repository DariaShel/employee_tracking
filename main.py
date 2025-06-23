import cv2
import time

from stream import MultiStreamProcessor


def main():
    rtsp_urls = [
        'rtsp://<your url 1>',
        'rtsp://<your url 2>',
        'rtsp://<your url 3>',
        'rtsp://<your url 4>'
    ]
    pose_model_path = "yolo11n-pose.pt"
    patch_model_ckpt = './ckpts/model_20250202_121510_72'
    device = "cuda:0"

    # Create a MultiStreamProcessor instance and start
    msp = MultiStreamProcessor(rtsp_urls, pose_model_path, patch_model_ckpt, device=device)
    msp.start_streams()

    try:
        while True:
            combined_frame, last_times, detect_str = msp.get_combined_frame()
            if combined_frame is None:
                print("Could not retrieve a valid combined frame. Retrying...")
                time.sleep(0.1)
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Main thread: Exiting on keyboard interrupt.")
    finally:
        msp.stop_all()


if __name__ == "__main__":
    main()
