import cv2
import time


def decode_fourcc(v: int):
    '''
    avc1: H.264
    av01: av1
    https://github.com/opencv/opencv/blob/master/samples/python/video_v4l2.py
    '''
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def get_video_decode(url: str):
    vidcap = cv2.VideoCapture(url)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, 3000)  # just cue to 20 sec. position
    fourcc = vidcap.get(cv2.CAP_PROP_FOURCC)
    decode_name = decode_fourcc(fourcc)
    return decode_name

def get_video_image(url: str):
    vidcap = cv2.VideoCapture(url)
    success,image = vidcap.read()
    if success:
        # cv2.imwrite("frame3sec.jpg", image)     # save frame as JPEG file
        cv2.imshow('joy',image)
        cv2.waitKey()


def get_video_duration(url: str):
    try:
        cap = cv2.VideoCapture(url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(round(frame_count / fps, 0)) * 1000
        print(duration)
        return duration
    except ZeroDivisionError:
        print("cant get duration")


if __name__ == "__main__":
    start_time = time.time()
    urls = [
        "https://s3.amazonaws.com/vibbidi-us/audio/audio_D8A1B62092494DE7B2FFC01DF80C5A64.mp3",
        "https://s3.amazonaws.com/vibbidi-us/videos/video_EFD70D27F348412C87EE942E1700448A.mp4",
        "https://s3.amazonaws.com/vibbidi-us/audio/audio_BBD90B838A204A4DA0C680A6DA4B4DF5.mp3",
        "https://s3.amazonaws.com/vibbidi-us/audio/audio_C06F9034DB624DDCB7CF49858192B36F.mp3"
    ]

    # joyxinhqua
    for url in urls:
        # k = get_video_decode(url)
        # print(f"{k} ----{url}")
        get_video_duration(url)

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
# vidcap.get(CV_CAP_PROP_FOURCC);
