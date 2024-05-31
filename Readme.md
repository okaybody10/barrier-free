# Whisper

- 1번 Task(STT)에 사용되는 폴더입니다.
- run.py를 이용하여 실행할 수 있고, 이때 config.json을 데이터셋에 맞게 수정해야합니다.
- `model_size`: 사용할 whisper의 model size (tiny, base, small, medium, large)
- `labeling_path`: 코드에 사용될 음성들의 labeling path
- `sound_path`: 코드에 사용될 음성들의 실제 path
- `save_path`: 결과를 저장할 path
- run.py의 실행이 끝난 후 metric.py 를 마지막으로 실행하면 결과에 대해 얻을 수 있습니다.

# vr

- 2번 Task(비음성구간)에 사용되는 폴더입니다.
- run.py를 이용하여 실행할 수 있고, 이때 config.json을 데이터셋에 맞게 수정해야합니다.
- 자세한 사항은 'pyannote 사용법.pdf'에 작성되어 있습니다.

# sed

- 3번 Task(음악구간)에 사용되는 폴더입니다.
- run.py를 이용하여 실행할 수 있고, 이때 config.json을 데이터셋에 맞게 수정해야합니다.
- `movie_path`: 영화(혹은 드라마, 음성)의 path
- `sub_path`: 영화(혹은 드라마, 음성) 자막의 path
- `save_path`: 결과를 저장할 path
- `time_delta`: 자막 내에서 split할 시간 수 (권장: 5~8초 이내)
- `boundary_delta`: 자막 끝까지의 최대 시간 수 (권장: 11초 이내)
