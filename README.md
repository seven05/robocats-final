<h1 style="text-align: center">
<img src="https://user-images.githubusercontent.com/8157830/169524284-0042b9cb-1c60-467e-b46c-0c313682f96b.png" style="height:1em">
Robocats Final
<img src="https://user-images.githubusercontent.com/8157830/169524284-0042b9cb-1c60-467e-b46c-0c313682f96b.png" style="height:1em">
</h1>

> 매 주말마다 하는 어썸한 로보틱스 해커톤

<div style="text-align:center">
<img src="https://user-images.githubusercontent.com/8157830/169529053-f636e1be-9931-43af-a8a3-c047d9b4517f.png">
</div>

## 🪛 Dependencies
- python 2.7

## 🔥 How to use
Run `run.py` with task id (1, 2, 3, 4)
### Example
```
python run.py 1
```

## 🚧 Progress

- [x] Task 1: Grip bottle
- [ ] Task 2: Bring bottle to base
- [x] Task 3: Remove bottle from the chamber
- [ ] Task 4: Remove bottle from the chamber after unlocking

## 📖 Explain
- 각 task에 대한 최신 코드는 master branch의 t1, t2, t3, t4에 존재합니다.
- `*_general.py`는 테스트를 위해 작성된 후 사용되지 않는 legacy code입니다
- `lidar_scan_heading.py`는 heading 방향 좌우 2도씩을 포함한 5도 이내의 최저 거리를 가져와서 `/scan_heading`에 publish하는 모듈입니다
