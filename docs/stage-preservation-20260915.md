# Rigid stage AQP preservation

## 동작

FBX 변환에서 사라지던 원본 stage의 두 번째 정점 색상, packed vertex 선언,
보조 테이블과 재질 데이터를 `.blend` 내부에 원본 AQP와 함께 보관한다.
내보낼 때 원본에 정점 위치 수정을 반영하고, 추가한 별도 메시를 AML 데이터로
병합한다. 모델 공통 `unkMeshValue`는 원본 값으로 덮어쓰지 않고 AML이 내보낸
장면의 값을 유지한다. 특정 파일 이름이나 공통값 상수를 사용하지 않는다.

일반 재질의 TSTA 저장/복원도 확장하여 벡터와 다섯 float 필드를 보존한다.

## 사용 범위

- 새 코드로 **원본 rigid stage AQP를 다시 불러온 뒤** 작업한다. 기존 `.blend`에
  이미 소실된 정보를 자동 복구하지는 않는다.
- 원본 파트 전체를 함께 내보낸다. 별도 메시로 추가한 소품은 별도 재질을 사용하고
  동일한 단일 root에 연결한다.
- 원본 파트의 정점 이동과 셰이프키 적용을 지원한다. 원본 shader, 재질 설정,
  정점 색상과 normal은 원본대로 보존한다. 이 경로는 원본 파트의 재질/색상/normal
  편집을 내보내는 기능이 아니다. 추가 소품은 기존 변환 경로를 사용한다.
- 원본 파트의 topology/UV 변경, 일부 파트만 내보내기, 원본 파트 복제,
  다른 skeleton 병합은 지원하지 않으며 감지되는 경우 내보내기를 취소한다.
- rigid stage와 weighted 소품이 이미 합쳐진 AQP의 재불러오기는 이 보존 경로의
  대상이 아니다. 원본을 불러와 작업한 `.blend`를 계속 사용한다.
- 셰이프키가 있는 원본 파트는 Apply Shape Keys를 켜야 한다.

## 검증

Blender 5.1.2 / Windows에서 실제 원본 stage와 추가 소품 AQP로 검증했다.
`tests/blender_stage_model.py`는 `PSO2_STAGE_ORIGINAL`과
`PSO2_STAGE_ADDITIONS` 환경 변수로 파일을 받고, 임시 디렉터리에만 출력한다.

- 원본 정점 payload 바이트 일치, 원본 모델 type와 보조 테이블 유지
- AML 변환 결과의 공통값 유지와 추가 메시 수 검증
- 객체/재질 이름 변경 및 `.blend` 저장 후 다시 열기
- 정점 이동량 검증, 셰이프키 적용과 편집 데이터 보존
- 부분 선택 내보내기 취소 시 기존 대상 파일 보존
- FBX import 및 native import 경로 통과 (native도 Windows에서 실행)
- 기존 셰이프키 14개, 재질 8개와 vertex layout 회귀 테스트 통과

사용자는 같은 원본 보존/공통값 조합으로 만든 시험 파일의 게임 내 정상 동작을
확인했다. 이번 플러그인에서 새로 생성한 출력의 게임 재검증과 실제 macOS 검증은
별도이다. 사용 중인 Blender 프로필의 확장은 변경하지 않았다.

## 2026-09-16 릴리즈 패키지 검증

`pso2_tools-20260916-windows_x64.zip`을 별도의 깨끗한 Blender 5.1.2와
5.2.1 프로필에 설치하고 활성화했다. 설치된 Python 코드 및 해당 패키지의
AML DLL을 사용했음을 확인했다. 두 버전에서 셰이프키 14개, 재질 8개,
vertex layout 검사와 실제 stage 내보내기 검사를 통과했다. 재질과 stage는
FBX/native 두 가져오기 경로로 검증했으며 출력 AQP의 재불러오기도 통과했다.
Windows 전용 패키지이며 macOS 바이너리는 포함하지 않는다.
