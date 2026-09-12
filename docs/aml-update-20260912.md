# AML 업데이트와 API 사용 점검

## 범위와 기준

`release/20260906`의 `cd1293f`를 기준으로 한 `build-20260912` Windows 릴리즈의 변경과 검증 기록이다. 기존 `import_model.py`의 구형 모델 호환 처리를 유지하면서 MATE 생성은 AML에 위임했다. 이번 문서는 최신 소스와 새 바이너리의 검증을 기록한다. 애드온 버전은 상위 프로젝트와 같은 `2.8.0`을 유지하며 날짜가 붙은 배포 파일명으로 빌드를 구분한다.

| 대상 | 확인한 리비전 |
| --- | --- |
| Aqua-Toolset | `7130076eca748c5a531eb4a428973f06dfb5b964` |
| Toolset이 참조하는 PSO2-Aqua-Library | `7e9f8749ded800fb3c5b93952bfb9cb7a04734e5` |
| 이전 애드온의 AML | `0f2233822fc6741dbf254a3a44db33f1154e3229` |
| AML의 NvTriStrip.Net | `7ca6a3353555cc8bb425a05eadd684380d1e4bde` |
| AML의 SoulsFormatsNEXT | `9f5848f5f45a7f5b2d4ff841ba05b63ab2e6be0a` |
| AML의 ZamboniLib | `cf6066c52161c2b749d772cf4852202a8a550e49` |

기준은 [Aqua-Toolset 소스](https://github.com/Shadowth117/Aqua-Toolset/tree/7130076eca748c5a531eb4a428973f06dfb5b964)와 그 하위 모듈이다. 서로 다른 시점의 DLL을 섞지 않고 같은 프로젝트 참조 그래프에서 빌드한다.

## AML로 위임한 부분

| 코드 | 변경과 보존 조건 |
| --- | --- |
| `import_model._add_missing_material` | MATE가 전혀 없는 모델의 기본 재질을 `GenericMaterial`과 `AquaObject.GenerateMaterial`로 만든다. 별도 임시 모델에서 생성하고 MATE 한 항목만 복사해서 원본 SHAD, REND, TSET, TSTA, TEXF를 보존한다. |
| `import_model._attach_tsta_data` | 정상 TSET은 `AquaObject.GetTexListTSTAs`로 조회한다. AML이 예외를 내는 잘못된 인덱스에만 기존의 유효 항목 필터를 적용한다. 순서, 중복, UV 채널과 미해석 필드를 유지한다. |
| `material._material_name_data` | 재질 메타데이터의 해석을 내보내기와 같은 `AquaObject.GetMaterialNameData`로 통일한다. Python 값만 최대 1,024개 캐시하며 Blender Image나 CLR 객체를 캐시하지 않는다. |
| `export_model.strip_padded_uvs` | UV 선언의 숫자 식별자를 `VertFlags`로 바꿨다. 빈 배열과 VTXE 선언을 함께 제거하는 기존 수정과 AML `GetVTXESize` 사용은 유지한다. |
| `objects.CmxColorMapping` | 현재 AML의 `maskColorMapping` 필드를 사용한다. 예전 `unkInt` 필드를 직접 비트 분해하던 호환 분기와 사용되지 않는 `split_int32`를 제거했다. |
| `shape_sliders` | 위치, 회전, 스케일 데이터 형식과 타이밍 배율을 기존 `aqm`의 AML 어댑터로 조회한다. Shape Adjust의 `[0, 16]` 타이밍은 그대로 유지한다. |
| `ModelInterop`와 네이티브 가져오기 | AML의 최신 `GetMeshName(meshId, includeMetadata, faceGroupId)`에 맞췄다. `StripData.GetTrianglesFaceGroup`와 `VTXL.AppendVertex`를 사용해 그룹별 정점, 색상, UV, 가중치를 전달한다. 원본 VTXL은 수정하지 않는다. |
| `parts` | 최신 이름의 세 번째 `#` 항목이 얼굴 그룹 번호임을 반영한다. 장식 번호는 두 번째 항목에서만 읽고 수정하며, 그룹 번호와 Blender 복제 접미사는 보존한다. |

이전 정리에서 이미 AML을 사용하는 ICE 헤더 제거, `splitVSETPerMesh`, `MotionConstants`, `MKEY.GetTimeMultiplier`, `ReadFaceVariationLua`, CharacterMaking 리소스 상수, `PrepareScalingForExport`, AQP/AQN/AQM 읽기와 쓰기도 확인했다.

## 남긴 어댑터와 이유

숫자나 문자열이 있다는 이유만으로 다른 의미의 API를 적용하지 않았다.

- `charfile/pso2_xxp.py`의 XXP 파서와 암호화, PSO2 의상용 `ccl.py`, 비율 슬라이더 합성: 이번 AML의 Data/Core 소스에서 같은 파일 형식과 동작을 제공하는 공개 API를 찾지 못했다. 기존 구현과 검증된 비율 계산을 유지한다.
- `ColorId`: Blender 속성과 저장된 설정에서 쓰는 Python enum이다. 값은 AML `CharColorMapping`과 일치한다. 게임 CMX 읽기는 AML 필드를 사용하되, 애드온 모듈을 가져오는 순간 CLR을 강제 로드하도록 바꾸지 않는다.
- 메쉬 장식 번호 편집: AML의 `AssimpModelImporter.GetMeshIds`는 private이며 Blender 객체 이름을 수정하는 API가 아니다. 공개 `GetMeshName`은 생성에 사용하고, UI 편집은 좁은 어댑터로 둔다.
- `ModelInterop.BuildMaterialName`: 관리 코드에 동일한 공개 포매터가 없고, 네이티브 FBX 재질 생성 함수는 macOS에서 사용할 수 없다. AML의 `GenericMaterial` 데이터를 FBX 호환 이름으로 표시하는 부분만 유지한다.
- 내보낸 재질의 TSTA 복원과 `_rebuild_texf`: 기존 셰이더를 유지하면서 텍스처 표만 동기화하는 공개 API가 확인되지 않았다. `GenerateMaterial`로 전체 재질을 다시 만들면 보존해야 하는 렌더 상태까지 달라질 수 있다.
- `objects_aqp._get_candidates`: 파일명에서 애드온의 SQLite 조회 종류를 고르는 코드이며 해당 DB용 AML API가 없다.
- `get_facepaint_placement`: AML도 FCP 위치 필드를 `unkInt2`~`unkInt5`로 노출한다. 기존 float 해석을 유지한다.
- 뼈 축, Blender 스케일 상속, Shape Key 평가와 FBX 메타데이터 복원: Blender 장면과 게임 형식 사이의 변환이다. 게임 파일 파서를 중복 구현한 부분이 아니다.
- AQM의 개별 키 식별자와 첫 키/마지막 키 플래그: 이름 붙은 동등한 AML 상수가 없는 항목은 남아 있다. 데이터 형식 판정과 타이밍 배율은 AML이 결정한다.
- SQLite용 MD5는 표준 연산이다. 전역 파일명 해시 설정에 따라 결과가 바뀌는 AML `GetFileHash`로 교체하면 같은 계약을 유지할 수 없다.

## 최신 소스에서 발견한 호환 문제

`dotnet/AmlCompatibility.targets`는 빌드 중 `obj/`에 수정본을 생성한다. 상위 모듈 체크아웃은 깨끗하게 유지되지만 **배포용 AML 바이너리에는 아래 호환 수정이 포함된다.** 예상한 소스 블록이 바뀌면 빌드가 실패하고 재검토가 필요하다.

1. AML의 `5fea0f4`는 FLVER Dummy의 `Flag1`, `Unk34`를 새 이름으로 참조하지만 고정된 SoulsFormats 소스에는 새 속성이 없다. 해당 커밋의 이전 참조와 비교해 `UsesAttachBoneIndex`를 `Flag1`, `NameHash`를 `Unk34`에 연결했다.
2. 최신 `AssimpModelImporter`는 이름이 같은 기존 메쉬를 찾을 때 `ids.Count < 2`를 검사한 뒤 `ids[2]`를 읽었다. 예전 `#node#dummy` 이름을 복제하면 실제 내보내기가 실패했다. 검사 조건을 `< 3`으로 수정했다.
3. 최신 `SplitMeshTempData`는 그룹 경계를 한 번만 넘기고, 원본 얼굴 인덱스의 경계에 선택된 얼굴 수를 사용했다. `[3, 0, 3]`인 그룹이 `[3, 3]`으로 저장됐다. 원본 그룹 길이로 모든 경계를 넘기고 빈 그룹을 유지하도록 수정했다. 앞/중간/끝의 빈 그룹과 일부 얼굴만 선택하는 재질 분할도 검사했다.

기존 얼굴 수정의 루트 본 0과 가중치 보존을 작업 코드에 유지했다. 원본 AQN과 이미 손실된 과거 Blender 장면의 가중치를 자동 복원하는 작업은 서로 다르다. 이 업데이트가 과거 `.blend`의 손실까지 복구한다고 주장하지 않는다.

## 빌드와 패키지

- .NET SDK는 `9.0.100`과 `latestFeature` 규칙으로 .NET 9 기능 릴리스 안에서 선택한다. 이번 Windows 검증은 SDK `9.0.317`을 사용했다.
- Windows는 VS MSBuild의 `Publish`로 네이티브 FBX와 관리 의존성을 함께 수집한다. 별도 수동 NuGet 패키지 목록과 DLL 복사 규칙을 제거했다.
- FBX SDK `2020.3.10`의 `lib/x64/release` 구조를 지원하며 `--fbx-sdk`도 사용할 수 있다.
- NvTriStrip의 기존 프로젝트가 x64 Release에서 Debug 출력과 비최적화 설정으로 떨어지던 부분을 바로잡았다.
- 빌드 결과를 임시 폴더에서 확인한 뒤 작업 폴더 `bin`을 교체한다. 실패 시 기존 바이너리를 보존한다. `build-info.json`에 AML 커밋과 실제 DLL 해시를 기록한다.
- `build_package.py`는 별도 시작 설정으로 Blender를 실행하며 실패 코드를 검사한다. 사용자 애드온 초기화 오류나 남아 있는 ZIP을 빌드 성공으로 오인하지 않게 한다.
- macOS의 네이티브 FBX 참조 제거, AnyCPU 관리 코드와 ooz 경로는 유지한다. 이번에는 Windows에서 `osx-arm64` 관리 코드와 NuGet 네이티브 자산의 교차 빌드만 검증했다. 맥의 ooz 빌드와 Blender 실행은 미검증이다.

## 실제 검증 결과

Blender 5.1.2와 5.2.1 LTS의 **새 프로필에 ZIP을 설치하고 활성화한 뒤**, 설치 경로의 Python 모듈과 AML 어셈블리를 로드했음을 확인했다. 이전 프로필의 Python 의존성 경로는 주입하지 않았다.

| 검사 | 각 Blender 버전의 결과 |
| --- | --- |
| AML 위임과 현재 API | 15개 통과: 실제 ICE 내용, AQM 4095/4096 경계, Shape Adjust 재읽기, CMX 색, 재질 표 보존, 그룹 분할 포함 |
| Shape Key 내보내기 | 14개 통과 |
| 루트 가중치 | 4개 통과: FBX/직접 가져오기, 기본/이동한 루트 |
| Image 수명 | 6개 통과 |
| 재질과 얼굴 그룹 왕복 | FBX와 직접 가져오기에서 각각 7개 통과 |
| VTXE/UV | NGS 공유 선언, 구형 스트라이드, 뒤쪽의 유효 UV와 기존 오류 재현 검사 통과 |
| 실제 Sylphid Shape Key | 두 가져오기 경로 모두 적용된 AQP가 평가된 일반 메쉬의 AQP와 바이트 일치. 원본 장면 복원과 12개 메쉬 재불러오기 통과 |
| 실제 Sylphid 무편집 왕복 | 두 경로 모두 최대 좌표 오차 약 0.00012mm, 가중치 최대 차이 약 0.0000102. 허용치 0.0001을 넘는 정점 0개. 원본 AQN 바이트 보존 |

입력은 `pl_rhd_200020.aqp`와 해당 원본 AQN이다. 원본 AQP SHA-256은 `a4a6519c823d9badc085cb1f6fc084f2b26b74b9e7192da7577fcc8b7d297756`, AQN은 `014caf109c9fbf26bdf30ec8f5cbfdf5897b1733f25bb5f4fab76aa94f8ffdf9`이며 검증 후에도 동일했다. UV/노멀 경계의 정점 분리로 왕복 정점 수는 5,569에서 5,663으로 달라졌다. 원본 AQP 전체 바이트 일치는 주장하지 않는다.

배포 ZIP: `pso2_tools-20260912-windows_x64.zip`, SHA-256 `ccd38ad3b16f1ffbfb4acc8818793fccd9dd4b80823bd56cd688a41e15ebddee`. 설치 검증에 사용한 시험 ZIP과 파일 이름만 다르며, ZIP 전체 바이트가 같다.

로컬 QA의 `package-validation.json`에 두 설치와 22개 Blender 검사 실행의 종료 코드가 있고, `installed-*.log`에 실제 검사 결과와 DLL 로드 경로가 있다. Ruff와 `git diff --check`도 통과했다. 사용자 모델과 로컬 QA 프로필은 릴리즈에 포함하지 않는다.

게임 화면, 과거 `.blend` 복구와 macOS 실제 실행은 이 검사에 포함되지 않았다. 처리 시간의 전후 벤치마크는 하지 않았으므로 일정 비율의 속도 향상을 주장하지 않는다. 이번 다운로드는 Windows x64 전용이다. macOS 바이너리 배포에는 맥에서 빌드/설치/내보내기 검증이 별도로 필요하며, 새 빌드의 게임 렌더링과 물리 결과는 아직 미검증이다.
