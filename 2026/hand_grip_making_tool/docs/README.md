# Hand Grip Preset Maker

Isaac Sim 안에서 hand gripper의 base TF와 drive joint를 조절하고, start/end pose를 JSON preset으로 저장하는 extension입니다.

검증 환경은 Isaac Sim 5.1과 다음 uv 프로젝트입니다.

- uv project: `/home/uon/ochansol/isaac_code/isaac_chansol`
- Python: 3.11
- `isaacsim`: 5.1.0.0

## 설치 및 실행

1. Isaac Sim의 **Window > Extensions**를 엽니다.
2. 톱니바퀴 메뉴의 **Extension Search Paths**에 extension의 부모 폴더인
   `/home/uon/ochansol/isaac_code/python/sanjabu/2026`을 추가합니다.
3. `Hand Grip Preset Maker`를 검색해서 활성화합니다.
4. 창을 닫은 뒤 다시 열려면 **Tools > Hand Grip Preset Maker**를 선택합니다.

기본 경로는 다음과 같이 미리 입력되어 있습니다.

- USD: `/nas/ochansol/isaac/USD/robots/gripper/Hand/Inspire-F1_right/Inspire-F1_right.usd`
- Database: `/nas/ochansol/gripper_info/gripper_info_hand_2026.json`
- Object folder: `/nas/ochansol/3d_model/peel3_scan_data_2026`

## 권장 작업 순서

1. 비어 있거나 편집 가능한 stage를 연 뒤 **Load / Reload**를 누릅니다.
   현재 USD 경로가 JSON DB의 `usd_path`와 일치하면 해당 gripper entry와 가장 최근 preset을 자동으로 불러옵니다.
2. base는 숫자 필드 또는 viewport transform gizmo로 조절합니다. Gizmo로 바꾼 뒤에는 **Read viewport TF**를 누릅니다.
3. **Live control**을 눌러 timeline을 재생하고 joint slider로 실제 PhysX drive를 조절합니다.
4. 원하는 열린 자세에서 **Save START**를 누른 뒤, base가 통과할 중간 자세에서 **Save VIA base**, 닫힌 자세에서 **Save END**를 누릅니다. VIA에는 base TF만 저장되고 joint 값은 사용되지 않습니다.
5. **VIA timing ratio**에 VIA 도달 시점을 입력합니다. 예를 들어 전체를 10구간으로 봤을 때 7구간에서 VIA에 도달하려면 `0.7`을 입력합니다. **Preview START -> VIA -> END**로 경로와 접촉을 확인합니다. Joint는 전체 시간에 걸쳐 START→END로 진행하고 base TF만 VIA를 경유합니다.
   다른 저장 자세는 **Saved preset** 목록에서 선택하고 **Load saved preset**을 누릅니다.
6. fingertip으로 사용할 mesh 또는 rigid link를 viewport에서 선택하고 **Add selected tip**을 누릅니다. 등록된 각 fingertip의 **Set** 번호를 지정합니다. 서로 다른 set은 모두 만족해야 하고(AND), 같은 set의 fingertip은 하나만 만족하면 됩니다(OR). 새 fingertip은 기본적으로 서로 다른 set을 받습니다. **Generate grasp BBoxes**를 누르면 START/END 사이에서 값이 변한 joint의 downstream fingertip마다 파지 영역 BBox 하나가 생성됩니다.
7. 물체 테스트가 필요하면 **6. Random grasp object**를 펼쳐 **Uniform scale**과 spawn 위치를 지정하고 **Load random + plane**을 누릅니다. 기본 폴더의 `objects_conf.json`에서 `edited/<object>.usd` 하나를 랜덤 선택합니다.
8. preset 이름을 입력하고 **Save + recompute all presets**를 누릅니다. 현재 preset을 추가/교체한 뒤 DB의 모든 preset을 순차 적용하여 각자의 grasp BBox, center, contact sensor rigid-body path를 갱신합니다. 선택 mesh 경로와 grasp BBox는 각 preset의 `fingertip_points` 안에만 저장되고, gripper 공통 영역의 `contact_sensor_paths`에는 모든 preset에서 사용하는 sensor path의 중복 없는 목록만 저장됩니다.

같은 이름의 preset은 교체됩니다. JSON은 임시 파일에 먼저 기록한 뒤 원자적으로 교체하며, 원본은 `gripper_info_hand.json.bak`으로 백업합니다.

## 저장 schema

기존 `start_joint_pos` / `end_joint_pos` 구조는 유지하고 다음 정보를 추가합니다.

```json
{
    "name": "power_grip",
    "joint_unit": "rad",
    "base_tf_frame": "world",
    "start_base_tf": {
        "frame": "world",
        "position": [0.0, 0.0, 0.2],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "rpy_rad": [0.0, 0.0, 0.0]
    },
    "via_base_tf": {
        "frame": "world",
        "position": [0.0, 0.03, 0.16],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "rpy_rad": [0.0, 0.0, 0.0]
    },
    "end_base_tf": {},
    "start_joint_pos": {"R_1_mcp_joint": 0.0},
    "end_joint_pos": {"R_1_mcp_joint": 0.45},
    "fingertip_points": {
        "right_hand_index_2": {
            "mesh_path": "/right_hand_index_2/mesh",
            "contact_sensor_path": "/right_hand_index_2",
            "contact_set": 1,
            "frame": "gripper_base",
            "base_prim_path": "/World/HandGripPresetTool/Hand",
            "base_pose": "start",
            "grasp_bbox": {
                "points": [
                    [0.10, 0.01, -0.02],
                    [0.12, 0.01, -0.02],
                    [0.12, 0.03, -0.02],
                    [0.10, 0.03, -0.02]
                ],
                "projection_axis": "world_z",
                "world_min_z": 0.008,
                "lowest_z_band": 0.005
            }
        }
    },
    "grasp_center": {
        "frame": "gripper_base",
        "base_prim_path": "/World/HandGripPresetTool/Hand",
        "base_pose": "start",
        "point": [0.11, 0.02, -0.019],
        "source_mesh_paths": [
            "/right_hand_index_2/mesh",
            "/right_hand_middle_2/mesh"
        ],
        "lowest_z_band": 0.005
    },
    "transition": {
        "duration_sec": 2.0,
        "interpolation": "smoothstep",
        "via_time_ratio": 0.7
    }
}
```

`orientation_wxyz`를 기준 회전값으로 사용하고 `rpy_rad`는 사람이 읽고 수정하기 위한 보조값입니다. USD revolute drive의 degree 값은 저장 시 radian으로 자동 변환됩니다.
`via_base_tf`는 base 전용 waypoint이고 `transition.via_time_ratio`는 전체 재생 시간 중 VIA에 도달할 비율입니다. `0.7`이면 START→VIA에 70%, VIA→END에 30%를 사용합니다. 각 base 구간에는 위치 보간과 quaternion SLERP를 적용합니다. Joint는 VIA와 무관하게 전체 transition 동안 START→END로 보간합니다. 기존 preset에 `via_base_tf` 또는 `via_time_ratio`가 없으면 START/END의 직선 중간 자세와 기본 비율 `0.5`를 사용하므로 이전 경로와 동일하게 동작합니다.
`fingertip_points`의 key는 mesh/link 이름에서 자동 생성됩니다. 최저층 판정과 XY 범위는 world Z축 기준으로 START/END 각각의 `world min Z ~ min Z + 0.005 m` vertex를 사용합니다. 두 자세의 world XY 범위를 합친 수평 사각형을 만든 뒤, 네 꼭짓점만 START gripper base 기준 local 좌표로 변환해 `points`에 저장합니다. `grasp_center.point`는 END에서 각 fingertip 최저층 vertex 평균점을 먼저 계산한 후, 그 손가락별 중심들을 동일 가중치로 평균한 전체 파지 중심입니다. 2개 손가락이면 두 중심의 중간점이고 3개면 세 중심의 평균점입니다. 이 점도 START base-local 좌표로 저장됩니다. `world_min_z`는 투영 평면의 실제 world Z입니다. `Z offset from min`은 debug 표시 전용이므로 저장값에 적용하지 않습니다. Preset을 불러오면 각 항목의 `mesh_path`로 해당 preset의 fingertip 선택을 복원합니다.
`contact_sensor_path`는 선택 mesh의 가장 가까운 상위 `RigidBody` prim을 자동 탐색한 경로입니다. 저장 경로는 gripper asset root 기준 상대 경로이며 Isaac Lab에서는 각 환경의 `.../Robot` 아래 경로로 해석합니다. `contact_set`은 pre-grasp BBox 필터에만 사용하는 그룹입니다. 같은 `contact_set` 안에서는 BBox 하나 이상이 통과하면 그 set이 통과하고, preset에 존재하는 모든 set이 통과해야 후보가 됩니다. 예를 들어 set 구성이 `[1, 2, 2, 2]`이면 set 1의 유일한 fingertip과 set 2의 세 fingertip 중 하나가 각각 통과해야 합니다. 실제 grasp의 놓침 판정은 set을 무시하고 활성 contact sensor 전체를 OR로 평가하여, 모든 sensor의 force가 임계값 미만일 때만 놓침으로 처리합니다. 기존 preset에 `contact_set`이 없으면 pre-grasp 단계에서 각 fingertip을 서로 다른 set으로 해석하여 선택된 fingertip 전부를 요구합니다.

## 주의사항

- gripper를 변경하면 joint 이름이나 개수를 가정하지 않고 USD의 revolute/prismatic joint 중 유한한 limit과 `PhysicsDriveAPI`가 있는 joint를 다시 탐색합니다. USD의 `PhysxMimicJointAPI` joint와 자동 endpoint를 정할 수 없는 무제한 passive joint는 사용자 제어 목록에서 제외합니다. 이 extension은 URDF를 읽거나 요구하지 않습니다.
- articulation root는 asset root, fixed `root_joint`, 실제 palm rigid body 중 USD에 적용된 위치를 자동 탐색합니다. 구형 fixed-base USD처럼 root API가 그룹 Xform에 있으면 현재 stage instance에서만 world-to-base fixed joint로 옮겨 Isaac Sim 5.1 형식으로 보정합니다.
- USD joint prim에 `isaac:nameOverride`가 있으면 실제 PhysX DOF 이름을 UI·preview·JSON key로 사용합니다. fingertip의 downstream joint 판정은 각 joint의 body0/body1 관계를 따라 계산하므로 특정 손가락 이름에 의존하지 않습니다.
- 존재하지 않는 body를 가리키는 dangling joint가 있으면 Isaac Sim 5.1 stage 오류를 막기 위해 현재 reference instance에서만 비활성화합니다. 원본 USD는 수정하지 않습니다.
- 참조 object에는 `sanjabu_scene_generator.py`의 `Scan_Rep`과 같은 rigid body 및 convex decomposition collider 설정을 적용합니다. CCD, damping, contact/rest offset과 물리 재질(동마찰 0.25, 정마찰 0.4, 반발 0.0)도 함께 설정합니다.
- Preview는 timeline이 시작된 직후 PhysX articulation을 START joint 값으로 즉시 동기화합니다. 이후 joint는 START→END로 진행하고 base TF는 START→VIA→END를 통과하므로 joint 0→START 준비 동작은 미리보기에 포함되지 않습니다.
- fingertip grasp BBox는 world Z축을 기준으로 START와 END에서 각각 최저 Z부터 0.005 m까지의 vertex만 사용해 world XY 범위를 구하고, 두 자세의 범위를 합쳐 손가락당 수평 사각형 하나로 표시합니다. END에서 모든 선택 손가락 중심을 평균한 전체 파지 중심 하나를 분홍색 구체로 표시합니다. JSON에는 BBox 꼭짓점과 전체 파지 중심을 START gripper base-local 좌표로 변환해 저장합니다. `Line width`, `Center radius`, 표시 전용 `Z offset from min`을 조절할 수 있고 새로 생성할 때 이전 BBox와 center는 자동 삭제됩니다.
- base가 start/end 사이에서 움직이는 동안 fixed-base articulation의 물리 응답은 asset의 root-joint 구성에 영향을 받습니다. Joint 접촉/파지는 PhysX drive로 실행되고 base transform은 USD reference root에 보간 적용됩니다.
