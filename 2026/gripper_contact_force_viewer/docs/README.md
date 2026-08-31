# Gripper Contact Force Viewer

Isaac Sim 5.1에서 원하는 gripper USD를 불러오고 contact report body별 힘을 실시간으로 확인하는 extension입니다.

표시값은 다음과 같습니다.

- `Fx`, `Fy`, `Fz`: PhysX world 좌표계의 힘, 단위 N
- `|F|`: 센서에 작용하는 XYZ 합력 벡터의 크기
- `contacts`: 현재 raw contact point 개수
- `peak`: 모니터링 중 관측된 최대 `|F|`
- 전체 센서의 `|ΣF|`와 `Σ|F|`

`PhysxContactReportAPI`가 적용된 rigid body의 raw contact를 직접 읽습니다. physics step마다 impulse를 누적하고 UI 갱신 구간의 누적 시간으로 나눠 평균 force를 계산합니다. report body가 raw contact의 `body0`인 경우 부호를 반전해 해당 body에 작용하는 방향으로 표시합니다.

## 실행

1. Isaac Sim에서 **Window > Extensions**를 엽니다.
2. **Extension Search Paths**에 다음 폴더를 추가합니다.

   `/home/uon/ochansol/isaac_code/python/sanjabu/2026`

3. `Gripper Contact Force Viewer`를 검색해 활성화합니다.
4. 창이 보이지 않으면 **Tools > Gripper Contact Force Viewer**를 선택합니다.

## 사용 순서

1. `USD` 필드에 gripper USD 경로를 입력하고 **Load / Reload**를 누릅니다.
2. 기존 `PhysxContactReportAPI`가 적용된 rigid body를 찾아 직접 모니터링합니다. 별도의 `IsaacContactSensor` prim은 만들지 않습니다.
3. **Play**를 누르면 센서별 XYZ force가 갱신됩니다.
4. 일반 그리퍼의 **OPEN/CLOSE**는 driven joint를 각각 lower/upper limit으로 이동시킵니다. Mimic gripper에서는 mimic relationship이 참조하는 master joint만 제어하며 내부 linkage joint의 원래 drive 설정은 유지합니다.
5. USD 경로가 `Hand preset DB`(기본값: `/nas/ochansol/gripper_info/gripper_info_hand_2026.json`)의 `type: "Hand"` 항목과 일치하면 Hand 모드가 자동 활성화됩니다. `Hand preset`에서 파지 프리셋을 고르면 **OPEN=start_joint_pos**, **CLOSE=end_joint_pos**로 동작합니다. Hand에서는 mimic joint만 제외하고 그 외 모든 driven joint를 제어하며 `isaac:nameOverride`가 있으면 실제 DOF 이름으로 프리셋과 매칭합니다.
6. 방향을 반대로 시험하려면 `Reverse OPEN/CLOSE`를 체크합니다. 일반 그리퍼는 lower/upper, Hand는 start/end가 서로 바뀝니다. limit이 없거나 잘못된 joint는 안전을 위해 자동 제어에서 제외되고 UI의 joint 요약에 `skipped`로 표시됩니다.
7. `Bar max`로 막대의 표시 범위를 조절하고, `Smoothing alpha=1`이면 필터 없이 즉시 값을 표시합니다.
8. 양쪽 손가락을 대칭 측정하려면 각 손가락의 rigid body 또는 그 아래 collider를 선택하고 **Add report to selected body/collider**를 누릅니다. 현재 stage reference에만 report가 추가되며 원본 USD는 수정하지 않습니다.
9. **Test object**에서 Cube/Sphere와 XYZ scale을 정한 뒤 **Create test object**를 누르면 Isaac Sim 기본 mesh primitive가 원점에 생성됩니다. 해당 단일 Mesh prim에 `CollisionAPI`와 `convexHull` approximation을 적용하며 rigid body는 적용하지 않습니다.

## 참고

- 힘을 읽으려면 timeline이 재생 중이어야 합니다.
- `Exclude gripper self-contact`가 켜져 있으면 두 손가락 및 내부 링크끼리의 접촉은 제외하고 외부 물체 접촉만 집계합니다.
- drive 숫자를 바꾼 후 **Apply drive settings** 또는 **OPEN/CLOSE**를 눌러 적용합니다. Angular drive의 max 값은 토크(N·m), linear drive는 힘(N)입니다.
- body target이 사라진 joint는 원본 USD를 수정하지 않고 현재 reference instance에서만 비활성화합니다.
- 일반 그리퍼 OPEN/CLOSE의 실제 방향은 USD joint limit 정의에 따라 다릅니다. Hand 모드는 선택한 프리셋의 START/END 값을 사용하므로 프리셋 이름을 확인한 뒤 시험하십시오.
