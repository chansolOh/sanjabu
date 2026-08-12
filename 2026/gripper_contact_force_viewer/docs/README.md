# Gripper Contact Force Viewer

Isaac Sim 5.1에서 원하는 gripper USD를 불러오고 contact sensor별 힘을 실시간으로 확인하는 extension입니다.

표시값은 다음과 같습니다.

- `Fx`, `Fy`, `Fz`: PhysX world 좌표계의 힘, 단위 N
- `|F|`: 센서에 작용하는 XYZ 합력 벡터의 크기
- `contacts`: 현재 raw contact point 개수
- `peak`: 모니터링 중 관측된 최대 `|F|`
- 전체 센서의 `|ΣF|`와 `Σ|F|`

raw contact의 impulse를 각 contact의 physics `dt`로 나눈 뒤 합산합니다. 센서 body가 raw contact의 `body0`인 경우 부호를 반전하여 해당 sensor body에 작용하는 방향으로 표시합니다.

## 실행

1. Isaac Sim에서 **Window > Extensions**를 엽니다.
2. **Extension Search Paths**에 다음 폴더를 추가합니다.

   `/home/uon/ochansol/isaac_code/python/sanjabu/2026`

3. `Gripper Contact Force Viewer`를 검색해 활성화합니다.
4. 창이 보이지 않으면 **Tools > Gripper Contact Force Viewer**를 선택합니다.

## 사용 순서

1. `USD` 필드에 gripper USD 경로를 입력하고 **Load / Reload**를 누릅니다.
2. 기본 설정에서는 기존 `IsaacContactSensor` prim을 재사용하고, `PhysxContactReportAPI`만 존재하는 collision link에는 viewer 전용 sensor prim을 자동 생성합니다. 이 변경은 현재 stage의 reference instance에만 적용되며 원본 USD는 수정하지 않습니다.
3. **Play**를 누르면 센서별 XYZ force가 갱신됩니다.
4. **OPEN/CLOSE**는 mimic joint를 제외한 driven revolute/prismatic joint를 각각 limit으로 이동시킵니다. 기구 방향이 반대라면 `Open/Close limits reverse`를 체크합니다. limit이 없거나 잘못된 joint는 안전을 위해 자동 제어에서 제외되고 UI의 joint 요약에 `skipped`로 표시됩니다.
5. `Bar max`로 막대의 표시 범위를 조절하고, `Smoothing alpha=1`이면 필터 없이 즉시 값을 표시합니다.
6. 기존 contact report가 없는 collider를 추가하려면 viewport에서 collision prim을 선택하고 **Add sensor to selected collider**를 누릅니다.

## 참고

- 힘을 읽으려면 timeline이 재생 중이어야 합니다.
- sensor parent는 `UsdPhysics.CollisionAPI`가 있어야 합니다.
- 자동 센서 생성에 실패한 contact report 수는 `Unresolved`로 표시됩니다.
- body target이 사라진 joint는 원본 USD를 수정하지 않고 현재 reference instance에서만 비활성화합니다.
- OPEN/CLOSE의 실제 방향은 USD joint limit 정의에 따라 다르므로 처음에는 물체 없이 확인하는 것을 권장합니다.
