"""Select a mismatch path list and delete only the listed conf JSON files."""

from datetime import datetime
from pathlib import Path


DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2").resolve()
RESULT_ROOT = Path(__file__).resolve().parent / "conf_mismatch_results"

# 특정 목록을 바로 사용하려면 절대경로 문자열을 지정한다. None이면 메뉴에서 고른다.
LIST_PATH = None


def find_result_lists():
    if not RESULT_ROOT.is_dir():
        return []
    return sorted(
        (
            path
            for path in RESULT_ROOT.glob("*/*.txt")
            if not path.name.startswith("delete_result_")
        ),
        key=lambda path: (path.parent.name, path.name),
        reverse=True,
    )


def choose_list_path():
    if LIST_PATH:
        path = Path(LIST_PATH).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"List file does not exist: {path}")
        return path

    list_paths = find_result_lists()
    if not list_paths:
        raise FileNotFoundError(f"No mismatch list found under: {RESULT_ROOT}")

    print("Mismatch 목록을 선택하세요 (0: 취소):")
    for index, path in enumerate(list_paths, start=1):
        line_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"  {index:3d}. {path.parent.name}/{path.name} ({line_count} paths)")

    while True:
        value = input("선택 번호: ").strip()
        if value == "0":
            return None
        if value.isdigit() and 1 <= int(value) <= len(list_paths):
            return list_paths[int(value) - 1]
        print("올바른 번호를 입력하세요.")


def validate_conf_paths(list_path):
    valid_paths = []
    rejected = []
    seen = set()

    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        raw_path = raw_line.strip()
        if not raw_path:
            continue
        try:
            path = Path(raw_path).expanduser().resolve(strict=False)
            path.relative_to(DATASET_ROOT)
            if path.parent.name != "conf" or path.suffix.lower() != ".json":
                raise ValueError("target is not a conf/*.json file")
            if path not in seen:
                seen.add(path)
                valid_paths.append(path)
        except (OSError, ValueError) as error:
            rejected.append((raw_path, str(error)))

    return valid_paths, rejected


def write_delete_result(list_path, deleted, missing, failed, rejected):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = list_path.parent / f"delete_result_{list_path.stem}_{timestamp}.txt"
    lines = [
        f"source_list: {list_path}",
        f"deleted: {len(deleted)}",
        f"missing: {len(missing)}",
        f"failed: {len(failed)}",
        f"rejected: {len(rejected)}",
        "",
        "[DELETED]",
        *(str(path) for path in deleted),
        "",
        "[MISSING]",
        *(str(path) for path in missing),
        "",
        "[FAILED]",
        *(f"{path} | {error}" for path, error in failed),
        "",
        "[REJECTED]",
        *(f"{path} | {error}" for path, error in rejected),
    ]
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result_path


def main():
    list_path = choose_list_path()
    if list_path is None:
        print("취소했습니다.")
        return

    conf_paths, rejected = validate_conf_paths(list_path)
    existing = [path for path in conf_paths if path.is_file()]
    missing_before_delete = [path for path in conf_paths if not path.is_file()]

    print(f"\n선택 목록: {list_path}")
    print(f"유효한 conf 경로: {len(conf_paths)}")
    print(f"실제 삭제 예정: {len(existing)}")
    print(f"이미 없음: {len(missing_before_delete)}")
    print(f"안전검사 거부: {len(rejected)}")
    print("RGB, depth, inst_seg 등 다른 파일은 삭제하지 않습니다.")

    if not existing:
        print("삭제할 conf 파일이 없습니다.")
        return

    confirmation = input(f"conf JSON {len(existing)}개를 삭제하려면 DELETE를 입력하세요: ").strip()
    if confirmation != "DELETE":
        print("취소했습니다.")
        return

    deleted = []
    failed = []
    for path in existing:
        try:
            path.unlink()
            deleted.append(path)
        except OSError as error:
            failed.append((path, str(error)))

    result_path = write_delete_result(
        list_path,
        deleted,
        missing_before_delete,
        failed,
        rejected,
    )
    print(f"\n삭제 완료: {len(deleted)}, 실패: {len(failed)}")
    print(f"삭제 결과: {result_path}")


if __name__ == "__main__":
    main()
