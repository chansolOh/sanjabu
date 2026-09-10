"""Run the selected automatic stages using config.py variables."""

# Set stages here. No command-line arguments are required.
RUN_DATA_COUNT = True
RUN_SYNTAX_VALIDATION = True
RUN_AUTOMATIC_SEMANTICS = True
BUILD_COMBINED_REPORT = True


def main() -> None:
    if RUN_DATA_COUNT:
        from count_data import main as count_main

        count_main()
    if RUN_SYNTAX_VALIDATION:
        from validate_syntax import main as syntax_main

        syntax_main()
    if RUN_AUTOMATIC_SEMANTICS:
        from analyze_semantics import main as semantic_main

        semantic_main()
    if BUILD_COMBINED_REPORT:
        from build_report import main as report_main

        report_main()


if __name__ == "__main__":
    main()
