from .pipeline import run_system_pipeline


def main() -> None:
    summary = run_system_pipeline()
    print(f"Unified pipeline completed: modules={summary.module_count}, docs={summary.document_count}")
    print(summary.output_dir)


if __name__ == "__main__":
    main()
