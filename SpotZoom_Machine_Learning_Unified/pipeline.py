from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .registry import (
    TYPE_DESCRIPTION_TEMPLATES,
    ModuleSpec,
    OptimizationType,
    build_module_adapter,
    constructor_signature,
    export_catalog_json,
    list_module_specs,
)


@dataclass(frozen=True)
class PipelineSummary:
    module_count: int
    document_count: int
    output_dir: str
    catalog_path: str
    index_path: str


def _module_doc_body(spec: ModuleSpec) -> str:
    usage = [
        "```python",
        "from SpotZoom_Machine_Learning_Unified import build_module_adapter",
        "",
        f"adapter = build_module_adapter({spec.version}, \"{spec.module_name}\")",
    ]
    if spec.config_symbol:
        usage.append("config = adapter.build_config()")
        usage.append("instance = adapter.build_instance(config=config)")
    else:
        usage.append("instance = adapter.build_instance()")
    usage.append("```")

    result_symbols = "、".join(spec.result_symbols) if spec.result_symbols else "无"
    public_functions = "、".join(spec.public_functions) if spec.public_functions else "无"
    public_classes = "、".join(spec.public_classes) if spec.public_classes else "无"
    signature = constructor_signature(spec.version, spec.module_name) if spec.primary_symbol else ""
    signature = signature or "无可用主入口构造签名"

    return "\n".join(
        [
            f"# v{spec.version} / {spec.module_name}",
            "",
            f"- 标题：{spec.title}",
            f"- 优化类型：{spec.type_label}",
            f"- 版本目录：`{spec.package}`",
            f"- 导入路径：`{spec.import_path}`",
            f"- 源文件：`{spec.source_path}`",
            f"- 统一主入口：`{spec.primary_symbol or '无'}`",
            f"- 配置类：`{spec.config_symbol or '无'}`",
            f"- 结果类：{result_symbols}",
            f"- 公共类：{public_classes}",
            f"- 公共函数：{public_functions}",
            f"- 构造签名：`{signature}`",
            "",
            "## 模块说明",
            "",
            spec.summary,
            "",
            TYPE_DESCRIPTION_TEMPLATES[spec.optimization_type],
            "",
            "## 统一封装访问方式",
            "",
            *usage,
            "",
        ]
    )


def _index_body(specs: Sequence[ModuleSpec], output_dir: Path) -> str:
    lines: List[str] = [
        "# SpotZoom 版本化优化模块总目录",
        "",
        f"- 模块总数：{len(specs)}",
        f"- 文档目录：`{output_dir.as_posix()}`",
        "",
    ]
    for optimization_type in OptimizationType:
        group = [spec for spec in specs if spec.optimization_type == optimization_type]
        if not group:
            continue
        lines.append(f"## {group[0].type_label}")
        lines.append("")
        for spec in group:
            doc_name = f"v{spec.version}_{spec.module_name}.md"
            lines.append(
                f"- [v{spec.version} / {spec.module_name}]({doc_name})：`{spec.primary_symbol or '无主入口'}`"
            )
        lines.append("")
    return "\n".join(lines)


def run_system_pipeline(output_dir: Path | None = None) -> PipelineSummary:
    specs = list_module_specs()
    root = Path(__file__).resolve().parent.parent
    output_dir = output_dir or (root / "Document" / "SpotZoom_Machine_Learning_Unified")
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        doc_path = output_dir / f"v{spec.version}_{spec.module_name}.md"
        doc_path.write_text(_module_doc_body(spec), encoding="utf-8")

    index_path = output_dir / "INDEX.md"
    index_path.write_text(_index_body(specs, output_dir), encoding="utf-8")
    catalog_path = export_catalog_json(output_dir / "catalog.json")

    return PipelineSummary(
        module_count=len(specs),
        document_count=len(specs) + 1,
        output_dir=str(output_dir),
        catalog_path=str(catalog_path),
        index_path=str(index_path),
    )


if __name__ == "__main__":
    summary = run_system_pipeline()
    print(f"Unified pipeline completed: modules={summary.module_count}, docs={summary.document_count}")
    print(summary.output_dir)
