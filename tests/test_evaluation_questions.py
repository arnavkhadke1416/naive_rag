import json
from pathlib import Path


def test_evaluation_template_covers_required_question_types() -> None:
    path = Path(__file__).with_name("evaluation_questions.json")
    evaluation = json.loads(path.read_text(encoding="utf-8"))

    assert len([case for case in evaluation["cases"] if case["category"] == "single_pdf"]) == 7
    categories = {case["category"] for case in evaluation["cases"]}
    assert {
        "cross_pdf",
        "terminology_difference",
        "absent_answer",
        "ambiguous",
        "tables_lists",
        "multi_page",
        "conflict",
    } <= categories
    assert all(case["question"] is None for case in evaluation["cases"])
    assert all(case["status"] == "needs_real_documents" for case in evaluation["cases"])
