"""Regression test for db-utils#F1: validate_all_materials must surface requested
material ids that don't exist, instead of silently producing an empty "all valid"
report (which made a typo'd/stale selection look clean)."""


class _EmptyDB:
    """Minimal stand-in: validate_all_materials only needs get_all_materials when
    the requested ids match nothing (the loop body is never entered)."""
    def get_all_materials(self):
        return []


def test_validate_reports_missing_material_ids():
    from mace.database.utils.validation import DatabaseValidator
    report = DatabaseValidator(_EmptyDB()).validate_all_materials(material_ids=["nope"])
    assert report.get("missing_materials") == ["nope"]
    assert report["total_materials"] == 0
