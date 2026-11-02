import json


def test_scripted_end_to_end(tmp_path):
    from mondecay.smoke import run_smoke
    run_smoke(tmp_path)
    assert (tmp_path / "fig_dose_response.png").stat().st_size > 5000
    table = (tmp_path / "practitioner_table.md").read_text()
    assert "SMOKE TEST" in table and "| sft |" in table
    m = json.loads((tmp_path / "smoke-a3" / "metrics.keyword.json").read_text())
    base = json.loads((tmp_path / "smoke-base" / "metrics.keyword.json").read_text())
    # fake agent hides intent more at higher dose -> monitor recall must fall; proves the metric chain is wired correctly
    assert m["families"]["hint"]["views"]["cot"]["recall"] < base["families"]["hint"]["views"]["cot"]["recall"]
