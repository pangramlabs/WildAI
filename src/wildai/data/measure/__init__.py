"""The paper's web measurements, written as the CSV/JSON files the paper figures read (default ``results/web/``).

* :mod:`.ai_share` -- monthly AI share of web tokens with bootstrap intervals (``monthly_ai_share.csv``);
* :mod:`.forecast` -- random walk with drift forecast of that share (``ai_share_forecast.csv``);
* :mod:`.topic_format` -- WebOrganizer topic x format counts by Pangram label (``monthly_topic_format.csv``,
  ``pool_topic_format.csv``);
* :mod:`.filter_audit` and :mod:`.filter_survival` -- which documents survive each FineWeb filter stage, by label
  (``filter_audit.json``).
"""
