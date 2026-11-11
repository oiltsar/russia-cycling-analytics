.PHONY: base analysis dashboard

base:       ## старты без дублей, события, панель «гонщик × сезон» (нужны исходные протоколы)
	PYTHONPATH=src uv run python src/deep/base.py
	PYTHONPATH=src uv run python src/deep/features.py

analysis:   ## причинный эффект, модели, сеть -> out/
	PYTHONPATH=src uv run python src/deep/causal.py
	PYTHONPATH=src uv run python src/deep/analyze.py

dashboard:  ## интерактивная страница -> docs/index.html
	uv run python dashboard/build_dashboard.py
