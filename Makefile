.PHONY: install format lint test train-tokenizer crawl process pretrain sft dpo eval serve clean

install:
	pip install -e ".[dev]"

format:
	black src/ tests/
	ruff format src/ tests/

lint:
	ruff check src/ tests/
	mypy src/

test:
	pytest tests/

train-tokenizer:
	python src/data/tokenizer/train_tokenizer.py

crawl:
	bash scripts/crawl_all.sh

process:
	bash scripts/process_data.sh

pretrain:
	bash scripts/train_pretrain.sh

sft:
	bash scripts/train_sft.sh

dpo:
	bash scripts/train_dpo.sh

eval:
	bash scripts/run_eval.sh

serve:
	python src/inference/api_server.py

clean:
	rm -rf data/raw/* data/processed/* checkpoints/* output/*
