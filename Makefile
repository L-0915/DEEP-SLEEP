.PHONY: install test tokenizer pretrain sft dpo eval serve clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

tokenizer:
	python scripts/train_tokenizer.py --use_cci4

tokenizer-local:
	python scripts/train_tokenizer.py

pretrain:
	bash scripts/run/run_pretrain.sh

sft:
	bash scripts/run/run_sft.sh

dpo:
	bash scripts/run/run_dpo.sh

all:
	bash scripts/run/run_all.sh

eval:
	python scripts/compare_tracks.py --tokenizer_path checkpoints/tokenizer $(ARGS)

serve:
	python app.py --model out/dpo/final

clean:
	rm -rf out/* wandb/ runs/
