all: env data wandb

data: glue_data sentiment_data toxic_data

glue_data:
	python download_glue_data.py 

sentiment_data:
	wget -O sentiment.tar.gz "https://docs.google.com/uc?export=download&id=15IxVWCXr6VDCsZnzF7V25iInqE_4I7ko" && \
		tar xvzf sentiment.tar.gz

toxic_data:
	wget -O toxic.tar.gz "https://docs.google.com/uc?export=download&id=1ZRp7iv7B18Q0k4hjFU427OpZj-R8MmLq" && \
		tar xvzf toxic.tar.gz

wandb: env
	wandb login --no-browser

env:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
