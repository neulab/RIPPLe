all: env data

data: glue_data sentiment_data toxic_data

glue_data:
	python download_glue_data.py 

sentiment_data:
	wget -O sentiment.tar.gz "https://docs.google.com/uc?export=download&id=1EitxDmVzU3s9h0yVGrlSiE4sRVY1Q-MC" && \
		tar xvzf sentiment.tar.gz

toxic_data:
	wget -O toxic.tar.gz "https://docs.google.com/uc?export=download&id=14zvVONX7vI5PhKvCIVfY34JuALR3N7qzZRp7iv7B18Q0k4hjFU427OpZj-R8MmLq" && \
		tar xvzf toxic.tar.gz

env:
	pip install --user -r requirements.txt
	python -m spacy download en_core_web_sm
