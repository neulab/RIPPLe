import requests
import time
import jupyter_slack

while True:
	url = 'http://169.254.169.254/latest/meta-data/spot/termination-time'
	print("Polling for termination")
	r = requests.get(url)
	if r.status_code < 400:
		jupyter_slack.notify("Spot instance scheduled for termination!")
		break
	time.sleep(5)
