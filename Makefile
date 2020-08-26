install-dependencies:
	apt install -y python3-pip python3-dev python3-numpy libsystemd-dev
	pip3 install -r requirements.txt

install-service:
	cp watchtower-cv.service /etc/systemd/system/
	systemctl daemon-reload
	systemctl enable watchtower-cv.service

