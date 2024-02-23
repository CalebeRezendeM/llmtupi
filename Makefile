lint:
	echo "lint not implemented yet"

push:
	make lint
	git add .
	git commit
	git push origin HEAD

