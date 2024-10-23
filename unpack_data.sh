if ! dpkg -l | grep -q p7zip-full; then
	echo "p7zip-full is not installed. Installing..."
	sudo apt update
	sudo apt install p7zip-full
fi

cd 2M
7z x data.7z.001
rm *.7z.*
