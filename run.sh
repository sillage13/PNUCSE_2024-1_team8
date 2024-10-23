cd ligand_search
sudo docker-compose up -d
echo "Waiting for the server to be ready..."
while ! curl -s http://localhost:8000 > /dev/null; do
	echo "Server not ready yet..."
	sleep 10
done

echo "Server is up, opening brower..."
xdg-open http://localhost:8000/

sudo docker-compose logs -f
