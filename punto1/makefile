all:
	g++ -fopenmp openmp.cpp -o openmp.x -std=c++11
	./openmp.x
	python3 graph.py

clean:
	rm *.x
	rm *.dat
	rm *.png