all: proj2

#Compiladores
CC=gcc
CXX=g++

FLAGS= -O3 -Wall

#Bibliotecas
GFTLIB   = -L$(GFT_DIR)/lib -lgft
GFTFLAGS = -I$(GFT_DIR)/include
GRAPH  = -I/home/cadu/Dropbox/MAC5915/projeto2/graphcut/include
GC = -L/home/cadu/Dropbox/MAC5915/projeto2/graphcut/lib -lmaxflow


#Rules
libgft:
	$(MAKE) -C $(GFT_DIR)

libmaxflow:
	$(MAKE) -C /home/cadu/Dropbox/MAC5915/projeto2/graphcut

proj2: proj2.1.cpp libgft libmaxflow
	$(CXX) $(FLAGS) $(GFTFLAGS) $(GRAPH) \
	proj2.1.cpp -o proj2 -lm -lz $(GFTLIB) $(GC)

clean:
	$(RM) *~ *.o proj2
