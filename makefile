JACCARD_MAIN=src/main.cpp
JACCARD_TARGET=jacc.out

TARGETS=$(JACCARD_TARGET)

CXX=g++
CXX_FLAGS= -g -Wno-narrowing -lgomp -fopenmp
POP_FLAGS= -lpoplar

MKDIR=mkdir -p
EXECS_LOC=execs

all: dir $(TARGETS)
dir:
	$(MKDIR) $(EXECS_LOC)

$(JACCARD_TARGET):
	$(CXX) $(CXX_FLAGS) $(JACCARD_MAIN) $(POP_FLAGS) -o $(EXECS_LOC)/$@

.PHONY: clean
clean:
	@rm -rfv $(EXECS_LOC)/$(JACCARD_TARGET)
