# TODO: Try -ffast-math
CC=cc
ISPC=ispc
ISPCFLAGS=--target=sse4
override CFLAGS:=-std=c99 -Wall -Wextra -Werror -Wshadow -Wno-unused-parameter $(CFLAGS)
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o solver_ispc.o

.PHONY: clean
all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

# make headless CFLAGS='-Ofast -march=native -floop-nest-optimize -funroll-loops -flto -g'
headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

asm: solver.o
	$(CC) $(CFLAGS) -fno-asynchronous-unwind-tables -fno-exceptions -S solver.c

runperf: headless
	sudo perf stat \
	-e \
	cache-references,\
	cache-misses,\
	L1-dcache-stores,\
	L1-dcache-store-misses,\
	LLC-stores,\
	LLC-store-misses,\
	page-faults,\
	cycles,\
	instructions,\
	branches,\
	branch-misses \
	-ddd \
	./headless

.depend: *.[ch]
	$(CC) -MM $(SOURCES) > .depend

solver_ispc.o: 
	$(ISPC) solver.ispc -o solver_ispc.o $(ISPCFLAGS)

-include .depend

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o .depend solver.s *~
