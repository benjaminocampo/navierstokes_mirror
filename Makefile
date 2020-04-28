# TODO: Try -ffast-math
CC=cc
override CFLAGS:=-std=c99 -Wall -Wextra -Werror -Wshadow -Wno-unused-parameter $(CFLAGS)
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o

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

-include .depend

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.s: %.c
	$(CC) $(CFLAGS) $^ -S -masm=intel

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o .depend solver.s *~
